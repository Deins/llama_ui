const std = @import("std");
const zt = @import("zt");
const ig = @import("ig.zig");
const nfd = @import("nfd");
const builtin = @import("builtin");
const llama = @import("llama");
const slog = std.log.scoped(.main);
const TokUi = @import("tokenizer_ui.zig");
const Atomic = std.atomic.Atomic;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Loader responsible of loading model on a separate thread
pub const Loader = struct {
    const Self = @This();
    pub const State = enum(u32) { failed, loading, finished };

    state: Atomic(State) = .{ .value = .finished },
    result: Atomic(?*llama.Model) = .{ .value = null },
    cancel: Atomic(bool) = .{ .value = false },
    progress: f32 = 0,
    load_time: u64 = 0,

    pub const reset = deinit;

    pub fn deinit(self: *Self) void {
        if (self.result.swap(null, .AcqRel)) |m| m.deinit();
    }

    fn progressCB(p: f32, ctx: ?*anyopaque) callconv(.C) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        @atomicStore(f32, &self.progress, p, .Release);
        return !self.cancel.load(.Acquire);
    }

    fn doLoad(self: *Self, params_: llama.Model.Params, path: [:0]const u8) void {
        defer gpa.allocator().free(path);
        std.debug.assert(self.state.load(.Acquire) == .loading);
        _ = progressCB(0, @ptrCast(self));
        var params = params_;
        params.progress_callback = &Self.progressCB;
        params.progress_callback_user_data = self;

        var timer = std.time.Timer.start() catch unreachable;
        const model = llama.Model.initFromFile(path, params) catch {
            if (self.cancel.load(.Acquire)) slog.err("Model load stopped - canceled. {s}", .{path}) else slog.err("Failed to loade model: '{s}'", .{path});
            _ = progressCB(-1, @ptrCast(self));
            self.state.store(.failed, .Release);
            return;
        };
        self.load_time = timer.read();

        self.result.store(model, .SeqCst);
        self.state.store(.finished, .SeqCst);
        slog.info("model loaded: '{s}' in ", .{std.fmt.fmtDuration(self.load_time)});
    }
};

pub const ModelRuntime = struct {
    pub const State = enum {
        unloaded,
        loading,
        ready,
        unloading,
        generating,
    };

    pub const WorkerState = enum(u8) { working, stopping, stopped };

    alloc: std.mem.Allocator = gpa.allocator(),
    state: State = .unloaded,
    target_state: ?State = null,
    model_path: ?[:0]const u8 = null,
    mparams: llama.Model.Params = .{},
    cparams: llama.Context.Params = .{},
    loader: Loader = .{},
    model: ?*llama.Model = null,
    model_ctx: ?*llama.Context = null,
    tok_ui: TokUi,
    history_tokens: usize = 0,
    history: std.ArrayList(u8),
    role: std.ArrayList(u8),
    input: std.ArrayList(u8),
    input_special: bool = true,
    input_templated: bool = true,
    gen_state: Atomic(WorkerState) = .{ .value = .stopped },
    prompt: ?llama.Prompt = null,

    pub fn init(alloc: std.mem.Allocator) !@This() {
        var role = std.ArrayList(u8).init(alloc);
        try role.appendSlice("system");
        return .{
            .alloc = alloc,
            .mparams = llama.Model.defaultParams(),
            .cparams = llama.Context.defaultParams(),
            .tok_ui = try TokUi.init(alloc),
            .history = std.ArrayList(u8).init(alloc),
            .input = std.ArrayList(u8).init(alloc),
            .role = role,
        };
    }

    pub fn deinit(self: *@This()) void {
        if (self.gen_state.swap(.stopping, .Release) != .stopped) {
            while (self.gen_state.load(.Monotonic) != .stopped) std.Thread.yield() catch std.atomic.spinLoopHint();
        }
        self.history.deinit();
        self.input.deinit();
        self.role.deinit();
        self.tok_ui.deinit();
        self.loader.deinit();
        if (self.prompt) |*p| p.deinit();
        if (self.model_ctx) |mc| mc.deinit();
        if (self.model) |m| m.deinit();
        if (self.model_path) |pth| self.alloc.free(pth);
    }

    pub fn updateState(self: *@This()) !void {
        switch (self.state) {
            .unloaded => {
                if (self.target_state != null) try self.switchStateTo(.loading);
            },
            .loading => {
                const lstate = self.loader.state.load(.SeqCst);
                self.model = self.loader.result.swap(null, .SeqCst);
                var done = false;
                if (self.model) |model| {
                    self.model_ctx = try llama.Context.initWithModel(model, self.cparams);
                    self.prompt = try llama.Prompt.init(gpa.allocator(), .{
                        .model = self.model.?,
                        .ctx = self.model_ctx.?,
                        .batch_size = self.cparams.n_batch,
                    });
                    done = true;
                } else if (lstate != .loading) done = true;
                if (done) try self.switchStateTo(.ready);
            },
            .ready => {
                if (self.target_state) |ts| switch (ts) {
                    .loading, .unloading, .unloaded => try self.switchStateTo(.unloading),
                    .generating => try self.switchStateTo(.generating),
                    .ready => {},
                };
            },
            .unloading => {
                const model = self.loader.result.swap(null, .Acquire);
                var done = self.loader.state.load(.Acquire) != .loading;
                if (model) |m| {
                    m.deinit();
                    done = true;
                }
                if (done) try self.switchStateTo(.unloaded);
            },
            .generating => {
                const prompt = &self.prompt.?;
                prompt.tokens_mutex.lock();
                defer prompt.tokens_mutex.unlock();
                if (prompt.tokens.items.len != self.history_tokens) {
                    self.history.clearRetainingCapacity();
                    var dtok = llama.Detokenizer{ .data = self.history };
                    defer self.history = dtok.data;

                    if (self.prompt) |p| {
                        for (p.tokens.items) |tok| _ = try dtok.detokenizeWithSpecial(self.model.?, tok);
                    }
                }
                if (self.gen_state.load(.Acquire) == .stopped) try self.switchStateTo(.ready);
            },
        }
    }

    pub fn switchStateTo(self: *@This(), state: State) !void {
        std.log.debug("switchStateTo {} from {}", .{ state, self.state });
        if (state == self.target_state) self.target_state = null;
        switch (state) {
            .loading => {
                std.debug.assert(self.state == .unloaded);

                if (state == self.state) {
                    try self.switchStateTo(.unloading);
                    return;
                }
                if (self.model_path) |path| {
                    self.loader.state.store(.loading, .Release);
                    self.loader.cancel.store(false, .Release);
                    self.loader.progress = 0;
                    const thread = try std.Thread.spawn(.{}, Loader.doLoad, .{ &self.loader, self.mparams, try self.alloc.dupeZ(u8, path) });
                    thread.setName("LoaderThread") catch {};
                } else {
                    slog.info("Can't load - missing model path!", .{});
                    self.state = .unloaded;
                    return;
                }
            },
            .generating => {
                std.debug.assert(self.state == .ready);
                const prev_gen_state = self.gen_state.swap(.working, .Acquire);
                std.debug.assert(prev_gen_state == .stopped); // invoking generatoin while previous still has not finished is not allowed
                const thread = try std.Thread.spawn(.{}, gen, .{self});
                defer thread.detach();
                thread.setName("GeneratorThread") catch {};
            },
            .unloading => {
                std.debug.assert(self.state == .ready or self.state == .loading);
                self.loader.cancel.store(true, .Release);
                if (self.loader.result.swap(null, .AcqRel)) |m| m.deinit();
                if (self.prompt) |*p| {
                    // TODO: keep text, reembed after reloading?
                    p.deinit();
                    self.prompt = null;
                }
                if (self.model_ctx) |mc| {
                    mc.deinit();
                    self.model_ctx = null;
                }
                if (self.model) |m| {
                    m.deinit();
                    self.model = null;
                }
            },
            .unloaded => {
                if (self.loader.result.swap(null, .AcqRel)) |m| m.deinit();
                std.debug.assert(self.state == .unloading);
            },
            .ready => {
                std.debug.assert(self.state == .loading or self.state == .generating);
                std.debug.assert(self.model != null);
                std.debug.assert(self.model_ctx != null);
                std.debug.assert(self.prompt != null);
            },
        }
        self.state = state;
    }

    pub fn reload(self: *@This()) !void {
        self.target_state = null;
        if (self.state == .loading) try self.switchStateTo(.unloading);
        self.target_state = .loading;
    }

    fn gen(self: *@This()) void {
        if (self.prompt) |*prompt| {
            const model = self.model.?;
            const eos = model.tokenEos();
            while (self.gen_state.load(.Acquire) == .working) {
                const tok = prompt.generateAppendOne() catch unreachable; // TODO: handle OOM
                if (tok == eos) break;
            }
        } else unreachable;
        self.gen_state.store(.stopped, .Release);
    }
};

/// AppData will be available through the context anywhere.
const ContextData = struct {
    /// just counts rendered frames
    frame: usize = 0,
    mrt: ?ModelRuntime = null,
};

const LlamaApp = zt.App(ContextData);

var show_im_demo: bool = false;
pub fn renderTick(context: *LlamaApp.Context) !void {
    const mrt: *ModelRuntime = &context.data.mrt.?; // context data
    const first_time = context.data.frame == 0;
    try mrt.updateState();

    { // Create root window filling entire screen
        const viewport = ig.igGetMainViewport();
        ig.igSetNextWindowPos(viewport.*.WorkPos, 1, .{});
        ig.igSetNextWindowSize(viewport.*.WorkSize, 1);
        ig.igSetNextWindowViewport(viewport.*.ID);
        ig.igPushStyleVar_Float(ig.ImGuiStyleVar_WindowRounding, 0.0);
        ig.igPushStyleVar_Float(ig.ImGuiStyleVar_WindowBorderSize, 0.0);
        errdefer ig.igPopStyleVar(2);
        // ig.igPushItemWidth(ig.igGetWindowWidth() * 0.5);
        // defer ig.igPopItemWidth();
        var flags = ig.ImGuiWindowFlags_NoResize | ig.ImGuiWindowFlags_NoBackground | ig.ImGuiWindowFlags_NoMove | ig.ImGuiWindowFlags_NoCollapse;
        flags |= ig.ImGuiWindowFlags_NoBringToFrontOnFocus | ig.ImGuiWindowFlags_NoNavFocus | ig.ImGuiWindowFlags_NoDecoration | ig.ImGuiWindowFlags_NoTitleBar;
        _ = ig.igBegin("DockSpace", null, flags);
        defer ig.igEnd();
        ig.igPopStyleVar(2);

        const dockspace_flags = ig.ImGuiDockNodeFlags_PassthruCentralNode;
        const dockspace_id = ig.igGetID_Str("MyDockSpace");
        _ = ig.igDockSpace(dockspace_id, .{}, dockspace_flags, null);
        if (first_time) {
            // _ = ig.igDockBuilderRemoveNode(dockspace_id);
            // _ = ig.igDockBuilderAddNode(dockspace_id, dockspace_flags);
            // _ = ig.igDockBuilderSetNodeSize(dockspace_id, viewport.*.Size);
        }

        try configWindow(context);
        try promptWindow(context);

        if (ig.igBegin("Tokenizer tool", null, ig.ImGuiWindowFlags_NoFocusOnAppearing)) {
            if (mrt.model) |m| try mrt.tok_ui.render(m) else ig.text("Model not loaded.");
        }
        ig.igEnd();

        if (show_im_demo) ig.igShowDemoWindow(null);
    }
}

pub fn promptWindow(context: *LlamaApp.Context) !void {
    const mrt = &context.data.mrt.?;

    if (ig.igBegin("Prompt history", null, ig.ImGuiWindowFlags_None)) {
        ig.text("History");
        ig.igSameLine(0, -1);
        if (ig.igButton("clear", .{})) {
            if (mrt.prompt) |*p| p.tokens.clearRetainingCapacity();
            mrt.history.clearRetainingCapacity();
        }

        var tmp: [512]u8 = undefined;
        _ = tmp;
        ig.igPushItemWidth(-1);
        if (try ig.inputTextArrayList("##prompt_history", "", &mrt.history, ig.ImGuiInputTextFlags_ReadOnly | ig.ImGuiInputTextFlags_Multiline)) {}
        ig.igPopItemWidth();
    }
    ig.igEnd();

    if (ig.igBegin("Prompt input", null, ig.ImGuiWindowFlags_None)) {
        _ = ig.igCheckbox("special tokens", &mrt.input_special);
        ig.igSameLine(0, -1);
        _ = ig.igCheckbox("templated", &mrt.input_templated);
        if (mrt.input_templated) {
            ig.igSameLine(0, -1);
            ig.igPushItemWidth(150);
            defer ig.igPopItemWidth();
            _ = try ig.inputTextArrayList("role", "role", &mrt.role, ig.ImGuiInputTextFlags_None);
        }

        switch (mrt.state) {
            .generating => {
                const r = 16;
                var aspace: ig.ImVec2 = .{};
                ig.igGetContentRegionAvail(&aspace);
                ig.igSetCursorPosX(aspace.x / 2 - r);
                ig.spinner("##GeneratingSpinner", r, 6, 0xFFFFFFAA);
                context.setAnimationFrames(0.05);
                const w = 100;
                ig.igSetCursorPosX(aspace.x / 2 - w / 2);
                if (ig.igButton("cancel", .{ .x = w, .y = 32 })) mrt.gen_state.store(.stopping, .Monotonic);
            },
            .ready => {
                ig.igPushItemWidth(-1);
                defer ig.igPopItemWidth();
                const input = &mrt.input;
                try input.ensureTotalCapacity(input.items.len + 1);
                input.allocatedSlice()[input.items.len] = 0; // ensure 0 termination
                const flags = ig.ImGuiInputTextFlags_Multiline | ig.ImGuiInputTextFlags_EnterReturnsTrue;
                if (try ig.inputTextArrayList("##prompt_input", "Prompt input", input, flags)) {
                    if (mrt.prompt) |*prompt| {
                        try prompt.appendText(input.items, true);
                        try mrt.switchStateTo(.generating);
                        input.clearRetainingCapacity();
                    }
                }
            },
            else => ig.igText("Model has not been loaded"),
        }
    }
    ig.igEnd();

    //igText("Input");
}

pub fn configWindow(context: *LlamaApp.Context) !void {
    const mrt: *ModelRuntime = &context.data.mrt.?; // context data
    var tmp: [512]u8 = undefined;
    ig.igSetNextWindowSizeConstraints(.{ .x = 300, .y = 128 }, .{ .x = 9999999, .y = 9999999 }, null, null);
    if (ig.igBegin("Config", null, ig.ImGuiWindowFlags_NoFocusOnAppearing)) {
        ig.igPushItemWidth(200);
        defer ig.igPopItemWidth();

        ig.igSeparatorText("Model");
        ig.igAlignTextToFramePadding();
        ig.text("model:");
        ig.igSameLine(0, 8);

        if (ig.igButton("browse", .{})) {
            var out_path: ?[*:0]nfd.nfdchar_t = null;
            const filter = [_]nfd.nfdfilteritem_t{.{ .name = "Model file", .spec = "gguf,bin" }};
            const res = nfd.NFD_OpenDialogU8(&out_path, &filter, filter.len, null);
            defer if (out_path != null) nfd.NFD_FreePathU8(out_path);
            if (res == nfd.NFD_OKAY) {
                if (mrt.model_path) |path| mrt.alloc.free(path);
                mrt.model_path = try mrt.alloc.dupeZ(u8, std.mem.span(out_path.?));
                try mrt.reload();
            } else if (res == nfd.NFD_CANCEL) {} else std.log.err("File dialog returned error: '{s}' ({})", .{ nfd.NFD_GetError(), res });
        }
        ig.igSameLine(0, 8);
        var model: [:0]const u8 = mrt.model_path orelse "<none>";
        if (std.mem.lastIndexOf(u8, model, if (builtin.target.os.tag == .windows) "\\" else "//")) |idx| model = model[idx + 1 ..];
        ig.text(model);
        const start_y = ig.igGetCursorPosY();
        ig.igAlignTextToFramePadding();
        switch (mrt.state) {
            .loading => {
                const p = mrt.loader.progress;
                if (p <= 0) {
                    // due to mmap & stuff initial progress doesn't update which takes longest, better display spinner/text
                    ig.text(loadingText("Loading", context.time.lifetime));
                    ig.igSameLine(90, 0);
                    ig.spinner("##loading_spinner", ig.igGetTextLineHeight() * 0.5, 2, 0xFFFFFFAA);
                    context.setAnimationFrames(0.05);
                } else {
                    ig.igSetCursorPosX(64);
                    var wa: ig.ImVec2 = .{};
                    ig.igGetContentRegionAvail(&wa);
                    ig.igProgressBar(p, .{ .x = wa.x }, null);
                    context.setAnimationFrames(0.05);
                }
            },
            .unloading => {
                ig.igTextColored(.{ .x = 1.0, .y = 1.0, .z = 0.0, .w = 1.0 }, loadingText("Unloading", context.time.lifetime));
                ig.igSameLine(100, 0);
                context.setAnimationFrames(0.05);
                ig.spinner("##loading_spinner", ig.igGetTextLineHeight() * 0.45, 2, 0xFFFFFFAA);
            },
            .unloaded => {
                if (mrt.loader.state.load(.Acquire) == .failed) {
                    ig.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "Model failed to load!");
                } else {
                    ig.igTextColored(.{ .x = 1.0, .y = 1.0, .z = 0.0, .w = 1.0 }, "Model not loaded!");
                    if (mrt.model_path) |path| {
                        _ = path;
                        ig.igSameLine(0, 8);
                        var sa: ig.ImVec2 = .{};
                        ig.igGetContentRegionAvail(&sa);
                        if (ig.igButton("RELOAD", .{ .x = sa.x })) try mrt.reload();
                    }
                }
            },
            .ready => {
                ig.igTextColored(.{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 }, @ptrCast(try std.fmt.bufPrint(tmp[0..], "Model loaded! ({})\x00", .{std.fmt.fmtDuration(mrt.loader.load_time)})));
                if (mrt.state != .loading) {} else {}
            },
            .generating => {
                ig.igTextColored(.{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 }, @ptrCast(try std.fmt.bufPrint(tmp[0..], "Generating\x00", .{})));
                ig.igSameLine(0, -1);
                if (ig.igButton("cancel", .{})) {
                    mrt.gen_state.store(.stopping, .Release);
                }
            },
        }
        ig.igSetCursorPosY(start_y + ig.igGetStyle().*.FramePadding.y * 2 + ig.GImGui.*.FontSize + 4);

        if (ig.igInputInt("gpu_layers", &mrt.mparams.n_gpu_layers, 4, 32, ig.ImGuiSliderFlags_None)) {
            mrt.mparams.n_gpu_layers = @max(0, mrt.mparams.n_gpu_layers);
            try mrt.reload();
        }

        if (ig.igCheckbox("use mmap", &mrt.mparams.use_mmap)) {} //mrt.invalidate();
        ig.igSameLine(0, 16);
        if (ig.igCheckbox("use mlock", &mrt.mparams.use_mlock)) try mrt.reload();

        if (ig.igCollapsingHeader_BoolPtr("Model info", null, ig.ImGuiTreeNodeFlags_None)) {
            if (mrt.model) |m| {
                ig.labelFmt("Description", m.desc(&tmp), "{s}");
                ig.label("Vocab lenght", m.nVocab());
                ig.label("Trained context lenght", m.nCtxTrain());
                ig.label("Embeding lenght", m.nEmbd());
                ig.label("RoPE frequency scaling factor", m.ropeFreqScaleTrain());

                ig.igSeparator();
                const kcount = m.metaCount();
                for (0..kcount) |kidx| {
                    var tmp2: [64]u8 = undefined;
                    const key = m.metaKeyByIndex(kidx, &tmp2) orelse "ERROR";
                    const val = m.metaValStrByIndex(kidx, &tmp) orelse "ERROR";
                    ig.labelFmt(key, val, "{s}");
                }
            } else ig.text("Model has not been loaded");
        }

        if (ig.igCollapsingHeader_BoolPtr("Context config", null, ig.ImGuiTreeNodeFlags_DefaultOpen)) {
            const p = &mrt.cparams;
            var mod = false;
            _ = (ig.inputScalar(u32, "seed", &p.seed));
            if (ig.inputScalar(u32, "n_ctx", &p.n_ctx)) mod = true;
            if (ig.inputScalar(u32, "n_batch", &p.n_batch)) mod = true;
            _ = (ig.inputScalar(u32, "n_threads", &p.n_threads));
            _ = (ig.inputScalar(u32, "n_threads_batch", &p.n_threads_batch));
            // todo: llama_rope_scaling_type
            if (ig.inputScalar(f32, "rope_freq_base", &p.rope_freq_base)) mod = true;
            if (ig.inputScalar(f32, "rope_freq_scale", &p.rope_freq_scale)) mod = true;
            if (ig.inputScalar(f32, "yarn_ext_factor", &p.yarn_ext_factor)) mod = true;
            if (ig.inputScalar(f32, "yarn_attn_factor", &p.yarn_attn_factor)) mod = true;
            if (ig.inputScalar(f32, "yarn_beta_fast", &p.yarn_beta_fast)) mod = true;
            if (ig.inputScalar(f32, "yarn_beta_slow", &p.yarn_beta_slow)) mod = true;
            if (ig.inputScalar(u32, "yarn_orig_ctx", &p.yarn_orig_ctx)) mod = true;
            //        enum ggml_type type_k; // data type for K cache
            //        enum ggml_type type_v; // data type for V cache
            // bool mul_mat_q;   // if true, use experimental mul_mat_q kernels (DEPRECATED - always true)
            // bool logits_all;  // the llama_eval() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
            _ = ig.igCheckbox("embedding mode only", &p.embedding);
            // bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
            if (ig.igCheckbox("offload_kqv", &p.offload_kqv)) mod = true;

            // TODO: if (mod) something something
        }
        if (ig.igCollapsingHeader_BoolPtr("App config", null, ig.ImGuiTreeNodeFlags_None)) {
            ig.igSeparatorText("debug");
            _ = ig.igCheckbox("imgui demo", &show_im_demo);

            ig.text("state: ");
            ig.igSameLine(0, -1);
            ig.text(@tagName(mrt.state));

            ig.text("target state: ");
            ig.igSameLine(0, -1);
            ig.text(if (mrt.target_state) |ts| @tagName(ts) else "<none>");
        }
    }
    ig.igEnd();
}

pub fn main() !void {
    const is_debug = (builtin.mode == .Debug);
    const do_cleanup = is_debug; // speed up program exit, by skipping cleanup, works as long as all threads are deatached
    defer if (gpa.deinit() == .leak) @panic("Memory leak detected!");
    { // llama
        llama.Backend.init(false);
        defer llama.Backend.deinit();
        slog.info("llama_system_info: {s}", .{llama.printSystemInfo()});
        llama.logSet(llama.utils.scopedLog, null);
    }

    var context = try LlamaApp.begin(gpa.allocator());
    defer if (do_cleanup) context.deinit();

    context.setWindowSize(1280, 720);
    context.setWindowTitle("llama.cpp.zig ui");
    context.setVsync(true);
    context.data.mrt = try ModelRuntime.init(gpa.allocator());
    defer if (do_cleanup) {
        context.data.mrt.?.deinit();
        context.data.mrt = null;
    };

    const io = ig.igGetIO();
    {
        const font_cfg = ig.ImFontConfig_ImFontConfig();
        defer ig.ImFontConfig_destroy(font_cfg);
        font_cfg.*.PixelSnapH = true;
        font_cfg.*.OversampleH = 2;
        font_cfg.*.OversampleV = 2;
        const new_font: *ig.ImFont = ig.ImFontAtlas_AddFontFromFileTTF(io.*.Fonts, zt.path("ttf/Hack-Regular.ttf").ptr, 14.0, font_cfg, ig.ImFontAtlas_GetGlyphRangesDefault(io.*.Fonts));
        context.rebuildFont();
        io.*.FontDefault = new_font;
    }
    io.*.ConfigWindowsResizeFromEdges = true; // don't work for some reason
    //igio.*.ConfigWindowsMoveFromTitleBarOnly = true;

    context.settings.energySaving = true;
    while (context.open) {
        context.beginFrame();

        try renderTick(context);

        context.endFrame();
        context.data.frame += 1;
        // some of our logic runs in same thread as UI, do modified energy saving with timeout of 6hz to avoid not updating at all
        // if (context._updateSeconds > 0.0) context._updateSeconds -= context.time.dt else @import("glfw").waitEventsTimeout(1.0 / 6.0);
        // context.settings.energySaving = false;
    }
}

pub const std_options = struct {
    pub const log_level = std.log.Level.debug;
    pub const log_scope_levels: []const std.log.ScopeLevel = &.{.{ .scope = .llama_cpp, .level = .info }};
};

pub fn loadingText(comptime prefix: [:0]const u8, t: f32) [:0]const u8 {
    const texts = [_][:0]const u8{ prefix ++ ".   ", prefix ++ "..  ", prefix ++ "..." };
    return texts[@as(usize, @intFromFloat(t * 2)) % texts.len];
}
