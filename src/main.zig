const std = @import("std");
const zt = @import("zt");
const ig = @import("imgui");
const nfd = @import("nfd");
const builtin = @import("builtin");
const llama = @import("llama");
const slog = std.log.scoped(.main);
const TokUi = @import("tokenizer_ui.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Loader responsible of loading model on a separate thread
pub const Loader = struct {
    const Self = @This();
    pub const State = enum { failed, cancel, loading, finished };

    state: State = .cancel,
    progress: f32 = 0,
    thread: ?std.Thread = null,
    result: ?*llama.Model = null,
    load_time: u64 = 0,
    // /// loader thread dumps all its data as soon as it is able to, instead of writing deatach
    // readonly_self_destruct: bool = false,

    pub const reset = deinit;

    pub fn deinit(self: *Self) void {
        if (self.thread) |t| {
            slog.info("loader - joining", .{});
            defer slog.info("loader - joined", .{});
            switch (self.getState()) {
                .failed, .cancel, .finished => {},
                .loading => @atomicStore(State, &self.state, .cancel, .Release),
            }
            t.join();
        }
        if (@atomicRmw(?*llama.Model, &self.result, .Xchg, null, .SeqCst)) |m| m.deinit();
        self.thread = null;
    }

    pub fn getProgress(self: *Self) f32 {
        return @atomicLoad(f32, self.progress, .Acquire);
    }

    pub fn getState(self: *Self) State {
        return @atomicLoad(State, &self.state, .Acquire);
    }

    fn progressCB(p: f32, ctx: ?*anyopaque) callconv(.C) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        @atomicStore(f32, &self.progress, p, .Release);
        return @atomicLoad(State, &self.state, .Acquire) == .loading;
    }

    fn doLoad(self: *Self, params: llama.ModelParams, path: [:0]const u8) void {
        var timer = std.time.Timer.start() catch unreachable;
        defer gpa.allocator().free(path);
        @atomicStore(State, &self.state, .loading, .Release);
        _ = progressCB(0, @ptrCast(self));
        const model = llama.Model.initFromFile(path, params) catch {
            _ = progressCB(-1, @ptrCast(self));
            @atomicStore(State, &self.state, .failed, .Release);
            slog.err("Failed to loade model: '{s}'", .{path});
            return;
        };
        @atomicStore(?*llama.Model, &self.result, model, .Release);
        @atomicStore(State, &self.state, .finished, .Release);
        self.load_time = timer.read();
        slog.info("model loaded: '{s}' in ", .{std.fmt.fmtDuration(self.load_time)});
    }
};

/// AppData will be available through the context anywhere.
const LlamaAppData = struct {
    alloc: std.mem.Allocator = gpa.allocator(),
    model_path: ?[:0]const u8 = null,
    mparams: llama.ModelParams = .{},
    cparams: llama.ContextParams = .{},
    loader: Loader = .{},
    model: ?*llama.Model = null,
    model_ctx: ?*llama.Context = null,
    prompt: ?llama.Prompt = null,
    /// just counts rendered frames
    frame: usize = 0,
    tok_ui: TokUi = undefined,

    pub fn init(alloc: std.mem.Allocator) !@This() {
        return .{
            .alloc = alloc,
            .mparams = llama.modelDefaultParams(),
            .cparams = llama.contextDefaultParams(),
            .tok_ui = try TokUi.init(alloc),
        };
    }

    pub fn deinit(self: *@This()) void {
        if (self.prompt) |*p| p.deinit();
        if (self.model_ctx) |mc| mc.deinit();
        if (self.model) |m| m.deinit();
        if (self.model_path) |pth| self.alloc.free(pth);
        self.tok_ui.deinit();
        self.loader.deinit();
    }

    /// due to model or other changes model needs reload
    pub fn invalidate(self: *@This()) void {
        self.reload(null) catch unreachable;
    }

    /// must be called during loading to grab loaded model and prepare context
    pub fn grabAndPrepModel(self: *@This()) !void {
        std.debug.assert(self.model == null);
        if (self.loader.getState() == .finished) {
            self.model = @atomicRmw(?*llama.Model, &self.loader.result, .Xchg, null, .SeqCst).?;
            self.loader.result = null;
            self.loader.deinit();

            self.model_ctx = try llama.Context.initWithModel(self.model.?, self.cparams);
            self.prompt = try llama.Prompt.init(gpa.allocator(), .{
                .model = self.model.?,
                .ctx = self.model_ctx.?,
                .batch_size = self.cparams.n_batch,
            });
        }
    }

    /// load/reload model from path
    /// @param path path to model or null to unload current
    pub fn reload(self: *@This(), path: ?[:0]const u8) !void {
        self.loader.deinit();
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
        if (path == null) {
            self.loader.state = .cancel;
            slog.info("unloaded model", .{});
            return;
        }
        slog.info("reloading model: '{s}'", .{self.model_path.?});

        @atomicStore(Loader.State, &self.loader.state, .loading, .Release);
        self.loader.progress = 0;

        var mparams = self.mparams;
        mparams.progress_callback = &Loader.progressCB;
        mparams.progress_callback_user_data = &self.loader;
        self.loader.thread = try std.Thread.spawn(.{}, Loader.doLoad, .{ &self.loader, mparams, try gpa.allocator().dupeZ(u8, path.?) });
        self.loader.thread.?.setName("LoaderThread") catch unreachable;
    }
};

const LlamaApp = zt.App(LlamaAppData);

pub fn renderTick(context: *LlamaApp.Context) !void {
    const cdata: *LlamaAppData = &context.data; // context data
    const first_time = context.data.frame == 0;
    if (cdata.model == null) try cdata.grabAndPrepModel();

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

        if (ig.igBegin("Tokenizer tool", null, ig.ImGuiWindowFlags_None)) {
            if (cdata.model) |m| try cdata.tok_ui.render(m) else igText("Model not loaded.");
        }
        ig.igEnd();

        ig.igShowDemoWindow(null);
    }
}

var prompt_text: [512:0]u8 = undefined;
pub fn promptWindow(context: *LlamaApp.Context) !void {
    const cdata = &context.data;
    // TODO: temporary, we should not detokenize every frame
    var dtok = llama.Detokenizer.init(gpa.allocator());
    defer dtok.deinit();
    if (cdata.prompt) |p| {
        for (p.tokens.items) |tok| _ = try dtok.detokenizeWithSpecial(cdata.model.?, tok);
    }

    if (ig.igBegin("Prompt history", null, ig.ImGuiWindowFlags_None)) {
        igText("History");
        var tmp: [512]u8 = undefined;
        _ = tmp;
        ig.igPushItemWidth(-1);
        _ = ig.igInputTextMultiline("prompt_history", &prompt_text, prompt_text.len, .{}, ig.ImGuiInputTextFlags_ReadOnly, null, null);
        ig.igPopItemWidth();
    }
    ig.igEnd();

    if (ig.igBegin("Prompt input", null, ig.ImGuiWindowFlags_None)) {
        var tmp: [512]u8 = undefined;
        _ = tmp;
        ig.igPushItemWidth(-1);
        _ = ig.igInputTextMultiline("prompt_history", &prompt_text, prompt_text.len, .{}, ig.ImGuiInputTextFlags_ReadOnly, null, null);
        ig.igPopItemWidth();
    }
    ig.igEnd();
    //igText("Input");
}

pub fn configWindow(context: *LlamaApp.Context) !void {
    const cdata: *LlamaAppData = &context.data; // context data
    var tmp: [512]u8 = undefined;
    ig.igSetNextWindowSizeConstraints(.{ .x = 300, .y = 128 }, .{ .x = 9999999, .y = 9999999 }, null, null);
    if (ig.igBegin("Config", null, ig.ImGuiWindowFlags_NoFocusOnAppearing)) {
        ig.igPushItemWidth(200);
        defer ig.igPopItemWidth();

        ig.igSeparatorText("Model");
        igText("model:");
        ig.igSameLine(0, 8);

        if (ig.igButton("browse", .{})) {
            var out_path: ?[*:0]nfd.nfdchar_t = null;
            const filter = [_]nfd.nfdfilteritem_t{.{ .name = "Model file", .spec = "gguf,bin" }};
            const res = nfd.NFD_OpenDialogU8(&out_path, &filter, filter.len, null);
            defer if (out_path != null) nfd.NFD_FreePathU8(out_path);
            if (res == nfd.NFD_OKAY) {
                if (context.data.model_path) |path| gpa.allocator().free(path);
                cdata.model_path = try gpa.allocator().dupeZ(u8, std.mem.span(out_path.?));
                try cdata.reload(cdata.model_path.?);
            } else if (res == nfd.NFD_CANCEL) {} else std.log.warn("File dialog error: {}", .{res});
        }
        ig.igSameLine(0, 8);
        var model: [:0]const u8 = cdata.model_path orelse "<none>";
        if (std.mem.lastIndexOf(u8, model, if (builtin.target.os.tag == .windows) "\\" else "//")) |idx| model = model[idx + 1 ..];
        igText(model);
        const start_y = ig.igGetCursorPosY();
        switch (cdata.loader.getState()) {
            .loading => {
                const p = cdata.loader.progress;
                if (p <= 0) {
                    // due to mmap & stuff initial progress doesn't update which takes longest, better display spinner/text
                    const texts = [_][:0]const u8{ "Loading.   ", "Loading..  ", "Loading..." };
                    const text = texts[@as(usize, @intFromFloat(context.time.lifetime * 2)) % texts.len];
                    igText(text);
                    ig.igSameLine(90, 0);
                    igSpinner("loading", ig.igGetTextLineHeight() * 0.45, 2, 0xFFFFFFAA);
                    context.setAnimationFrames(0.05);
                } else {
                    ig.igSetCursorPosX(64);
                    var wa: ig.ImVec2 = .{};
                    ig.igGetContentRegionAvail(&wa);
                    ig.igProgressBar(p, .{ .x = wa.x }, null);
                    context.setAnimationFrames(0.05);
                }
            },
            .cancel => {
                ig.igTextColored(.{ .x = 1.0, .y = 1.0, .z = 0.0, .w = 1.0 }, "Model not loaded!");
                if (cdata.model_path) |path| {
                    ig.igSameLine(0, 8);
                    var sa: ig.ImVec2 = .{};
                    ig.igGetContentRegionAvail(&sa);
                    if (ig.igButton("RELOAD", .{ .x = sa.x })) try cdata.reload(path);
                }
            },
            .failed => ig.igTextColored(.{ .x = 1.0, .y = 0.0, .z = 0.0, .w = 1.0 }, "Model failed to load!"),
            .finished => {
                ig.igTextColored(.{ .x = 0.0, .y = 1.0, .z = 0.0, .w = 1.0 }, @ptrCast(try std.fmt.bufPrint(tmp[0..], "Model loaded! ({})\x00", .{std.fmt.fmtDuration(cdata.loader.load_time)})));
            },
        }
        ig.igSetCursorPosY(start_y + ig.igGetStyle().*.FramePadding.y * 2 + ig.GImGui.*.FontSize + 2);

        if (ig.igInputInt("gpu_layers", &cdata.mparams.n_gpu_layers, 4, 32, ig.ImGuiSliderFlags_None)) {
            cdata.mparams.n_gpu_layers = @max(0, cdata.mparams.n_gpu_layers);
            cdata.invalidate();
        }

        if (ig.igCheckbox("use mmap", &cdata.mparams.use_mmap)) cdata.invalidate();
        ig.igSameLine(0, 16);
        if (ig.igCheckbox("use mlock", &cdata.mparams.use_mlock)) cdata.invalidate();

        if (ig.igCollapsingHeader_BoolPtr("Model info", null, ig.ImGuiTreeNodeFlags_None)) {
            if (cdata.model) |m| {
                igLabelFmt("Description", m.desc(&tmp), "{s}");
                igLabel("Vocab lenght", m.nVocab());
                igLabel("Trained context lenght", m.nCtxTrain());
                igLabel("Embeding lenght", m.nEmbd());
                igLabel("RoPE frequency scaling factor", m.ropeFreqScaleTrain());

                ig.igSeparator();
                const kcount = m.metaCount();
                for (0..kcount) |kidx| {
                    var tmp2: [64]u8 = undefined;
                    const key = m.metaKeyByIndex(kidx, &tmp2) orelse "ERROR";
                    const val = m.metaValStrByIndex(kidx, &tmp) orelse "ERROR";
                    igLabelFmt(key, val, "{s}");
                }
            } else igText("Model has not been loaded");
        }

        if (ig.igCollapsingHeader_BoolPtr("Context config", null, ig.ImGuiTreeNodeFlags_DefaultOpen)) {
            const p = &cdata.cparams;
            var mod = false;
            _ = (igInputScalar(u32, "seed", &p.seed));
            if (igInputScalar(u32, "n_ctx", &p.n_ctx)) mod = true;
            if (igInputScalar(u32, "n_batch", &p.n_batch)) mod = true;
            _ = (igInputScalar(u32, "n_threads", &p.n_threads));
            _ = (igInputScalar(u32, "n_threads_batch", &p.n_threads_batch));
            // todo: llama_rope_scaling_type
            if (igInputScalar(f32, "rope_freq_base", &p.rope_freq_base)) mod = true;
            if (igInputScalar(f32, "rope_freq_scale", &p.rope_freq_scale)) mod = true;
            if (igInputScalar(f32, "yarn_ext_factor", &p.yarn_ext_factor)) mod = true;
            if (igInputScalar(f32, "yarn_attn_factor", &p.yarn_attn_factor)) mod = true;
            if (igInputScalar(f32, "yarn_beta_fast", &p.yarn_beta_fast)) mod = true;
            if (igInputScalar(f32, "yarn_beta_slow", &p.yarn_beta_slow)) mod = true;
            if (igInputScalar(u32, "yarn_orig_ctx", &p.yarn_orig_ctx)) mod = true;
            //        enum ggml_type type_k; // data type for K cache
            //        enum ggml_type type_v; // data type for V cache
            // bool mul_mat_q;   // if true, use experimental mul_mat_q kernels (DEPRECATED - always true)
            // bool logits_all;  // the llama_eval() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
            _ = ig.igCheckbox("embedding mode only", &p.embedding);
            // bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
            if (ig.igCheckbox("offload_kqv", &p.offload_kqv)) mod = true;

            // TODO: if (mod) something something
        }
    }
    ig.igEnd();
}

pub fn main() !void {
    const is_debug = (builtin.mode == .Debug);
    defer if (gpa.deinit() == .leak) @panic("Memory leak detected!");
    { // llama
        llama.Backend.init(false);
        defer llama.Backend.deinit();
        slog.info("llama_system_info: {s}", .{llama.printSystemInfo()});
        llama.logSet(llama.utils.scopedLog, null);
    }

    var context = try LlamaApp.begin(gpa.allocator());
    defer context.deinit();

    context.setWindowSize(1280, 720);
    context.setWindowTitle("llama.cpp.zig ui");
    context.setVsync(true);
    context.data = try LlamaAppData.init(gpa.allocator());
    defer {
        if (!is_debug) {
            if (context.data.loader.thread) |t| {
                t.detach(); // results & state doesn't matter on exit - not worth waiting for it to finish, deatach!
                context.data.loader.thread = null;
            }
        }
        context.data.deinit();
    }

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

//
// ImGui helpers

pub fn igLabelFmt(label: []const u8, v: anytype, comptime fmt: []const u8) void {
    var tmp: [511:0]u8 = undefined;
    var str = std.fmt.bufPrint(&tmp, fmt ++ "\x00", .{v}) catch blk: {
        std.debug.assert(false);
        break :blk tmp[0..];
    };
    ig.igLabelText(label.ptr, "%s", str.ptr);
    //ig.igTextEx(label.ptr, @ptrFromInt(@intFromPtr(label.ptr) + label.len), ig.ImGuiTextFlags_None);
    //ig.igSameLine(0, 32);
    //ig.igTextEx(@ptrCast(str), @ptrFromInt(@intFromPtr(str.ptr) + str.len - 1), ig.ImGuiTextFlags_None);
}

pub inline fn igLabel(label: []const u8, v: anytype) void {
    igLabelFmt(label, v, "{}");
}

pub inline fn igText(txt: [:0]const u8) void {
    ig.igTextUnformatted(txt.ptr, @ptrFromInt(@intFromPtr(txt.ptr) + txt.len));
}

pub fn igInputScalarFull(comptime T: type, label: [:0]const u8, val: *T, step_: ?T, step_fast_: ?T, format: [*c]const u8, flags: ig.ImGuiInputTextFlags) bool {
    const igt = switch (@typeInfo(T)) {
        .Int => |it| switch (it.bits) {
            8 => if (it.signedness == .signed) ig.ImGuiDataType_I8 else ig.ImGuiDataType_U8,
            16 => if (it.signedness == .signed) ig.ImGuiDataType_I16 else ig.ImGuiDataType_U16,
            32 => if (it.signedness == .signed) ig.ImGuiDataType_I32 else ig.ImGuiDataType_U32,
            64 => if (it.signedness == .signed) ig.ImGuiDataType_I64 else ig.ImGuiDataType_U64,
            else => unreachable,
        },
        .Float => |ft| switch (ft.bits) {
            32 => ig.ImGuiDataType_Float,
            64 => ig.ImGuiDataType_Double,
            else => unreachable,
        },
        else => unreachable,
    };
    const step: ?*const T = if (step_) |s| &s else null;
    const step_fast: ?*const T = if (step_fast_) |s| &s else null;
    return ig.igInputScalar(label.ptr, igt, @ptrCast(val), @ptrCast(step), @ptrCast(step_fast), format, flags);
}

pub fn igInputScalar(comptime T: type, label: [:0]const u8, val: *T) bool {
    return igInputScalarFull(T, label, val, null, null, null, ig.ImGuiInputFlags_None);
}

pub fn igSpinner(label: [:0]const u8, radius: f32, thickness: f32, color: ig.ImU32) void {
    //const window = ig.igGetCurrentWindow();
    // if (window.SkipItems)
    //     return;

    const style = ig.igGetStyle();
    const id = ig.igGetID_Str(label);

    var pos: ig.ImVec2 = .{};
    ig.igGetCursorScreenPos(&pos);
    var size: ig.ImVec2 = .{ .x = (radius) * 2, .y = (radius + style.*.FramePadding.y) * 2 };
    const bb: ig.ImRect = .{ .Min = pos, .Max = .{ .x = pos.x + size.x, .y = pos.y + size.y } };
    ig.igItemSize_Rect(bb, 0);
    if (!ig.igItemAdd(bb, id, null, ig.ImGuiItemFlags_None))
        return;

    // Render
    const draw_list = ig.igGetWindowDrawList();
    ig.ImDrawList_PathClear(draw_list);

    const time = ig.igGetTime();
    const num_segments: f32 = 30;
    var start: f32 = @floatCast(@abs(std.math.sin(time * 1.8) * (num_segments - 5)));

    const a_min = std.math.pi * 2 * start / num_segments;
    const a_max = std.math.pi * 2 * (num_segments - 3) / num_segments;

    const centre = ig.ImVec2{ .x = pos.x + radius, .y = pos.y + radius + style.*.FramePadding.y };

    for (0..@intFromFloat(num_segments)) |ii| {
        const i: f32 = @floatFromInt(ii);
        const a = a_min + (i / num_segments) * (a_max - a_min);
        ig.ImDrawList_PathLineTo(draw_list, .{ .x = centre.x + radius * @as(f32, @floatCast(std.math.cos(a + time * 8))), .y = centre.y + radius * @as(f32, @floatCast(std.math.sin(a + time * 8))) });
    }

    ig.ImDrawList_PathStroke(draw_list, color, ig.ImDrawFlags_None, thickness);
}
