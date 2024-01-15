const std = @import("std");
const builtin = @import("builtin");
const ig = @import("ig.zig");
const main = @import("main.zig");
const nfd = @import("nfd");
const llama = @import("llama");

const AppContext = main.AppContext;
const ModelRuntime = main.ModelRuntime;

var show_im_demo = false;

pub fn configWindow(context: *AppContext) !void {
    const mrt: *ModelRuntime = &context.data.mrt.?; // context data
    var tmp: [512]u8 = undefined;
    ig.igSetNextWindowSizeConstraints(.{ .x = 300, .y = 128 }, .{ .x = 9999999, .y = 9999999 }, null, null);

    defer ig.igEnd();
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
            { // context extenison
                if (mrt.prompt) |prompt| mrt.prompt_ctx_extension = prompt.ctx_extension;
                const TagType = std.meta.Tag(llama.Prompt.ContextExtensionStrategy);
                var tag: TagType = mrt.prompt_ctx_extension;
                var str_arr: []const [*:0]const u8 = comptime blk: {
                    const tags = std.meta.tags(TagType);
                    var str_arr: [tags.len][*:0]const u8 = undefined;
                    for (0..tags.len) |i| str_arr[i] = @tagName(tags[i]);
                    break :blk &str_arr;
                };
                const combo_mod = ig.igCombo("Context extension", @ptrCast(&tag), str_arr, -1);
                if (combo_mod) {
                    switch (tag) {
                        .none => mrt.prompt_ctx_extension = TagType.none,
                        .sliding_window => mrt.prompt_ctx_extension = TagType.sliding_window,
                        .shifting => mrt.prompt_ctx_extension = .{ .shifting = .{} },
                        .self_extend => mrt.prompt_ctx_extension = .{ .self_extend = .{} },
                    }
                }

                // extension params
                switch (mrt.prompt_ctx_extension) {
                    .none => {},
                    .sliding_window => {},
                    .shifting => |*cfg| {
                        _ = (ig.inputScalar(usize, "keep_first_n", &cfg.keep_first_n));
                    },
                    .self_extend => |*cfg| {
                        _ = (ig.inputScalar(i32, "i", &cfg.i));
                        _ = (ig.inputScalar(i32, "n", &cfg.n));
                        _ = (ig.inputScalar(i32, "w", &cfg.w));
                    },
                }
                if (mrt.prompt) |*prompt| prompt.ctx_extension = mrt.prompt_ctx_extension;
                ig.igSeparator();
            }

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

            if (mod) try mrt.recreateCtx();
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
    if (show_im_demo) ig.igShowDemoWindow(null);
}

pub fn loadingText(comptime prefix: [:0]const u8, t: f32) [:0]const u8 {
    const texts = [_][:0]const u8{ prefix ++ ".   ", prefix ++ "..  ", prefix ++ "..." };
    return texts[@as(usize, @intFromFloat(t * 2)) % texts.len];
}
