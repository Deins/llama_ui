const std = @import("std");
const ig = @import("ig.zig");
const main = @import("main.zig");
const llama = @import("llama");
const Prompt = llama.Prompt;
const AppContext = main.AppContext;
const ModelRuntime = main.ModelRuntime;

history_tokens: usize = 0,
history: std.ArrayList(u8),
role: std.ArrayList(u8),
input: std.ArrayList(u8),
input_special: bool = true,
template: Template = .raw,
template_gen: ?llama.utils.TemplatedPrompt = null,
submit_generate: bool = true,

pub const Template = enum(c_int) {
    raw,
    chatml,
    basic_chat,
    alpaca,
    custom,
};

pub const model_unavailable_label = "Model unavailable";

pub fn init(alloc: std.mem.Allocator) !@This() {
    var role = std.ArrayList(u8).init(alloc);
    try role.appendSlice("system");
    return .{
        .history = std.ArrayList(u8).init(alloc),
        .input = std.ArrayList(u8).init(alloc),
        .role = role,
    };
}

pub fn deinit(self: *@This()) void {
    self.history.deinit();
    self.input.deinit();
    self.role.deinit();
}

pub fn updateHistory(self: *@This(), mrt: *ModelRuntime) !void {
    const prompt = mrt.prompt.?;
    if (prompt.tokens.items.len != self.history_tokens) {
        self.history.clearRetainingCapacity();
        var dtok = llama.Detokenizer{ .data = self.history };
        defer self.history = dtok.data;
        for (prompt.tokens.items) |tok| _ = try dtok.detokenizeWithSpecial(mrt.model.?, tok);
    }
}

pub fn draw(self: *@This(), context: *AppContext) !void {
    const mrt = &context.data.mrt.?;
    var gen = false;
    const model_available = mrt.model != null;
    if (model_available) {
        if (mrt.prompt) |*prompt| {
            prompt.tokens_mutex.lock();
            defer prompt.tokens_mutex.unlock();
            try self.updateHistory(&context.data.mrt.?);
        }
    }

    defer ig.igEnd();
    if (ig.igBegin("Prompt history", null, ig.ImGuiWindowFlags_None)) {
        if (ig.igButton("clear", .{})) {
            if (mrt.prompt) |*p| p.clearRetainingCapacity();
            self.history.clearRetainingCapacity();
        }
        ig.igSameLine(0, -1);
        ig.igAlignTextToFramePadding();
        ig.textFmt(64, "prompt: {} context: {}/{}", .{ if (mrt.prompt) |p| p.tokens.items.len else 0, if (mrt.prompt) |p| p.ctx_used else 0, if (mrt.model_ctx) |c| c.nCtx() else 0 });

        // ig.igPushStyleColor_U32(ig.ImGuiCol_ChildBg, 0xFFFFFFFF);
        // defer ig.igPopStyleColor(1);
        // _ = ig.igBeginChild_Str("##prompt_history_area", .{}, false, 0);
        // defer ig.igEndChild();

        if (try ig.inputTextArrayList("##prompt_history", .{ .x = -1, .y = -1 }, "", &self.history, ig.ImGuiInputTextFlags_ReadOnly | ig.ImGuiInputTextFlags_Multiline)) {}
    }

    defer ig.igEnd();
    if (ig.igBegin("Prompt input", null, ig.ImGuiWindowFlags_None)) {
        _ = ig.igCheckbox("special tokens", &self.input_special);
        ig.igSameLine(0, -1);
        //_ = ig.igCheckbox("templated", &mrt.input_templated);

        ig.igPushItemWidth(100);
        if (ig.igCombo_Str("template", @ptrCast(&self.template), "Raw\x00ChatML\x00Basic chat\x00Alpaca\x00Custom\x00", -1)) {
            if (self.template_gen) |*tg| tg.deinit();
            self.template_gen = null;
            if (self.template != .raw) {
                self.template_gen = llama.utils.TemplatedPrompt.init(
                    self.input.allocator,
                    switch (self.template) {
                        .raw => unreachable,
                        .chatml => llama.utils.TemplatedPrompt.template_chatml,
                        .basic_chat => llama.utils.TemplatedPrompt.template_basic_chat,
                        .alpaca => llama.utils.TemplatedPrompt.template_alpaca,
                        .custom => .{}, // todo
                    },
                );
            }
        }
        ig.igPopItemWidth();

        if (self.template != .raw) {
            ig.igSameLine(0, -1);
            ig.igPushItemWidth(150);
            defer ig.igPopItemWidth();
            _ = try ig.inputTextArrayList("role", .{}, "role", &self.role, ig.ImGuiInputTextFlags_None);
        }
        ig.igSameLine(0, -1);
        _ = ig.igCheckbox("gen on submit", &self.submit_generate);
        if (ig.igIsItemHovered(ig.ImGuiHoveredFlags_DelayShort) and ig.igBeginTooltip()) {
            ig.text("Start generating on submit, otherwise has to be called manually.");
            ig.igEndTooltip();
        }
        ig.igSameLine(0, -1);
        if (ig.igButton("generate", .{})) {
            if (mrt.state != .ready) ig.igOpenPopup_Str(model_unavailable_label, ig.ImGuiPopupFlags_NoOpenOverExistingPopup) else gen = true;
        }

        switch (mrt.state) {
            .generating => {
                const r = 12;
                var aspace: ig.ImVec2 = .{};
                ig.igGetContentRegionAvail(&aspace);
                ig.igSetCursorPosX(aspace.x / 2 - r);
                ig.spinner("##GeneratingSpinner", r, 4, 0xFFFFFFAA);
                context.setAnimationFrames(0.05);
                const w = 100;
                ig.igSetCursorPosX(aspace.x / 2 - w / 2);
                if (ig.igButton("cancel", .{ .x = w, .y = 32 })) mrt.gen_state.store(.stopping, .Monotonic);
            },
            else => {
                ig.igPushItemWidth(-1);
                defer ig.igPopItemWidth();
                const input = &self.input;
                try input.ensureTotalCapacity(input.items.len + 1);
                input.allocatedSlice()[input.items.len] = 0; // ensure 0 termination
                const flags = ig.ImGuiInputTextFlags_Multiline | ig.ImGuiInputTextFlags_EnterReturnsTrue;
                if (try ig.inputTextArrayList("##prompt_input", .{ .y = -1 }, "Prompt input\nCtrl+Enter to submit", input, flags)) {
                    if (mrt.state != .ready)
                        ig.igOpenPopup_Str(model_unavailable_label, ig.ImGuiPopupFlags_NoOpenOverExistingPopup)
                    else {
                        self.history_tokens += 12345; // fake value just to force reloading
                        if (mrt.prompt) |*prompt| {
                            if (self.template_gen) |*tg| {
                                defer tg.clearRetainingCapacity();
                                try tg.add(self.role.items, input.items);
                                try prompt.appendText(tg.text.items, true);
                            } else try prompt.appendText(input.items, true);

                            input.clearRetainingCapacity();
                            if (self.submit_generate) gen = true;
                        }
                    }
                }
            },
        }

        if (gen) {
            if (self.template_gen) |*tg| {
                defer tg.clearRetainingCapacity();
                try tg.add("assistant", "{{generate}}");
                try mrt.prompt.?.appendText(tg.text.items, true);
            }
            try mrt.switchStateTo(.generating);
        }
    }

    // popup rendering (closed by default)
    var center: ig.ImVec2 = undefined;
    ig.ImGuiViewport_GetCenter(&center, ig.igGetMainViewport());
    ig.igSetNextWindowPos(center, ig.ImGuiCond_Always, .{ .x = 0.5, .y = 0.5 });
    if (ig.igBeginPopupModal(model_unavailable_label, null, ig.ImGuiWindowFlags_AlwaysAutoResize | ig.ImGuiWindowFlags_NoSavedSettings | ig.ImGuiWindowFlags_NoMove)) {
        const bw = 120;
        switch (mrt.state) {
            .unloaded => ig.text("Model has not been loaded.\nLoad model first (see config section)"),
            .loading, .unloading => ig.text("Model is still loading"),
            .generating => {
                ig.text("Model is still generating");
                ig.igSetCursorPosX(ig.getSpaceAvail().x / 2 - bw / 2);
                if (ig.igButton("Stop generating", .{ .x = bw })) mrt.gen_state.store(.stopping, .Monotonic);
            },
            .ready => ig.igCloseCurrentPopup(),
        }
        ig.igSetCursorPosX(ig.getSpaceAvail().x / 2 - bw / 2);
        if (ig.igButton("ok", .{ .x = bw })) ig.igCloseCurrentPopup();
        ig.igEndPopup();
    }
}
