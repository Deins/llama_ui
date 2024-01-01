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
input_templated: bool = true,

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

    if (ig.igBegin("Prompt history", null, ig.ImGuiWindowFlags_None)) {
        ig.text("History");
        ig.igSameLine(0, -1);
        if (ig.igButton("clear", .{})) {
            if (mrt.prompt) |*p| p.tokens.clearRetainingCapacity();
            self.history.clearRetainingCapacity();
        }

        // ig.igPushStyleColor_U32(ig.ImGuiCol_ChildBg, 0xFFFFFFFF);
        // defer ig.igPopStyleColor(1);
        // _ = ig.igBeginChild_Str("##prompt_history_area", .{}, false, 0);
        // defer ig.igEndChild();

        var tmp: [512]u8 = undefined;
        _ = tmp;
        if (try ig.inputTextArrayList("##prompt_history", .{ .x = -1, .y = -1 }, "", &self.history, ig.ImGuiInputTextFlags_ReadOnly | ig.ImGuiInputTextFlags_Multiline)) {}
    }
    ig.igEnd();

    if (ig.igBegin("Prompt input", null, ig.ImGuiWindowFlags_None)) {
        _ = ig.igCheckbox("special tokens", &self.input_special);
        ig.igSameLine(0, -1);
        //_ = ig.igCheckbox("templated", &mrt.input_templated);

        var i: i32 = 0;
        _ = ig.igCombo_Str("template", &i, "A\x00B\x00C\x00", -1);

        if (self.input_templated) {
            ig.igSameLine(0, -1);
            ig.igPushItemWidth(150);
            defer ig.igPopItemWidth();
            _ = try ig.inputTextArrayList("role", .{}, "role", &self.role, ig.ImGuiInputTextFlags_None);
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
                const input = &self.input;
                try input.ensureTotalCapacity(input.items.len + 1);
                input.allocatedSlice()[input.items.len] = 0; // ensure 0 termination
                const flags = ig.ImGuiInputTextFlags_Multiline | ig.ImGuiInputTextFlags_EnterReturnsTrue;
                if (try ig.inputTextArrayList("##prompt_input", .{ .y = -1 }, "Prompt input", input, flags)) {
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
