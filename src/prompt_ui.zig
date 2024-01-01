const std = @import("std");
const ig = @import("ig.zig");
const main = @import("main.zig");
const AppContext = main.AppContext;

pub fn promptUi(context: *AppContext) !void {
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
        //_ = ig.igCheckbox("templated", &mrt.input_templated);

        var i: i32 = 0;
        _ = ig.igCombo_Str("template", &i, "A\x00B\x00C\x00", -1);

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
