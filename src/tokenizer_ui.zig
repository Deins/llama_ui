const std = @import("std");
const ig = @import("ig.zig");
const llama = @import("llama");

const Self = @This();

input: std.ArrayList(u8),
tokenizer: llama.Tokenizer,
// tokenization params
add_bos: bool = false,
special: bool = true,

pub fn init(alloc: std.mem.Allocator) !Self {
    var s: Self = .{
        .input = std.ArrayList(u8).init(alloc),
        .tokenizer = llama.Tokenizer.init(alloc),
    };
    //try s.input.append(0);
    return s;
}

pub fn deinit(self: *Self) void {
    self.input.deinit();
    self.tokenizer.deinit();
}

pub fn render(self: *Self, model: *llama.Model) !void {
    var refresh = false;
    if (ig.igCheckbox("add_bos", &self.add_bos)) refresh = true;
    ig.igSameLine(0, 8);
    if (ig.igCheckbox("special", &self.special)) refresh = true;
    const flags = ig.ImGuiInputTextFlags_AllowTabInput | ig.ImGuiInputTextFlags_Multiline;
    ig.igPushItemWidth(-1);

    if (try ig.inputTextArrayList("##input", "Input text to tokenize here...", &self.input, flags))
        refresh = true;

    ig.igPopItemWidth();
    if (refresh) {
        self.tokenizer.clearRetainingCapacity();
        try self.tokenizer.tokenize(model, self.input.items, self.add_bos, self.special);
    }

    // Draw output
    ig.igText("Input len: %d; Tokens: %d", self.input.items.len, self.tokenizer.data.items.len);
    // if (self.tokenizer.data.items.len > 1)
    {
        if (ig.igBeginTable("tokenizer", 5, ig.ImGuiTableFlags_Borders | ig.ImGuiTableFlags_SizingFixedFit, .{}, 0)) {
            ig.igTableSetupColumn("Token", ig.ImGuiTableColumnFlags_None, 0, 0);
            ig.igTableSetupColumn("Type", ig.ImGuiTableColumnFlags_None, 0, 0);
            ig.igTableSetupColumn("Text", ig.ImGuiTableColumnFlags_None, 0, 0);
            ig.igTableSetupColumn("Unicode codepoints", ig.ImGuiTableColumnFlags_None, 0, 0);
            ig.igTableSetupColumn("Score", ig.ImGuiTableColumnFlags_None, 0, 0);
            ig.igTableHeadersRow();

            for (self.tokenizer.data.items) |tok| {
                ig.igTableNextRow(ig.ImGuiTableRowFlags_None, 0);
                var temp: [64]u8 = undefined;
                var str: []const u8 = undefined;
                var text = model.tokenGetTextSlice(tok);

                // token
                _ = ig.igTableNextColumn();
                str = std.fmt.bufPrint(&temp, "{}", .{tok}) catch "ERROR";
                ig.igTextUnformatted(str.ptr, str[str.len..].ptr);

                // type
                _ = ig.igTableNextColumn();
                str = @tagName(model.tokenGetType(tok))["LLAMA_TOKEN_TYPE_".len..];
                ig.igTextUnformatted(str.ptr, str[str.len..].ptr);

                // text
                _ = ig.igTableNextColumn();
                str = text;
                ig.igTextUnformatted(str.ptr, str[str.len..].ptr);

                // unicode
                _ = ig.igTableNextColumn();
                var it = std.unicode.Utf8Iterator{ .bytes = text, .i = 0 };
                var had_prev = false;
                while (it.nextCodepoint()) |codepoint| {
                    if (had_prev) ig.igSameLine(0, 8) else had_prev = true;
                    str = @ptrCast(std.fmt.bufPrint(&temp, "U+{X}", .{(codepoint)}) catch "ERROR");
                    ig.igTextUnformatted(str.ptr, str[str.len..].ptr);
                }

                // score
                _ = !ig.igTableNextColumn();
                ig.igText("%d", model.tokenGetScore(tok));
            }
            ig.igEndTable();
        }
    }
}
