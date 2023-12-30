const std = @import("std");
const ig = @import("imgui");
const llama = @import("llama");

const Self = @This();

/// NOTE: 0 terminated
input: std.ArrayList(u8),
tokenizer: llama.Tokenizer,
//detokenizer: llama.Detokenizer,
// tokenization params
add_bos: bool = false,
special: bool = true,

pub fn init(alloc: std.mem.Allocator) !Self {
    var s: Self = .{
        .input = try std.ArrayList(u8).initCapacity(alloc, 512),
        .tokenizer = llama.Tokenizer.init(alloc),
    };
    try s.input.append(0);
    return s;
}

pub fn deinit(self: *Self) void {
    self.input.deinit();
    self.tokenizer.deinit();
}

/// returns input slice without 0 terminator
pub fn slice(self: Self) [:0]u8 {
    return @ptrCast(self.input.items.ptr[0 .. self.input.items.len - 1]);
}

pub fn render(self: *Self, model: *llama.Model) !void {
    var refresh = false;
    if (ig.igCheckbox("add_bos", &self.add_bos)) refresh = true;
    ig.igSameLine(0, 8);
    if (ig.igCheckbox("special", &self.special)) refresh = true;
    const flags = ig.ImGuiInputTextFlags_CallbackResize | ig.ImGuiInputTextFlags_AllowTabInput | ig.ImGuiInputTextFlags_Multiline;
    ig.igPushItemWidth(-1);

    if (ig.igInputTextEx("##input", "Input text to tokenize here...", self.input.items.ptr, @intCast(self.input.capacity), .{}, flags, @ptrCast(&inputResizeCB), self)) {
        refresh = true;
        self.input.items = self.input.items[0 .. std.mem.indexOf(u8, self.input.items.ptr[0..self.input.capacity], "\x00").? + 1];
    }
    ig.igPopItemWidth();
    if (refresh) {
        self.tokenizer.clearRetainingCapacity();
        try self.tokenizer.tokenize(model, self.slice(), self.add_bos, self.special);
        //self.detokenizer.clearRetainingCapacity();
        //try self.detokenizer.detokenize()
    }

    // Draw output
    ig.igText("Input len: %d; Tokens: %d", self.slice().len, self.tokenizer.data.items.len);
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

fn inputResizeCB(data: *ig.ImGuiInputTextCallbackData) callconv(.C) c_int {
    const self: *Self = @ptrCast(@alignCast(data.UserData));
    if (data.EventFlag == ig.ImGuiInputTextFlags_CallbackResize) {
        self.input.resize(@intCast(data.BufSize)) catch {
            data.BufTextLen = @intCast(self.input.capacity - 1); // Text length (in bytes)               // Read-write   // [Resize,Completion,History,Always] Exclude zero-terminator storage. In C land: == strlen(some_text), in C++ land: string.length()
            data.BufSize = @intCast(self.input.capacity); // Buffer size (in bytes) = capacity+1  // Read-only    // [Resize,Completion,History,Always] Include zero-terminator storage. In C land == ARRAYSIZE(my_char_array), in C++ land: string.capacity(+1)
        };
        data.Buf = self.input.items.ptr; // Text buffer                          // Read-write   // [Resize] Can replace pointer / [Completion,History,Always] Only write to pointed data, don't replace the actual pointer!
    }
    return 0;
}
