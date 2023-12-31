//! Imgui helpers & utilities
const std = @import("std");
const ig = @import("imgui");
pub usingnamespace ig;

pub fn labelFmt(label_txt: []const u8, v: anytype, comptime fmt: []const u8) void {
    var tmp: [511:0]u8 = undefined;
    var str = std.fmt.bufPrint(&tmp, fmt ++ "\x00", .{v}) catch blk: {
        std.debug.assert(false);
        break :blk tmp[0..];
    };
    ig.igLabelText(label_txt.ptr, "%s", str.ptr);
    //ig.igTextEx(label.ptr, @ptrFromInt(@intFromPtr(label.ptr) + label.len), ig.ImGuiTextFlags_None);
    //ig.igSameLine(0, 32);
    //ig.igTextEx(@ptrCast(str), @ptrFromInt(@intFromPtr(str.ptr) + str.len - 1), ig.ImGuiTextFlags_None);
}

pub inline fn label(label_txt: []const u8, v: anytype) void {
    labelFmt(label_txt, v, "{}");
}

pub inline fn text(txt: [:0]const u8) void {
    ig.igTextUnformatted(txt.ptr, @ptrFromInt(@intFromPtr(txt.ptr) + txt.len));
}

pub fn inputScalarFull(comptime T: type, label_txt: [:0]const u8, val: *T, step_: ?T, step_fast_: ?T, format: [*c]const u8, flags: ig.ImGuiInputTextFlags) bool {
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
    return ig.igInputScalar(label_txt.ptr, igt, @ptrCast(val), @ptrCast(step), @ptrCast(step_fast), format, flags);
}

pub fn inputScalar(comptime T: type, label_txt: [:0]const u8, val: *T) bool {
    return inputScalarFull(T, label_txt, val, null, null, null, ig.ImGuiInputFlags_None);
}

pub fn inputTextArrayList(label_txt: [:0]const u8, hint: [:0]const u8, input: *std.ArrayList(u8), flags: ig.ImGuiInputTextFlags) !bool {
    try input.ensureTotalCapacity(input.items.len + 1);
    input.allocatedSlice()[input.items.len] = 0; // ensure 0 termination
    return (ig.igInputTextEx(label_txt, hint, input.items.ptr, @intCast(input.capacity), .{}, flags, @ptrCast(&inputResizeCB), input));
}

fn inputResizeCB(data: *ig.ImGuiInputTextCallbackData) callconv(.C) c_int {
    const input: *std.ArrayList(u8) = @ptrCast(@alignCast(data.UserData));
    if (data.EventFlag == ig.ImGuiInputTextFlags_CallbackResize) {
        //std.log.info("Resize from {} to {}; text_len {}", .{ input.capacity, data.BufSize, data.BufTextLen });
        input.resize(@intCast(data.BufTextLen)) catch {
            data.BufTextLen = @intCast(input.capacity - 1); // Text length (in bytes)               // Read-write   // [Resize,Completion,History,Always] Exclude zero-terminator storage. In C land: == strlen(some_text), in C++ land: string.length()
        };
        data.BufSize = @intCast(input.capacity); // Buffer size (in bytes) = capacity+1  // Read-only    // [Resize,Completion,History,Always] Include zero-terminator storage. In C land == ARRAYSIZE(my_char_array), in C++ land: string.capacity(+1)
        data.Buf = input.items.ptr; // Text buffer                          // Read-write   // [Resize] Can replace pointer / [Completion,History,Always] Only write to pointed data, don't replace the actual pointer!
    }
    return 0;
}

pub fn spinner(label_txt: [:0]const u8, radius: f32, thickness: f32, color: ig.ImU32) void {
    //const window = ig.igGetCurrentWindow();
    // if (window.SkipItems)
    //     return;

    const style = ig.igGetStyle();
    const id = ig.igGetID_Str(label_txt);

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
