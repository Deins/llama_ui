// modified zig-imgui glfw example code
const std = @import("std");
const builtin = @import("builtin");
const imgui = @import("imgui");
const glfw = @import("core").core.glfw;
const assert = std.debug.assert;
pub const tracy = @import("ztracy");

const GLFW_HEADER_VERSION = glfw.version.major * 1000 + glfw.version.minor * 100;
const GLFW_HAS_NEW_CURSORS = true; //@hasDecl(glfw, "GLFW_RESIZE_NESW_CURSOR") and (GLFW_HEADER_VERSION >= 3400); // 3.4+ GLFW_RESIZE_ALL_CURSOR, GLFW_RESIZE_NESW_CURSOR, GLFW_RESIZE_NWSE_CURSOR, GLFW_NOT_ALLOWED_CURSOR
const GLFW_HAS_GAMEPAD_API = (GLFW_HEADER_VERSION >= 3300); // 3.3+ glfwGetGamepadState() new api
const GLFW_HAS_GET_KEY_NAME = (GLFW_HEADER_VERSION >= 3200); // 3.2+ glfwGetKeyName()

const IS_EMSCRIPTEN = builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64; //@import("platform").platform == .emscripten;

const Key = glfw.Key;

const Data = struct {
    Window: ?glfw.Window = null,
    Time: f64 = 0,
    MouseWindow: ?glfw.Window = null,
    MouseCursors: [imgui.MouseCursor.COUNT]?glfw.Cursor = [_]?glfw.Cursor{null} ** imgui.MouseCursor.COUNT,
    LastValidMousePos: imgui.Vec2 = .{ .x = 0, .y = 0 },
    InstalledCallbacks: bool = false,
};

// Backend data stored in io.BackendPlatformUserData to allow support for multiple Dear ImGui contexts
// It is STRONGLY preferred that you use docking branch with multi-viewports (== single Dear ImGui context + multiple windows) instead of multiple Dear ImGui contexts.
// FIXME: multi-context support is not well tested and probably dysfunctional in this backend.
// - Because glfwPollEvents() process all windows and some events may be called outside of it, you will need to register your own callbacks
//   (passing install_callbacks=false in ImGui_ImplGlfw_InitXXX functions), set the current dear imgui context and then call our callbacks.
// - Otherwise we may need to store a GLFWWindow* -> ImGuiContext* map and handle this in the backend, adding a little bit of extra complexity to it.
// FIXME: some shared resources (mouse cursor shape, gamepad) are mishandled when using multi-context.
fn GetBackendData() ?*Data {
    return if (imgui.GetCurrentContext() != null) @ptrCast(@alignCast(imgui.GetIO().BackendPlatformUserData)) else null;
}

// Functions
fn GetClipboardText(user_data: ?*anyopaque) callconv(.C) ?[*:0]const u8 {
    _ = user_data;
    return glfw.getClipboardString() catch null;
}

fn SetClipboardText(user_data: ?*anyopaque, text: ?[*:0]const u8) callconv(.C) void {
    _ = user_data;
    glfw.setClipboardString(text orelse "") catch {};
}

fn KeyToImGuiKey(key: Key) imgui.Key {
    return switch (key) {
        .tab => .Tab,
        .left => .LeftArrow,
        .right => .RightArrow,
        .up => .UpArrow,
        .down => .DownArrow,
        .page_up => .PageUp,
        .page_down => .PageDown,
        .home => .Home,
        .end => .End,
        .insert => .Insert,
        .delete => .Delete,
        .backspace => .Backspace,
        .space => .Space,
        .enter => .Enter,
        .escape => .Escape,
        .apostrophe => .Apostrophe,
        .comma => .Comma,
        .minus => .Minus,
        .period => .Period,
        .slash => .Slash,
        .semicolon => .Semicolon,
        .equal => .Equal,
        .left_bracket => .LeftBracket,
        .backslash => .Backslash,
        .right_bracket => .RightBracket,
        .grave_accent => .GraveAccent,
        .caps_lock => .CapsLock,
        .scroll_lock => .ScrollLock,
        .num_lock => .NumLock,
        .print_screen => .PrintScreen,
        .pause => .Pause,
        .kp_0 => .Keypad0,
        .kp_1 => .Keypad1,
        .kp_2 => .Keypad2,
        .kp_3 => .Keypad3,
        .kp_4 => .Keypad4,
        .kp_5 => .Keypad5,
        .kp_6 => .Keypad6,
        .kp_7 => .Keypad7,
        .kp_8 => .Keypad8,
        .kp_9 => .Keypad9,
        .kp_decimal => .KeypadDecimal,
        .kp_divide => .KeypadDivide,
        .kp_multiply => .KeypadMultiply,
        .kp_subtract => .KeypadSubtract,
        .kp_add => .KeypadAdd,
        .kp_enter => .KeypadEnter,
        .kp_equal => .KeypadEqual,
        .left_shift => .LeftShift,
        .left_control => .LeftCtrl,
        .left_alt => .LeftAlt,
        .left_super => .LeftSuper,
        .right_shift => .RightShift,
        .right_control => .RightCtrl,
        .right_alt => .RightAlt,
        .right_super => .RightSuper,
        .menu => .Menu,
        .zero => .@"0",
        .one => .@"1",
        .two => .@"2",
        .three => .@"3",
        .four => .@"4",
        .five => .@"5",
        .six => .@"6",
        .seven => .@"7",
        .eight => .@"8",
        .nine => .@"9",
        .a => .A,
        .b => .B,
        .c => .C,
        .d => .D,
        .e => .E,
        .f => .F,
        .g => .G,
        .h => .H,
        .i => .I,
        .j => .J,
        .k => .K,
        .l => .L,
        .m => .M,
        .n => .N,
        .o => .O,
        .p => .P,
        .q => .Q,
        .r => .R,
        .s => .S,
        .t => .T,
        .u => .U,
        .v => .V,
        .w => .W,
        .x => .X,
        .y => .Y,
        .z => .Z,
        .F1 => .F1,
        .F2 => .F2,
        .F3 => .F3,
        .F4 => .F4,
        .F5 => .F5,
        .F6 => .F6,
        .F7 => .F7,
        .F8 => .F8,
        .F9 => .F9,
        .F10 => .F10,
        .F11 => .F11,
        .F12 => .F12,
        else => .None,
    };
}

fn UpdateKeyModifiers(mods: glfw.Mods) void {
    const io = imgui.GetIO();
    io.AddKeyEvent(.ModCtrl, (mods.control));
    io.AddKeyEvent(.ModShift, (mods.shift));
    io.AddKeyEvent(.ModAlt, (mods.alt));
    io.AddKeyEvent(.ModSuper, (mods.super));
}

pub fn MouseButtonCallback(window: glfw.Window, button: glfw.MouseButton, action: glfw.Action, mods: glfw.Mods) void {
    _ = window;
    UpdateKeyModifiers(mods);

    const io = imgui.GetIO();
    if (@intFromEnum(button) < imgui.MouseButton.COUNT)
        io.AddMouseButtonEvent(@intFromEnum(button), action == .press);
}

pub fn ScrollCallback(window: glfw.Window, xoffset: f64, yoffset: f64) void {
    _ = window;
    const io = imgui.GetIO();
    io.AddMouseWheelEvent(@as(f32, @floatCast(xoffset)), @as(f32, @floatCast(yoffset)));
}

fn TranslateUntranslatedKey(raw_key: Key, scancode: i32) Key {
    if (GLFW_HAS_GET_KEY_NAME and !IS_EMSCRIPTEN) {
        // GLFW 3.1+ attempts to "untranslate" keys, which goes the opposite of what every other framework does, making using lettered shortcuts difficult.
        // (It had reasons to do so: namely GLFW is/was more likely to be used for WASD-type game controls rather than lettered shortcuts, but IHMO the 3.1 change could have been done differently)
        // See https://github.com/glfw/glfw/issues/1502 for details.
        // Adding a workaround to undo this (so our keys are translated->untranslated->translated, likely a lossy process).
        // This won't cover edge cases but this is at least going to cover common cases.
        if (@intFromEnum(raw_key) >= @intFromEnum(Key.kp_0) and @intFromEnum(raw_key) <= @intFromEnum(Key.kp_equal))
            return raw_key;
        const kn = raw_key.getName(scancode) catch unreachable;
        if (kn) |key_name| {
            if (key_name[0] != 0 and key_name[1] == 0) {
                const char_names = "`-=[]\\,;\'./";
                const char_keys = [_]Key{ .grave_accent, .minus, .equal, .left_bracket, .right_bracket, .backslash, .comma, .semicolon, .apostrophe, .period, .slash };
                comptime assert(char_names.len == char_keys.len);
                if (key_name[0] >= '0' and key_name[0] <= '9') {
                    return @as(Key, @enumFromInt(@intFromEnum(Key.zero) + (key_name[0] - '0')));
                } else if (key_name[0] >= 'A' and key_name[0] <= 'Z') {
                    return @as(Key, @enumFromInt(@intFromEnum(Key.a) + (key_name[0] - 'A')));
                } else if (key_name[0] >= 'a' and key_name[0] <= 'z') {
                    return @as(Key, @enumFromInt(@intFromEnum(Key.a) + (key_name[0] - 'a')));
                } else if (std.mem.indexOfScalar(u8, char_names, key_name[0])) |idx| {
                    return char_keys[idx];
                }
            }
        }
        // if (action == GLFW_PRESS) std.debug.print("key {} scancode {} name '{s}'\n", .{ key, scancode, key_name });
    }
    return raw_key;
}

pub fn KeyCallback(window: glfw.Window, key: Key, scancode: i32, action: glfw.Action, mods: glfw.Mods) void {
    _ = window;
    if (action != .press and action != .release)
        return;

    // Workaround: X11 does not include current pressed/released modifier key in 'mods' flags. https://github.com/glfw/glfw/issues/1630
    var key_mods = mods;
    if (key == .left_control or key == .right_control)
        key_mods.control = (action == .press);
    if (key == .left_shift or key == .right_shift)
        key_mods.shift = (action == .press);
    if (key == .left_alt or key == .right_alt)
        key_mods.alt = (action == .press);
    if (key == .left_super or key == .right_super)
        key_mods.super = (action == .press);
    UpdateKeyModifiers(key_mods);

    const keycode = TranslateUntranslatedKey(key, scancode);

    const io = imgui.GetIO();
    const imgui_key = KeyToImGuiKey(keycode);
    io.AddKeyEvent(imgui_key, (action == .press));
    io.SetKeyEventNativeData(imgui_key, @intFromEnum(keycode), scancode); // To support legacy indexing (<1.87 user code)
}

pub fn WindowFocusCallback(window: glfw.Window, focused: bool) void {
    _ = window;
    const io = imgui.GetIO();
    io.AddFocusEvent(focused);
}

pub fn CursorPosCallback(window: glfw.Window, x: f64, y: f64) void {
    _ = window;
    const bd = GetBackendData().?;
    const io = imgui.GetIO();
    io.AddMousePosEvent(@as(f32, @floatCast(x)), @as(f32, @floatCast(y)));
    bd.LastValidMousePos = .{ .x = @as(f32, @floatCast(x)), .y = @as(f32, @floatCast(y)) };
}

// Workaround: X11 seems to send spurious Leave/Enter events which would make us lose our position,
// so we back it up and restore on Leave/Enter (see https://github.com/ocornut/imgui/issues/4984)
pub fn CursorEnterCallback(window: glfw.Window, entered: bool) void {
    const bd = GetBackendData().?;
    const io = imgui.GetIO();
    if (entered) {
        bd.MouseWindow = window;
        io.AddMousePosEvent(bd.LastValidMousePos.x, bd.LastValidMousePos.y);
    } else if (entered and bd.MouseWindow.?.handle == window.handle) {
        bd.LastValidMousePos = io.MousePos;
        bd.MouseWindow = null;
        io.AddMousePosEvent(-imgui.FLT_MAX, -imgui.FLT_MAX);
    }
}

pub fn CharCallback(window: glfw.Window, c: u32) void {
    const io = imgui.GetIO();
    io.AddInputCharacter(c);
    _ = window;
}

pub fn MonitorCallback(monitor: *glfw.GLFWmonitor, event: i32) void {
    // Unused in 'master' branch but 'docking' branch will use this, so we declare it ahead of it so if you have to install callbacks you can install this one too.
    _ = monitor;
    _ = event;
}

pub fn InstallCallbacks(window: glfw.Window) void {
    const bd = GetBackendData().?;
    assert(bd.InstalledCallbacks == false); // Callbacks already installed!
    assert(bd.Window != null and bd.Window.?.handle == window.handle);

    const focus_cb = struct {
        fn cb(w: glfw.Window, focused: bool) void {
            WindowFocusCallback(w, focused);
        }
    }.cb;
    window.setFocusCallback(focus_cb);
    window.setCursorEnterCallback(struct {
        fn cb(w: glfw.Window, entered: bool) void {
            CursorEnterCallback(w, entered);
        }
    }.cb);
    window.setCursorPosCallback(struct {
        fn cb(w: glfw.Window, x: f64, y: f64) void {
            CursorPosCallback(w, x, y);
        }
    }.cb);
    window.setMouseButtonCallback(struct {
        fn cb(w: glfw.Window, button: glfw.MouseButton, action: glfw.Action, mods: glfw.Mods) void {
            MouseButtonCallback(w, button, action, mods);
        }
    }.cb);
    window.setScrollCallback(struct {
        fn cb(w: glfw.Window, xoffset: f64, yoffset: f64) void {
            ScrollCallback(w, xoffset, yoffset);
        }
    }.cb);
    window.setKeyCallback(struct {
        fn cb(w: glfw.Window, key: Key, scancode: i32, action: glfw.Action, mods: glfw.Mods) void {
            KeyCallback(w, key, scancode, action, mods);
        }
    }.cb);
    window.setCharCallback(struct {
        fn cb(w: glfw.Window, c: u21) void {
            CharCallback(w, @as(u32, @intCast(c)));
        }
    }.cb);
    //glfw.Monitor.setCallback(&MonitorCallback);
}

pub fn Init(window: glfw.Window) !void {
    const t_imgui_init = tracy.ZoneN(@src(), "init imgui_glfw");
    defer t_imgui_init.End();

    const ctx = imgui.CreateContext();
    imgui.SetCurrentContext(ctx);
    const io = imgui.GetIO();
    if (IS_EMSCRIPTEN) io.IniFilename = null;
    assert(io.BackendPlatformUserData == null); // Already initialized a platform backend!

    // Setup backend capabilities flags
    const bd = @as(*Data, @ptrCast(@alignCast(imgui.MemAlloc(@sizeOf(Data)))));
    bd.* = .{
        .Window = window,
        .Time = 0,
    };

    io.BackendPlatformUserData = bd;
    io.BackendFlags.HasMouseCursors = !IS_EMSCRIPTEN; // We can honor GetMouseCursor() values (optional)
    io.BackendFlags.HasSetMousePos = !IS_EMSCRIPTEN; // We can honor io.WantSetMousePos requests (optional, rarely used)

    io.SetClipboardTextFn = SetClipboardText;
    io.GetClipboardTextFn = GetClipboardText;
    //io.ClipboardUserData = window;

    // Set platform dependent data in viewport
    // if (builtin.os.tag == .windows) {
    //     imgui.GetMainViewport().?.PlatformHandleRaw = glfw.glfwGetWin32Window(window);
    // }

    // Create mouse cursors
    // (By design, on X11 cursors are user configurable and some cursors may be missing. When a cursor doesn't exist,
    // GLFW will emit an error which will often be printed by the app, so we temporarily disable error reporting.
    // Missing cursors will return NULL and our _UpdateMouseCursor() function will use the Arrow cursor instead.)
    //const prev_error_callback = glfw.glfwSetErrorCallback(null);

    if (!IS_EMSCRIPTEN) {
        bd.MouseCursors[@intFromEnum(imgui.MouseCursor.Arrow)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.arrow);
        bd.MouseCursors[@intFromEnum(imgui.MouseCursor.TextInput)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.ibeam);
        bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeNS)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.resize_ns);
        bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeEW)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.resize_ew);
        bd.MouseCursors[@intFromEnum(imgui.MouseCursor.Hand)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.pointing_hand);
        if (GLFW_HAS_NEW_CURSORS) {
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeAll)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.resize_all);
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeNESW)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.resize_nesw);
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeNWSE)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.resize_nwse);
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.NotAllowed)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.not_allowed);
        } else {
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeAll)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.arrow);
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeNESW)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.arrow);
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.ResizeNWSE)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.arrow);
            bd.MouseCursors[@intFromEnum(imgui.MouseCursor.NotAllowed)] = try glfw.Cursor.createStandard(glfw.Cursor.Shape.arrow);
        }
    }
    //_ = glfw.glfwSetErrorCallback(prev_error_callback);

    // Chain GLFW callbacks: our callbacks will call the user's previously installed callbacks, if any.
    // if (install_callbacks)
    InstallCallbacks(window);
}

pub fn Shutdown() void {
    const bd = GetBackendData();
    assert(bd != null); // No platform backend to shutdown, or already shutdown?
    const io = imgui.GetIO();

    for (bd.?.MouseCursors) |cursor|
        if (cursor) |c| glfw.glfwDestroyCursor(c);

    io.BackendPlatformName = null;
    io.BackendPlatformUserData = null;
    imgui.MemFree(bd.?);
}

fn UpdateMouseData() void {
    const bd = GetBackendData().?;
    const io = imgui.GetIO();

    const focused_atrib = bd.Window.?.getAttrib(.focused) catch 1;
    const is_app_focused = if (IS_EMSCRIPTEN) true else (focused_atrib != 0);
    if (is_app_focused) {
        // (Optional) Set OS mouse position from Dear ImGui if requested (rarely used, only when ImGuiConfigFlags_NavEnableSetMousePos is enabled by user)
        if (io.WantSetMousePos)
            bd.Window.?.setCursorPos(io.MousePos.x, io.MousePos.y) catch unreachable;

        // (Optional) Fallback to provide mouse position when focused (ImGui_ImplGlfw_CursorPosCallback already provides this when hovered or captured)
        if (is_app_focused and bd.MouseWindow == null) {
            const cursor_pos = bd.Window.?.getCursorPos() catch unreachable;
            io.AddMousePosEvent(@as(f32, @floatCast(cursor_pos.xpos)), @as(f32, @floatCast(cursor_pos.ypos)));
            bd.LastValidMousePos = .{ .x = @as(f32, @floatCast(cursor_pos.xpos)), .y = @as(f32, @floatCast(cursor_pos.ypos)) };
        }
    }
}

fn UpdateMouseCursor() void {
    const bd = GetBackendData().?;
    const io = imgui.GetIO();
    if ((io.ConfigFlags.NoMouseCursorChange) or bd.Window.?.getInputModeCursor() == .disabled)
        return;

    const imgui_cursor = imgui.GetMouseCursor();
    if (imgui_cursor == .None or io.MouseDrawCursor) {
        // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor
        bd.Window.?.setInputModeCursor(.hidden) catch unreachable;
    } else {
        // Show OS mouse cursor
        // FIXME-PLATFORM: Unfocused windows seems to fail changing the mouse cursor with GLFW 3.2, but 3.3 works here.
        bd.Window.?.setCursor(bd.MouseCursors[@as(usize, @intCast(@intFromEnum(imgui_cursor)))] orelse bd.MouseCursors[@as(usize, @intCast(@intFromEnum(imgui.MouseCursor.Arrow)))]) catch unreachable;
        bd.Window.?.setInputModeCursor(.normal) catch unreachable;
    }
}

// Update gamepad inputs
inline fn Saturate(v: f32) f32 {
    return if (v < 0) 0 else if (v > 1) 1 else v;
}

fn UpdateGamepads() void {
    const io = imgui.GetIO();
    if (!io.ConfigFlags.NavEnableGamepad)
        return;

    io.BackendFlags.HasGamepad = false;
    if (IS_EMSCRIPTEN) return;

    const InputKind = enum { Button, Analog };
    const Mapping = struct { kind: InputKind, key: imgui.Key, btn: c_int, low: f32 = 0, high: f32 = 0 };
    const mappings = [_]Mapping{
        .{ .kind = .Button, .key = .GamepadStart, .btn = @intFromEnum(glfw.GamepadButton.start) },
        .{ .kind = .Button, .key = .GamepadBack, .btn = @intFromEnum(glfw.GamepadButton.back) },
        .{ .kind = .Button, .key = .GamepadFaceDown, .btn = @intFromEnum(glfw.GamepadButton.a) }, // Xbox A, PS Cross
        .{ .kind = .Button, .key = .GamepadFaceRight, .btn = @intFromEnum(glfw.GamepadButton.b) }, // Xbox B, PS Circle
        .{ .kind = .Button, .key = .GamepadFaceLeft, .btn = @intFromEnum(glfw.GamepadButton.x) }, // Xbox X, PS Square
        .{ .kind = .Button, .key = .GamepadFaceUp, .btn = @intFromEnum(glfw.GamepadButton.y) }, // Xbox Y, PS Triangle
        .{ .kind = .Button, .key = .GamepadDpadLeft, .btn = @intFromEnum(glfw.GamepadButton.dpad_left) },
        .{ .kind = .Button, .key = .GamepadDpadRight, .btn = @intFromEnum(glfw.GamepadButton.dpad_right) },
        .{ .kind = .Button, .key = .GamepadDpadUp, .btn = @intFromEnum(glfw.GamepadButton.dpad_up) },
        .{ .kind = .Button, .key = .GamepadDpadDown, .btn = @intFromEnum(glfw.GamepadButton.dpad_down) },
        .{ .kind = .Button, .key = .GamepadL1, .btn = @intFromEnum(glfw.GamepadButton.left_bumper) },
        .{ .kind = .Button, .key = .GamepadR1, .btn = @intFromEnum(glfw.GamepadButton.right_bumper) },
        .{ .kind = .Analog, .key = .GamepadL2, .btn = @intFromEnum(glfw.GamepadAxis.left_trigger), .low = -0.75, .high = 1.0 },
        .{ .kind = .Analog, .key = .GamepadR2, .btn = @intFromEnum(glfw.GamepadAxis.right_trigger), .low = -0.75, .high = 1.0 },
        .{ .kind = .Button, .key = .GamepadL3, .btn = @intFromEnum(glfw.GamepadButton.left_thumb) },
        .{ .kind = .Button, .key = .GamepadR3, .btn = @intFromEnum(glfw.GamepadButton.right_thumb) },
        .{ .kind = .Analog, .key = .GamepadLStickLeft, .btn = @intFromEnum(glfw.GamepadAxis.left_x), .low = -0.25, .high = -1.0 },
        .{ .kind = .Analog, .key = .GamepadLStickRight, .btn = @intFromEnum(glfw.GamepadAxis.left_x), .low = 0.25, .high = 1.0 },
        .{ .kind = .Analog, .key = .GamepadLStickUp, .btn = @intFromEnum(glfw.GamepadAxis.left_y), .low = -0.25, .high = -1.0 },
        .{ .kind = .Analog, .key = .GamepadLStickDown, .btn = @intFromEnum(glfw.GamepadAxis.left_y), .low = 0.25, .high = 1.0 },
        .{ .kind = .Analog, .key = .GamepadRStickLeft, .btn = @intFromEnum(glfw.GamepadAxis.right_x), .low = -0.25, .high = -1.0 },
        .{ .kind = .Analog, .key = .GamepadRStickRight, .btn = @intFromEnum(glfw.GamepadAxis.right_x), .low = 0.25, .high = 1.0 },
        .{ .kind = .Analog, .key = .GamepadRStickUp, .btn = @intFromEnum(glfw.GamepadAxis.right_y), .low = -0.25, .high = -1.0 },
        .{ .kind = .Analog, .key = .GamepadRStickDown, .btn = @intFromEnum(glfw.GamepadAxis.right_y), .low = 0.25, .high = 1.0 },
    };

    if (GLFW_HAS_GAMEPAD_API) {
        const joystick: glfw.Joystick = .{ .jid = .one };
        var gamepad = joystick.getGamepadState() orelse return;
        inline for (mappings) |m| switch (m.kind) {
            .Button => io.AddKeyEvent(m.key, gamepad.buttons[m.btn] != 0),
            .Analog => {
                var v = gamepad.axes[m.btn];
                v = (v - m.low) / (m.high - m.low);
                io.AddKeyAnalogEvent(m.key, v > 0.1, Saturate(v));
            },
        };
    } else {
        var axes_count: c_int = 0;
        var buttons_count: c_int = 0;
        const axes = glfw.glfwGetJoystickAxes(glfw.GLFW_JOYSTICK_1, &axes_count);
        const buttons = glfw.glfwGetJoystickButtons(glfw.GLFW_JOYSTICK_1, &buttons_count);
        if (axes_count == 0 or buttons_count == 0)
            return;

        inline for (mappings) |m| switch (m.kind) {
            .Button => io.AddKeyEvent(m.key, m.btn > buttons_count and buttons.?[m.btn] != 0),
            .Analog => {
                var v: f32 = if (m.btn < axes_count) axes.?[m.btn] else m.low;
                v = (v - m.low) / (m.high - m.low);
                io.AddKeyAnalogEvent(m.key, v > 0.1, Saturate(v));
            },
        };
    }
    io.BackendFlags.HasGamepad = true;
}

pub fn NewFrame() void {
    const bd = GetBackendData().?; // Did you call ImGui_ImplGlfw_InitForXXX()?
    const io = imgui.GetIO();

    // Setup display size (every frame to accommodate for window resizing)
    const size = bd.Window.?.getSize() catch unreachable;
    const w = size.width;
    const h = size.height;
    // const win_ctx = bd.Window.?.getUserPointer(common.WindowContext).?;
    // const display_w = win_ctx.target_desc.width;
    // const display_h = win_ctx.target_desc.height;
    const fb_size = bd.Window.?.getFramebufferSize() catch unreachable;
    const display_w = fb_size.width;
    const display_h = fb_size.height;

    io.DisplaySize = .{ .x = @as(f32, @floatFromInt(w)), .y = @as(f32, @floatFromInt(h)) };
    if (w > 0 and h > 0) {
        io.DisplayFramebufferScale = .{
            .x = @as(f32, @floatFromInt(display_w)) / @as(f32, @floatFromInt(w)),
            .y = @as(f32, @floatFromInt(display_h)) / @as(f32, @floatFromInt(h)),
        };
    }

    // Setup time step
    const current_time = glfw.getTime();
    io.DeltaTime = if (bd.Time > 0) @as(f32, @floatCast(current_time - bd.Time)) else (1.0 / 60.0);
    bd.Time = current_time;

    UpdateMouseData();
    UpdateMouseCursor();

    // Update game controllers (if enabled and available)
    UpdateGamepads();
}
