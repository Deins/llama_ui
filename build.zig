const std = @import("std");
const ztBuild = @import("libs/zt/build.zig");
const mach_glfw = @import("mach_glfw");
const llama = @import("llama");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // llama
    const use_clblast = b.option(bool, "clblast", "Use clblast acceleration") orelse false;
    const opencl_includes = b.option([]const u8, "opencl_includes", "Path to OpenCL headers");
    const opencl_libs = b.option([]const u8, "opencl_libs", "Path to OpenCL libs");
    const want_opencl = use_clblast;
    const opencl_maybe = if (!want_opencl) null else if (opencl_includes != null or opencl_libs != null) llama.clblast.OpenCL{ .include_path = (opencl_includes orelse ""), .lib_path = (opencl_libs orelse "") } else llama.clblast.OpenCL.fromOCL(b, target);
    if (use_clblast and opencl_maybe == null) @panic("OpenCL not found. Please specify include or libs manually if its installed!");
    const llama_zig_dep = b.dependency("llama", .{ .target = target, .optimize = optimize });
    var llama_zig = llama.Context.init(llama_zig_dep.builder, .{
        .target = target,
        .optimize = optimize,
        .opencl = opencl_maybe,
        .clblast = use_clblast,
    });

    const file_dialog = b.addTranslateC(.{
        .target = target,
        .optimize = optimize,
        .source_file = .{ .path = "libs/native_file_dialog/src/include/nfd.h" },
    });
    const file_dialog_module = file_dialog.addModule("nfd");

    var exe = b.addExecutable(.{
        .name = "llama_ui",
        .target = target,
        .optimize = optimize,
        .root_source_file = .{ .path = "src/main.zig" },
    });
    llama_zig.link(exe);
    exe.addModule("llama", llama_zig.module);
    exe.linkLibCpp();
    exe.want_lto = false; // on windows: undefined symbol: __declspec(dllimport) _create_locale
    try ztBuild.link(b, exe);
    try ztBuild.addBinaryContent(b.pathFromRoot("assets"));
    b.installArtifact(exe);

    // file dialog
    exe.addModule("nfd", file_dialog_module);
    exe.addIncludePath(.{ .path = "libs/native_file_dialog/src/include/" });
    switch (target.getOsTag()) {
        .windows => {
            exe.addCSourceFile(.{ .file = .{ .path = "libs/native_file_dialog/src/nfd_win.cpp" }, .flags = &.{} });
            exe.linkSystemLibraryNeeded("Ole32");
        },
        .linux => {
            const nfd_portal = b.option(bool, "nfd_portal", "Linux, use the portal implementation instead of GTK") orelse false;
            if (nfd_portal) exe.addCSourceFile(.{ .file = .{ .path = "libs/native_file_dialog/src/nfd_portal.cpp" }, .flags = &.{} }) else exe.addCSourceFile(.{ .file = .{ .path = "libs/native_file_dialog/src/nfd_gtk.cpp" }, .flags = &.{} });
        },
        .macos => exe.addCSourceFile(.{ .file = .{ .path = "libs/native_file_dialog/src/nfd_cocoa.cpp" }, .flags = &.{} }),
        else => @panic("no file dialog for this platform!"),
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // // Creates a step for unit testing. This only builds the test executable
    // // but does not run it.
    // const unit_tests = b.addTest(.{
    //     .root_source_file = .{ .path = "src/main.zig" },
    //     .target = target,
    //     .optimize = optimize,
    // });

    // const run_unit_tests = b.addRunArtifact(unit_tests);

    // // Similar to creating the run step earlier, this exposes a `test` step to
    // // the `zig build --help` menu, providing a way for the user to request
    // // running the unit tests.
    // const test_step = b.step("test", "Run unit tests");
    // test_step.dependOn(&run_unit_tests.step);
}
