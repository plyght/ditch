const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // GPU acceleration. The Metal backend needs Apple's SDK headers for its
    // Objective-C shim, which Zig does not ship, so it is off unless asked for
    // explicitly: `zig build -Dmetal` on a Mac with Xcode's command line tools.
    // Everything else, cross-compiling to aarch64-macos included, builds the
    // CPU path exactly as before.
    const metal = b.option(bool, "metal", "Build the Metal backend (macOS target with Apple's SDK; default false)") orelse false;
    if (metal and target.result.os.tag != .macos) {
        std.debug.print("-Dmetal needs a macOS target ({s} was requested)\n", .{@tagName(target.result.os.tag)});
        std.process.exit(1);
    }
    const options = b.addOptions();
    options.addOption(bool, "metal", metal);

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addLua(b, root);
    root.addOptions("build_options", options);
    if (metal) addMetal(b, root);

    const exe = b.addExecutable(.{
        .name = "ditch",
        .root_module = root,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run ditch");
    run_step.dependOn(&run_cmd.step);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    addLua(b, test_mod);
    test_mod.addOptions("build_options", options);
    if (metal) addMetal(b, test_mod);
    const tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // End-to-end test: builds ditch in ReleaseFast itself and runs the whole
    // pipeline on a synthetic fixture model.
    const e2e = b.addSystemCommand(&.{ "bash", "tests/e2e.sh" });
    e2e.setEnvironmentVariable("ZIG", b.graph.zig_exe);
    e2e.setCwd(b.path("."));
    const e2e_step = b.step("e2e", "Run the end-to-end pipeline test (bash tests/e2e.sh)");
    e2e_step.dependOn(&e2e.step);

    // Type-checks the Zig half of the Metal backend for an Apple silicon
    // target without Apple's SDK: the shim is only declared here, never
    // linked, so this runs anywhere. The shaders themselves need `xcrun
    // metal`, which the macOS CI job runs.
    const metal_mod = b.createModule(.{
        .root_source_file = b.path("src/metal_check.zig"),
        .target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos }),
        .optimize = .ReleaseFast,
    });
    metal_mod.addOptions("build_options", options);
    const metal_obj = b.addObject(.{ .name = "ditch-metal-check", .root_module = metal_mod });
    const metal_check = b.step("metal-check", "Compile the Metal backend for aarch64-macos without linking");
    metal_check.dependOn(&metal_obj.step);
}

/// Compiles the Objective-C shim of the Metal backend and links the frameworks
/// it needs. Only ever called for macOS targets (see `-Dmetal` above).
fn addMetal(b: *std.Build, mod: *std.Build.Module) void {
    mod.link_libc = true;
    mod.addIncludePath(b.path("src/metal"));
    mod.addCSourceFile(.{
        .file = b.path("src/metal/shim.m"),
        .flags = &.{ "-x", "objective-c", "-fno-objc-arc", "-fobjc-exceptions", "-O2" },
    });
    mod.linkFramework("Metal", .{});
    mod.linkFramework("Foundation", .{});
}

/// Lua 5.4 sources are compiled into the executable so configuration files
/// can be written in Lua. Sources live in vendor/lua54 (MIT licensed).
fn addLua(b: *std.Build, mod: *std.Build.Module) void {
    mod.link_libc = true;
    mod.addIncludePath(b.path("vendor/lua54"));
    mod.addCSourceFiles(.{
        .root = b.path("vendor/lua54"),
        .files = &lua_sources,
        .flags = &.{ "-std=gnu99", "-DLUA_COMPAT_5_3", "-O2" },
    });
}

const lua_sources = [_][]const u8{
    "lapi.c",    "lauxlib.c",  "lbaselib.c", "lcode.c",   "lcorolib.c", "lctype.c",   "ldblib.c",
    "ldebug.c",  "ldo.c",      "ldump.c",    "lfunc.c",   "lgc.c",      "linit.c",    "liolib.c",
    "llex.c",    "lmathlib.c", "lmem.c",     "loadlib.c", "lobject.c",  "lopcodes.c", "loslib.c",
    "lparser.c", "lstate.c",   "lstring.c",  "lstrlib.c", "ltable.c",   "ltablib.c",  "ltm.c",
    "lundump.c", "lutf8lib.c", "lvm.c",      "lzio.c",
};
