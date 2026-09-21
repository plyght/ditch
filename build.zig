const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addLua(b, root);

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
