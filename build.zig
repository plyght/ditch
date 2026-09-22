const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const is_macos = target.result.os.tag == .macos;

    // Apple's Accelerate framework (cblas_sgemm) as the matmul backend for
    // prefill-shaped calls; compiled in when building for macOS, ignored
    // everywhere else. `--accelerate=false` turns it off at run time.
    const accelerate = b.option(bool, "accelerate", "Use Apple's Accelerate framework for large matrix products (macOS only, default: true on macOS)") orelse is_macos;
    // Kernel shape overrides; 0 means "pick from the target's register file"
    // (see `src/tensor.zig`). Only for benchmarking a machine by hand.
    const vector_width = b.option(u32, "vector-width", "Kernel vector width in f32 lanes (0 = choose from the target, the default)") orelse 0;
    const tile_rows = b.option(u32, "tile-rows", "Weight rows per register tile (0 = choose from the target, the default)") orelse 0;
    const tile_inputs = b.option(u32, "tile-inputs", "Input rows per register tile (0 = choose from the target, the default)") orelse 0;

    const options = b.addOptions();
    options.addOption(bool, "accelerate", accelerate and is_macos);
    options.addOption(u32, "vector_width", vector_width);
    options.addOption(u32, "tile_rows", tile_rows);
    options.addOption(u32, "tile_inputs", tile_inputs);

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addLua(b, root);
    addPerf(root, options);

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
    addPerf(test_mod, options);
    // `-Dtest-filter=name`: run only the tests whose name contains it. Needed
    // to run the suite under qemu, whose mmap emulation cannot serve the
    // file-mapping tests.
    const test_filters = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this substring (may be repeated)") orelse &.{};
    const tests = b.addTest(.{
        .root_module = test_mod,
        .filters = test_filters,
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

/// Build options the kernels read. Accelerate itself is not linked: the macOS
/// release binaries are cross-compiled from Linux, where the framework does
/// not exist to link against, so `src/tensor.zig` resolves `cblas_sgemm` with
/// `dlopen` at startup and keeps the Zig kernels when that fails.
fn addPerf(mod: *std.Build.Module, options: *std.Build.Step.Options) void {
    mod.addOptions("build_options", options);
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
