const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const is_macos = target.result.os.tag == .macos;

    // Apple's Accelerate framework (cblas_sgemm) as the matmul backend for
    // prefill-shaped calls; compiled in when building for macOS, ignored
    // everywhere else. `--accelerate=false` turns it off at run time. The
    // framework is not linked: the macOS release binaries are cross-compiled
    // from Linux, where it does not exist, so src/tensor.zig resolves
    // cblas_sgemm with dlopen and keeps the Zig kernels when that fails.
    const accelerate = b.option(bool, "accelerate", "Use Apple's Accelerate framework for large matrix products (macOS only, default: true on macOS)") orelse is_macos;
    // Kernel shape overrides; 0 means "pick from the target's register file"
    // (see `src/tensor.zig`). Only for benchmarking a machine by hand.
    const vector_width = b.option(u32, "vector-width", "Kernel vector width in f32 lanes (0 = choose from the target, the default)") orelse 0;
    const tile_rows = b.option(u32, "tile-rows", "Weight rows per register tile (0 = choose from the target, the default)") orelse 0;
    const tile_inputs = b.option(u32, "tile-inputs", "Input rows per register tile (0 = choose from the target, the default)") orelse 0;

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
    options.addOption(bool, "accelerate", accelerate and is_macos);
    options.addOption(u32, "vector_width", vector_width);
    options.addOption(u32, "tile_rows", tile_rows);
    options.addOption(u32, "tile_inputs", tile_inputs);
    // `zig build test -Dupdate-snapshot`: rewrite tests/config_variants/snapshot.txt
    // after an intended change to how a model definition reads config.json.
    options.addOption(bool, "update_snapshot", b.option(bool, "update-snapshot", "Rewrite the parsed-config snapshot of the model definitions (tests/config_variants/snapshot.txt)") orelse false);

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    addLua(b, root);
    addHarness(b, root);
    const definitions = modelDefinitions(b);
    root.addImport("model_definitions", definitions);
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
    addHarness(b, test_mod);
    test_mod.addImport("model_definitions", definitions);
    test_mod.addOptions("build_options", options);
    if (metal) addMetal(b, test_mod);
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

/// `ditch verify`'s reference harness: the Python files in tools/ that run a
/// model's official implementation, embedded (`@embedFile("harness/<name>")`)
/// so the binary carries the exact copies in the repository.
fn addHarness(b: *std.Build, mod: *std.Build.Module) void {
    const files = [_][]const u8{
        "verify_reference.py", "probe_reference.py",  "ref_stream.py",  "ref_lazy_moe.py", "lazy_checkpoint.py",
        "ref_deepseek_v4.py",  "ref_deepseek_v41.py", "ref_kimi_k3.py", "ref_mimo_v2.py",  "check_abliteration.py",
    };
    for (files) |f| mod.addAnonymousImport(b.fmt("harness/{s}", .{f}), .{ .root_source_file = b.path(b.fmt("tools/{s}", .{f})) });
}

/// The built-in Lua model definitions: every `*.lua` file of src/models
/// (families) and src/models/lib (libraries a definition can `require`),
/// embedded by a generated module, so adding a family is adding a file.
fn modelDefinitions(b: *std.Build) *std.Build.Module {
    const io = b.graph.io;
    const wf = b.addWriteFiles();
    var index: std.Io.Writer.Allocating = .init(b.allocator);
    const w = &index.writer;
    w.writeAll("pub const File = struct { name: []const u8, source: []const u8 };\n") catch @panic("OOM");
    for ([_][]const u8{ "files", "libs" }, [_][]const u8{ "src/models", "src/models/lib" }) |decl, sub| {
        var names: std.ArrayList([]const u8) = .empty;
        if (b.build_root.handle.openDir(io, sub, .{ .iterate = true })) |dir_const| {
            var dir = dir_const;
            defer dir.close(io);
            var it = dir.iterate();
            while (it.next(io) catch null) |entry| {
                if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".lua")) continue;
                if (std.mem.eql(u8, entry.name, "prelude.lua")) continue;
                names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
            }
        } else |_| {}
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        w.print("pub const {s} = [_]File{{\n", .{decl}) catch @panic("OOM");
        for (names.items) |name| {
            const dest = b.fmt("{s}/{s}", .{ decl, name });
            _ = wf.addCopyFile(b.path(b.fmt("{s}/{s}", .{ sub, name })), dest);
            w.print("    .{{ .name = \"{s}\", .source = @embedFile(\"{s}\") }},\n", .{ name, dest }) catch @panic("OOM");
        }
        w.writeAll("};\n") catch @panic("OOM");
    }
    _ = wf.addCopyFile(b.path("src/models/prelude.lua"), "prelude.lua");
    w.writeAll("pub const prelude = @embedFile(\"prelude.lua\");\n") catch @panic("OOM");
    const root = wf.add("model_definitions.zig", index.written());
    return b.createModule(.{ .root_source_file = root });
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
