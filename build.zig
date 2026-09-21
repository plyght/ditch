const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

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

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
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
