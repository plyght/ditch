//! The Vulkan backend on a real device, when this machine has one (a GPU, or
//! Mesa's lavapipe, which is how CI runs it): every kernel against the CPU
//! reference, both memory paths, the residency cache and the dispatch grid
//! for more rows than one grid dimension holds. Skipped where no Vulkan
//! device exists, so `zig build test` stays green everywhere.

const std = @import("std");
const compute = @import("../compute.zig");
const selftest = @import("../selftest.zig");
const tensor = @import("../tensor.zig");

const testing = std.testing;

fn openOrSkip(gpa: std.mem.Allocator, io: std.Io, budget: u64) !compute.Device {
    if (!compute.vulkan_supported) return error.SkipZigTest;
    const r = compute.select(gpa, .vulkan, .{ .io = io, .memory_budget = budget }) catch return error.SkipZigTest;
    return r.device;
}

test "the Vulkan backend matches the CPU reference on every kernel" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const pool = tensor.Pool.init(io, 1);
    var dev = try openOrSkip(gpa, io, 0);
    defer dev.deinit();
    var report = try selftest.check(gpa, &pool, &dev, 4242);
    defer gpa.free(report.kernels);
    for (report.kernels) |k| {
        if (!k.ok) std.debug.print("{s}: max abs {e}, max rel {e}\n", .{ k.name, k.max_abs, k.max_rel });
        try testing.expect(k.on_device);
    }
    try testing.expect(!report.failed());
}

test "resident weight tiles are uploaded once and give the same result" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const pool = tensor.Pool.init(io, 1);
    var dev = try openOrSkip(gpa, io, 64 << 20);
    defer dev.deinit();
    dev.min_macs = 0;
    const stable_before = compute.weights_stable;
    defer compute.weights_stable = stable_before;
    compute.weights_stable = true;

    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const rows = 70;
    const cols = 45;
    const n = 3;
    const wf = try gpa.alloc(f32, rows * cols);
    defer gpa.free(wf);
    for (wf) |*v| v.* = rnd.floatNorm(f32);
    const wbytes = try gpa.alloc(u8, rows * cols * 2);
    defer gpa.free(wbytes);
    tensor.convertFromF32(.bf16, wf, wbytes);
    const w = tensor.Weight{ .data = wbytes, .dtype = .bf16, .rows = rows, .cols = cols };
    const x = try gpa.alloc(f32, n * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);

    const want = try gpa.alloc(f32, n * rows);
    defer gpa.free(want);
    try tensor.matmulT(&pool, gpa, want, x, n, w, null);
    const served_before = compute.served.load(.monotonic);
    for (0..3) |_| {
        const got = try gpa.alloc(f32, n * rows);
        defer gpa.free(got);
        try compute.matmulTOn(&dev, &pool, gpa, got, x, n, w, null);
        for (got, want) |g, e| try testing.expectApproxEqAbs(e, g, 1e-4);
    }
    try testing.expectEqual(served_before + 3, compute.served.load(.monotonic));
    // The same tile, three calls: uploaded once, then served from the cache.
    const backend = @import("backend.zig");
    const self: *backend.Backend = @ptrCast(@alignCast(dev.ctx.?));
    try testing.expectEqual(@as(u64, 1), self.residency.misses);
    try testing.expectEqual(@as(u64, 2), self.residency.hits);
}

test "row reductions over more rows than one grid dimension allows" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const pool = tensor.Pool.init(io, 1);
    var dev = try openOrSkip(gpa, io, 0);
    defer dev.deinit();
    dev.min_macs = 0;
    // A vocabulary-sized matrix: 70000 rows is more than the 65535
    // workgroups every device must allow in x.
    const rows = 70000;
    const cols = 3;
    const wf = try gpa.alloc(f32, rows * cols);
    defer gpa.free(wf);
    for (wf, 0..) |*v, i| v.* = @floatFromInt(i % 7);
    const w = tensor.Weight{ .data = std.mem.sliceAsBytes(wf), .dtype = .f32, .rows = rows, .cols = cols };
    const want = try gpa.alloc(f32, rows);
    defer gpa.free(want);
    const got = try gpa.alloc(f32, rows);
    defer gpa.free(got);
    try tensor.rowNorms(&pool, gpa, want, w);
    try compute.rowNormsOn(&dev, &pool, gpa, got, w);
    for (got, want) |g, e| try testing.expectApproxEqAbs(e, g, 1e-5);

    const x = try gpa.alloc(f32, rows * cols);
    defer gpa.free(x);
    @memcpy(x, wf);
    try compute.softmaxRowsOn(&dev, x, rows, cols);
    for (0..rows) |r| {
        const row = x[r * cols ..][0..cols];
        try testing.expectApproxEqAbs(@as(f32, 1), row[0] + row[1] + row[2], 1e-5);
    }
}
