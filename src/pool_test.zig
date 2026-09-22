//! Regression tests for `tensor.Pool`'s resident worker threads.
//!
//! The case that matters here is the very first job: a worker thread that has
//! not run a single instruction yet when the job is published must still take
//! part in it. If it instead adopts the already-bumped generation as its
//! baseline, it waits for the *next* job and never reports this one as done,
//! and the caller waits for it forever — a hang, not a wrong answer, because
//! the calling thread does claim and run every chunk itself.
//!
//! These tests therefore start a persistent pool and dispatch immediately,
//! many times over, which is exactly the window that race lives in.

const std = @import("std");
const tensor = @import("tensor.zig");

const Counter = struct {
    items: []std.atomic.Value(u32),

    fn touch(self: *const Counter, start: usize, end: usize) void {
        for (start..end) |i| _ = self.items[i].fetchAdd(1, .monotonic);
    }
};

test "a job dispatched immediately after the pool starts completes" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const n = 4096;
    const items = try gpa.alloc(std.atomic.Value(u32), n);
    defer gpa.free(items);

    // Many short-lived pools: each one dispatches in the window between
    // spawning its workers and those workers first being scheduled.
    for (0..32) |_| {
        for (items) |*it| it.* = .init(0);
        var pool = tensor.Pool.initPersistent(gpa, io, 4);
        defer pool.deinit();
        const ctx = Counter{ .items = items };
        pool.parallelFor(n, &ctx, Counter.touch);
        for (items) |*it| try std.testing.expectEqual(@as(u32, 1), it.load(.monotonic));
    }
}

test "consecutive jobs on one pool each run exactly once per element" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    const n = 1000;
    const items = try gpa.alloc(std.atomic.Value(u32), n);
    defer gpa.free(items);
    for (items) |*it| it.* = .init(0);

    var pool = tensor.Pool.initPersistent(gpa, io, 3);
    defer pool.deinit();
    const ctx = Counter{ .items = items };
    const rounds = 50;
    for (0..rounds) |_| pool.parallelFor(n, &ctx, Counter.touch);
    for (items) |*it| try std.testing.expectEqual(@as(u32, rounds), it.load(.monotonic));

    // A single-element range and an empty one are handled without a dispatch.
    pool.parallelFor(0, &ctx, Counter.touch);
    pool.parallelFor(1, &ctx, Counter.touch);
    try std.testing.expectEqual(@as(u32, rounds + 1), items[0].load(.monotonic));
    try std.testing.expectEqual(@as(u32, rounds), items[1].load(.monotonic));
}
