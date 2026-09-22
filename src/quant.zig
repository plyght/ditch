//! ggml block quantisation formats (https://github.com/ggml-org/ggml): the
//! block layouts of Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q6_K and Q8_K, their
//! dequantisation, and the reference quantisers for the simple formats.
//! Everything follows ggml-quants.c so that files are bit-compatible with
//! llama.cpp.

const std = @import("std");
const tensor = @import("tensor.zig");

const f16ToF32 = tensor.f16ToF32;
const f32ToF16 = tensor.f32ToF16;

pub const qk = 32;
pub const qk_k = 256;
pub const k_scale_size = 12;

/// Element count of one block.
pub fn blockSize(dtype: tensor.DType) usize {
    return switch (dtype) {
        .f32, .f16, .bf16, .i64 => 1,
        .q8_0, .q4_0, .q4_1, .q5_0, .q5_1 => qk,
        .q4_k, .q6_k, .q8_k => qk_k,
    };
}

/// Byte size of one block.
pub fn blockBytes(dtype: tensor.DType) usize {
    return switch (dtype) {
        .f32 => 4,
        .f16, .bf16 => 2,
        .i64 => 8,
        .q8_0 => 2 + qk, // 34
        .q4_0 => 2 + qk / 2, // 18
        .q4_1 => 4 + qk / 2, // 20
        .q5_0 => 2 + 4 + qk / 2, // 22
        .q5_1 => 4 + 4 + qk / 2, // 24
        .q4_k => 4 + k_scale_size + qk_k / 2, // 144
        .q6_k => qk_k / 2 + qk_k / 4 + qk_k / 16 + 2, // 210
        .q8_k => 4 + qk_k + qk_k / 16 * 2, // 292
    };
}

inline fn readF16(bytes: []const u8) f32 {
    return f16ToF32(std.mem.readInt(u16, bytes[0..2], .little));
}

inline fn writeF16(bytes: []u8, v: f32) void {
    std.mem.writeInt(u16, bytes[0..2], f32ToF16(v), .little);
}

// ---------------------------------------------------------------------------
// Dequantisation
// ---------------------------------------------------------------------------

/// Dequantises `out.len` elements (a whole number of blocks) of `dtype` data.
pub fn dequantize(dtype: tensor.DType, bytes: []const u8, out: []f32) void {
    const bs = blockSize(dtype);
    const bb = blockBytes(dtype);
    std.debug.assert(out.len % bs == 0);
    const nb = out.len / bs;
    std.debug.assert(bytes.len >= nb * bb);
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const b = bytes[i * bb ..][0..bb];
        const y = out[i * bs ..][0..bs];
        switch (dtype) {
            .q8_0 => dequantQ8_0(b, y),
            .q4_0 => dequantQ4_0(b, y),
            .q4_1 => dequantQ4_1(b, y),
            .q5_0 => dequantQ5_0(b, y),
            .q5_1 => dequantQ5_1(b, y),
            .q4_k => dequantQ4K(b, y),
            .q6_k => dequantQ6K(b, y),
            .q8_k => dequantQ8K(b, y),
            .f32, .f16, .bf16, .i64 => unreachable,
        }
    }
}

fn dequantQ8_0(b: []const u8, y: []f32) void {
    const d = readF16(b);
    const qs = b[2..][0..qk];
    for (0..qk) |j| y[j] = @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j])))) * d;
}

fn dequantQ4_0(b: []const u8, y: []f32) void {
    const d = readF16(b);
    const qs = b[2..][0 .. qk / 2];
    for (0..qk / 2) |j| {
        const x0: i32 = @as(i32, qs[j] & 0x0f) - 8;
        const x1: i32 = @as(i32, qs[j] >> 4) - 8;
        y[j] = @as(f32, @floatFromInt(x0)) * d;
        y[j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
    }
}

fn dequantQ4_1(b: []const u8, y: []f32) void {
    const d = readF16(b);
    const m = readF16(b[2..]);
    const qs = b[4..][0 .. qk / 2];
    for (0..qk / 2) |j| {
        y[j] = @as(f32, @floatFromInt(qs[j] & 0x0f)) * d + m;
        y[j + qk / 2] = @as(f32, @floatFromInt(qs[j] >> 4)) * d + m;
    }
}

fn dequantQ5_0(b: []const u8, y: []f32) void {
    const d = readF16(b);
    const qh = std.mem.readInt(u32, b[2..6], .little);
    const qs = b[6..][0 .. qk / 2];
    for (0..qk / 2) |j| {
        const sh: u5 = @intCast(j);
        const xh0: u8 = @intCast(((qh >> sh) << 4) & 0x10);
        const xh1: u8 = @intCast((qh >> (sh + 12)) & 0x10);
        const x0: i32 = @as(i32, (qs[j] & 0x0f) | xh0) - 16;
        const x1: i32 = @as(i32, (qs[j] >> 4) | xh1) - 16;
        y[j] = @as(f32, @floatFromInt(x0)) * d;
        y[j + qk / 2] = @as(f32, @floatFromInt(x1)) * d;
    }
}

fn dequantQ5_1(b: []const u8, y: []f32) void {
    const d = readF16(b);
    const m = readF16(b[2..]);
    const qh = std.mem.readInt(u32, b[4..8], .little);
    const qs = b[8..][0 .. qk / 2];
    for (0..qk / 2) |j| {
        const sh: u5 = @intCast(j);
        const xh0: u8 = @intCast(((qh >> sh) << 4) & 0x10);
        const xh1: u8 = @intCast((qh >> (sh + 12)) & 0x10);
        const x0: u8 = (qs[j] & 0x0f) | xh0;
        const x1: u8 = (qs[j] >> 4) | xh1;
        y[j] = @as(f32, @floatFromInt(x0)) * d + m;
        y[j + qk / 2] = @as(f32, @floatFromInt(x1)) * d + m;
    }
}

/// 6-bit scale/min pairs of a Q4_K super-block (`get_scale_min_k4`).
fn scaleMinK4(j: usize, q: []const u8) struct { d: u8, m: u8 } {
    if (j < 4) return .{ .d = q[j] & 63, .m = q[j + 4] & 63 };
    return .{
        .d = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
        .m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
    };
}

fn dequantQ4K(b: []const u8, y: []f32) void {
    const d = readF16(b);
    const min = readF16(b[2..]);
    const scales = b[4..][0..k_scale_size];
    var q: []const u8 = b[4 + k_scale_size ..][0 .. qk_k / 2];
    var is: usize = 0;
    var o: usize = 0;
    var j: usize = 0;
    while (j < qk_k) : (j += 64) {
        const s1 = scaleMinK4(is, scales);
        const d1 = d * @as(f32, @floatFromInt(s1.d));
        const m1 = min * @as(f32, @floatFromInt(s1.m));
        const s2 = scaleMinK4(is + 1, scales);
        const d2 = d * @as(f32, @floatFromInt(s2.d));
        const m2 = min * @as(f32, @floatFromInt(s2.m));
        for (0..32) |l| {
            y[o + l] = d1 * @as(f32, @floatFromInt(q[l] & 0xf)) - m1;
        }
        for (0..32) |l| {
            y[o + 32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
        }
        o += 64;
        q = q[32..];
        is += 2;
    }
}

fn dequantQ6K(b: []const u8, y: []f32) void {
    var ql: []const u8 = b[0..][0 .. qk_k / 2];
    var qh: []const u8 = b[qk_k / 2 ..][0 .. qk_k / 4];
    var sc: []const u8 = b[qk_k / 2 + qk_k / 4 ..][0 .. qk_k / 16];
    const d = readF16(b[qk_k / 2 + qk_k / 4 + qk_k / 16 ..]);
    var o: usize = 0;
    var n: usize = 0;
    while (n < qk_k) : (n += 128) {
        for (0..32) |l| {
            const is = l / 16;
            const q1: i32 = @as(i32, (ql[l] & 0xf) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const q2: i32 = @as(i32, (ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[o + l] = d * sci(sc[is]) * @as(f32, @floatFromInt(q1));
            y[o + l + 32] = d * sci(sc[is + 2]) * @as(f32, @floatFromInt(q2));
            y[o + l + 64] = d * sci(sc[is + 4]) * @as(f32, @floatFromInt(q3));
            y[o + l + 96] = d * sci(sc[is + 6]) * @as(f32, @floatFromInt(q4));
        }
        o += 128;
        ql = ql[64..];
        qh = qh[32..];
        sc = sc[8..];
    }
}

inline fn sci(v: u8) f32 {
    return @floatFromInt(@as(i8, @bitCast(v)));
}

fn dequantQ8K(b: []const u8, y: []f32) void {
    const d: f32 = @bitCast(std.mem.readInt(u32, b[0..4], .little));
    const qs = b[4..][0..qk_k];
    for (0..qk_k) |j| y[j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
}

// ---------------------------------------------------------------------------
// Quantisation (reference implementations)
// ---------------------------------------------------------------------------

/// True if `dtype` can be produced by `quantize`.
pub fn canQuantize(dtype: tensor.DType) bool {
    return switch (dtype) {
        .q8_0, .q4_0, .q4_1, .q5_0, .q5_1 => true,
        else => false,
    };
}

/// Quantises `src` (a whole number of blocks) into `out`.
pub fn quantize(dtype: tensor.DType, src: []const f32, out: []u8) void {
    const bs = blockSize(dtype);
    const bb = blockBytes(dtype);
    std.debug.assert(src.len % bs == 0);
    const nb = src.len / bs;
    std.debug.assert(out.len >= nb * bb);
    var i: usize = 0;
    while (i < nb) : (i += 1) {
        const x = src[i * bs ..][0..bs];
        const b = out[i * bb ..][0..bb];
        switch (dtype) {
            .q8_0 => quantQ8_0(x, b),
            .q4_0 => quantQ4_0(x, b),
            .q4_1 => quantQ4_1(x, b),
            .q5_0 => quantQ5_0(x, b),
            .q5_1 => quantQ5_1(x, b),
            else => unreachable,
        }
    }
}

fn quantQ8_0(x: []const f32, b: []u8) void {
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(v));
    const d = amax / 127.0;
    const id: f32 = if (d != 0) 1.0 / d else 0.0;
    writeF16(b, d);
    for (0..qk) |j| {
        const q: i8 = @intFromFloat(@round(x[j] * id));
        b[2 + j] = @bitCast(q);
    }
}

/// `(int8_t)(v)` in C: truncation toward zero.
inline fn trunc8(v: f32) i32 {
    return @intFromFloat(@trunc(v));
}

fn quantQ4_0(x: []const f32, b: []u8) void {
    var amax: f32 = 0;
    var max: f32 = 0;
    for (x) |v| {
        if (amax < @abs(v)) {
            amax = @abs(v);
            max = v;
        }
    }
    const d = max / -8.0;
    const id: f32 = if (d != 0) 1.0 / d else 0.0;
    writeF16(b, d);
    for (0..qk / 2) |j| {
        const x0 = x[j] * id;
        const x1 = x[j + qk / 2] * id;
        const xi0: u8 = @intCast(@min(15, trunc8(x0 + 8.5)));
        const xi1: u8 = @intCast(@min(15, trunc8(x1 + 8.5)));
        b[2 + j] = xi0 | (xi1 << 4);
    }
}

fn quantQ4_1(x: []const f32, b: []u8) void {
    var min: f32 = std.math.floatMax(f32);
    var max: f32 = -std.math.floatMax(f32);
    for (x) |v| {
        min = @min(min, v);
        max = @max(max, v);
    }
    const d = (max - min) / 15.0;
    const id: f32 = if (d != 0) 1.0 / d else 0.0;
    writeF16(b, d);
    writeF16(b[2..], min);
    for (0..qk / 2) |j| {
        const x0 = (x[j] - min) * id;
        const x1 = (x[j + qk / 2] - min) * id;
        const xi0: u8 = @intCast(@min(15, trunc8(x0 + 0.5)));
        const xi1: u8 = @intCast(@min(15, trunc8(x1 + 0.5)));
        b[4 + j] = xi0 | (xi1 << 4);
    }
}

fn quantQ5_0(x: []const f32, b: []u8) void {
    var amax: f32 = 0;
    var max: f32 = 0;
    for (x) |v| {
        if (amax < @abs(v)) {
            amax = @abs(v);
            max = v;
        }
    }
    const d = max / -16.0;
    const id: f32 = if (d != 0) 1.0 / d else 0.0;
    writeF16(b, d);
    var qh: u32 = 0;
    for (0..qk / 2) |j| {
        const x0 = x[j] * id;
        const x1 = x[j + qk / 2] * id;
        const xi0: u8 = @intCast(@min(31, trunc8(x0 + 16.5)));
        const xi1: u8 = @intCast(@min(31, trunc8(x1 + 16.5)));
        b[6 + j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
        const sh: u5 = @intCast(j);
        qh |= @as(u32, (xi0 & 0x10) >> 4) << sh;
        qh |= @as(u32, (xi1 & 0x10) >> 4) << (sh + qk / 2);
    }
    std.mem.writeInt(u32, b[2..6], qh, .little);
}

fn quantQ5_1(x: []const f32, b: []u8) void {
    var min: f32 = std.math.floatMax(f32);
    var max: f32 = -std.math.floatMax(f32);
    for (x) |v| {
        min = @min(min, v);
        max = @max(max, v);
    }
    const d = (max - min) / 31.0;
    const id: f32 = if (d != 0) 1.0 / d else 0.0;
    writeF16(b, d);
    writeF16(b[2..], min);
    var qh: u32 = 0;
    for (0..qk / 2) |j| {
        const x0 = (x[j] - min) * id;
        const x1 = (x[j + qk / 2] - min) * id;
        const xi0: u8 = @intCast(@min(31, trunc8(x0 + 0.5)));
        const xi1: u8 = @intCast(@min(31, trunc8(x1 + 0.5)));
        b[8 + j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
        const sh: u5 = @intCast(j);
        qh |= @as(u32, (xi0 & 0x10) >> 4) << sh;
        qh |= @as(u32, (xi1 & 0x10) >> 4) << (sh + qk / 2);
    }
    std.mem.writeInt(u32, b[4..8], qh, .little);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn testValues(buf: []f32) void {
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    for (buf) |*v| v.* = (r.float(f32) * 2.0 - 1.0) * 2.0;
    buf[0] = 0;
    buf[5] = 2.5;
    buf[17] = -2.5;
}

test "q8_0 quantise then dequantise stays within half a step" {
    var x: [64]f32 = undefined;
    testValues(&x);
    var blocks: [2 * 34]u8 = undefined;
    quantize(.q8_0, &x, &blocks);
    var y: [64]f32 = undefined;
    dequantize(.q8_0, &blocks, &y);
    for (0..2) |blk| {
        var amax: f32 = 0;
        for (x[blk * 32 ..][0..32]) |v| amax = @max(amax, @abs(v));
        const d = f16ToF32(f32ToF16(amax / 127.0));
        for (0..32) |j| {
            const i = blk * 32 + j;
            // half a quantisation step plus the f16 rounding of the scale
            try std.testing.expect(@abs(x[i] - y[i]) <= d * 0.5 + amax * 1e-3);
        }
    }
    // the largest element is reproduced (almost) exactly
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), y[5], 2.5 * 1e-3);
    // known block: d = 2.5 / 127, q[5] = 127
    try std.testing.expectEqual(@as(u8, 127), blocks[2 + 5]);
}

test "q4_0 q4_1 q5_0 q5_1 round trip error bounds" {
    var x: [64]f32 = undefined;
    testValues(&x);
    const cases = [_]struct { dtype: tensor.DType, steps: f32 }{
        .{ .dtype = .q4_0, .steps = 8 },
        .{ .dtype = .q4_1, .steps = 15 },
        .{ .dtype = .q5_0, .steps = 16 },
        .{ .dtype = .q5_1, .steps = 31 },
    };
    for (cases) |c| {
        var blocks: [2 * 24]u8 = undefined;
        quantize(c.dtype, &x, &blocks);
        var y: [64]f32 = undefined;
        dequantize(c.dtype, blocks[0 .. 2 * blockBytes(c.dtype)], &y);
        for (0..2) |blk| {
            var lo: f32 = std.math.floatMax(f32);
            var hi: f32 = -std.math.floatMax(f32);
            var amax: f32 = 0;
            for (x[blk * 32 ..][0..32]) |v| {
                lo = @min(lo, v);
                hi = @max(hi, v);
                amax = @max(amax, @abs(v));
            }
            const step = switch (c.dtype) {
                .q4_0, .q5_0 => amax / c.steps,
                else => (hi - lo) / c.steps,
            };
            for (0..32) |j| {
                const i = blk * 32 + j;
                // Symmetric formats clamp the element opposite the maximum to a full step.
                try std.testing.expect(@abs(x[i] - y[i]) <= step * 1.01 + amax * 2e-3);
            }
        }
    }
}

test "q4_K dequantisation of a hand-built block" {
    var b = [_]u8{0} ** 144;
    writeF16(b[0..], 0.5); // d
    writeF16(b[2..], 0.25); // dmin
    // scales: sub-block 0 scale 4 min 2, sub-block 1 scale 8 min 1 (6-bit fields, j < 4)
    b[4 + 0] = 4;
    b[4 + 4] = 2;
    b[4 + 1] = 8;
    b[4 + 5] = 1;
    // sub-block 4 (j = 4): d = (q[8] & 0xF) | ((q[0] >> 6) << 4), m = (q[8] >> 4) | ((q[4] >> 6) << 4)
    b[4 + 8] = 0x35; // d = 5, m = 3
    b[4 + 0] |= 0x40; // d += 16 -> 21
    // quants: first 32 elements low nibbles, next 32 high nibbles of the same bytes
    const q = b[16..];
    q[0] = 0x3 | (0x9 << 4); // element 0 = 3 (scale 4, min 2), element 32 = 9 (scale 8, min 1)
    q[64] = 0x7; // element 128 = 7 -> sub-block 4
    var y: [256]f32 = undefined;
    dequantize(.q4_k, &b, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 * 4 * 3 - 0.25 * 2), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 * 8 * 9 - 0.25 * 1), y[32], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25 * 2), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 * 21 * 7 - 0.25 * 3), y[128], 1e-6);
}

test "q6_K dequantisation of a hand-built block" {
    var b = [_]u8{0} ** 210;
    writeF16(b[208..], 0.125); // d at the end
    const ql = b[0..128];
    const qh = b[128..192];
    const sc = b[192..208];
    sc[0] = @bitCast(@as(i8, 3)); // scale of elements 0..15
    sc[2] = @bitCast(@as(i8, -2)); // scale of elements 32..47
    // element 0: low nibble of ql[0], high bits 0..1 of qh[0]; value 0x2A = 42 -> 42 - 32 = 10
    ql[0] = 0xA;
    qh[0] = 0x2;
    // element 32: low nibble of ql[32], bits 2..3 of qh[0]; value 0x11 = 17 -> -15
    ql[32] = 0x1;
    qh[0] |= 0x1 << 2;
    var y: [256]f32 = undefined;
    dequantize(.q6_k, &b, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125 * 3 * 10), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125 * -2 * -15), y[32], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125 * 3 * -32), y[1], 1e-6);
}

test "q8_K dequantisation" {
    var b = [_]u8{0} ** 292;
    const d: f32 = 0.75;
    std.mem.writeInt(u32, b[0..4], @bitCast(d), .little);
    b[4 + 3] = @bitCast(@as(i8, -100));
    b[4 + 255] = 64;
    var y: [256]f32 = undefined;
    dequantize(.q8_k, &b, &y);
    try std.testing.expectApproxEqAbs(@as(f32, -75), y[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 48), y[255], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), y[0]);
}
