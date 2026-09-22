//! The compute backend seam: every kernel the forward pass needs, dispatched
//! to a `Device`.
//!
//! The CPU backend is the reference implementation and the default: its
//! functions are thin wrappers that call `tensor.zig` with the same arguments,
//! so a CPU run is bit-identical to one that never went through this file.
//! Another backend (today: Metal, see `src/metal/`) may implement any subset of
//! the operations; anything it does not implement, or refuses for a given shape
//! or dtype, falls back to the CPU kernel. That is what makes the seam safe for
//! ditch's access pattern: weights are memory-mapped or streamed tile by tile
//! and may be quantised, and a backend that cannot take a tile simply lets the
//! CPU have it.
//!
//! What is dispatched to a device today
//! ------------------------------------
//! `matmulT`, `matvecTMulti` and `rowNorms` — the operations that dominate the
//! FLOPs and that read a whole weight tile, which is exactly the unit a device
//! can take, compute on and drop again. The element-wise kernels (norms,
//! softmax, rope, gated activations, per-head attention) are called per row or
//! per head from inside the thread pool, on data that lives in host memory; a
//! device round trip per row would cost more than the arithmetic. Their device
//! kernels exist and are checked by `ditch selftest`, ready for a forward pass
//! that keeps activations resident, but the forward pass uses the CPU for them.
//! `Device.dispatched` documents this per operation.
//!
//! Device memory
//! -------------
//! A device never needs the whole model. `matmulT` hands the backend one weight
//! tile; the backend uploads it, computes and drops it ("upload, compute,
//! drop"). With `--gpu-memory N` a `Residency` cache keeps the hottest tiles on
//! the device up to N bytes, but only when the host-side weights are stable
//! (memory-mapped): in streamed and warp modes the same host address is reused
//! for different tensors, so caching by address would be wrong and is disabled.

const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const Pool = tensor.Pool;
const Weight = tensor.Weight;
const Delta = tensor.Delta;

pub const metal_supported = builtin.os.tag == .macos and @import("build_options").metal;

/// A backend may refuse any call; the caller then runs the CPU kernel.
pub const Error = error{
    /// This backend does not implement the operation for this shape or dtype.
    Unsupported,
    /// The device rejected the work (out of device memory, command failure).
    DeviceFailed,
    OutOfMemory,
};

pub const Kind = enum {
    /// Probe for a usable GPU, fall back to the CPU with a note on stderr.
    auto,
    cpu,
    metal,

    pub fn parse(s: []const u8) ?Kind {
        inline for (@typeInfo(Kind).@"enum".fields) |f| {
            if (std.ascii.eqlIgnoreCase(s, f.name)) return @enumFromInt(f.value);
        }
        if (std.ascii.eqlIgnoreCase(s, "gpu")) return .auto;
        return null;
    }

    /// The backends that can be asked for by name, for help texts and errors.
    pub const names = "auto|cpu|metal";
};

/// Element-wise activation, mirrored from `tensor.Activation` so a backend can
/// switch on a stable integer.
pub const ActivationCode = u32;

pub const RopeStyle = enum(u32) { neox = 0, gptj = 1 };

/// Everything a backend implements. A null entry means "not implemented";
/// the CPU kernel is used instead. Every function may also return
/// `error.Unsupported` for a shape or dtype it cannot take.
pub const VTable = struct {
    deinit: ?*const fn (ctx: *anyopaque) void = null,

    /// `out[n][w.rows] = x[n][w.cols] @ W^T`.
    matmulT: ?*const fn (ctx: *anyopaque, out: []f32, x: []const f32, n: usize, w: Weight) Error!void = null,
    /// `out[q][w.cols] = y[q][w.rows] @ W`.
    matvecTMulti: ?*const fn (ctx: *anyopaque, out: []f32, w: Weight, y: []const f32, q: usize) Error!void = null,
    /// `out[w.rows] = ||W_i||_2`.
    rowNorms: ?*const fn (ctx: *anyopaque, out: []f32, w: Weight) Error!void = null,

    /// `scores[p] = scale * dot(q, k[p * stride ..][0..q.len])`.
    attentionScores: ?*const fn (ctx: *anyopaque, scores: []f32, q: []const f32, k: []const f32, stride: usize, scale: f32) Error!void = null,
    /// `out[j] = Σ_p scores[p] * v[p * stride + j]`.
    attentionValues: ?*const fn (ctx: *anyopaque, out: []f32, scores: []const f32, v: []const f32, stride: usize) Error!void = null,
    /// `out[i][j] = act(gate[i][j]) * up[i][j]` over `n` rows of `len`.
    gatedActivation: ?*const fn (ctx: *anyopaque, act: ActivationCode, out: []f32, gate: []const f32, up: ?[]const f32, n: usize, len: usize, in_stride: usize, out_stride: usize) Error!void = null,
    /// RMS normalisation of `n` rows of `len` elements.
    rmsnormRows: ?*const fn (ctx: *anyopaque, out: []f32, x: []const f32, weight: []const f32, n: usize, len: usize, eps: f32, gemma_style: bool) Error!void = null,
    /// LayerNorm of `n` rows of `len` elements.
    layernormRows: ?*const fn (ctx: *anyopaque, out: []f32, x: []const f32, weight: []const f32, bias: ?[]const f32, n: usize, len: usize, eps: f32, one_plus: bool) Error!void = null,
    /// In-place softmax of `n` rows of `len` elements.
    softmaxRows: ?*const fn (ctx: *anyopaque, x: []f32, n: usize, len: usize) Error!void = null,
    /// Rotary embeddings of `n` head vectors of `dim` elements at `pos[i]`.
    ropeRows: ?*const fn (ctx: *anyopaque, x: []f32, n: usize, dim: usize, cos: []const f32, sin: []const f32, half: usize, pos: []const u32, style: RopeStyle) Error!void = null,
};

/// The CPU backend implements every operation itself, so its vtable is empty
/// and every call takes the fallback path into `tensor.zig`.
const cpu_vtable = VTable{};

pub const Device = struct {
    kind: Kind,
    /// Human-readable name, e.g. "cpu" or "Apple M2 Pro (Metal)".
    name: []const u8,
    vtable: *const VTable = &cpu_vtable,
    ctx: ?*anyopaque = null,
    /// Bytes of device memory the residency cache may hold (0 = none).
    memory_budget: u64 = 0,
    /// Work below this many multiply-accumulates stays on the CPU whatever the
    /// backend implements (`ditch selftest` sets it to zero so that even the
    /// smallest shape is checked on the device).
    min_macs: u64 = min_device_macs,

    pub fn deinit(self: *Device) void {
        if (self.ctx) |c| if (self.vtable.deinit) |f| f(c);
        self.ctx = null;
        self.vtable = &cpu_vtable;
    }

    pub fn isCpu(self: *const Device) bool {
        return self.ctx == null;
    }

    /// Whether the forward pass actually routes `op` to this device (see the
    /// module comment): only the weight-tile operations are dispatched.
    pub fn dispatched(self: *const Device, comptime op: []const u8) bool {
        if (self.isCpu()) return false;
        const dispatched_ops = [_][]const u8{ "matmulT", "matvecTMulti", "rowNorms" };
        inline for (dispatched_ops) |d| {
            if (std.mem.eql(u8, d, op)) return @field(self.vtable, op) != null;
        }
        return false;
    }
};

pub const cpu_device = Device{ .kind = .cpu, .name = "cpu" };

/// The device every kernel in this file uses. Selected once at start-up
/// (`selectInto`) and read-only afterwards, so kernels may run on any thread.
pub var active: Device = cpu_device;

/// Whether host weight pointers stay valid for the run (memory-mapped mode).
/// Streamed and warp modes reuse buffers, so device-side caching by address is
/// unsound there and residency is disabled.
pub var weights_stable: bool = false;

pub fn setWeightsStable(stable: bool) void {
    weights_stable = stable;
}

/// Work below this many multiply-accumulates stays on the CPU: a device
/// dispatch plus the tile upload costs more than the arithmetic saves.
pub const min_device_macs: u64 = 1 << 20;

/// Weight-tile operations the active device actually served (it neither
/// declined them nor fell below `min_macs`). Lets a run report, and a test
/// assert, that the GPU did the work rather than silently leaving it all to the
/// CPU.
pub var served: std.atomic.Value(u64) = .init(0);

fn noteServed() void {
    _ = served.fetchAdd(1, .monotonic);
}

fn deviceWorthIt(dev: *const Device, n: usize, rows: usize, cols: usize) bool {
    const macs = @as(u64, n) * @as(u64, rows) * @as(u64, cols);
    return macs >= dev.min_macs;
}

// ---------------------------------------------------------------------------
// Device selection
// ---------------------------------------------------------------------------

pub const SelectOptions = struct {
    /// Device memory the backend may keep resident for hot weights.
    memory_budget: u64 = 0,
    /// Lets a backend put waiting threads to sleep (`std.Io.Mutex`). A GPU
    /// backend needs it; without one it reports itself unavailable, which
    /// `auto` turns into the CPU.
    io: ?std.Io = null,
    /// Overrides `Device.min_macs` (see `Settings.device_min_macs`).
    min_macs: ?u64 = null,
};

pub const SelectResult = struct {
    device: Device,
    /// Set when `auto` (or an explicit backend that then failed to probe as
    /// requested) fell back to the CPU; one line for stderr.
    note: ?[]const u8 = null,
};

/// Opens `kind`, or returns an error for an explicitly requested backend that
/// is not available. `auto` never fails: it falls back to the CPU with a note.
pub fn select(gpa: Allocator, kind: Kind, opts: SelectOptions) !SelectResult {
    var r = try selectDevice(gpa, kind, opts);
    if (opts.min_macs) |m| r.device.min_macs = m;
    return r;
}

fn selectDevice(gpa: Allocator, kind: Kind, opts: SelectOptions) !SelectResult {
    switch (kind) {
        .cpu => return .{ .device = cpu_device },
        .metal => {
            if (!metal_supported) return error.DeviceUnavailable;
            const dev = openMetal(gpa, opts) catch return error.DeviceUnavailable;
            return .{ .device = dev };
        },
        .auto => {
            if (metal_supported) {
                if (openMetal(gpa, opts)) |dev| {
                    return .{ .device = dev };
                } else |_| {
                    return .{ .device = cpu_device, .note = "no usable Metal device: running on the CPU" };
                }
            }
            // On a Mac the user may well expect Metal, so say why it is not there.
            if (builtin.os.tag == .macos) {
                return .{ .device = cpu_device, .note = "this build has no GPU backend (rebuild with -Dmetal): running on the CPU" };
            }
            return .{ .device = cpu_device, .note = null };
        },
    }
}

fn openMetal(gpa: Allocator, opts: SelectOptions) !Device {
    if (!metal_supported) return error.DeviceUnavailable;
    const metal = @import("metal/backend.zig");
    const io = opts.io orelse return error.DeviceUnavailable;
    return metal.open(gpa, io, opts.memory_budget);
}

/// Selects `kind` into `active`. Returns the note, if any, for stderr.
pub fn selectInto(gpa: Allocator, kind: Kind, opts: SelectOptions) !?[]const u8 {
    const r = try select(gpa, kind, opts);
    active = r.device;
    return r.note;
}

pub fn shutdown() void {
    active.deinit();
    active = cpu_device;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// `out[n][rows] = x[n][cols] @ W^T (+ x @ (B A)^T)`; see `tensor.matmulT`.
pub fn matmulT(pool: *const Pool, gpa: Allocator, out: []f32, x: []const f32, n: usize, w: Weight, delta: ?*const Delta) !void {
    return matmulTOn(&active, pool, gpa, out, x, n, w, delta);
}

pub fn matmulTOn(dev: *const Device, pool: *const Pool, gpa: Allocator, out: []f32, x: []const f32, n: usize, w: Weight, delta: ?*const Delta) !void {
    if (dev.vtable.matmulT) |f| {
        if (n > 0 and w.rows > 0 and deviceWorthIt(dev, n, w.rows, w.cols)) {
            if (f(dev.ctx.?, out, x, n, w)) |_| {
                noteServed();
                if (delta) |d| try applyDelta(gpa, out, x, n, w, d);
                return;
            } else |err| switch (err) {
                error.Unsupported => {},
                else => |e| return e,
            }
        }
    }
    return tensor.matmulT(pool, gpa, out, x, n, w, delta);
}

/// The low-rank correction of `tensor.matmulT`, applied on the host after a
/// device matmul. Same arithmetic and same order as the CPU kernel.
fn applyDelta(gpa: Allocator, out: []f32, x: []const f32, n: usize, w: Weight, dl: *const Delta) !void {
    const xa = try gpa.alloc(f32, n * dl.rank);
    defer gpa.free(xa);
    for (0..n) |i| {
        const xi = x[i * w.cols ..][0..w.cols];
        for (0..dl.rank) |j| xa[i * dl.rank + j] = tensor.dot(dl.a[j * w.cols ..][0..w.cols], xi);
    }
    for (0..n) |i| {
        const xai = xa[i * dl.rank ..][0..dl.rank];
        for (0..w.rows) |r| {
            var v: f32 = 0;
            for (0..dl.rank) |j| v += dl.b[r * dl.rank + j] * xai[j];
            out[i * w.rows + r] += v;
        }
    }
}

/// `out[q][cols] = y[q][rows] @ W`; see `tensor.matvecTMulti`.
pub fn matvecTMulti(pool: *const Pool, gpa: Allocator, out: []f32, w: Weight, y: []const f32, q: usize) !void {
    return matvecTMultiOn(&active, pool, gpa, out, w, y, q);
}

pub fn matvecTMultiOn(dev: *const Device, pool: *const Pool, gpa: Allocator, out: []f32, w: Weight, y: []const f32, q: usize) !void {
    if (dev.vtable.matvecTMulti) |f| {
        if (q > 0 and w.rows > 0 and deviceWorthIt(dev, q, w.rows, w.cols)) {
            if (f(dev.ctx.?, out, w, y, q)) |_| {
                noteServed();
                return;
            } else |err| switch (err) {
                error.Unsupported => {},
                else => |e| return e,
            }
        }
    }
    return tensor.matvecTMulti(pool, gpa, out, w, y, q);
}

/// `out[cols] = W^T y`.
pub fn matvecT(pool: *const Pool, gpa: Allocator, out: []f32, w: Weight, y: []const f32) !void {
    return matvecTMulti(pool, gpa, out, w, y, 1);
}

/// `out[rows] = ||W_i||_2`.
pub fn rowNorms(pool: *const Pool, gpa: Allocator, out: []f32, w: Weight) !void {
    return rowNormsOn(&active, pool, gpa, out, w);
}

pub fn rowNormsOn(dev: *const Device, pool: *const Pool, gpa: Allocator, out: []f32, w: Weight) !void {
    if (dev.vtable.rowNorms) |f| {
        if (w.rows > 0 and deviceWorthIt(dev, 1, w.rows, w.cols)) {
            if (f(dev.ctx.?, out, w)) |_| {
                noteServed();
                return;
            } else |err| switch (err) {
                error.Unsupported => {},
                else => |e| return e,
            }
        }
    }
    return tensor.rowNorms(pool, gpa, out, w);
}

// The kernels below are called per row or per head from inside the thread
// pool; the forward pass always runs them on the CPU (see the module comment).
// The `*On` forms exist so `ditch selftest` can check a device's versions
// against these reference implementations.

pub fn attentionScores(scores: []f32, q: []const f32, k: [*]const f32, stride: usize, scale: f32) void {
    tensor.attentionScores(scores, q, k, stride, scale);
}

pub fn attentionScoresOn(dev: *const Device, scores: []f32, q: []const f32, k: []const f32, stride: usize, scale: f32) !void {
    if (dev.vtable.attentionScores) |f| {
        if (f(dev.ctx.?, scores, q, k, stride, scale)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    tensor.attentionScores(scores, q, k.ptr, stride, scale);
}

pub fn attentionValues(out: []f32, scores: []const f32, v: [*]const f32, stride: usize) void {
    tensor.attentionValues(out, scores, v, stride);
}

pub fn attentionValuesOn(dev: *const Device, out: []f32, scores: []const f32, v: []const f32, stride: usize) !void {
    if (dev.vtable.attentionValues) |f| {
        if (f(dev.ctx.?, out, scores, v, stride)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    tensor.attentionValues(out, scores, v.ptr, stride);
}

pub fn gatedActivation(pool: *const Pool, act: tensor.Activation, out: []f32, gate: []const f32, up: ?[]const f32, n: usize, len: usize, in_stride: usize, out_stride: usize) void {
    tensor.gatedActivation(pool, act, out, gate, up, n, len, in_stride, out_stride);
}

pub fn gatedActivationOn(dev: *const Device, pool: *const Pool, act: tensor.Activation, out: []f32, gate: []const f32, up: ?[]const f32, n: usize, len: usize, in_stride: usize, out_stride: usize) !void {
    if (dev.vtable.gatedActivation) |f| {
        if (f(dev.ctx.?, @intFromEnum(act), out, gate, up, n, len, in_stride, out_stride)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    tensor.gatedActivation(pool, act, out, gate, up, n, len, in_stride, out_stride);
}

pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32, gemma_style: bool) void {
    tensor.rmsnorm(out, x, weight, eps, gemma_style);
}

/// `n` rows of `len` elements, each normalised independently.
pub fn rmsnormRowsOn(dev: *const Device, out: []f32, x: []const f32, weight: []const f32, n: usize, len: usize, eps: f32, gemma_style: bool) !void {
    if (dev.vtable.rmsnormRows) |f| {
        if (f(dev.ctx.?, out, x, weight, n, len, eps, gemma_style)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    for (0..n) |i| tensor.rmsnorm(out[i * len ..][0..len], x[i * len ..][0..len], weight, eps, gemma_style);
}

pub fn layernorm(out: []f32, x: []const f32, weight: []const f32, bias: ?[]const f32, eps: f32, one_plus: bool) void {
    tensor.layernorm(out, x, weight, bias, eps, one_plus);
}

pub fn layernormRowsOn(dev: *const Device, out: []f32, x: []const f32, weight: []const f32, bias: ?[]const f32, n: usize, len: usize, eps: f32, one_plus: bool) !void {
    if (dev.vtable.layernormRows) |f| {
        if (f(dev.ctx.?, out, x, weight, bias, n, len, eps, one_plus)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    for (0..n) |i| tensor.layernorm(out[i * len ..][0..len], x[i * len ..][0..len], weight, bias, eps, one_plus);
}

pub fn softmaxInPlace(x: []f32) void {
    tensor.softmaxInPlace(x);
}

pub fn softmaxRowsOn(dev: *const Device, x: []f32, n: usize, len: usize) !void {
    if (dev.vtable.softmaxRows) |f| {
        if (f(dev.ctx.?, x, n, len)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    for (0..n) |i| tensor.softmaxInPlace(x[i * len ..][0..len]);
}

pub fn logSoftmax(out: []f32, x: []const f32) void {
    tensor.logSoftmax(out, x);
}

pub fn applyRope(x: []f32, cos_row: []const f32, sin_row: []const f32) void {
    tensor.applyRope(x, cos_row, sin_row);
}

pub fn applyRopeInterleaved(x: []f32, cos_row: []const f32, sin_row: []const f32) void {
    tensor.applyRopeInterleaved(x, cos_row, sin_row);
}

/// `n` head vectors of `dim` elements, rotated at `pos[i]` against a
/// `[len][half]` cosine/sine table.
pub fn ropeRowsOn(dev: *const Device, x: []f32, n: usize, dim: usize, cos: []const f32, sin: []const f32, half: usize, pos: []const u32, style: RopeStyle) !void {
    if (dev.vtable.ropeRows) |f| {
        if (f(dev.ctx.?, x, n, dim, cos, sin, half, pos, style)) |_| {
            return;
        } else |err| switch (err) {
            error.Unsupported => {},
            else => |e| return e,
        }
    }
    for (0..n) |i| {
        const row = x[i * dim ..][0..dim];
        const c = cos[pos[i] * half ..][0..half];
        const s = sin[pos[i] * half ..][0..half];
        switch (style) {
            .neox => tensor.applyRope(row, c, s),
            .gptj => tensor.applyRopeInterleaved(row, c, s),
        }
    }
}

pub fn softcap(x: []f32, cap: f32) void {
    tensor.softcap(x, cap);
}

// ---------------------------------------------------------------------------
// Device-side residency of hot weights
// ---------------------------------------------------------------------------

/// A byte-budgeted LRU of device buffers holding converted weight tiles,
/// keyed by the host address and length of the tile. A backend consults it
/// before uploading; a miss uploads and inserts, evicting the least recently
/// used entries until the new one fits. With a zero budget every tile is
/// uploaded, used and dropped again.
///
/// Keying by host address is only sound while those addresses stay valid and
/// unique, i.e. in memory-mapped mode; `enabled` is false otherwise (see
/// `weights_stable`).
pub const Residency = struct {
    budget: u64,
    used: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(Entry) = .empty,
    gpa: Allocator,
    release_ctx: *anyopaque,
    release: *const fn (ctx: *anyopaque, handle: *anyopaque) void,
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,

    pub const Entry = struct {
        addr: usize,
        len: usize,
        bytes: u64,
        handle: *anyopaque,
        used_at: u64,
    };

    pub fn init(gpa: Allocator, budget: u64, release_ctx: *anyopaque, release: *const fn (*anyopaque, *anyopaque) void) Residency {
        return .{ .gpa = gpa, .budget = budget, .release_ctx = release_ctx, .release = release };
    }

    pub fn deinit(self: *Residency) void {
        for (self.entries.items) |e| self.release(self.release_ctx, e.handle);
        self.entries.deinit(self.gpa);
        self.used = 0;
    }

    pub fn enabled(self: *const Residency) bool {
        return self.budget > 0 and weights_stable;
    }

    /// The cached buffer for `data`, marked as just used.
    pub fn get(self: *Residency, data: []const u8) ?*anyopaque {
        if (!self.enabled()) return null;
        const addr = @intFromPtr(data.ptr);
        for (self.entries.items) |*e| {
            if (e.addr == addr and e.len == data.len) {
                self.clock += 1;
                e.used_at = self.clock;
                self.hits += 1;
                return e.handle;
            }
        }
        self.misses += 1;
        return null;
    }

    /// Inserts `handle` for `data`, evicting least recently used entries first.
    /// Returns false when the entry does not fit at all, in which case the
    /// caller owns `handle` and must release it after use.
    pub fn put(self: *Residency, data: []const u8, bytes: u64, handle: *anyopaque) bool {
        if (!self.enabled() or bytes > self.budget) return false;
        while (self.used + bytes > self.budget) {
            if (!self.evictOldest()) return false;
        }
        self.clock += 1;
        self.entries.append(self.gpa, .{
            .addr = @intFromPtr(data.ptr),
            .len = data.len,
            .bytes = bytes,
            .handle = handle,
            .used_at = self.clock,
        }) catch return false;
        self.used += bytes;
        return true;
    }

    fn evictOldest(self: *Residency) bool {
        if (self.entries.items.len == 0) return false;
        var oldest: usize = 0;
        for (self.entries.items, 0..) |e, i| {
            if (e.used_at < self.entries.items[oldest].used_at) oldest = i;
        }
        const e = self.entries.swapRemove(oldest);
        self.used -= @min(self.used, e.bytes);
        self.evictions += 1;
        self.release(self.release_ctx, e.handle);
        return true;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "the CPU backend is the default and matches tensor.zig exactly" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const pool = Pool.init(threaded.io(), 1);
    try testing.expect(active.isCpu());
    try testing.expectEqualStrings("cpu", active.name);
    // The CPU backend dispatches nothing: it *is* the kernels.
    try testing.expect(!active.dispatched("matmulT"));
    try testing.expect(!active.dispatched("softmaxRows"));

    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const rows = 37;
    const cols = 48;
    const n = 5;
    const wf = try gpa.alloc(f32, rows * cols);
    defer gpa.free(wf);
    for (wf) |*v| v.* = rnd.floatNorm(f32);
    const x = try gpa.alloc(f32, n * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const w = Weight{ .data = std.mem.sliceAsBytes(wf), .dtype = .f32, .rows = rows, .cols = cols };

    const a = try gpa.alloc(f32, n * rows);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, n * rows);
    defer gpa.free(b);
    try tensor.matmulT(&pool, gpa, a, x, n, w, null);
    try matmulT(&pool, gpa, b, x, n, w, null);
    try testing.expectEqualSlices(f32, a, b);

    // With a delta too, including the host-side correction path.
    const da = try gpa.alloc(f32, 2 * cols);
    defer gpa.free(da);
    const db = try gpa.alloc(f32, rows * 2);
    defer gpa.free(db);
    for (da) |*v| v.* = rnd.floatNorm(f32) * 0.1;
    for (db) |*v| v.* = rnd.floatNorm(f32) * 0.1;
    const delta = Delta{ .rank = 2, .a = da, .b = db };
    try tensor.matmulT(&pool, gpa, a, x, n, w, &delta);
    try matmulT(&pool, gpa, b, x, n, w, &delta);
    try testing.expectEqualSlices(f32, a, b);

    // matvecT and rowNorms.
    const ya = try gpa.alloc(f32, 3 * rows);
    defer gpa.free(ya);
    for (ya) |*v| v.* = rnd.floatNorm(f32);
    const oa = try gpa.alloc(f32, 3 * cols);
    defer gpa.free(oa);
    const ob = try gpa.alloc(f32, 3 * cols);
    defer gpa.free(ob);
    try tensor.matvecTMulti(&pool, gpa, oa, w, ya, 3);
    try matvecTMulti(&pool, gpa, ob, w, ya, 3);
    try testing.expectEqualSlices(f32, oa, ob);
    const na = try gpa.alloc(f32, rows);
    defer gpa.free(na);
    const nb = try gpa.alloc(f32, rows);
    defer gpa.free(nb);
    try tensor.rowNorms(&pool, gpa, na, w);
    try rowNorms(&pool, gpa, nb, w);
    try testing.expectEqualSlices(f32, na, nb);
}

test "the host-side delta correction equals the fused one" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const pool = Pool.init(threaded.io(), 1);
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const rows = 9;
    const cols = 16;
    const n = 3;
    const wf = try gpa.alloc(f32, rows * cols);
    defer gpa.free(wf);
    for (wf) |*v| v.* = rnd.floatNorm(f32);
    const x = try gpa.alloc(f32, n * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const w = Weight{ .data = std.mem.sliceAsBytes(wf), .dtype = .f32, .rows = rows, .cols = cols };
    const da = try gpa.alloc(f32, cols);
    defer gpa.free(da);
    const db = try gpa.alloc(f32, rows);
    defer gpa.free(db);
    for (da) |*v| v.* = rnd.floatNorm(f32);
    for (db) |*v| v.* = rnd.floatNorm(f32);
    const delta = Delta{ .rank = 1, .a = da, .b = db };

    const fused = try gpa.alloc(f32, n * rows);
    defer gpa.free(fused);
    try tensor.matmulT(&pool, gpa, fused, x, n, w, &delta);
    const split = try gpa.alloc(f32, n * rows);
    defer gpa.free(split);
    try tensor.matmulT(&pool, gpa, split, x, n, w, null);
    try applyDelta(gpa, split, x, n, w, &delta);
    for (fused, split) |a, b| try testing.expectApproxEqAbs(a, b, 1e-5);
}

test "device kinds parse" {
    try testing.expectEqual(Kind.cpu, Kind.parse("cpu").?);
    try testing.expectEqual(Kind.auto, Kind.parse("AUTO").?);
    try testing.expectEqual(Kind.auto, Kind.parse("gpu").?);
    try testing.expectEqual(Kind.metal, Kind.parse("metal").?);
    try testing.expect(Kind.parse("cuda") == null);
}

test "selecting cpu and auto never fails and metal is refused off-Apple" {
    const gpa = testing.allocator;
    var r = try select(gpa, .cpu, .{});
    try testing.expect(r.device.isCpu());
    r = try select(gpa, .auto, .{});
    try testing.expect(r.device.isCpu() or r.device.kind == .metal);
    if (!metal_supported) {
        try testing.expectError(error.DeviceUnavailable, select(gpa, .metal, .{}));
    }
}

test "select applies a min_macs override, and a device serves only what crosses it" {
    const gpa = testing.allocator;
    const r = try select(gpa, .cpu, .{ .min_macs = 0 });
    try testing.expectEqual(@as(u64, 0), r.device.min_macs);
    const d = try select(gpa, .cpu, .{});
    try testing.expectEqual(min_device_macs, d.device.min_macs);
    // A 2x2x2 product is 8 MACs: below the default threshold, above zero.
    var dev = d.device;
    try testing.expect(!deviceWorthIt(&dev, 2, 2, 2));
    dev.min_macs = 0;
    try testing.expect(deviceWorthIt(&dev, 2, 2, 2));
}

test "residency keeps hot tiles and evicts the least recently used" {
    const gpa = testing.allocator;
    const Fake = struct {
        released: usize = 0,
        fn release(ctx: *anyopaque, handle: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.released += 1;
            _ = handle;
        }
    };
    var fake = Fake{};
    var res = Residency.init(gpa, 200, &fake, Fake.release);
    defer res.deinit();
    const stable_before = weights_stable;
    defer weights_stable = stable_before;
    weights_stable = true;

    var storage: [3][64]u8 = undefined;
    var handles: [3]u8 = .{ 0, 1, 2 };
    const a = storage[0][0..];
    const b = storage[1][0..];
    const c = storage[2][0..];

    try testing.expect(res.get(a) == null);
    try testing.expect(res.put(a, 100, &handles[0]));
    try testing.expect(res.put(b, 100, &handles[1]));
    try testing.expect(res.get(a) != null); // `a` is now the most recent
    // `c` does not fit: `b` (least recently used) is evicted.
    try testing.expect(res.put(c, 100, &handles[2]));
    try testing.expectEqual(@as(usize, 1), fake.released);
    try testing.expect(res.get(b) == null);
    try testing.expect(res.get(a) != null);
    try testing.expect(res.get(c) != null);
    // An entry bigger than the whole budget is refused outright.
    try testing.expect(!res.put(b, 1000, &handles[1]));

    // Without stable host weights nothing is cached at all.
    weights_stable = false;
    try testing.expect(res.get(a) == null);
    try testing.expect(!res.put(a, 8, &handles[0]));
}
