//! `ditch selftest [--device X]`: checks a compute backend against the CPU
//! reference, kernel by kernel, on random inputs over a sweep of shapes
//! (including the edge shapes: single rows, single columns, sizes that are not
//! a multiple of a tile or a vector width). For every kernel it reports the
//! largest absolute and relative error against the CPU result and whether that
//! is inside the kernel's tolerance.
//!
//! This is the harness that makes a GPU backend trustworthy: the CPU kernels in
//! `tensor.zig` are the reference implementation, the selftest is the contract,
//! and a backend that passes it computes what ditch's CPU path computes to f32
//! rounding. Running it with `--device cpu` compares the CPU backend with
//! itself and must report exactly zero error, which is how the harness itself
//! is tested where no GPU exists.
//!
//! Exit codes: 0 when every kernel is inside tolerance, 1 when one is not (a
//! kernel a backend does not implement is reported as "cpu" and is not a
//! failure), 2 for a usage error such as an unavailable device.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const compute = @import("compute.zig");
const config = @import("config.zig");
const tensor = @import("tensor.zig");

const Weight = tensor.Weight;
const Device = compute.Device;
const Pool = tensor.Pool;

/// An element passes when it is within `abs` or within `rel` of the reference;
/// the kernel passes when every element does. The values allow for a different
/// summation order in f32, not for a different algorithm.
pub const Tolerance = struct {
    abs: f64,
    rel: f64,
};

const tol_matmul = Tolerance{ .abs = 1e-4, .rel = 1e-3 };
const tol_elementwise = Tolerance{ .abs = 1e-5, .rel = 1e-4 };
const tol_reduction = Tolerance{ .abs = 1e-4, .rel = 1e-4 };

pub const KernelReport = struct {
    name: []const u8,
    /// Shape cases compared.
    cases: usize = 0,
    /// Elements compared.
    elements: usize = 0,
    max_abs: f64 = 0,
    max_rel: f64 = 0,
    tol: Tolerance,
    ok: bool = true,
    /// True when the backend implements this kernel; false when every case
    /// fell back to the CPU (then the comparison is trivially exact).
    on_device: bool = false,
};

pub const Report = struct {
    device: []const u8,
    kernels: []KernelReport,

    pub fn failed(self: *const Report) bool {
        for (self.kernels) |k| if (!k.ok) return true;
        return false;
    }
};

const Harness = struct {
    gpa: Allocator,
    pool: *const Pool,
    dev: *const Device,
    rnd: std.Random,
    reports: std.ArrayList(KernelReport) = .empty,
    current: *KernelReport = undefined,

    fn begin(self: *Harness, name: []const u8, tol: Tolerance, on_device: bool) !void {
        try self.reports.append(self.gpa, .{ .name = name, .tol = tol, .on_device = on_device });
        self.current = &self.reports.items[self.reports.items.len - 1];
    }

    /// Compares one case against the reference.
    fn compare(self: *Harness, got: []const f32, want: []const f32) void {
        const r = self.current;
        r.cases += 1;
        r.elements += want.len;
        for (got, want) |g, w| {
            const a: f64 = @abs(@as(f64, g) - @as(f64, w));
            const rel: f64 = a / @max(@abs(@as(f64, w)), 1e-30);
            if (a > r.max_abs) r.max_abs = a;
            // A relative error is only meaningful where the reference is not
            // vanishing; below the absolute tolerance it is reported as zero.
            if (a > r.tol.abs and rel > r.max_rel) r.max_rel = rel;
            if (a > r.tol.abs and rel > r.tol.rel) r.ok = false;
            if (std.math.isNan(g) != std.math.isNan(w)) r.ok = false;
        }
    }

    fn randVec(self: *Harness, n: usize) ![]f32 {
        const v = try self.gpa.alloc(f32, n);
        for (v) |*x| x.* = self.rnd.floatNorm(f32);
        return v;
    }

    /// Random weights stored in `dtype`, plus the f32 values the storage
    /// actually holds (so the reference and the device see the same numbers).
    fn randWeight(self: *Harness, rows: usize, cols: usize, dtype: tensor.DType) !Weight {
        const vals = try self.gpa.alloc(f32, rows * cols);
        defer self.gpa.free(vals);
        for (vals) |*x| x.* = self.rnd.floatNorm(f32);
        const bytes = try self.gpa.alloc(u8, dtype.byteLen(rows * cols, cols));
        tensor.convertFromF32(dtype, vals, bytes);
        return .{ .data = bytes, .dtype = dtype, .rows = rows, .cols = cols };
    }

    fn freeWeight(self: *Harness, w: Weight) void {
        self.gpa.free(@constCast(w.data));
    }
};

const Shape = struct { n: usize, rows: usize, cols: usize };

/// Shape sweep for the matrix kernels: powers of two, odd sizes that are not a
/// multiple of the vector width or the device tile, single rows and columns.
const matmul_shapes = [_]Shape{
    .{ .n = 1, .rows = 1, .cols = 1 },
    .{ .n = 1, .rows = 1, .cols = 16 },
    .{ .n = 1, .rows = 3, .cols = 17 },
    .{ .n = 1, .rows = 16, .cols = 16 },
    .{ .n = 1, .rows = 64, .cols = 128 },
    .{ .n = 2, .rows = 17, .cols = 33 },
    .{ .n = 4, .rows = 32, .cols = 64 },
    .{ .n = 5, .rows = 65, .cols = 31 },
    .{ .n = 8, .rows = 128, .cols = 256 },
    .{ .n = 13, .rows = 37, .cols = 48 },
};

const weight_dtypes = [_]tensor.DType{ .f32, .f16, .bf16 };

/// Runs every comparison. The caller owns `Report.kernels`.
pub fn check(gpa: Allocator, pool: *const Pool, dev: *const Device, seed: u64) !Report {
    var prng = std.Random.DefaultPrng.init(seed);
    // Every shape is checked on the device, including ones the forward pass
    // would keep on the CPU because they are too small to be worth a dispatch.
    var checked = dev.*;
    checked.min_macs = 0;
    var h = Harness{ .gpa = gpa, .pool = pool, .dev = &checked, .rnd = prng.random() };
    errdefer h.reports.deinit(gpa);

    try checkMatmul(&h);
    try checkMatvec(&h);
    try checkRowNorms(&h);
    try checkAttention(&h);
    try checkGated(&h);
    try checkNorms(&h);
    try checkSoftmax(&h);
    try checkRope(&h);

    return .{ .device = dev.name, .kernels = try h.reports.toOwnedSlice(gpa) };
}

fn checkMatmul(h: *Harness) !void {
    try h.begin("matmul", tol_matmul, h.dev.vtable.matmulT != null);
    for (weight_dtypes) |dt| {
        for (matmul_shapes) |s| {
            const w = try h.randWeight(s.rows, s.cols, dt);
            defer h.freeWeight(w);
            const x = try h.randVec(s.n * s.cols);
            defer h.gpa.free(x);
            const want = try h.gpa.alloc(f32, s.n * s.rows);
            defer h.gpa.free(want);
            const got = try h.gpa.alloc(f32, s.n * s.rows);
            defer h.gpa.free(got);
            try tensor.matmulT(h.pool, h.gpa, want, x, s.n, w, null);
            try compute.matmulTOn(h.dev, h.pool, h.gpa, got, x, s.n, w, null);
            h.compare(got, want);
        }
    }
    // A low-rank delta: the device computes the base product and the host adds
    // the correction, so the result must still match the fused CPU kernel.
    const s = matmul_shapes[matmul_shapes.len - 1];
    const w = try h.randWeight(s.rows, s.cols, .f32);
    defer h.freeWeight(w);
    const x = try h.randVec(s.n * s.cols);
    defer h.gpa.free(x);
    const da = try h.randVec(2 * s.cols);
    defer h.gpa.free(da);
    const db = try h.randVec(s.rows * 2);
    defer h.gpa.free(db);
    const delta = tensor.Delta{ .rank = 2, .a = da, .b = db };
    const want = try h.gpa.alloc(f32, s.n * s.rows);
    defer h.gpa.free(want);
    const got = try h.gpa.alloc(f32, s.n * s.rows);
    defer h.gpa.free(got);
    try tensor.matmulT(h.pool, h.gpa, want, x, s.n, w, &delta);
    try compute.matmulTOn(h.dev, h.pool, h.gpa, got, x, s.n, w, &delta);
    h.compare(got, want);
}

fn checkMatvec(h: *Harness) !void {
    try h.begin("matvec_t", tol_matmul, h.dev.vtable.matvecTMulti != null);
    for (weight_dtypes) |dt| {
        for (matmul_shapes) |s| {
            const q = @max(s.n / 2, 1);
            const w = try h.randWeight(s.rows, s.cols, dt);
            defer h.freeWeight(w);
            const y = try h.randVec(q * s.rows);
            defer h.gpa.free(y);
            const want = try h.gpa.alloc(f32, q * s.cols);
            defer h.gpa.free(want);
            const got = try h.gpa.alloc(f32, q * s.cols);
            defer h.gpa.free(got);
            try tensor.matvecTMulti(h.pool, h.gpa, want, w, y, q);
            try compute.matvecTMultiOn(h.dev, h.pool, h.gpa, got, w, y, q);
            h.compare(got, want);
        }
    }
}

fn checkRowNorms(h: *Harness) !void {
    try h.begin("row_norms", tol_reduction, h.dev.vtable.rowNorms != null);
    for (weight_dtypes) |dt| {
        for (matmul_shapes) |s| {
            const w = try h.randWeight(s.rows, s.cols, dt);
            defer h.freeWeight(w);
            const want = try h.gpa.alloc(f32, s.rows);
            defer h.gpa.free(want);
            const got = try h.gpa.alloc(f32, s.rows);
            defer h.gpa.free(got);
            try tensor.rowNorms(h.pool, h.gpa, want, w);
            try compute.rowNormsOn(h.dev, h.pool, h.gpa, got, w);
            h.compare(got, want);
        }
    }
}

const attn_cases = [_]struct { keys: usize, hd: usize, vd: usize, stride: usize }{
    .{ .keys = 1, .hd = 16, .vd = 16, .stride = 16 },
    .{ .keys = 7, .hd = 16, .vd = 16, .stride = 32 },
    .{ .keys = 33, .hd = 64, .vd = 64, .stride = 128 },
    .{ .keys = 128, .hd = 32, .vd = 32, .stride = 64 },
};

fn checkAttention(h: *Harness) !void {
    try h.begin("attn_scores", tol_reduction, h.dev.vtable.attentionScores != null);
    for (attn_cases) |c| {
        const q = try h.randVec(c.hd);
        defer h.gpa.free(q);
        const k = try h.randVec(c.keys * c.stride);
        defer h.gpa.free(k);
        const want = try h.gpa.alloc(f32, c.keys);
        defer h.gpa.free(want);
        const got = try h.gpa.alloc(f32, c.keys);
        defer h.gpa.free(got);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hd)));
        tensor.attentionScores(want, q, k.ptr, c.stride, scale);
        try compute.attentionScoresOn(h.dev, got, q, k, c.stride, scale);
        h.compare(got, want);
    }
    try h.begin("attn_values", tol_reduction, h.dev.vtable.attentionValues != null);
    for (attn_cases) |c| {
        const scores = try h.randVec(c.keys);
        defer h.gpa.free(scores);
        const v = try h.randVec(c.keys * c.stride);
        defer h.gpa.free(v);
        const want = try h.gpa.alloc(f32, c.vd);
        defer h.gpa.free(want);
        const got = try h.gpa.alloc(f32, c.vd);
        defer h.gpa.free(got);
        tensor.attentionValues(want, scores, v.ptr, c.stride);
        try compute.attentionValuesOn(h.dev, got, scores, v, c.stride);
        h.compare(got, want);
    }
}

fn checkGated(h: *Harness) !void {
    try h.begin("gated_activation", tol_elementwise, h.dev.vtable.gatedActivation != null);
    const cases = [_]struct { n: usize, len: usize }{
        .{ .n = 1, .len = 1 },
        .{ .n = 1, .len = 17 },
        .{ .n = 3, .len = 64 },
        .{ .n = 8, .len = 129 },
    };
    inline for (@typeInfo(tensor.Activation).@"enum".fields) |f| {
        const act: tensor.Activation = @enumFromInt(f.value);
        for (cases) |c| {
            for ([_]bool{ true, false }) |with_up| {
                const gate = try h.randVec(c.n * c.len);
                defer h.gpa.free(gate);
                const up = try h.randVec(c.n * c.len);
                defer h.gpa.free(up);
                const want = try h.gpa.alloc(f32, c.n * c.len);
                defer h.gpa.free(want);
                const got = try h.gpa.alloc(f32, c.n * c.len);
                defer h.gpa.free(got);
                const up_arg: ?[]const f32 = if (with_up) up else null;
                tensor.gatedActivation(h.pool, act, want, gate, up_arg, c.n, c.len, c.len, c.len);
                try compute.gatedActivationOn(h.dev, h.pool, act, got, gate, up_arg, c.n, c.len, c.len, c.len);
                h.compare(got, want);
            }
        }
    }
}

fn checkNorms(h: *Harness) !void {
    const cases = [_]struct { n: usize, len: usize }{
        .{ .n = 1, .len = 1 },
        .{ .n = 1, .len = 15 },
        .{ .n = 2, .len = 64 },
        .{ .n = 5, .len = 257 },
        .{ .n = 3, .len = 1024 },
    };
    try h.begin("rmsnorm", tol_elementwise, h.dev.vtable.rmsnormRows != null);
    for (cases) |c| {
        for ([_]u2{ 0, 1, 2 }) |mode| { // weighted, gemma-style (1 + w), non-parametric
            const x = try h.randVec(c.n * c.len);
            defer h.gpa.free(x);
            const w = try h.randVec(c.len);
            defer h.gpa.free(w);
            const want = try h.gpa.alloc(f32, c.n * c.len);
            defer h.gpa.free(want);
            const got = try h.gpa.alloc(f32, c.n * c.len);
            defer h.gpa.free(got);
            const weight: []const f32 = if (mode == 2) &.{} else w;
            const gemma = mode == 1;
            for (0..c.n) |i| tensor.rmsnorm(want[i * c.len ..][0..c.len], x[i * c.len ..][0..c.len], weight, 1e-6, gemma);
            try compute.rmsnormRowsOn(h.dev, got, x, weight, c.n, c.len, 1e-6, gemma);
            h.compare(got, want);
        }
    }
    try h.begin("layernorm", tol_elementwise, h.dev.vtable.layernormRows != null);
    for (cases) |c| {
        for ([_]u2{ 0, 1, 2 }) |mode| { // weight+bias, (1 + w), non-parametric
            const x = try h.randVec(c.n * c.len);
            defer h.gpa.free(x);
            const w = try h.randVec(c.len);
            defer h.gpa.free(w);
            const b = try h.randVec(c.len);
            defer h.gpa.free(b);
            const want = try h.gpa.alloc(f32, c.n * c.len);
            defer h.gpa.free(want);
            const got = try h.gpa.alloc(f32, c.n * c.len);
            defer h.gpa.free(got);
            const weight: []const f32 = if (mode == 2) &.{} else w;
            const bias: ?[]const f32 = if (mode == 0) b else null;
            const one_plus = mode == 1;
            for (0..c.n) |i| tensor.layernorm(want[i * c.len ..][0..c.len], x[i * c.len ..][0..c.len], weight, bias, 1e-5, one_plus);
            try compute.layernormRowsOn(h.dev, got, x, weight, bias, c.n, c.len, 1e-5, one_plus);
            h.compare(got, want);
        }
    }
}

fn checkSoftmax(h: *Harness) !void {
    try h.begin("softmax", tol_elementwise, h.dev.vtable.softmaxRows != null);
    const cases = [_]struct { n: usize, len: usize }{
        .{ .n = 1, .len = 1 },
        .{ .n = 1, .len = 13 },
        .{ .n = 4, .len = 100 },
        .{ .n = 2, .len = 1024 },
    };
    for (cases) |c| {
        const x = try h.randVec(c.n * c.len);
        defer h.gpa.free(x);
        // A wide range of logits, to exercise the max subtraction.
        for (x) |*v| v.* *= 8;
        const want = try h.gpa.alloc(f32, c.n * c.len);
        defer h.gpa.free(want);
        @memcpy(want, x);
        const got = try h.gpa.alloc(f32, c.n * c.len);
        defer h.gpa.free(got);
        @memcpy(got, x);
        for (0..c.n) |i| tensor.softmaxInPlace(want[i * c.len ..][0..c.len]);
        try compute.softmaxRowsOn(h.dev, got, c.n, c.len);
        h.compare(got, want);
    }
}

fn checkRope(h: *Harness) !void {
    try h.begin("rope", tol_elementwise, h.dev.vtable.ropeRows != null);
    const cases = [_]struct { n: usize, dim: usize }{
        .{ .n = 1, .dim = 2 },
        .{ .n = 1, .dim = 64 },
        .{ .n = 7, .dim = 128 },
        .{ .n = 16, .dim = 96 },
    };
    const positions = 64;
    for (cases) |c| {
        const half = c.dim / 2;
        const cos = try h.randVec(positions * half);
        defer h.gpa.free(cos);
        const sin = try h.randVec(positions * half);
        defer h.gpa.free(sin);
        const pos = try h.gpa.alloc(u32, c.n);
        defer h.gpa.free(pos);
        for (pos, 0..) |*p, i| p.* = @intCast(i % positions);
        for ([_]compute.RopeStyle{ .neox, .gptj }) |style| {
            const x = try h.randVec(c.n * c.dim);
            defer h.gpa.free(x);
            const want = try h.gpa.alloc(f32, c.n * c.dim);
            defer h.gpa.free(want);
            @memcpy(want, x);
            const got = try h.gpa.alloc(f32, c.n * c.dim);
            defer h.gpa.free(got);
            @memcpy(got, x);
            for (0..c.n) |i| {
                const row = want[i * c.dim ..][0..c.dim];
                const cr = cos[pos[i] * half ..][0..half];
                const sr = sin[pos[i] * half ..][0..half];
                switch (style) {
                    .neox => tensor.applyRope(row, cr, sr),
                    .gptj => tensor.applyRopeInterleaved(row, cr, sr),
                }
            }
            try compute.ropeRowsOn(h.dev, got, c.n, c.dim, cos, sin, half, pos, style);
            h.compare(got, want);
        }
    }
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// `ditch selftest`: messages to `out` (stderr), the table or JSON to
/// `result_out` (stdout). Exits 1 if a kernel is outside its tolerance.
pub fn run(gpa: Allocator, settings: *config.Settings, pool: *const Pool, out: *Io.Writer, result_out: *Io.Writer) !void {
    const kind = settings.deviceKind() orelse {
        std.log.err("unknown device: {s} (expected {s})", .{ settings.device, compute.Kind.names });
        std.process.exit(2);
    };
    var selected = compute.select(gpa, kind, .{ .memory_budget = settings.gpu_memory, .io = pool.io }) catch {
        if (compute.unavailable_reason) |why| {
            std.log.err("device {s} is not available: {s}", .{ settings.device, why });
        } else {
            std.log.err("device {s} is not available on this build or machine", .{settings.device});
        }
        std.process.exit(2);
    };
    defer selected.device.deinit();
    if (selected.note) |n| try out.print("{s}\n", .{n});
    try out.print("\nChecking the {s} backend against the CPU reference kernels...\n", .{selected.device.name});
    var info_buf: [512]u8 = undefined;
    var info_w: Io.Writer = .fixed(&info_buf);
    compute.describe(&selected.device, &info_w) catch {};
    const info = info_w.buffered();
    if (info.len > 0) try out.print("{s}\n", .{info});
    try out.flush();

    const seed = settings.seed orelse 0xd17c4;
    var report = try check(gpa, pool, &selected.device, seed);
    defer gpa.free(report.kernels);

    if (settings.json) {
        var js: std.json.Stringify = .{ .writer = result_out };
        try js.beginObject();
        try js.objectField("device");
        try js.write(report.device);
        try js.objectField("device_info");
        try js.write(info);
        try js.objectField("seed");
        try js.write(seed);
        try js.objectField("passed");
        try js.write(!report.failed());
        try js.objectField("kernels");
        try js.beginArray();
        for (report.kernels) |k| {
            try js.beginObject();
            try js.objectField("kernel");
            try js.write(k.name);
            try js.objectField("on_device");
            try js.write(k.on_device);
            try js.objectField("cases");
            try js.write(k.cases);
            try js.objectField("elements");
            try js.write(k.elements);
            try js.objectField("max_abs_error");
            try js.write(k.max_abs);
            try js.objectField("max_rel_error");
            try js.write(k.max_rel);
            try js.objectField("tolerance_abs");
            try js.write(k.tol.abs);
            try js.objectField("tolerance_rel");
            try js.write(k.tol.rel);
            try js.objectField("ok");
            try js.write(k.ok);
            try js.endObject();
        }
        try js.endArray();
        try js.endObject();
        try result_out.writeAll("\n");
    } else if (settings.plain) {
        for (report.kernels) |k| {
            try result_out.print("{s}: max_abs={e:.3} max_rel={e:.3} cases={d} on_device={} ok={}\n", .{ k.name, k.max_abs, k.max_rel, k.cases, k.on_device, k.ok });
        }
    } else {
        try result_out.print("\nBackend selftest: {s} (random inputs, seed {d}; rerun with --seed)\n\n", .{ report.device, seed });
        try result_out.writeAll("| Kernel | Where | Cases | Elements | Max abs err | Max rel err | Tolerance | Status |\n");
        try result_out.writeAll("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n");
        for (report.kernels) |k| {
            try result_out.print("| {s} | {s} | {d} | {d} | {e:.3} | {e:.3} | {e:.0}/{e:.0} | {s} |\n", .{
                k.name,
                if (k.on_device) "device" else "cpu",
                k.cases,
                k.elements,
                k.max_abs,
                k.max_rel,
                k.tol.abs,
                k.tol.rel,
                if (k.ok) "pass" else "FAIL",
            });
        }
        try result_out.writeAll("\n");
    }
    try result_out.flush();

    if (report.failed()) {
        try out.writeAll("Selftest failed: a kernel is outside its tolerance (see the table above).\n");
        try out.flush();
        std.process.exit(1);
    }
    try out.writeAll("Selftest passed: every kernel matches the CPU reference within tolerance.\n");
    try out.flush();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "the CPU backend reproduces the reference kernels exactly" {
    const gpa = testing.allocator;
    var threaded: Io.Threaded = .init_single_threaded;
    const pool = Pool.init(threaded.io(), 1);
    var report = try check(gpa, &pool, &compute.cpu_device, 12345);
    defer gpa.free(report.kernels);
    try testing.expect(!report.failed());
    try testing.expect(report.kernels.len >= 9);
    for (report.kernels) |k| {
        try testing.expect(k.cases > 0);
        try testing.expect(k.elements > 0);
        // The CPU backend runs the reference kernel itself: bit for bit equal.
        try testing.expectEqual(@as(f64, 0), k.max_abs);
        try testing.expectEqual(@as(f64, 0), k.max_rel);
        try testing.expect(!k.on_device);
    }
}

/// A fake backend whose matmul is deliberately wrong, to prove the harness
/// notices a broken kernel instead of rubber-stamping it.
const Wrong = struct {
    var bias: f32 = 0;

    fn matmulT(_: *anyopaque, out: []f32, x: []const f32, n: usize, w: Weight) compute.Error!void {
        var threaded: Io.Threaded = .init_single_threaded;
        const pool = Pool.init(threaded.io(), 1);
        tensor.matmulT(&pool, testing.allocator, out, x, n, w, null) catch return error.DeviceFailed;
        for (out[0 .. n * w.rows]) |*v| v.* += bias;
    }

    const vtable = compute.VTable{ .matmulT = matmulT };
};

test "the harness catches a backend whose kernel is wrong" {
    const gpa = testing.allocator;
    var threaded: Io.Threaded = .init_single_threaded;
    const pool = Pool.init(threaded.io(), 1);
    var dummy: u8 = 0;
    const dev = Device{ .kind = .metal, .name = "wrong (test)", .vtable = &Wrong.vtable, .ctx = &dummy };

    // An error far above the tolerance fails the matmul kernel only.
    Wrong.bias = 0.05;
    var report = try check(gpa, &pool, &dev, 99);
    defer gpa.free(report.kernels);
    try testing.expect(report.failed());
    try testing.expectEqualStrings("matmul", report.kernels[0].name);
    try testing.expect(!report.kernels[0].ok);
    try testing.expect(report.kernels[0].on_device);
    try testing.expect(report.kernels[0].max_abs >= 0.04);
    for (report.kernels[1..]) |k| try testing.expect(k.ok);

    // An error inside the tolerance passes, as a different summation order would.
    Wrong.bias = 1e-7;
    var ok_report = try check(gpa, &pool, &dev, 99);
    defer gpa.free(ok_report.kernels);
    try testing.expect(!ok_report.failed());
    Wrong.bias = 0;
}
