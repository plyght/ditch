//! Multi-direction refusal bases.
//!
//! Heretic removes one direction per layer: the normalised difference between
//! the mean "bad" and "good" residuals. With `n_directions = K > 1` an
//! orthonormal basis of K directions is removed instead:
//!
//! * direction 1 is the difference of means (optionally orthogonalised
//!   against the good mean, as for a single direction);
//! * directions 2..K are the top principal components of the per-prompt bad
//!   residuals centred on the good mean, after projecting out direction 1
//!   (and the good direction, when orthogonalising).
//!
//! The per-prompt covariance `C = Σ_p z_p z_pᵀ` (`z_p = r_p - good_mean`) is
//! `hidden × hidden` per layer entry, too large to keep for big models, and the
//! prompts are only seen once. It is therefore estimated from a randomised
//! Nyström sketch accumulated in a single streaming pass: with a fixed
//! Gaussian test matrix `Ω` (`hidden × m`, `m = 4K`) every prompt contributes
//! `y_p = Ωᵀ z_p` (m values) to
//!
//!     Y = C Ω = Σ_p z_p y_pᵀ        (hidden × m)
//!     W = Ωᵀ C Ω = Σ_p y_p y_pᵀ     (m × m)
//!
//! and `C ≈ Y W⁺ Yᵀ = B Bᵀ` with `B = Y W^{-1/2}`. The top eigenvectors of
//! the projected `P B (P B)ᵀ` (P removing the earlier directions) are the
//! remaining directions; they are obtained from the small `m × m` Gram matrix
//! of `P B`. The sketch is exact when the centred residuals span at most `m`
//! dimensions and, with oversampling factor 4, very accurate whenever the
//! spectrum has K dominant directions (the interesting case). Memory is
//! `entries × (hidden × m + m × m)` doubles.
//!
//! Principal components have arbitrary sign; for interpolation between layer
//! entries every direction is oriented consistently with the same direction
//! of the previous entry.

const std = @import("std");
const tensor = @import("tensor.zig");
const abliterate = @import("abliterate.zig");

const Allocator = std.mem.Allocator;

/// Streaming covariance sketch of residuals centred on `center` (`[entries][hidden]`).
pub const Sketch = struct {
    gpa: Allocator,
    entries: usize,
    hidden: usize,
    m: usize,
    center: []const f32,
    /// Test matrix `Ω` stored as `[m][hidden]`.
    omega: []f32,
    /// `Y = C Ω` stored as `[entries][m][hidden]`.
    y: []f64,
    /// `W = Ωᵀ C Ω` stored as `[entries][m][m]`.
    w: []f64,
    count: usize = 0,
    /// Scratch for one projection `y_p`.
    proj: []f64,
    /// Scratch for one centred residual.
    z: []f64,

    pub fn init(gpa: Allocator, entries: usize, hidden: usize, k: usize, center: []const f32, seed: u64) !Sketch {
        std.debug.assert(center.len == entries * hidden);
        const m = @min(@max(4 * k, 8), hidden);
        const omega = try gpa.alloc(f32, m * hidden);
        errdefer gpa.free(omega);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x5eed_d1ec_7105);
        const rand = prng.random();
        for (omega) |*x| x.* = rand.floatNorm(f32);
        const y = try gpa.alloc(f64, entries * m * hidden);
        errdefer gpa.free(y);
        @memset(y, 0);
        const w = try gpa.alloc(f64, entries * m * m);
        errdefer gpa.free(w);
        @memset(w, 0);
        const proj = try gpa.alloc(f64, m);
        errdefer gpa.free(proj);
        const z = try gpa.alloc(f64, hidden);
        return .{ .gpa = gpa, .entries = entries, .hidden = hidden, .m = m, .center = center, .omega = omega, .y = y, .w = w, .proj = proj, .z = z };
    }

    pub fn deinit(self: *Sketch) void {
        self.gpa.free(self.omega);
        self.gpa.free(self.y);
        self.gpa.free(self.w);
        self.gpa.free(self.proj);
        self.gpa.free(self.z);
    }

    /// Adds one prompt's residual at layer `entry`.
    pub fn add(self: *Sketch, entry: usize, residual: []const f32) void {
        const h = self.hidden;
        const m = self.m;
        const c = self.center[entry * h ..][0..h];
        for (self.z, 0..) |*zi, i| zi.* = @as(f64, residual[i]) - c[i];
        for (0..m) |j| {
            const om = self.omega[j * h ..][0..h];
            var acc: f64 = 0;
            for (self.z, 0..) |zi, i| acc += zi * om[i];
            self.proj[j] = acc;
        }
        const y = self.y[entry * m * h ..][0 .. m * h];
        for (0..m) |j| {
            const yj = y[j * h ..][0..h];
            const pj = self.proj[j];
            for (yj, 0..) |*v, i| v.* += self.z[i] * pj;
        }
        const w = self.w[entry * m * m ..][0 .. m * m];
        for (0..m) |j| for (0..m) |l| {
            w[j * m + l] += self.proj[j] * self.proj[l];
        };
        if (entry == 0) self.count += 1;
    }

    /// Nyström factor `B = Y W^{-1/2}` for `entry`, as `[hidden][r]` column vectors
    /// stored row-wise `[r][hidden]` (r = number of significant eigenvalues of W).
    fn factor(self: *const Sketch, gpa: Allocator, entry: usize) ![]f64 {
        const h = self.hidden;
        const m = self.m;
        const w = try gpa.dupe(f64, self.w[entry * m * m ..][0 .. m * m]);
        defer gpa.free(w);
        const vecs = try gpa.alloc(f64, m * m);
        defer gpa.free(vecs);
        symmetricEigen(w, vecs, m);
        var max_ev: f64 = 0;
        for (0..m) |j| max_ev = @max(max_ev, w[j * m + j]);
        const y = self.y[entry * m * h ..][0 .. m * h];
        const b = try gpa.alloc(f64, m * h);
        errdefer gpa.free(b);
        var r: usize = 0;
        for (0..m) |j| {
            const ev = w[j * m + j];
            if (ev <= max_ev * 1e-9 or ev <= 0) continue;
            const s = 1.0 / @sqrt(ev);
            const col = b[r * h ..][0..h];
            @memset(col, 0);
            for (0..m) |l| {
                const coef = vecs[l * m + j] * s;
                if (coef == 0) continue;
                const yl = y[l * h ..][0..h];
                for (col, 0..) |*x, i| x.* += coef * yl[i];
            }
            r += 1;
        }
        return gpa.realloc(b, r * h);
    }
};

/// Computes the `[entries][k][hidden]` orthonormal direction basis. `sketch`
/// is required for `k > 1`.
pub fn computeBasis(gpa: Allocator, good_means: []const f32, bad_means: []const f32, entries: usize, hidden: usize, k: usize, orthogonalize: bool, sketch: ?*const Sketch) ![]f32 {
    const first = try abliterate.computeDirections(gpa, good_means, bad_means, entries, hidden, orthogonalize);
    defer gpa.free(first);
    if (k == 1) return gpa.dupe(f32, first);
    const sk = sketch orelse return error.SketchRequired;
    std.debug.assert(sk.entries == entries and sk.hidden == hidden);
    const dirs = try gpa.alloc(f32, entries * k * hidden);
    errdefer gpa.free(dirs);
    @memset(dirs, 0);
    const good_dir = try gpa.alloc(f32, hidden);
    defer gpa.free(good_dir);
    for (0..entries) |e| {
        const basis = dirs[e * k * hidden ..][0 .. k * hidden];
        @memcpy(basis[0..hidden], first[e * hidden ..][0..hidden]);
        @memcpy(good_dir, good_means[e * hidden ..][0..hidden]);
        tensor.normalize(good_dir);

        const b = try sk.factor(gpa, e);
        defer gpa.free(b);
        const r = b.len / hidden;
        // Project the earlier directions out of every column of B.
        for (0..r) |c| {
            const col = b[c * hidden ..][0..hidden];
            removeF64(col, basis[0..hidden]);
            if (orthogonalize) removeF64(col, good_dir);
        }
        // Gram matrix G = Bᵀ B (r × r); its eigenvectors give the principal components.
        const gram = try gpa.alloc(f64, r * r);
        defer gpa.free(gram);
        for (0..r) |i| for (0..r) |j| {
            var acc: f64 = 0;
            for (b[i * hidden ..][0..hidden], b[j * hidden ..][0..hidden]) |x, y| acc += x * y;
            gram[i * r + j] = acc;
        };
        const vecs = try gpa.alloc(f64, r * r);
        defer gpa.free(vecs);
        symmetricEigen(gram, vecs, r);
        const order = try gpa.alloc(usize, r);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, gram, struct {
            fn lt(g: []const f64, x: usize, y: usize) bool {
                const n = std.math.sqrt(g.len);
                return g[x * n + x] > g[y * n + y];
            }
        }.lt);
        for (1..k) |d| {
            const dir = basis[d * hidden ..][0..hidden];
            @memset(dir, 0);
            if (d - 1 >= r) continue;
            const src = order[d - 1];
            const lambda = gram[src * r + src];
            if (lambda <= 0) continue;
            for (0..r) |c| {
                const coef: f32 = @floatCast(vecs[c * r + src] / @sqrt(lambda));
                const col = b[c * hidden ..][0..hidden];
                for (dir, 0..) |*x, i| x.* += coef * @as(f32, @floatCast(col[i]));
            }
        }
        abliterate.orthonormalize(basis, k, hidden);
        // Orient every component like the previous entry's.
        if (e > 0) {
            const prev = dirs[(e - 1) * k * hidden ..][0 .. k * hidden];
            for (1..k) |d| {
                const dir = basis[d * hidden ..][0..hidden];
                if (tensor.dot(dir, prev[d * hidden ..][0..hidden]) < 0) tensor.scale(dir, -1.0);
            }
        }
    }
    return dirs;
}

fn removeF64(col: []f64, unit: []const f32) void {
    var p: f64 = 0;
    for (col, 0..) |x, i| p += x * unit[i];
    for (col, 0..) |*x, i| x.* -= p * unit[i];
}

/// Symmetric eigen-decomposition (cyclic Jacobi, f64) of an n×n row-major
/// matrix `m` (overwritten by its diagonalisation). Eigenvectors are the
/// columns of `vecs`.
pub fn symmetricEigen(m: []f64, vecs: []f64, n: usize) void {
    @memset(vecs, 0);
    for (0..n) |i| vecs[i * n + i] = 1;
    var sweep: usize = 0;
    while (sweep < 60) : (sweep += 1) {
        var off: f64 = 0;
        var diag: f64 = 0;
        for (0..n) |i| for (0..n) |j| {
            if (i != j) off += m[i * n + j] * m[i * n + j] else diag += m[i * n + j] * m[i * n + j];
        };
        if (off <= 1e-24 * @max(diag, 1e-300)) break;
        for (0..n) |p| {
            for (p + 1..n) |q| {
                const apq = m[p * n + q];
                if (@abs(apq) < 1e-300) continue;
                const theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                const t = if (theta == 0) 1.0 else std.math.sign(theta) / (@abs(theta) + @sqrt(theta * theta + 1.0));
                const cs = 1.0 / @sqrt(t * t + 1.0);
                const sn = t * cs;
                for (0..n) |k| {
                    const akp = m[k * n + p];
                    const akq = m[k * n + q];
                    m[k * n + p] = cs * akp - sn * akq;
                    m[k * n + q] = sn * akp + cs * akq;
                }
                for (0..n) |k| {
                    const apk = m[p * n + k];
                    const aqk = m[q * n + k];
                    m[p * n + k] = cs * apk - sn * aqk;
                    m[q * n + k] = sn * apk + cs * aqk;
                }
                for (0..n) |k| {
                    const vkp = vecs[k * n + p];
                    const vkq = vecs[k * n + q];
                    vecs[k * n + p] = cs * vkp - sn * vkq;
                    vecs[k * n + q] = sn * vkp + cs * vkq;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "symmetric eigen decomposition" {
    var m = [_]f64{ 2, 1, 0, 1, 2, 0, 0, 0, 5 };
    var vecs: [9]f64 = undefined;
    symmetricEigen(&m, &vecs, 3);
    var evs = [_]f64{ m[0], m[4], m[8] };
    std.mem.sort(f64, &evs, {}, std.sort.asc(f64));
    try std.testing.expectApproxEqAbs(@as(f64, 1), evs[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 3), evs[1], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 5), evs[2], 1e-9);
}

test "two planted directions are recovered with n_directions = 2" {
    const gpa = std.testing.allocator;
    const hidden = 24;
    const entries = 2;
    const n = 300;
    var prng = std.Random.DefaultPrng.init(11);
    const rand = prng.random();
    // Orthonormal planted directions d1 (mean shift) and d2 (variance only).
    var planted: [2 * hidden]f32 = undefined;
    for (&planted) |*x| x.* = rand.floatNorm(f32);
    abliterate.orthonormalize(&planted, 2, hidden);
    const d1 = planted[0..hidden];
    const d2 = planted[hidden..];
    // Good residuals: small noise around a fixed offset; bad residuals add
    // 3 d1 (plus jitter) and a zero-mean spread of ±2 along d2.
    var good: [entries * hidden]f32 = undefined;
    for (&good) |*x| x.* = 0.5 * rand.floatNorm(f32);
    var bad_sum: [entries * hidden]f64 = [_]f64{0} ** (entries * hidden);
    var sketch = try Sketch.init(gpa, entries, hidden, 2, &good, 3);
    defer sketch.deinit();
    var r: [hidden]f32 = undefined;
    for (0..n) |_| {
        const a = 3.0 + 0.3 * rand.floatNorm(f32);
        const b = 2.0 * rand.floatNorm(f32);
        for (0..entries) |e| {
            for (&r, 0..) |*x, i| x.* = good[e * hidden + i] + a * d1[i] + b * d2[i] + 0.1 * rand.floatNorm(f32);
            sketch.add(e, &r);
            for (0..hidden) |i| bad_sum[e * hidden + i] += r[i];
        }
    }
    var bad: [entries * hidden]f32 = undefined;
    for (&bad, 0..) |*x, i| x.* = @floatCast(bad_sum[i] / n);
    const dirs = try computeBasis(gpa, &good, &bad, entries, hidden, 2, false, &sketch);
    defer gpa.free(dirs);
    for (0..entries) |e| {
        const basis = dirs[e * 2 * hidden ..][0 .. 2 * hidden];
        const first = basis[0..hidden];
        const second = basis[hidden..];
        try std.testing.expect(@abs(tensor.dot(first, d1)) > 0.95);
        try std.testing.expect(@abs(tensor.dot(second, d2)) > 0.95);
        try std.testing.expect(@abs(tensor.dot(first, second)) < 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1), tensor.norm2(second), 1e-4);
    }
    // Consistent orientation across entries.
    try std.testing.expect(tensor.dot(dirs[hidden .. 2 * hidden], dirs[3 * hidden .. 4 * hidden]) > 0);
    // k = 1 reproduces the single-direction result exactly.
    const single = try computeBasis(gpa, &good, &bad, entries, hidden, 1, true, null);
    defer gpa.free(single);
    const expected = try abliterate.computeDirections(gpa, &good, &bad, entries, hidden, true);
    defer gpa.free(expected);
    try std.testing.expectEqualSlices(f32, expected, single);
}
