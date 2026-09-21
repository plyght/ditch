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

// ---------------------------------------------------------------------------
// Direction estimation methods
// ---------------------------------------------------------------------------

/// How the first (refusal) direction of every layer entry is estimated.
pub const Method = enum {
    /// Heretic: the normalised difference of the mean residuals.
    mean,
    /// The difference of means whitened by the pooled per-coordinate
    /// variance (a diagonal, shrinkage-regularised Fisher discriminant):
    /// `d ∝ (μ_bad − μ_good) / (σ² + ε)`. Coordinates that differ between the
    /// prompt sets but also vary a lot within them (massive-activation
    /// dimensions, position and template features) are down-weighted; the
    /// direction points where the two sets are actually separable. The
    /// covariance is taken as diagonal, so this targets coordinate-aligned
    /// nuisance variance; it is not a full LDA.
    separating,

    pub fn parse(s: []const u8) ?Method {
        inline for (@typeInfo(Method).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

/// Streaming per-coordinate first and second moments of residuals,
/// `[entries][hidden]` doubles each.
pub const Moments = struct {
    gpa: Allocator,
    entries: usize,
    hidden: usize,
    sum: []f64,
    sq: []f64,
    count: usize = 0,

    pub fn init(gpa: Allocator, entries: usize, hidden: usize) !Moments {
        const sum = try gpa.alloc(f64, entries * hidden);
        errdefer gpa.free(sum);
        @memset(sum, 0);
        const sq = try gpa.alloc(f64, entries * hidden);
        @memset(sq, 0);
        return .{ .gpa = gpa, .entries = entries, .hidden = hidden, .sum = sum, .sq = sq };
    }

    pub fn deinit(self: *Moments) void {
        self.gpa.free(self.sum);
        self.gpa.free(self.sq);
    }

    pub fn add(self: *Moments, entry: usize, residual: []const f32) void {
        const h = self.hidden;
        const s = self.sum[entry * h ..][0..h];
        const q = self.sq[entry * h ..][0..h];
        for (residual, 0..) |x, i| {
            s[i] += x;
            q[i] += @as(f64, x) * x;
        }
        if (entry == 0) self.count += 1;
    }

    /// Population variance of coordinate `i` at `entry`.
    pub fn variance(self: *const Moments, entry: usize, i: usize) f64 {
        const n: f64 = @floatFromInt(@max(self.count, 1));
        const m = self.sum[entry * self.hidden + i] / n;
        return @max(self.sq[entry * self.hidden + i] / n - m * m, 0);
    }
};

/// Unit "separating" directions `[entries][hidden]` (see `Method.separating`).
/// `shrinkage` sets `ε = shrinkage · mean(σ²)` per entry (0 = no
/// regularisation, which is unsafe for coordinates with a near-zero
/// variance). With `orthogonalize`, the component along the good mean is
/// removed, as for the mean direction.
pub fn computeSeparating(gpa: Allocator, good_means: []const f32, bad_means: []const f32, good: *const Moments, bad: *const Moments, entries: usize, hidden: usize, shrinkage: f32, orthogonalize: bool) ![]f32 {
    std.debug.assert(good.entries == entries and bad.entries == entries and good.hidden == hidden and bad.hidden == hidden);
    const dirs = try gpa.alloc(f32, entries * hidden);
    errdefer gpa.free(dirs);
    const good_dir = try gpa.alloc(f32, hidden);
    defer gpa.free(good_dir);
    const ng: f64 = @floatFromInt(@max(good.count, 1));
    const nb: f64 = @floatFromInt(@max(bad.count, 1));
    for (0..entries) |e| {
        const d = dirs[e * hidden ..][0..hidden];
        const g = good_means[e * hidden ..][0..hidden];
        const b = bad_means[e * hidden ..][0..hidden];
        // Pooled (sample-size weighted) variance per coordinate and its mean.
        var mean_var: f64 = 0;
        for (0..hidden) |i| mean_var += (ng * good.variance(e, i) + nb * bad.variance(e, i)) / (ng + nb);
        mean_var /= @floatFromInt(hidden);
        const eps = @as(f64, shrinkage) * mean_var + 1e-30;
        for (0..hidden) |i| {
            const pooled = (ng * good.variance(e, i) + nb * bad.variance(e, i)) / (ng + nb);
            d[i] = @floatCast((@as(f64, b[i]) - g[i]) / (pooled + eps));
        }
        tensor.normalize(d);
        if (orthogonalize) {
            @memcpy(good_dir, g);
            tensor.normalize(good_dir);
            const proj = tensor.dot(d, good_dir);
            tensor.axpy(d, -proj, good_dir);
            tensor.normalize(d);
        }
    }
    return dirs;
}

// ---------------------------------------------------------------------------
// Separation scores
// ---------------------------------------------------------------------------

/// Area under the ROC curve of the scalar projections: the probability that a
/// random bad prompt projects higher than a random good one (ties count half).
/// 0.5 = no separation, 1.0 = perfectly separable by a threshold.
pub fn auroc(gpa: Allocator, good: []const f32, bad: []const f32) !f64 {
    if (good.len == 0 or bad.len == 0) return 0.5;
    const Item = struct { v: f32, bad: bool };
    const items = try gpa.alloc(Item, good.len + bad.len);
    defer gpa.free(items);
    for (good, 0..) |v, i| items[i] = .{ .v = v, .bad = false };
    for (bad, 0..) |v, i| items[good.len + i] = .{ .v = v, .bad = true };
    std.mem.sort(Item, items, {}, struct {
        fn lt(_: void, a: Item, b: Item) bool {
            return a.v < b.v;
        }
    }.lt);
    // Rank-sum (Mann-Whitney) with average ranks for ties.
    var rank_sum_bad: f64 = 0;
    var i: usize = 0;
    while (i < items.len) {
        var j = i;
        while (j < items.len and items[j].v == items[i].v) j += 1;
        const avg_rank = (@as(f64, @floatFromInt(i + 1)) + @as(f64, @floatFromInt(j))) / 2.0;
        for (items[i..j]) |it| if (it.bad) {
            rank_sum_bad += avg_rank;
        };
        i = j;
    }
    const nb: f64 = @floatFromInt(bad.len);
    const ng: f64 = @floatFromInt(good.len);
    return (rank_sum_bad - nb * (nb + 1) / 2.0) / (nb * ng);
}

/// Standardised mean difference of the projections, `(μ_bad − μ_good) / σ_pooled`.
pub fn dprime(good: []const f32, bad: []const f32) f64 {
    const stats = struct {
        fn of(x: []const f32) [2]f64 {
            var s: f64 = 0;
            var q: f64 = 0;
            for (x) |v| {
                s += v;
                q += @as(f64, v) * v;
            }
            const n: f64 = @floatFromInt(@max(x.len, 1));
            const m = s / n;
            return .{ m, @max(q / n - m * m, 0) };
        }
    };
    const g = stats.of(good);
    const b = stats.of(bad);
    const sd = @sqrt((g[1] + b[1]) / 2.0);
    return (b[0] - g[0]) / @max(sd, 1e-12);
}

/// Per-entry separation of the two prompt sets along the entry's direction.
pub const Separation = struct {
    auroc: []f64,
    dprime: []f64,

    pub fn deinit(self: *Separation, gpa: Allocator) void {
        gpa.free(self.auroc);
        gpa.free(self.dprime);
    }
};

/// `good_proj` / `bad_proj` are `[prompts][entries]` projections of every
/// prompt's residual onto its entry's direction (see `Engine.getProjections`).
pub fn separation(gpa: Allocator, good_proj: []const f32, bad_proj: []const f32, entries: usize) !Separation {
    const ng = good_proj.len / entries;
    const nb = bad_proj.len / entries;
    const au = try gpa.alloc(f64, entries);
    errdefer gpa.free(au);
    const dp = try gpa.alloc(f64, entries);
    errdefer gpa.free(dp);
    const g = try gpa.alloc(f32, ng);
    defer gpa.free(g);
    const b = try gpa.alloc(f32, nb);
    defer gpa.free(b);
    for (0..entries) |e| {
        for (0..ng) |p| g[p] = good_proj[p * entries + e];
        for (0..nb) |p| b[p] = bad_proj[p * entries + e];
        au[e] = try auroc(gpa, g, b);
        dp[e] = dprime(g, b);
    }
    return .{ .auroc = au, .dprime = dp };
}

/// Bounds of the TPE's `direction_index` parameter (0 = output of the first
/// layer, i.e. entry 1; `last = num_layers − 1`).
pub const Range = struct { low: f32, high: f32 };

/// AUROC slack (absolute) below the best entry within which an entry still
/// counts as "highly separating" for `autoRange`.
pub const auto_range_tolerance = 0.01;

/// The hull of the layer entries whose AUROC is within `auto_range_tolerance`
/// of the best one, as a `direction_index` range; widened to at least one
/// layer so the sampler has an interval. `auroc[0]` (the embeddings) is never
/// a candidate, like in heretic.
pub fn autoRange(auroc_by_entry: []const f64) Range {
    const entries = auroc_by_entry.len;
    const last: f32 = @floatFromInt(entries - 2);
    var best: f64 = -1;
    for (auroc_by_entry[1..]) |a| best = @max(best, a);
    var lo: usize = entries;
    var hi: usize = 0;
    for (auroc_by_entry[1..], 1..) |a, e| if (a >= best - auto_range_tolerance) {
        lo = @min(lo, e);
        hi = @max(hi, e);
    };
    var low: f32 = @floatFromInt(lo - 1);
    var high: f32 = @floatFromInt(hi - 1);
    if (high - low < 1.0) {
        low = @max(low - 0.5, 0);
        high = @min(high + 0.5, last);
        if (high - low < 1.0) {
            if (low > 0) low = @max(high - 1.0, 0) else high = @min(low + 1.0, last);
        }
    }
    return .{ .low = low, .high = high };
}

/// Computes the `[entries][k][hidden]` orthonormal direction basis from the
/// mean directions. `sketch` is required for `k > 1`.
pub fn computeBasis(gpa: Allocator, good_means: []const f32, bad_means: []const f32, entries: usize, hidden: usize, k: usize, orthogonalize: bool, sketch: ?*const Sketch) ![]f32 {
    const first = try abliterate.computeDirections(gpa, good_means, bad_means, entries, hidden, orthogonalize);
    defer gpa.free(first);
    return computeBasisFrom(gpa, first, good_means, entries, hidden, k, orthogonalize, sketch);
}

/// Like `computeBasis` with the first direction of every entry given
/// (`[entries][hidden]`, unit vectors from any `Method`).
pub fn computeBasisFrom(gpa: Allocator, first: []const f32, good_means: []const f32, entries: usize, hidden: usize, k: usize, orthogonalize: bool, sketch: ?*const Sketch) ![]f32 {
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

test "separating direction recovers a planted refusal feature that the mean direction misses" {
    // Two orthogonal planted directions: r (the "refusal" feature, spread over
    // the first 15 coordinates) shifts the bad prompts by 1.0 with a spread of
    // 0.1 in both sets, so it separates them almost perfectly; the last
    // coordinate (a "massive activation" dimension) shifts them by 2.0 but
    // spreads by 5.0, so it barely separates anything. The difference of means
    // is dominated by that coordinate (cosine with r ~0.45) while the
    // variance-whitened direction follows r, and the projection AUROC shows
    // which one separates. (Per-coordinate whitening targets exactly such
    // coordinate-aligned nuisance variance; a nuisance spread evenly over all
    // coordinates would need a full covariance.)
    const gpa = std.testing.allocator;
    const hidden = 16;
    const entries = 1;
    const n = 400;
    var prng = std.Random.DefaultPrng.init(21);
    const rand = prng.random();
    var planted: [2 * hidden]f32 = [_]f32{0} ** (2 * hidden);
    for (planted[0 .. hidden - 1]) |*x| x.* = rand.floatNorm(f32);
    planted[2 * hidden - 1] = 1;
    abliterate.orthonormalize(&planted, 2, hidden);
    const r = planted[0..hidden];
    const nz = planted[hidden..];
    var good_m = try Moments.init(gpa, entries, hidden);
    defer good_m.deinit();
    var bad_m = try Moments.init(gpa, entries, hidden);
    defer bad_m.deinit();
    var good_sum = [_]f64{0} ** hidden;
    var bad_sum = [_]f64{0} ** hidden;
    var good_pts: [n * hidden]f32 = undefined;
    var bad_pts: [n * hidden]f32 = undefined;
    for (0..n) |p| {
        const g = good_pts[p * hidden ..][0..hidden];
        const b = bad_pts[p * hidden ..][0..hidden];
        const gr = 0.1 * rand.floatNorm(f32);
        const gn = 5.0 * rand.floatNorm(f32);
        const br = 1.0 + 0.1 * rand.floatNorm(f32);
        const bn = 2.0 + 5.0 * rand.floatNorm(f32);
        for (0..hidden) |i| {
            g[i] = gr * r[i] + gn * nz[i] + 0.05 * rand.floatNorm(f32);
            b[i] = br * r[i] + bn * nz[i] + 0.05 * rand.floatNorm(f32);
            good_sum[i] += g[i];
            bad_sum[i] += b[i];
        }
        good_m.add(0, g);
        bad_m.add(0, b);
    }
    var good_mean: [hidden]f32 = undefined;
    var bad_mean: [hidden]f32 = undefined;
    for (0..hidden) |i| {
        good_mean[i] = @floatCast(good_sum[i] / n);
        bad_mean[i] = @floatCast(bad_sum[i] / n);
    }
    const mean_dir = try abliterate.computeDirections(gpa, &good_mean, &bad_mean, entries, hidden, false);
    defer gpa.free(mean_dir);
    const sep_dir = try computeSeparating(gpa, &good_mean, &bad_mean, &good_m, &bad_m, entries, hidden, 0.1, false);
    defer gpa.free(sep_dir);
    try std.testing.expectApproxEqAbs(@as(f32, 1), tensor.norm2(sep_dir), 1e-5);
    try std.testing.expect(@abs(tensor.dot(mean_dir, r)) < 0.6);
    try std.testing.expect(@abs(tensor.dot(sep_dir, r)) > 0.95);
    // Projections onto the two candidate directions and their AUROC.
    var gp_mean: [n]f32 = undefined;
    var bp_mean: [n]f32 = undefined;
    var gp_sep: [n]f32 = undefined;
    var bp_sep: [n]f32 = undefined;
    for (0..n) |p| {
        gp_mean[p] = tensor.dot(good_pts[p * hidden ..][0..hidden], mean_dir);
        bp_mean[p] = tensor.dot(bad_pts[p * hidden ..][0..hidden], mean_dir);
        gp_sep[p] = tensor.dot(good_pts[p * hidden ..][0..hidden], sep_dir);
        bp_sep[p] = tensor.dot(bad_pts[p * hidden ..][0..hidden], sep_dir);
    }
    const au_mean = try auroc(gpa, &gp_mean, &bp_mean);
    const au_sep = try auroc(gpa, &gp_sep, &bp_sep);
    try std.testing.expect(au_mean < 0.85);
    try std.testing.expect(au_sep > 0.99);
    try std.testing.expect(dprime(&gp_sep, &bp_sep) > dprime(&gp_mean, &bp_mean));
    // The orthogonalised variant is still a unit vector orthogonal to the good mean.
    const sep_orth = try computeSeparating(gpa, &good_mean, &bad_mean, &good_m, &bad_m, entries, hidden, 0.1, true);
    defer gpa.free(sep_orth);
    var gdir: [hidden]f32 = good_mean;
    tensor.normalize(&gdir);
    try std.testing.expect(@abs(tensor.dot(sep_orth, &gdir)) < 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1), tensor.norm2(sep_orth), 1e-5);
}

test "auroc and separation table" {
    const gpa = std.testing.allocator;
    // Perfect separation, no separation, ties, and the inverted case.
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try auroc(gpa, &.{ 0, 1, 2 }, &.{ 3, 4 }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try auroc(gpa, &.{ 3, 4 }, &.{ 0, 1, 2 }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try auroc(gpa, &.{ 1, 1 }, &.{ 1, 1 }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), try auroc(gpa, &.{ 0, 2 }, &.{ 1, 3 }), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try auroc(gpa, &.{}, &.{1}), 1e-12);
    // [prompts][entries] projections: entry 0 separates, entry 1 does not.
    const good = [_]f32{ 0, 5, 1, 6, 2, 4 };
    const bad = [_]f32{ 3, 5, 4, 6, 5, 4 };
    var sep = try separation(gpa, &good, &bad, 2);
    defer sep.deinit(gpa);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), sep.auroc[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), sep.auroc[1], 1e-12);
    try std.testing.expect(sep.dprime[0] > 3);
    try std.testing.expectApproxEqAbs(@as(f64, 0), sep.dprime[1], 1e-12);
}

test "auto direction range is the hull of the best-separating layers" {
    // 8 layers (9 entries): entries 4..6 are within tolerance of the best.
    const a = [_]f64{ 1.0, 0.6, 0.7, 0.8, 0.995, 1.0, 0.99, 0.9, 0.8 };
    const r = autoRange(&a);
    try std.testing.expectEqual(@as(f32, 3), r.low);
    try std.testing.expectEqual(@as(f32, 5), r.high);
    // A single best entry is widened to a one-layer interval.
    const b = [_]f64{ 0.5, 0.5, 0.5, 0.9, 0.5, 0.5 };
    const rb = autoRange(&b);
    try std.testing.expect(rb.high - rb.low >= 1.0 - 1e-6);
    try std.testing.expect(rb.low <= 2 and rb.high >= 2);
    try std.testing.expect(rb.low >= 0 and rb.high <= 4);
    // The embedding entry never counts, and the range stays inside [0, last].
    const c = [_]f64{ 1.0, 0.9, 0.5, 0.5 };
    const rc = autoRange(&c);
    try std.testing.expectEqual(@as(f32, 0), rc.low);
    try std.testing.expectEqual(@as(f32, 1), rc.high);
}
