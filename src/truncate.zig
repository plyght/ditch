//! `ditch truncate`: a checkpoint of a few decoder layers of a released
//! model, for checking a family layer by layer against a reference on its
//! real weights without downloading the whole model.
//!
//! Only what the cut keeps is read: each shard's header by a range read (the
//! 8-byte length, then the JSON), then the embedding, the final norm, the LM
//! head, every other tensor outside the decoder layers, and the chosen layers,
//! streamed straight into one `model.safetensors`. Neighbouring tensors are
//! merged into requests of up to 32 MB, run concurrently by a few workers; each
//! worker holds one request's bytes at a time, so memory stays bounded however
//! large the model. The output file is created at its full size first (sparse:
//! what is never written takes no space) and every request writes its bytes in
//! place. Quantisation stays exactly as stored: tensors are copied byte for
//! byte with their dtype and shape.
//!
//! Kept layers are renumbered 0..K-1 in the order given, in both spellings of
//! a layer tensor (`model.layers.N.` and DeepSeek's unprefixed `layers.N.`).
//! config.json gets `num_hidden_layers = K` and every per-layer list cut to the
//! kept layers (`editConfig`), so the result loads like any other checkpoint.
//! `kindLayers` picks the fewest layers that cover every layer kind of a
//! family, for families whose kinds are far apart.
//!
//! The model is a local directory, a Hub id (`owner/name` or `hf://owner/name`,
//! revision `main` or `--model-commit`) or an `http(s)://` base URL serving the
//! model files; remote reads go through `hf.Http` (token, redirects, retries,
//! 429 back-off). This is the Zig counterpart of tools/truncate_checkpoint.py
//! and writes the same bytes.

const std = @import("std");
const Io = std.Io;
const hf = @import("hf.zig");
const remote = @import("remote.zig");
const config = @import("config.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const ObjectMap = std.json.ObjectMap;
const JsonArray = std.json.Array;

/// Largest range request, and the size neighbouring tensors are merged up to.
pub const default_request_size: u64 = 32 << 20;
/// Requests in flight. Each holds at most one request's bytes (two for a
/// per-layer table gather), so the memory in use is bounded by this times
/// `default_request_size`.
pub const default_connections: u32 = 8;

/// Which decoder layers to keep.
pub const Layers = union(enum) {
    /// Layers 0..n-1.
    first: usize,
    /// These layers, renumbered 0..len-1 in this order.
    list: []const usize,
    /// `kindLayers`: the fewest layers covering every layer kind.
    kinds,
};

/// Rows of one tensor to write (`--rows NAME=FILE`): the tensor keeps its
/// full shape and size in the output, only these rows are fetched and
/// written, the rest of it stays a hole of the sparse file.
pub const RowSelection = struct {
    /// Tensor name, in the source or in the cut.
    name: []const u8,
    rows: []const u64,
};

pub const Options = struct {
    /// Local directory, `owner/name`, `hf://owner/name` or an `http(s)://` base URL.
    model: []const u8,
    out_dir: []const u8,
    layers: Layers,
    /// Tensors whose name starts with one of these are left out.
    drop: []const []const u8 = &.{},
    rows: []const RowSelection = &.{},
    /// Hub revision (default `main`).
    revision: ?[]const u8 = null,
    connections: u32 = default_connections,
    request_size: u64 = default_request_size,
    /// Progress lines ("x GB of y"); null for none.
    progress: ?*Io.Writer = null,
};

pub const Result = struct {
    /// The kept layers of the source, in their new order (owned by the caller's allocator).
    layers: []usize,
    /// num_hidden_layers of the source.
    source_layers: usize,
    tensors: usize,
    /// Size of the tensor data in the output (holes included).
    logical_bytes: u64,
    /// Bytes read from the source for tensor data.
    fetched_bytes: u64,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.layers);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// config.json
// ---------------------------------------------------------------------------

/// Per-layer lists cut to the kept layers (one entry per layer).
const cut_lists = [_][]const u8{ "layer_types", "mlp_layer_types", "num_attention_heads_per_layer", "compress_ratios", "is_moe_layer", "sliding_windows", "hybrid_layer_pattern", "moe_layer_freq", "no_rope_layers", "intermediate_size", "activation_sparsity_pattern" };
/// Lists of layer ids: the kept ids stay, renumbered.
const id_lists = [_][]const u8{ "kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids", "dspark_target_layer_ids", "moe_layers" };

/// The integer value of a JSON number (as parsed, or kept as text by
/// `parse_numbers = false`); booleans count as 0 and 1, as in Python.
fn asInt(v: Value) ?i64 {
    return switch (v) {
        .integer => |i| i,
        .number_string => |s| std.fmt.parseInt(i64, s, 10) catch null,
        .bool => |b| @intFromBool(b),
        else => null,
    };
}

/// Python truthiness of a JSON value.
fn truthy(v: Value) bool {
    return switch (v) {
        .null => false,
        .bool => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .number_string => |s| (std.fmt.parseFloat(f64, s) catch 1) != 0,
        .string => |s| s.len > 0,
        .array => |arr| arr.items.len > 0,
        .object => |o| o.count() > 0,
    };
}

/// The object holding the language model's settings: `text_config` when
/// present (multimodal checkpoints), else the top level.
pub fn textConfig(root: *Value) !*ObjectMap {
    if (root.* != .object) return error.InvalidConfig;
    if (root.object.getPtr("text_config")) |tc| if (tc.* == .object) return &tc.object;
    return &root.object;
}

fn numLayers(tc: ObjectMap) !usize {
    const v = tc.get("num_hidden_layers") orelse {
        std.log.err("config.json has no num_hidden_layers", .{});
        return error.InvalidConfig;
    };
    const n = asInt(v) orelse return error.InvalidConfig;
    if (n <= 0) return error.InvalidConfig;
    return @intCast(n);
}

/// The new index of source layer `old`, or null when it is not kept.
fn newId(layers: []const usize, old: i64) ?i64 {
    if (old < 0) return null;
    for (layers, 0..) |l, i| if (l == old) return @intCast(i);
    return null;
}

/// `items[j]` for every `j` of `picks`.
fn pick(a: Allocator, items: []const Value, picks: []const usize, key: []const u8) !Value {
    var out = JsonArray.init(a);
    for (picks) |j| {
        if (j >= items.len) {
            std.log.err("config.json: {s} has {d} entries; entry {d} is needed", .{ key, items.len, j });
            return error.InvalidConfig;
        }
        try out.append(items[j]);
    }
    return .{ .array = out };
}

/// Rewrites a parsed config.json for a cut that keeps `layers` (source
/// indices, renumbered 0..len-1 in this order), exactly as
/// tools/truncate_checkpoint.py does: on `text_config` when present, else the
/// top level, `num_hidden_layers` becomes the kept count; every per-layer list
/// is cut to the kept layers (MiMo's layer kinds written out from its
/// patterns, MiniMax M3's `sparse_attention_config` lists too); lists of
/// layer ids keep the kept ids, renumbered (a list paired with one keeps the
/// matching entries); Kimi / GLM `linear_attn_config` ids (1-based unless a 0
/// appears); Nemotron-H's `hybrid_override_pattern`; the dense-prefix count
/// and Gemma's KV-shared count recounted over the kept layers; and no
/// multi-token-prediction layers when a dropped prefix names them. Every other
/// field, and the key order, stays as it was. `a` should be an arena: the
/// replaced values are not freed.
pub fn editConfig(a: Allocator, root: *Value, layers: []const usize, drop: []const []const u8) !void {
    const tc = try textConfig(root);
    const n_orig = try numLayers(tc.*);
    try tc.put(a, "num_hidden_layers", .{ .integer = @intCast(layers.len) });
    for (cut_lists) |key| if (tc.getPtr(key)) |v| if (v.* == .array) {
        v.* = try pick(a, v.array.items, layers, key);
    };
    // MiMo V2: transformers derives the layer kinds from the layer index
    // (full at 0 and every 6th) unless `layer_types` says; in a cut the index
    // no longer tells, so the kinds are written out from the patterns.
    if (tc.get("hybrid_layer_pattern")) |hp| if (hp == .array and !tc.contains("layer_types")) {
        var kinds = JsonArray.init(a);
        for (hp.array.items) |x| try kinds.append(.{ .string = if (truthy(x)) "sliding_attention" else "full_attention" });
        try tc.put(a, "layer_types", .{ .array = kinds });
        if (tc.get("moe_layer_freq")) |mf| if (mf == .array) {
            var mlp = JsonArray.init(a);
            for (mf.array.items) |x| try mlp.append(.{ .string = if (truthy(x)) "sparse" else "dense" });
            try tc.put(a, "mlp_layer_types", .{ .array = mlp });
        };
    };
    // MiniMax M3 keeps its per-layer sparse-attention lists in a dictionary.
    if (tc.getPtr("sparse_attention_config")) |sac| if (sac.* == .object) {
        for (sac.object.keys(), sac.object.values()) |key, *v| {
            if (v.* == .array and v.array.items.len == n_orig) v.* = try pick(a, v.array.items, layers, key);
        }
    };
    for (id_lists) |key| if (tc.getPtr(key)) |v| if (v.* == .array) {
        var keep = std.ArrayList(usize).empty;
        for (v.array.items, 0..) |x, j| {
            if (asInt(x)) |id| if (newId(layers, id) != null) try keep.append(a, j);
        }
        if (std.mem.eql(u8, key, "engram_layer_ids")) {
            if (tc.getPtr("engram_num_embeddings")) |p| if (p.* == .array) {
                p.* = try pick(a, p.array.items, keep.items, "engram_num_embeddings");
            };
        }
        var ids = JsonArray.init(a);
        for (keep.items) |j| try ids.append(.{ .integer = newId(layers, asInt(v.array.items[j]).?).? });
        v.* = .{ .array = ids };
    };
    // Kimi-Linear / Kimi K3 name their layer kinds with 1-based ids,
    // GLM-5.3-Flash with 0-based ones (its lists contain a 0).
    if (tc.getPtr("linear_attn_config")) |lac| if (lac.* == .object) {
        const one: i64 = if (linearIdsZeroBased(lac.object)) 0 else 1;
        for ([_][]const u8{ "kda_layers", "full_attn_layers" }) |key| if (lac.object.getPtr(key)) |v| if (v.* == .array) {
            var ids = JsonArray.init(a);
            for (v.array.items) |x| {
                const id = asInt(x) orelse continue;
                if (newId(layers, id - one)) |n| try ids.append(.{ .integer = n + one });
            }
            v.* = .{ .array = ids };
        };
    };
    // Nemotron-H spells its layer kinds as one character per layer.
    if (tc.getPtr("hybrid_override_pattern")) |p| if (p.* == .string) {
        const pattern = p.string;
        const cut = try a.alloc(u8, layers.len);
        for (layers, cut) |j, *c| {
            if (j >= pattern.len) {
                std.log.err("config.json: hybrid_override_pattern has {d} characters; layer {d} is needed", .{ pattern.len, j });
                return error.InvalidConfig;
            }
            c.* = pattern[j];
        }
        p.* = .{ .string = cut };
    };
    // The dense-prefix count, for a cut that skips layers: the kept layers
    // that were dense (DeepSeek V3.2 --layers 0,3 keeps one dense layer, not three).
    if (tc.getPtr("first_k_dense_replace")) |p| if (asInt(p.*)) |k| {
        var count: i64 = 0;
        for (layers) |j| count += @intFromBool(@as(i64, @intCast(j)) < k);
        p.* = .{ .integer = count };
    };
    // Gemma 4's KV-shared layers are the last num_kv_shared_layers: count the
    // kept ones (each then reads the last kept non-shared layer of its type).
    if (tc.getPtr("num_kv_shared_layers")) |p| if (asInt(p.*)) |k| if (k > 0) {
        var count: i64 = 0;
        const first = @as(i64, @intCast(n_orig)) - k;
        for (layers) |j| count += @intFromBool(@as(i64, @intCast(j)) >= first);
        p.* = .{ .integer = count };
    };
    if (tc.getPtr("candidate_source_layer_id")) |p| {
        const old = asInt(p.*);
        p.* = .{ .integer = if (old) |o| newId(layers, o) orelse -1 else -1 };
    }
    for (drop) |d| if (std.mem.indexOf(u8, d, "mtp") != null) {
        if (tc.getPtr("num_nextn_predict_layers")) |p| p.* = .{ .integer = 0 };
        break;
    };
}

fn linearIdsZeroBased(lac: ObjectMap) bool {
    for ([_][]const u8{ "kda_layers", "full_attn_layers" }) |key| if (lac.get(key)) |v| if (v == .array) {
        for (v.array.items) |x| if (asInt(x)) |id| if (id == 0) return true;
    };
    return false;
}

/// Parses config.json keeping every number as written (so nothing is
/// reformatted), into `a`.
pub fn parseConfig(a: Allocator, text: []const u8) !Value {
    return std.json.parseFromSliceLeaky(Value, a, text, .{ .parse_numbers = false }) catch |err| {
        std.log.err("config.json is not valid JSON: {s}", .{@errorName(err)});
        return error.InvalidConfig;
    };
}

/// Renders a config as tools/truncate_checkpoint.py writes it (`json.dump`
/// with `indent=1`: one space per level, non-ASCII escaped, no final newline).
pub fn renderConfig(a: Allocator, root: Value) ![]u8 {
    return std.json.Stringify.valueAlloc(a, root, .{ .whitespace = .indent_1, .escape_unicode = true });
}

/// `editConfig` on config.json text; returns the new text (in `a`).
pub fn editConfigText(a: Allocator, text: []const u8, layers: []const usize, drop: []const []const u8) ![]u8 {
    var root = try parseConfig(a, text);
    try editConfig(a, &root, layers, drop);
    return renderConfig(a, root);
}

// ---------------------------------------------------------------------------
// Layer kinds
// ---------------------------------------------------------------------------

/// The smallest ascending set of layers that covers every layer kind of a
/// model, from its parsed config.json: the first layer of every distinct
/// combination of the per-layer lists (layer_types, mlp_layer_types,
/// hybrid_layer_pattern, moe_layer_freq, is_moe_layer, compress_ratios, the
/// other lists `editConfig` cuts and MiniMax M3's sparse-attention lists),
/// membership of the id lists (KV / index sources, engram layers, ...), KDA
/// versus full attention (`linear_attn_config`), the Nemotron-H pattern
/// character, dense versus MoE (`first_k_dense_replace`) and KV-shared versus
/// not (`num_kv_shared_layers`). Layer 0 is always in it, and so is every
/// layer a kept one refers to: the KV / index source a compressed layer reads,
/// the candidate source, and the layer a Gemma KV-shared layer takes its keys
/// and values from. Kinds that a cut does not rewrite, a formula of the layer
/// index (`full_attention_interval`, `decoder_sparse_step`,
/// `sliding_window_pattern` without `layer_types`, Jamba's periods, ...) or a
/// list `editConfig` leaves alone (`mlp_only_layers`, LFM2's `full_attn_idxs`,
/// any other list with one entry per layer), only survive a prefix, so with
/// one of them the result is every layer up to the last one picked.
/// Returned in `a`.
pub fn kindLayers(a: Allocator, root: Value) ![]usize {
    var r = root;
    const tc = (try textConfig(&r)).*;
    const n = try numLayers(tc);
    const sigs = try a.alloc(std.ArrayList(u8), n);
    for (sigs) |*s| s.* = .empty;
    var prefix = false;

    for (cut_lists) |key| if (tc.get(key)) |v| if (v == .array) {
        for (sigs, 0..) |*s, i| try sigValue(a, s, key, if (i < v.array.items.len) v.array.items[i] else null);
    };
    if (tc.get("sparse_attention_config")) |sac| if (sac == .object) {
        for (sac.object.keys(), sac.object.values()) |key, v| if (v == .array and v.array.items.len == n) {
            for (sigs, 0..) |*s, i| try sigValue(a, s, key, v.array.items[i]);
        };
    };
    for (id_lists) |key| if (tc.get(key)) |v| if (v == .array) {
        for (sigs, 0..) |*s, i| try sigFlag(a, s, key, inIds(v.array.items, @intCast(i), 0));
    };
    if (tc.get("linear_attn_config")) |lac| if (lac == .object) {
        const one: i64 = if (linearIdsZeroBased(lac.object)) 0 else 1;
        for ([_][]const u8{ "kda_layers", "full_attn_layers" }) |key| if (lac.object.get(key)) |v| if (v == .array) {
            for (sigs, 0..) |*s, i| try sigFlag(a, s, key, inIds(v.array.items, @intCast(i), one));
        };
    };
    if (tc.get("hybrid_override_pattern")) |p| if (p == .string) {
        for (sigs, 0..) |*s, i| try s.print(a, "hybrid_override_pattern={c};", .{if (i < p.string.len) p.string[i] else '-'});
    };
    if (tc.get("first_k_dense_replace")) |v| if (asInt(v)) |k| {
        for (sigs, 0..) |*s, i| try sigFlag(a, s, "dense", @as(i64, @intCast(i)) < k);
    };
    var shared_from: ?usize = null;
    if (tc.get("num_kv_shared_layers")) |v| if (asInt(v)) |k| if (k > 0 and k < n) {
        shared_from = n - @as(usize, @intCast(k));
    };
    if (shared_from) |first| for (sigs, 0..) |*s, i| try sigFlag(a, s, "kv_shared", i >= first);

    // Kinds given by a formula of the layer index.
    const has_layer_types = tc.get("layer_types") != null;
    const periodic = [_]struct { key: []const u8, needs_no_layer_types: bool, offset: u64 }{
        .{ .key = "full_attention_interval", .needs_no_layer_types = true, .offset = 1 },
        .{ .key = "sliding_window_pattern", .needs_no_layer_types = true, .offset = 1 },
        .{ .key = "decoder_sparse_step", .needs_no_layer_types = false, .offset = 1 },
        .{ .key = "interleave_moe_layer_step", .needs_no_layer_types = false, .offset = 1 },
        .{ .key = "moe_layer_freq", .needs_no_layer_types = false, .offset = 0 },
    };
    for (periodic) |pd| if (tc.get(pd.key)) |v| if (asInt(v)) |k| if (k > 1 and !(pd.needs_no_layer_types and has_layer_types)) {
        prefix = true;
        const period: u64 = @intCast(k);
        for (sigs, 0..) |*s, i| try sigFlag(a, s, pd.key, (i + pd.offset) % period == 0);
    };
    for ([_][2][]const u8{ .{ "attn_layer_period", "attn_layer_offset" }, .{ "expert_layer_period", "expert_layer_offset" } }) |po| {
        const period = asInt(tc.get(po[0]) orelse continue) orelse continue;
        if (period <= 1) continue;
        const offset = if (tc.get(po[1])) |o| asInt(o) orelse 0 else 0;
        prefix = true;
        for (sigs, 0..) |*s, i| try sigFlag(a, s, po[0], @mod(@as(i64, @intCast(i)), period) == offset);
    }
    // Layer-id lists the cut does not renumber (Qwen MoE's dense layers, LFM2's attention layers).
    for ([_][]const u8{ "mlp_only_layers", "full_attn_idxs" }) |key| if (tc.get(key)) |v| if (v == .array and v.array.items.len > 0) {
        prefix = true;
        for (sigs, 0..) |*s, i| try sigFlag(a, s, key, inIds(v.array.items, @intCast(i), 0));
    };
    // Any other list with one entry per layer is taken for a per-layer
    // setting the cut does not rewrite, which likewise only survives a prefix.
    for (tc.keys(), tc.values()) |key, v| {
        if (v != .array or v.array.items.len != n or n < 2) continue;
        if (isListed(&cut_lists, key) or isListed(&id_lists, key) or isListed(&.{ "engram_num_embeddings", "mlp_only_layers", "full_attn_idxs", "architectures" }, key)) continue;
        prefix = true;
        for (sigs, 0..) |*s, i| try sigValue(a, s, key, v.array.items[i]);
    }

    // The first layer of every signature.
    var keep = try a.alloc(bool, n);
    @memset(keep, false);
    keep[0] = true;
    var seen = std.StringHashMapUnmanaged(void).empty;
    for (sigs, 0..) |s, i| {
        const gop = try seen.getOrPut(a, s.items);
        if (!gop.found_existing) keep[i] = true;
    }

    // The layers the kept ones refer to, until nothing changes.
    const ratios: ?[]const Value = if (tc.get("compress_ratios")) |v| (if (v == .array) v.array.items else null) else null;
    const layer_types: ?[]const Value = if (tc.get("layer_types")) |v| (if (v == .array) v.array.items else null) else null;
    var changed = true;
    while (changed) {
        changed = false;
        for (0..n) |i| {
            if (!keep[i]) continue;
            // A compressed layer reads the cache (and index) of the last source at or before it.
            const compressed = if (ratios) |rs| (i < rs.len and truthy(rs[i])) else true;
            if (compressed) for ([_][]const u8{ "kv_source_layer_ids", "index_source_layer_ids" }) |key| {
                const v = tc.get(key) orelse continue;
                if (v != .array) continue;
                var best: ?usize = null;
                for (v.array.items) |x| if (asInt(x)) |id| if (id >= 0 and id <= i) {
                    const u: usize = @intCast(id);
                    if (best == null or u > best.?) best = u;
                };
                if (best) |b| if (!keep[b]) {
                    keep[b] = true;
                    changed = true;
                };
            };
            // A KV-shared layer reads the last non-shared layer of its kind.
            if (shared_from) |first| if (i >= first) {
                var j = first;
                while (j > 0) {
                    j -= 1;
                    const same = if (layer_types) |lt| (j < lt.len and i < lt.len and valueEql(lt[j], lt[i])) else true;
                    if (same) {
                        if (!keep[j]) {
                            keep[j] = true;
                            changed = true;
                        }
                        break;
                    }
                }
            };
        }
    }
    if (tc.get("candidate_source_layer_id")) |v| if (asInt(v)) |c| if (c >= 0 and c < n) {
        keep[@intCast(c)] = true;
    };

    var out = std.ArrayList(usize).empty;
    var last: usize = 0;
    for (keep, 0..) |k, i| if (k) {
        last = i;
    };
    for (keep, 0..) |k, i| if (k or (prefix and i <= last)) try out.append(a, i);
    return out.items;
}

fn sigValue(a: Allocator, s: *std.ArrayList(u8), key: []const u8, v: ?Value) !void {
    try s.appendSlice(a, key);
    try s.append(a, '=');
    if (v) |x| {
        const text = try std.json.Stringify.valueAlloc(a, x, .{});
        try s.appendSlice(a, text);
    } else try s.append(a, '-');
    try s.append(a, ';');
}

fn sigFlag(a: Allocator, s: *std.ArrayList(u8), key: []const u8, flag: bool) !void {
    try s.print(a, "{s}={d};", .{ key, @intFromBool(flag) });
}

fn isListed(list: []const []const u8, key: []const u8) bool {
    for (list) |k| if (std.mem.eql(u8, k, key)) return true;
    return false;
}

/// Whether layer `i` is in a list of layer ids counted from `one`.
fn inIds(items: []const Value, i: i64, one: i64) bool {
    for (items) |x| if (asInt(x)) |id| if (id - one == i) return true;
    return false;
}

fn valueEql(x: Value, y: Value) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .string => |s| std.mem.eql(u8, s, y.string),
        .number_string => |s| std.mem.eql(u8, s, y.number_string),
        .integer => |v| v == y.integer,
        .bool => |v| v == y.bool,
        .float => |v| v == y.float,
        .null => true,
        else => false,
    };
}

// ---------------------------------------------------------------------------
// Tensor names
// ---------------------------------------------------------------------------

/// Where the layer index sits in a tensor name: the digits of the first
/// `layers.N.` that starts the name or follows a '.' (the Python tool's
/// `(?:^|\.)layers\.(\d+)\.`).
pub const LayerRef = struct { start: usize, end: usize, index: u64 };

pub fn layerOf(name: []const u8) ?LayerRef {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, name, from, "layers.")) |q| {
        from = q + 1;
        if (q != 0 and name[q - 1] != '.') continue;
        const ds = q + "layers.".len;
        var e = ds;
        while (e < name.len and std.ascii.isDigit(name[e])) e += 1;
        if (e == ds or e >= name.len or name[e] != '.') continue;
        const index = std.fmt.parseInt(u64, name[ds..e], 10) catch continue;
        return .{ .start = ds, .end = e, .index = index };
    }
    return null;
}

/// The tensor's name in the cut (its layer renumbered), in `a`; null when
/// the cut leaves it out.
pub fn cutName(a: Allocator, name: []const u8, layers: []const usize, drop: []const []const u8) !?[]const u8 {
    for (drop) |d| if (std.mem.startsWith(u8, name, d)) return null;
    const ref = layerOf(name) orelse return name;
    const idx = newId(layers, std.math.cast(i64, ref.index) orelse return null) orelse return null;
    return try std.fmt.allocPrint(a, "{s}{d}{s}", .{ name[0..ref.start], idx, name[ref.end..] });
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// Small files copied into the cut: top-level, not hidden, and not the shard index.
pub fn isSmallFile(name: []const u8) bool {
    if (name.len == 0 or name[0] == '.' or std.mem.indexOfScalar(u8, name, '/') != null) return false;
    if (std.mem.eql(u8, name, "model.safetensors.index.json")) return false;
    for ([_][]const u8{ ".json", ".jinja", ".model", ".txt", ".tiktoken", ".py" }) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}

/// Small files looked for at a plain base URL, which cannot be listed.
const url_small_files = [_][]const u8{ "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "vocab.txt", "chat_template.jinja", "chat_template.json", "tokenizer.model", "tiktoken.model", "preprocessor_config.json", "processor_config.json" };

const Shard = struct {
    name: []const u8,
    /// Remote: the file's URL.
    url: []const u8 = "",
    /// Local: the open file.
    file: ?Io.File = null,
};

const Source = struct {
    io: Io,
    http: ?*hf.Http,
    /// Allocator of the buffers `fetch` returns.
    alloc: Allocator,
    /// Local directory (null for a remote model).
    dir: ?Io.Dir = null,
    /// Remote: base URL of the model files, ending with '/'.
    base_url: []const u8 = "",
    /// Remote Hub models: the id and revision (the file list comes from the Hub API).
    hub_id: ?[]const u8 = null,
    revision: []const u8 = "main",

    fn open(a: Allocator, gpa: Allocator, io: Io, http: ?*hf.Http, model: []const u8, revision: ?[]const u8) !Source {
        const rev = revision orelse "main";
        if (!remote.isRemoteId(model) and hf.isLocalDir(io, model)) {
            return .{ .io = io, .http = null, .alloc = gpa, .dir = try Io.Dir.cwd().openDir(io, model, .{ .iterate = true }) };
        }
        const h = http orelse {
            std.log.err("model directory not found: {s}", .{model});
            return error.ModelNotFound;
        };
        if (std.mem.startsWith(u8, model, "http://") or std.mem.startsWith(u8, model, "https://")) {
            const base = if (std.mem.endsWith(u8, model, "/")) model else try std.fmt.allocPrint(a, "{s}/", .{model});
            return .{ .io = io, .http = h, .alloc = h.gpa, .base_url = base, .revision = rev };
        }
        const id = remote.hubId(model);
        if (std.mem.indexOfScalar(u8, id, '/') == null or std.mem.startsWith(u8, id, ".") or std.mem.startsWith(u8, id, "/")) {
            std.log.err("model directory not found: {s} (a Hub model is owner/name)", .{model});
            return error.ModelNotFound;
        }
        return .{
            .io = io,
            .http = h,
            .alloc = h.gpa,
            .base_url = try std.fmt.allocPrint(a, "https://huggingface.co/{s}/resolve/{s}/", .{ id, rev }),
            .hub_id = id,
            .revision = rev,
        };
    }

    fn close(self: *Source) void {
        if (self.dir) |*d| d.close(self.io);
    }

    fn url(self: *const Source, a: Allocator, name: []const u8) ![]const u8 {
        return std.fmt.allocPrint(a, "{s}{s}", .{ self.base_url, name });
    }

    /// The small files to copy (for a base URL: the usual names, which may be missing).
    fn smallFiles(self: *const Source, a: Allocator) ![]const []const u8 {
        var names = std.ArrayList([]const u8).empty;
        if (self.dir) |d| {
            var it = d.iterate();
            while (try it.next(self.io)) |e| {
                if (e.kind != .file and e.kind != .sym_link) continue;
                if (isSmallFile(e.name)) try names.append(a, try a.dupe(u8, e.name));
            }
        } else if (self.hub_id) |id| {
            const api = try std.fmt.allocPrint(a, "https://huggingface.co/api/models/{s}/revision/{s}", .{ id, self.revision });
            const text = self.http.?.get(api) catch |err| {
                switch (err) {
                    error.NotFound => std.log.err("model {s} not found on Hugging Face (revision {s})", .{ id, self.revision }),
                    error.Forbidden => std.log.err("access to {s} denied; gated models require HF_TOKEN to be set", .{id}),
                    else => std.log.err("could not list the files of {s}: {s}", .{ id, @errorName(err) }),
                }
                return error.ModelNotFound;
            };
            defer self.alloc.free(text);
            const info = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch return error.InvalidResponse;
            if (info != .object) return error.InvalidResponse;
            const siblings = info.object.get("siblings") orelse return error.InvalidResponse;
            if (siblings != .array) return error.InvalidResponse;
            for (siblings.array.items) |s| {
                if (s != .object) continue;
                const name = s.object.get("rfilename") orelse continue;
                if (name == .string and isSmallFile(name.string)) try names.append(a, try a.dupe(u8, name.string));
            }
        } else {
            try names.appendSlice(a, &url_small_files);
        }
        std.mem.sort([]const u8, names.items, {}, lessThan);
        return names.items;
    }

    /// Copies small file `name` into `out`; false when the source has no such file.
    fn copySmall(self: *const Source, a: Allocator, name: []const u8, out: Io.Dir) !bool {
        if (self.dir) |d| {
            d.copyFile(name, out, name, self.io, .{}) catch |err| switch (err) {
                error.FileNotFound => return false,
                else => return err,
            };
            return true;
        }
        self.http.?.download(out, name, try self.url(a, name), null) catch |err| switch (err) {
            error.NotFound => {
                // The download's partial file, left by the 404.
                out.deleteFile(self.io, try std.fmt.allocPrint(a, "{s}.part", .{name})) catch {};
                return false;
            },
            error.Forbidden => {
                std.log.err("access to {s} denied; gated models require HF_TOKEN to be set", .{self.base_url});
                return error.ModelNotFound;
            },
            else => return err,
        };
        return true;
    }

    /// A whole small file (the shard index), or null when there is none.
    fn readSmall(self: *const Source, a: Allocator, name: []const u8) !?[]u8 {
        if (self.dir) |d| {
            return d.readFileAlloc(self.io, name, a, .limited(1 << 30)) catch |err| switch (err) {
                error.FileNotFound => return null,
                else => return err,
            };
        }
        const body = self.http.?.get(try self.url(a, name)) catch |err| switch (err) {
            error.NotFound => return null,
            else => return err,
        };
        defer self.alloc.free(body);
        return try a.dupe(u8, body);
    }

    fn openShard(self: *const Source, a: Allocator, name: []const u8) !Shard {
        if (self.dir) |d| {
            const file = d.openFile(self.io, name, .{}) catch |err| {
                std.log.err("could not open the shard {s}: {s}", .{ name, @errorName(err) });
                return err;
            };
            return .{ .name = name, .file = file };
        }
        return .{ .name = name, .url = try self.url(a, name) };
    }

    /// Bytes `[start, start + len)` of a shard, owned by `self.alloc`.
    fn fetch(self: *const Source, shard: *const Shard, start: u64, len: u64) ![]u8 {
        if (shard.file) |f| {
            const buf = try self.alloc.alloc(u8, @intCast(len));
            errdefer self.alloc.free(buf);
            const got = try f.readPositionalAll(self.io, buf, start);
            if (got != len) {
                std.log.err("{s}: {d} bytes at {d} are past the end of the file", .{ shard.name, len, start });
                return error.ShortRead;
            }
            return buf;
        }
        if (len == 0) return self.alloc.alloc(u8, 0);
        const body = try self.http.?.getRange(shard.url, start, start + len - 1);
        if (body.len != len) {
            self.alloc.free(body);
            std.log.err("{s}: asked for {d} bytes at {d}, got a different length", .{ shard.url, len, start });
            return error.ShortRead;
        }
        return body;
    }
};

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// A per-layer table cut to the kept layers' blocks (Gemma 3n / Gemma 4).
const Slice = struct {
    /// 0: row blocks (`per_layer_model_projection` `[layers * width, hidden]`);
    /// 1: column blocks of every row (`embed_tokens_per_layer` `[vocab, layers * width]`).
    axis: u1,
    width: u64,
    rows: u64,
    cols: u64,
    elem: u64,
};

const Entry = struct {
    source_name: []const u8,
    /// Name in the cut.
    name: []const u8,
    dtype: []const u8,
    shape: []const u64,
    shard: u32,
    /// Absolute offset of the tensor's bytes in the shard.
    start: u64,
    /// Bytes in the output.
    len: u64,
    slice: ?Slice = null,
    rows: ?[]const u64 = null,
    /// Offset in the output's data section.
    offset: u64 = 0,
};

const Job = struct {
    shard: u32,
    src: u64,
    len: u64,
    /// Absolute offset in the output file.
    dst: u64,
    gather: ?Gather = null,
};

/// Rows of a `[rows, layers * width]` table, keeping the kept layers' columns.
const Gather = struct { rows: u64, row_bytes: u64, seg: u64 };

fn jsonInt(v: Value) !u64 {
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else error.InvalidCheckpoint,
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch error.InvalidCheckpoint,
        else => error.InvalidCheckpoint,
    };
}

/// Reads a shard's header (the 8-byte length, then the JSON) into `a`.
fn readHeader(src: *const Source, a: Allocator, shard: *const Shard) !struct { value: Value, data_start: u64 } {
    const len_bytes = try src.fetch(shard, 0, 8);
    defer src.alloc.free(len_bytes);
    const n = std.mem.readInt(u64, len_bytes[0..8], .little);
    if (n == 0 or n > 1 << 30) {
        std.log.err("{s}: not a safetensors file (header length {d})", .{ shard.name, n });
        return error.InvalidCheckpoint;
    }
    const text = try src.fetch(shard, 8, n);
    defer src.alloc.free(text);
    const value = std.json.parseFromSliceLeaky(Value, a, text, .{ .parse_numbers = false }) catch {
        std.log.err("{s}: the safetensors header is not valid JSON", .{shard.name});
        return error.InvalidCheckpoint;
    };
    if (value != .object) return error.InvalidCheckpoint;
    return .{ .value = value, .data_start = 8 + n };
}

/// The safetensors header of the cut, padded with spaces to a multiple of 8.
fn renderHeader(a: Allocator, entries: []const Entry) ![]u8 {
    var buf: Io.Writer.Allocating = .init(a);
    const w = &buf.writer;
    try w.writeByte('{');
    for (entries, 0..) |e, i| {
        if (i > 0) try w.writeAll(", ");
        try std.json.Stringify.value(e.name, .{ .escape_unicode = true }, w);
        try w.writeAll(": {\"dtype\": ");
        try std.json.Stringify.value(e.dtype, .{ .escape_unicode = true }, w);
        try w.writeAll(", \"shape\": [");
        for (e.shape, 0..) |d, j| {
            if (j > 0) try w.writeAll(", ");
            try w.print("{d}", .{d});
        }
        try w.print("], \"data_offsets\": [{d}, {d}]}}", .{ e.offset, e.offset + e.len });
    }
    try w.writeByte('}');
    while (buf.written().len % 8 != 0) try w.writeByte(' ');
    return buf.written();
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

const Run = struct {
    io: Io,
    src: *const Source,
    shards: []const Shard,
    jobs: []const Job,
    out: Io.File,
    layers: []const usize,
    total: u64,
    progress: ?*Io.Writer,
    next: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),
    fetched: std.atomic.Value(u64) = .init(0),
    failed: std.atomic.Value(u16) = .init(0),
    print_lock: Io.Mutex = .init,

    fn worker(run: *Run) Io.Cancelable!void {
        while (run.failed.load(.acquire) == 0) {
            const i = run.next.fetchAdd(1, .monotonic);
            if (i >= run.jobs.len) return;
            run.doJob(run.jobs[i]) catch |err| {
                _ = run.failed.cmpxchgStrong(0, @intFromError(err), .acq_rel, .monotonic);
                return;
            };
        }
    }

    fn doJob(run: *Run, job: Job) !void {
        const alloc = run.src.alloc;
        const shard = &run.shards[job.shard];
        const data = try run.src.fetch(shard, job.src, job.len);
        defer alloc.free(data);
        if (job.gather) |g| {
            const seg: usize = @intCast(g.seg);
            const row: usize = @intCast(g.row_bytes);
            const cut = try alloc.alloc(u8, @intCast(g.rows * run.layers.len * g.seg));
            defer alloc.free(cut);
            var o: usize = 0;
            for (0..@intCast(g.rows)) |r| for (run.layers) |l| {
                @memcpy(cut[o..][0..seg], data[r * row + l * seg ..][0..seg]);
                o += seg;
            };
            try run.out.writePositionalAll(run.io, cut, job.dst);
        } else {
            try run.out.writePositionalAll(run.io, data, job.dst);
        }
        const fetched = run.fetched.fetchAdd(job.len, .monotonic) + job.len;
        const done = run.done.fetchAdd(1, .monotonic);
        if (run.progress) |w| if (done % 50 == 0) {
            run.print_lock.lockUncancelable(run.io);
            defer run.print_lock.unlock(run.io);
            w.print("{d:.2} GB of {d:.2}\n", .{ gb(fetched), gb(run.total) }) catch {};
            w.flush() catch {};
        };
    }
};

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

/// Splits `[src, src + len)` → `dst` into requests of at most `chunk`,
/// merging with the previous request when both sides continue it (tensors
/// stored next to each other in a shard are next to each other in the cut).
fn addCopy(a: Allocator, jobs: *std.ArrayList(Job), shard: u32, src: u64, len: u64, dst: u64, chunk: u64) !void {
    var done: u64 = 0;
    while (done < len) {
        const m = @min(chunk, len - done);
        if (jobs.items.len > 0) {
            const j = &jobs.items[jobs.items.len - 1];
            if (j.gather == null and j.shard == shard and j.src + j.len == src + done and j.dst + j.len == dst + done and j.len + m <= chunk) {
                j.len += m;
                done += m;
                continue;
            }
        }
        try jobs.append(a, .{ .shard = shard, .src = src + done, .len = m, .dst = dst + done });
        done += m;
    }
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

/// Writes the cut of `opts.model` into `opts.out_dir`: the small files, the
/// edited config.json and one `model.safetensors`. See the module comment.
pub fn truncate(gpa: Allocator, io: Io, http: ?*hf.Http, opts: Options) !Result {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const chunk = @max(opts.request_size, 1 << 16);

    var src = try Source.open(a, gpa, io, http, opts.model, opts.revision);
    defer src.close();
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, opts.out_dir);
    var out_dir = try cwd.openDir(io, opts.out_dir, .{});
    defer out_dir.close(io);

    // Small files, then config.json edited in place.
    for (try src.smallFiles(a)) |name| _ = try src.copySmall(a, name, out_dir);
    const cfg_text = out_dir.readFileAlloc(io, "config.json", a, .limited(64 << 20)) catch |err| {
        std.log.err("{s} has no config.json ({s})", .{ opts.model, @errorName(err) });
        return error.ModelNotFound;
    };
    var root = try parseConfig(a, cfg_text);
    const tc = try textConfig(&root);
    const n_orig = try numLayers(tc.*);
    const layers: []const usize = switch (opts.layers) {
        .first => |k| blk: {
            const l = try a.alloc(usize, k);
            for (l, 0..) |*x, i| x.* = i;
            break :blk l;
        },
        .list => |l| l,
        .kinds => try kindLayers(a, root),
    };
    if (layers.len == 0) {
        std.log.err("the cut keeps no layers", .{});
        return error.InvalidLayers;
    }
    for (layers, 0..) |l, i| {
        if (l >= n_orig) {
            std.log.err("layer {d} is out of range: {s} has {d} layers", .{ l, opts.model, n_orig });
            return error.InvalidLayers;
        }
        for (layers[0..i]) |m| if (m == l) {
            std.log.err("layer {d} is listed twice", .{l});
            return error.InvalidLayers;
        };
    }
    const ple_width: u64 = if (tc.get("hidden_size_per_layer_input")) |v| (if (asInt(v)) |w| (if (w > 0) @as(u64, @intCast(w)) else 0) else 0) else 0;
    try editConfig(a, &root, layers, opts.drop);
    try out_dir.writeFile(io, .{ .sub_path = "config.json", .data = try renderConfig(a, root) });

    // Shards holding a kept tensor.
    var shard_names = std.ArrayList([]const u8).empty;
    if (try src.readSmall(a, "model.safetensors.index.json")) |text| {
        const index = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch {
            std.log.err("model.safetensors.index.json is not valid JSON", .{});
            return error.InvalidCheckpoint;
        };
        const wm = if (index == .object) index.object.get("weight_map") else null;
        if (wm == null or wm.? != .object) {
            std.log.err("model.safetensors.index.json has no weight_map", .{});
            return error.InvalidCheckpoint;
        }
        var it = wm.?.object.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .string) continue;
            const shard = kv.value_ptr.string;
            if (try cutName(a, kv.key_ptr.*, layers, opts.drop) == null) continue;
            for (shard_names.items) |s| {
                if (std.mem.eql(u8, s, shard)) break;
            } else try shard_names.append(a, shard);
        }
        std.mem.sort([]const u8, shard_names.items, {}, lessThan);
    } else try shard_names.append(a, "model.safetensors");

    // Pass 1: headers.
    const shards = try a.alloc(Shard, shard_names.items.len);
    var opened: usize = 0;
    defer for (shards[0..opened]) |s| if (s.file) |f| f.close(io);
    for (shard_names.items, shards) |name, *s| {
        s.* = try src.openShard(a, name);
        opened += 1;
    }
    var entries = std.ArrayList(Entry).empty;
    for (shards, 0..) |*shard, si| {
        const header = try readHeader(&src, a, shard);
        const Item = struct { key: []const u8, value: Value, begin: u64 };
        var items = std.ArrayList(Item).empty;
        var it = header.value.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const v = kv.value_ptr.*;
            if (v != .object) return error.InvalidCheckpoint;
            const offs = v.object.get("data_offsets") orelse return error.InvalidCheckpoint;
            if (offs != .array or offs.array.items.len != 2) return error.InvalidCheckpoint;
            try items.append(a, .{ .key = kv.key_ptr.*, .value = v, .begin = try jsonInt(offs.array.items[0]) });
        }
        std.mem.sort(Item, items.items, {}, struct {
            fn lt(_: void, x: Item, y: Item) bool {
                return x.begin < y.begin;
            }
        }.lt);
        for (items.items) |item| {
            const name = (try cutName(a, item.key, layers, opts.drop)) orelse continue;
            const offs = item.value.object.get("data_offsets").?.array.items;
            const begin = item.begin;
            const end = try jsonInt(offs[1]);
            if (end < begin) return error.InvalidCheckpoint;
            const dtype = item.value.object.get("dtype") orelse return error.InvalidCheckpoint;
            if (dtype != .string) return error.InvalidCheckpoint;
            const shape_v = item.value.object.get("shape") orelse return error.InvalidCheckpoint;
            if (shape_v != .array) return error.InvalidCheckpoint;
            var shape = try a.alloc(u64, shape_v.array.items.len);
            for (shape_v.array.items, shape) |d, *s| s.* = try jsonInt(d);
            var entry: Entry = .{
                .source_name = item.key,
                .name = name,
                .dtype = dtype.string,
                .shape = shape,
                .shard = @intCast(si),
                .start = header.data_start + begin,
                .len = end - begin,
            };
            // Gemma 3n / Gemma 4's per-layer inputs are one table for every
            // layer (`embed_tokens_per_layer` `[vocab, layers * width]`, and the
            // projection `[layers * width, hidden]`): cut to the kept layers' slices.
            if (ple_width > 0 and shape.len == 2 and shape[0] > 0 and shape[1] > 0) {
                const axis: ?u1 = if (std.mem.endsWith(u8, item.key, "embed_tokens_per_layer.weight") and shape[1] == n_orig * ple_width)
                    1
                else if (std.mem.endsWith(u8, item.key, "per_layer_model_projection.weight") and shape[0] == n_orig * ple_width)
                    0
                else
                    null;
                if (axis) |ax| {
                    const elem = (end - begin) / (shape[0] * shape[1]);
                    entry.slice = .{ .axis = ax, .width = ple_width, .rows = shape[0], .cols = shape[1], .elem = elem };
                    shape = try a.dupe(u64, shape);
                    shape[ax] = layers.len * ple_width;
                    entry.shape = shape;
                    entry.len = (end - begin) / n_orig * layers.len;
                }
            }
            try entries.append(a, entry);
        }
    }

    // --rows selections.
    for (opts.rows) |sel| {
        const e = for (entries.items) |*x| {
            if (std.mem.eql(u8, x.name, sel.name) or std.mem.eql(u8, x.source_name, sel.name)) break x;
        } else {
            std.log.err("--rows: no tensor {s} in the cut", .{sel.name});
            return error.UnknownTensor;
        };
        if (e.slice != null or e.shape.len == 0 or e.shape[0] == 0) {
            std.log.err("--rows: {s} cannot be read by rows", .{sel.name});
            return error.InvalidRows;
        }
        const rows = try a.dupe(u64, sel.rows);
        std.mem.sort(u64, rows, {}, std.sort.asc(u64));
        for (rows) |r| if (r >= e.shape[0]) {
            std.log.err("--rows: row {d} is out of range for {s} ({d} rows)", .{ r, sel.name, e.shape[0] });
            return error.InvalidRows;
        };
        e.rows = rows;
    }

    // Layout.
    var off: u64 = 0;
    for (entries.items) |*e| {
        e.offset = off;
        off += e.len;
    }
    const hb = try renderHeader(a, entries.items);
    const base: u64 = 8 + hb.len;

    // Requests.
    var jobs = std.ArrayList(Job).empty;
    var gathers = std.ArrayList(Job).empty;
    for (entries.items) |e| {
        const dst = base + e.offset;
        if (e.slice) |sl| {
            if (sl.axis == 0) {
                // Whole row blocks: one range per kept layer.
                const blk = sl.width * sl.cols * sl.elem;
                for (layers, 0..) |l, i| {
                    var o: u64 = 0;
                    while (o < blk) : (o += chunk) {
                        try jobs.append(a, .{ .shard = e.shard, .src = e.start + l * blk + o, .len = @min(chunk, blk - o), .dst = dst + i * blk + o });
                    }
                }
            } else {
                const row = sl.cols * sl.elem;
                const seg = sl.width * sl.elem;
                const step = @max(1, chunk / row);
                var r: u64 = 0;
                while (r < sl.rows) : (r += step) {
                    const r1 = @min(r + step, sl.rows);
                    try gathers.append(a, .{
                        .shard = e.shard,
                        .src = e.start + r * row,
                        .len = (r1 - r) * row,
                        .dst = dst + r * layers.len * seg,
                        .gather = .{ .rows = r1 - r, .row_bytes = row, .seg = seg },
                    });
                }
            }
            continue;
        }
        if (e.rows) |rows| {
            const row = e.len / e.shape[0];
            var i: usize = 0;
            while (i < rows.len) {
                var j = i + 1;
                while (j < rows.len and rows[j] <= rows[j - 1] + 1) j += 1;
                const r0 = rows[i];
                const r1 = rows[j - 1] + 1;
                try addCopy(a, &jobs, e.shard, e.start + r0 * row, (r1 - r0) * row, dst + r0 * row, chunk);
                i = j;
            }
            continue;
        }
        try addCopy(a, &jobs, e.shard, e.start, e.len, dst, chunk);
    }
    try jobs.appendSlice(a, gathers.items);
    var total: u64 = 0;
    for (jobs.items) |j| total += j.len;

    // The output: header, then a sparse file of the full size filled in place.
    const path = try std.fs.path.join(a, &.{ opts.out_dir, "model.safetensors" });
    const file = cwd.createFile(io, path, .{ .read = true, .truncate = true }) catch |err| {
        std.log.err("could not create {s}: {s}", .{ path, @errorName(err) });
        return err;
    };
    defer file.close(io);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, hb.len, .little);
    try file.writePositionalAll(io, &len_bytes, 0);
    try file.writePositionalAll(io, hb, 8);
    try file.setLength(io, base + off);

    var run: Run = .{
        .io = io,
        .src = &src,
        .shards = shards,
        .jobs = jobs.items,
        .out = file,
        .layers = layers,
        .total = total,
        .progress = opts.progress,
    };
    const workers = @max(1, @min(opts.connections, jobs.items.len));
    var group: Io.Group = .init;
    for (0..workers) |_| {
        // Without a concurrent task (a single-threaded Io) the work runs here.
        group.concurrent(io, Run.worker, .{&run}) catch {
            Run.worker(&run) catch {};
        };
    }
    group.await(io) catch {};
    const code = run.failed.load(.acquire);
    if (code != 0) {
        const err = @errorFromInt(code);
        std.log.err("truncating {s} failed: {s}", .{ opts.model, @errorName(err) });
        return err;
    }
    if (opts.progress) |w| {
        try w.print("{d:.2} GB of {d:.2}\n", .{ gb(run.fetched.load(.monotonic)), gb(total) });
        try w.flush();
    }
    return .{
        .layers = try gpa.dupe(usize, layers),
        .source_layers = n_orig,
        .tensors = entries.items.len,
        .logical_bytes = off,
        .fetched_bytes = run.fetched.load(.monotonic),
    };
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

/// Parses a comma-separated list of layer indices (`--layers 0,1,20`), in `a`.
pub fn parseLayerList(a: Allocator, text: []const u8) ![]usize {
    var out = std.ArrayList(usize).empty;
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| try out.append(a, std.fmt.parseInt(usize, t, 10) catch return error.InvalidLayers);
    if (out.items.len == 0) return error.InvalidLayers;
    return out.items;
}

/// Reads `--rows NAME=FILE`: the tensor name and the row indices in FILE,
/// one per line (blank lines and `#` comments ignored), in `a`.
pub fn parseRowsArg(a: Allocator, io: Io, spec: []const u8) !RowSelection {
    const eq = std.mem.lastIndexOfScalar(u8, spec, '=') orelse return error.InvalidRows;
    const name = spec[0..eq];
    const path = spec[eq + 1 ..];
    if (name.len == 0 or path.len == 0) return error.InvalidRows;
    const text = Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30)) catch |err| {
        std.log.err("--rows: could not read {s}: {s}", .{ path, @errorName(err) });
        return err;
    };
    var rows = std.ArrayList(u64).empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len == 0 or t[0] == '#') continue;
        try rows.append(a, std.fmt.parseInt(u64, t, 10) catch {
            std.log.err("--rows: {s}: not a row index: {s}", .{ path, t });
            return error.InvalidRows;
        });
    }
    return .{ .name = name, .rows = rows.items };
}

/// The default output directory, as the Python tool names it: `models/<id>-L<K>`.
fn defaultOut(a: Allocator, model: []const u8, k: usize) ![]const u8 {
    const id = try a.dupe(u8, std.mem.trimEnd(u8, remote.hubId(model), "/"));
    var name = std.ArrayList(u8).empty;
    for (id) |c| {
        if (c == '/') try name.appendSlice(a, "__") else if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.') try name.append(a, c) else try name.append(a, '_');
    }
    return std.fmt.allocPrint(a, "models/{s}-L{d}", .{ name.items, k });
}

/// `ditch truncate MODEL K OUT`, `ditch truncate MODEL --layers 0,1,5 OUT`,
/// `ditch truncate MODEL --kinds OUT`: progress on `out`, the summary line
/// on `result`. Usage errors exit with status 2.
pub fn runCli(gpa: Allocator, arena: Allocator, io: Io, http: *hf.Http, settings: *const config.Settings, out: *Io.Writer, result: *Io.Writer) !void {
    if (settings.model.len == 0) usage("ditch truncate needs a model: ditch truncate <MODEL> <K> <OUT>", .{});
    var args = settings.positionals;
    var layers: Layers = undefined;
    var modes: usize = 0;
    if (settings.truncate_kinds) {
        layers = .kinds;
        modes += 1;
    }
    if (settings.truncate_layers) |text| {
        layers = .{ .list = parseLayerList(arena, text) catch usage("--layers takes comma-separated layer indices, e.g. 0,1,20 (got {s})", .{text}) };
        modes += 1;
    }
    if (modes == 0) {
        if (args.len == 0) usage("ditch truncate needs the number of layers to keep: ditch truncate <MODEL> <K> <OUT> (or --layers, --kinds)", .{});
        const k = std.fmt.parseInt(usize, args[0], 10) catch usage("expected the number of layers to keep, got {s}", .{args[0]});
        if (k == 0) usage("the cut must keep at least one layer", .{});
        layers = .{ .first = k };
        args = args[1..];
        modes = 1;
    }
    if (modes > 1) usage("give one of <K>, --layers and --kinds", .{});
    if (args.len > 1) usage("unexpected argument {s}", .{args[1]});
    const out_dir: []const u8 = if (args.len == 1) args[0] else if (settings.save_directory) |d| d else switch (layers) {
        .first => |k| try defaultOut(arena, settings.model, k),
        .list => |l| try defaultOut(arena, settings.model, l.len),
        .kinds => usage("--kinds needs an output directory: ditch truncate <MODEL> --kinds <OUT>", .{}),
    };
    var rows = try arena.alloc(RowSelection, settings.truncate_rows.len);
    for (settings.truncate_rows, 0..) |spec, i| {
        rows[i] = parseRowsArg(arena, io, spec) catch |err| switch (err) {
            error.InvalidRows => usage("--rows takes NAME=FILE (got {s})", .{spec}),
            else => return err,
        };
    }
    try out.print("Truncating {s} into {s}\n", .{ settings.model, out_dir });
    try out.flush();
    var res = try truncate(gpa, io, http, .{
        .model = settings.model,
        .out_dir = out_dir,
        .layers = layers,
        .drop = settings.truncate_drop,
        .rows = rows,
        .revision = settings.model_commit,
        .connections = if (settings.remote_connections == 0) default_connections else settings.remote_connections,
        .progress = out,
    });
    defer res.deinit(gpa);
    try out.print("Kept layers ", .{});
    for (res.layers, 0..) |l, i| try out.print("{s}{d}", .{ if (i > 0) "," else "", l });
    try out.print(" of {d}\n", .{res.source_layers});
    try out.flush();
    try result.print("{s} {d} tensors {d:.3} GB logical, {d:.3} GB fetched\n", .{ out_dir, res.tensors, gb(res.logical_bytes), gb(res.fetched_bytes) });
    try result.flush();
}

fn usage(comptime fmt: []const u8, args: anytype) noreturn {
    std.log.err(fmt, args);
    std.process.exit(2);
}
