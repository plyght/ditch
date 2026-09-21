//! GGUF as output: writes the abliterated model as one llama.cpp-compatible
//! file (`model.gguf`) following the conventions of llama.cpp's
//! `convert_hf_to_gguf.py`: its tensor names, the q/k row permutation of the
//! llama family, gemma norms stored as `1 + w`, stacked 3-D expert tensors,
//! the architecture and tokenizer metadata llama.cpp reads, and 1-D tensors in
//! f32. Abliteration deltas are merged exactly like export.zig does. Tensors
//! are streamed row chunk by row chunk through the model's weight store, so
//! peak memory is one chunk regardless of model size.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const quant = @import("quant.zig");
const gguf = @import("gguf.zig");
const gguf_model = @import("gguf_model.zig");
const model_mod = @import("model.zig");
const moe = @import("moe.zig");
const stream = @import("stream.zig");
const safetensors = @import("safetensors.zig");
const export_mod = @import("export.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;
const DType = tensor.DType;

pub const file_name = "model.gguf";

pub const Options = struct {
    /// Storage type of the 2-D matrices: f16, bf16, f32, q8_0 (or q4_0, q4_1,
    /// q5_0, q5_1). null keeps every tensor's source type (a GGUF input keeps
    /// its quantisation; edited tensors of a type ditch cannot produce, such as
    /// Q4_K, are re-quantised to Q8_0).
    dtype: ?DType = null,
    /// `general.name`.
    name: []const u8 = "",
    /// Markdown written next to the file as README.md.
    readme_body: ?[]const u8 = null,
};

const Which = enum { gate, up, down };

const Src = union(enum) {
    /// A `[rows][cols]` matrix, optionally with a delta (rows indexed in Hugging
    /// Face order) and llama.cpp's head permutation (`n_head`).
    matrix: struct { ref: moe.MatrixRef, delta: ?tensor.Delta = null, n_head: ?usize = null },
    /// A 1-D tensor written as f32, optionally `+ 1` (gemma norms) or permuted (llama q/k biases).
    vector: struct { ref: stream.WeightRef, plus_one: bool = false, n_head: ?usize = null },
    /// The routed experts of a MoE layer, stacked `[E][rows][cols]`.
    experts: struct { layer: usize, which: Which },
    /// A GGUF tensor of the source file copied verbatim.
    raw: gguf.TensorInfo,
    /// Computed f32 values (`rope_freqs.weight`).
    values: []const f32,
};

const Entry = struct {
    name: []const u8,
    /// Row-major (Hugging Face) shape.
    shape: []const usize,
    dtype: DType,
    src: Src,
};

const Planner = struct {
    a: Allocator,
    model: *const Model,
    opts: Options,
    entries: std.ArrayList(Entry) = .empty,
    /// Count of 2-D matrices per dtype, for `general.file_type`.
    counts: std.EnumArray(DType, usize) = .initFill(0),

    fn matrixDtype(self: *Planner, name: []const u8, src: DType, cols: usize, edited: bool, embed_or_output: bool) DType {
        var d: DType = undefined;
        if (self.opts.dtype) |want| {
            // Like llama.cpp's converter, keep the embeddings and the output
            // projection in 16 bits when the rest is quantised.
            d = if (embed_or_output and want.isQuantized()) .f16 else want;
        } else if (!src.isQuantized()) {
            d = src;
        } else if (edited and !quant.canQuantize(src)) {
            std.log.warn("{s}: an edited {s} tensor is re-quantised to Q8_0 (ditch cannot produce {s})", .{ name, src.safetensorsName(), src.safetensorsName() });
            d = .q8_0;
        } else {
            d = src;
        }
        if (cols % d.blockSize() != 0) {
            // Rows must hold whole blocks; llama.cpp's converter falls back the same way.
            std.log.warn("{s}: {d} columns are not a multiple of the {s} block size; storing it as F16", .{ name, cols, d.safetensorsName() });
            d = .f16;
        }
        return d;
    }

    fn hfName(self: *Planner, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(self.a, fmt, args);
    }

    fn addMatrix(self: *Planner, gguf_name: []const u8, hf_name: []const u8, delta: ?tensor.Delta, n_head: ?usize, embed_or_output: bool) !void {
        const ref = try self.model.ref(hf_name);
        const shape = try self.a.alloc(usize, 2);
        shape[0] = ref.rows;
        shape[1] = ref.cols;
        const dt = self.matrixDtype(gguf_name, ref.dtype, ref.cols, delta != null, embed_or_output);
        self.counts.getPtr(dt).* += 1;
        try self.entries.append(self.a, .{ .name = gguf_name, .shape = shape, .dtype = dt, .src = .{ .matrix = .{ .ref = .{ .ref = ref }, .delta = delta, .n_head = n_head } } });
    }

    /// Adds a 1-D tensor if it exists.
    fn addVectorOpt(self: *Planner, gguf_name: []const u8, hf_name: []const u8, plus_one: bool, n_head: ?usize) !bool {
        const ref = self.model.store.lookup(hf_name) orelse return false;
        const shape = try self.a.alloc(usize, 1);
        shape[0] = ref.rows * ref.cols;
        try self.entries.append(self.a, .{ .name = gguf_name, .shape = shape, .dtype = .f32, .src = .{ .vector = .{ .ref = ref, .plus_one = plus_one, .n_head = n_head } } });
        return true;
    }

    fn addVector(self: *Planner, gguf_name: []const u8, hf_name: []const u8, plus_one: bool) !void {
        if (!try self.addVectorOpt(gguf_name, hf_name, plus_one, null)) {
            std.log.err("missing tensor: {s}", .{hf_name});
            return error.MissingWeights;
        }
    }

    fn addExperts(self: *Planner, gguf_name: []const u8, layer: usize, which: Which) !void {
        const m = &self.model.layers[layer].moe.?;
        const ex0 = m.experts[0];
        const mref = switch (which) {
            .gate => ex0.gate_ref,
            .up => ex0.up_ref,
            .down => ex0.down_ref,
        };
        const shape = try self.a.alloc(usize, 3);
        shape[0] = m.experts.len;
        shape[1] = mref.rows();
        shape[2] = mref.cols();
        const edited = which == .down and m.anyExpertDelta();
        const dt = self.matrixDtype(gguf_name, mref.ref.dtype, mref.cols(), edited, false);
        self.counts.getPtr(dt).* += 1;
        try self.entries.append(self.a, .{ .name = gguf_name, .shape = shape, .dtype = dt, .src = .{ .experts = .{ .layer = layer, .which = which } } });
    }

    fn plan(self: *Planner) !void {
        const model = self.model;
        const c = &model.config;
        const gemma = gguf_model.isGemma(c.arch);
        const permute = gguf_model.permutesQk(c.arch);
        const p = model.prefix;

        try self.addMatrix("token_embd.weight", model.embed_ref.name, null, null, true);
        try self.addVector("output_norm.weight", try self.hfName("{s}norm.weight", .{p}), gemma);
        if (!std.mem.eql(u8, model.lm_head_ref.name, model.embed_ref.name)) {
            try self.addMatrix("output.weight", model.lm_head_ref.name, null, null, true);
        }
        if (ropeFactors(self.a, c)) |factors| {
            const shape = try self.a.alloc(usize, 1);
            shape[0] = factors.len;
            try self.entries.append(self.a, .{ .name = "rope_freqs.weight", .shape = shape, .dtype = .f32, .src = .{ .values = factors } });
        }

        for (model.layers, 0..) |*layer, i| {
            const lp = try self.hfName("{s}layers.{d}.", .{ p, i });
            const B = struct {
                fn n(a: Allocator, li: usize, suffix: []const u8) ![]const u8 {
                    return std.fmt.allocPrint(a, "blk.{d}.{s}", .{ li, suffix });
                }
            };
            try self.addVector(try B.n(self.a, i, "attn_norm.weight"), try self.hfName("{s}input_layernorm.weight", .{lp}), gemma);
            try self.addMatrix(try B.n(self.a, i, "attn_q.weight"), try self.hfName("{s}self_attn.q_proj.weight", .{lp}), null, if (permute) c.num_heads else null, false);
            try self.addMatrix(try B.n(self.a, i, "attn_k.weight"), try self.hfName("{s}self_attn.k_proj.weight", .{lp}), null, if (permute) c.num_kv_heads else null, false);
            try self.addMatrix(try B.n(self.a, i, "attn_v.weight"), try self.hfName("{s}self_attn.v_proj.weight", .{lp}), null, null, false);
            try self.addMatrix(try B.n(self.a, i, "attn_output.weight"), try self.hfName("{s}self_attn.o_proj.weight", .{lp}), model.getDelta(i, .attn_o_proj), null, false);
            _ = try self.addVectorOpt(try B.n(self.a, i, "attn_q.bias"), try self.hfName("{s}self_attn.q_proj.bias", .{lp}), false, if (permute) c.num_heads else null);
            _ = try self.addVectorOpt(try B.n(self.a, i, "attn_k.bias"), try self.hfName("{s}self_attn.k_proj.bias", .{lp}), false, if (permute) c.num_kv_heads else null);
            _ = try self.addVectorOpt(try B.n(self.a, i, "attn_v.bias"), try self.hfName("{s}self_attn.v_proj.bias", .{lp}), false, null);
            _ = try self.addVectorOpt(try B.n(self.a, i, "attn_output.bias"), try self.hfName("{s}self_attn.o_proj.bias", .{lp}), false, null);
            _ = try self.addVectorOpt(try B.n(self.a, i, "attn_q_norm.weight"), try self.hfName("{s}self_attn.q_norm.weight", .{lp}), gemma, null);
            _ = try self.addVectorOpt(try B.n(self.a, i, "attn_k_norm.weight"), try self.hfName("{s}self_attn.k_norm.weight", .{lp}), gemma, null);
            if (gemma) {
                try self.addVector(try B.n(self.a, i, "post_attention_norm.weight"), try self.hfName("{s}post_attention_layernorm.weight", .{lp}), true);
                try self.addVector(try B.n(self.a, i, "ffn_norm.weight"), try self.hfName("{s}pre_feedforward_layernorm.weight", .{lp}), true);
                try self.addVector(try B.n(self.a, i, "post_ffw_norm.weight"), try self.hfName("{s}post_feedforward_layernorm.weight", .{lp}), true);
            } else {
                try self.addVector(try B.n(self.a, i, "ffn_norm.weight"), try self.hfName("{s}post_attention_layernorm.weight", .{lp}), false);
            }
            if (layer.moe) |*m| {
                try self.addMatrix(try B.n(self.a, i, "ffn_gate_inp.weight"), m.router_ref.name, null, null, false);
                try self.addExperts(try B.n(self.a, i, "ffn_gate_exps.weight"), i, .gate);
                try self.addExperts(try B.n(self.a, i, "ffn_up_exps.weight"), i, .up);
                try self.addExperts(try B.n(self.a, i, "ffn_down_exps.weight"), i, .down);
                if (m.shared) |*sh| {
                    try self.addSharedMatrix(try B.n(self.a, i, "ffn_gate_shexp.weight"), sh.gate_ref, null);
                    try self.addSharedMatrix(try B.n(self.a, i, "ffn_up_shexp.weight"), sh.up_ref, null);
                    try self.addSharedMatrix(try B.n(self.a, i, "ffn_down_shexp.weight"), sh.down_ref, sh.down_delta);
                    _ = try self.addVectorOpt(try B.n(self.a, i, "ffn_gate_inp_shexp.weight"), try self.hfName("{s}mlp.shared_expert_gate.weight", .{lp}), false, null);
                }
            } else {
                try self.addMatrix(try B.n(self.a, i, "ffn_gate.weight"), try self.hfName("{s}mlp.gate_proj.weight", .{lp}), null, null, false);
                try self.addMatrix(try B.n(self.a, i, "ffn_up.weight"), try self.hfName("{s}mlp.up_proj.weight", .{lp}), null, null, false);
                try self.addMatrix(try B.n(self.a, i, "ffn_down.weight"), try self.hfName("{s}mlp.down_proj.weight", .{lp}), model.getDelta(i, .mlp_down_proj), null, false);
            }
        }

        // Tensors of a GGUF source ditch does not interpret are passed through.
        if (model.gguf) |g| {
            for (g.extra) |t| {
                var dup = false;
                for (self.entries.items) |e| dup = dup or std.mem.eql(u8, e.name, t.name);
                if (dup) continue;
                const dt = t.dtype() orelse continue;
                try self.entries.append(self.a, .{ .name = t.name, .shape = try t.shape(self.a), .dtype = dt, .src = .{ .raw = t } });
            }
        }
    }

    fn addSharedMatrix(self: *Planner, gguf_name: []const u8, mref: moe.MatrixRef, delta: ?tensor.Delta) !void {
        const shape = try self.a.alloc(usize, 2);
        shape[0] = mref.rows();
        shape[1] = mref.cols();
        const dt = self.matrixDtype(gguf_name, mref.ref.dtype, mref.cols(), delta != null, false);
        self.counts.getPtr(dt).* += 1;
        try self.entries.append(self.a, .{ .name = gguf_name, .shape = shape, .dtype = dt, .src = .{ .matrix = .{ .ref = mref, .delta = delta } } });
    }

    /// `general.file_type`: the most used quantised matrix type ("mostly
    /// Q8_0"), or the most used floating-point type when nothing is quantised.
    fn fileType(self: *const Planner) u32 {
        var best: DType = .f16;
        var best_n: usize = 0;
        inline for (std.meta.fields(DType)) |f| {
            const d: DType = @enumFromInt(f.value);
            const n = self.counts.get(d);
            const better = if (d.isQuantized() == best.isQuantized()) n > best_n else d.isQuantized() and n > 0;
            if (better) {
                best_n = n;
                best = d;
            }
        }
        return gguf.fileType(best);
    }
};

/// llama.cpp stores the llama3 rope scaling as per-frequency factors (`rope_freqs.weight`).
fn ropeFactors(a: Allocator, c: *const model_mod.Config) ?[]const f32 {
    switch (c.rope_scaling) {
        .factors => |f| return f,
        .llama3 => |s| return gguf_model.llama3Factors(a, c.rope_theta, c.head_dim, s) catch null,
        else => return null,
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

fn addArchKeys(w: *gguf.Writer, a: Allocator, model: *const Model, file_type: u32) !void {
    const c = &model.config;
    const arch = gguf_model.archName(c.arch);
    const K = struct {
        a: Allocator,
        arch: []const u8,
        fn key(self: @This(), suffix: []const u8) ![]const u8 {
            return std.fmt.allocPrint(self.a, "{s}.{s}", .{ self.arch, suffix });
        }
    };
    const k = K{ .a = a, .arch = arch };
    try w.addString("general.architecture", arch);
    try w.addString("general.type", "model");
    try w.addU32("general.quantization_version", 2);
    try w.addU32("general.file_type", file_type);
    try w.addU32(try k.key("context_length"), @intCast(c.max_position_embeddings));
    try w.addU32(try k.key("embedding_length"), @intCast(c.hidden_size));
    try w.addU32(try k.key("block_count"), @intCast(c.num_layers));
    try w.addU32(try k.key("feed_forward_length"), @intCast(c.intermediate_size));
    try w.addU32(try k.key("attention.head_count"), @intCast(c.num_heads));
    try w.addU32(try k.key("attention.head_count_kv"), @intCast(c.num_kv_heads));
    try w.addF32(try k.key("attention.layer_norm_rms_epsilon"), c.rms_norm_eps);
    try w.addU32(try k.key("attention.key_length"), @intCast(c.head_dim));
    try w.addU32(try k.key("attention.value_length"), @intCast(c.head_dim));
    try w.addU32(try k.key("rope.dimension_count"), @intCast(c.head_dim));
    try w.addF32(try k.key("rope.freq_base"), c.rope_theta);
    try w.addU32(try k.key("vocab_size"), @intCast(c.vocab_size));
    switch (c.rope_scaling) {
        .linear => |f| {
            try w.addString(try k.key("rope.scaling.type"), "linear");
            try w.addF32(try k.key("rope.scaling.factor"), f);
        },
        else => {},
    }
    if (c.attn_logit_softcapping) |v| try w.addF32(try k.key("attn_logit_softcapping"), v);
    if (c.final_logit_softcapping) |v| try w.addF32(try k.key("final_logit_softcapping"), v);
    if (gguf_model.isGemma(c.arch)) {
        if (c.sliding_window) |sw| try w.addU32(try k.key("attention.sliding_window"), @intCast(sw));
        if (std.mem.eql(u8, c.arch.model_type, "gemma3")) {
            // Pattern of sliding layers (every n-th layer is global).
            var pattern: usize = 0;
            for (c.sliding_layers, 0..) |s, i| if (!s) {
                pattern = i + 1;
                break;
            };
            if (pattern > 0) try w.addU32(try k.key("attention.sliding_window_pattern"), @intCast(pattern));
            try w.addF32(try k.key("rope.freq_base_swa"), c.rope_local_theta);
        }
    }
    if (c.num_experts > 0) {
        try w.addU32(try k.key("expert_count"), @intCast(c.num_experts));
        try w.addU32(try k.key("expert_used_count"), @intCast(c.num_experts_per_tok));
        try w.addU32(try k.key("expert_feed_forward_length"), @intCast(c.moe_intermediate_size));
        for (model.layers) |*l| {
            if (l.moe) |*m| if (m.shared) |*sh| {
                try w.addU32(try k.key("expert_shared_feed_forward_length"), @intCast(sh.gate_ref.rows()));
                break;
            };
        }
    }
}

fn isByteToken(s: []const u8) bool {
    return s.len == 6 and std.mem.startsWith(u8, s, "<0x") and s[5] == '>';
}

fn addVocab(w: *gguf.Writer, a: Allocator, model: *const Model) !void {
    const tok = model.tokenizer;
    const byte_level = switch (tok.pre) {
        .byte_level_regex, .byte_level_plain => true,
        else => false,
    };
    const pre: []const u8 = switch (tok.pre) {
        .byte_level_regex => |kind| switch (kind) {
            .qwen2 => "qwen2",
            .llama3 => "llama-bpe",
            .gpt2 => "gpt-2",
        },
        else => "default",
    };
    try w.addString("tokenizer.ggml.model", if (byte_level) "gpt2" else "llama");
    try w.addString("tokenizer.ggml.pre", pre);

    const n = tok.id_to_token.len;
    const tokens = try a.alloc([]const u8, n);
    const types = try a.alloc(i32, n);
    for (0..n) |id| {
        var s = tok.id_to_token[id];
        var t: gguf_model.TokenType = .normal;
        if (tok.added_by_id.get(@intCast(id))) |ai| {
            t = if (tok.added[ai].special) .control else .user_defined;
        } else if (s.len == 0) {
            s = try std.fmt.allocPrint(a, "[PAD{d}]", .{id});
            t = .unused;
        } else if (!byte_level) {
            if (isByteToken(s)) t = .byte else if (tok.unk_id != null and tok.unk_id.? == id) t = .unknown;
        }
        tokens[id] = s;
        types[id] = @intFromEnum(t);
    }
    try w.addStringArray("tokenizer.ggml.tokens", tokens);
    try w.addI32Array("tokenizer.ggml.token_type", types);

    // Merges in rank order.
    const merges = try a.alloc([]const u8, tok.merges.count());
    @memset(merges, "");
    var it = tok.merges.iterator();
    while (it.next()) |e| {
        const rank = e.value_ptr.*;
        if (rank < merges.len) merges[rank] = e.key_ptr.*;
    }
    if (merges.len > 0) try w.addStringArray("tokenizer.ggml.merges", merges);
    if (!byte_level) {
        // SentencePiece scores: llama.cpp merges the pair whose result scores
        // highest, so a token produced by merge rank r gets score -(r + 1).
        const scores = try a.alloc(f32, n);
        @memset(scores, 0);
        for (merges, 0..) |m, rank| {
            const sp = std.mem.indexOfScalar(u8, m, ' ') orelse continue;
            const joined = try std.mem.concat(a, u8, &.{ m[0..sp], m[sp + 1 ..] });
            const id = tok.vocab.get(joined) orelse continue;
            const score = -@as(f32, @floatFromInt(rank + 1));
            if (scores[id] == 0 or score > scores[id]) scores[id] = score;
        }
        try w.addF32Array("tokenizer.ggml.scores", scores);
        try w.addBool("tokenizer.ggml.add_space_prefix", tok.prepend != null);
    }
    if (tok.bos_id) |b| try w.addU32("tokenizer.ggml.bos_token_id", b);
    const eos: ?u32 = if (model.eos_ids.len > 0) model.eos_ids[0] else tok.eos_id;
    if (eos) |e| try w.addU32("tokenizer.ggml.eos_token_id", e);
    if (model.eos_ids.len > 1) try w.addU32("tokenizer.ggml.eot_token_id", model.eos_ids[1]);
    if (tok.unk_id) |u| try w.addU32("tokenizer.ggml.unknown_token_id", u);
    try w.addU32("tokenizer.ggml.padding_token_id", model.pad_id);
    try w.addBool("tokenizer.ggml.add_bos_token", tok.add_bos);
    try w.addBool("tokenizer.ggml.add_eos_token", false);
    if (model.chat_template) |t| try w.addString("tokenizer.chat_template", t);
}

// ---------------------------------------------------------------------------
// Tensor data
// ---------------------------------------------------------------------------

const Ctx = struct {
    gpa: Allocator,
    io: Io,
    model: *const Model,
    out: *Io.Writer,
    w: *const gguf.Writer,
};

/// Inverse of `safetensors.llamaPermutedRow`: the Hugging Face row stored at GGUF row `g`.
fn hfRowOf(g: usize, head_dim: usize) usize {
    const half = head_dim / 2;
    const h = g / head_dim;
    const r = g % head_dim;
    return h * head_dim + (r % 2) * half + r / 2;
}

fn rowsPerChunk(model: *const Model, src_dtype: DType, out_dtype: DType, rows: usize, cols: usize, align_rows: usize) usize {
    var chunk_bytes: u64 = export_mod.convert_chunk_bytes;
    if (model.budget) |b| {
        if (b.limited()) chunk_bytes = @min(chunk_bytes, b.limitBytes() / 16);
    }
    const row_bytes = @max(1, @max(src_dtype.rowBytes(cols), out_dtype.rowBytes(cols)) + 4 * cols);
    var n: usize = @intCast(@max(1, @min(@as(u64, rows), chunk_bytes / row_bytes)));
    if (align_rows > 1) n = @max(align_rows, n / align_rows * align_rows);
    return @min(rows, n);
}

/// Streams one `[rows][cols]` matrix: rows of `mref` (Hugging Face order) are
/// converted, the delta merged, the head permutation applied and the result
/// quantised chunk by chunk.
fn writeMatrix(ctx: Ctx, mref: moe.MatrixRef, delta: ?tensor.Delta, n_head: ?usize, out_dtype: DType) !void {
    const store: *stream.WeightStore = @constCast(&ctx.model.store);
    const gpa = ctx.gpa;
    const rows = mref.rows();
    const cols = mref.cols();
    const src_dtype = mref.ref.dtype;
    const hd: usize = if (n_head) |h| rows / h else 1;
    if (n_head != null and (rows % (2 * n_head.?) != 0)) return error.InvalidConfig;
    const per = rowsPerChunk(ctx.model, src_dtype, out_dtype, rows, cols, if (n_head != null) hd else 1);
    const plain = delta == null and out_dtype == src_dtype;
    const out_rb = out_dtype.rowBytes(cols);
    const chunk: []u8 = if (plain) &.{} else try gpa.alloc(u8, per * out_rb);
    defer if (chunk.len > 0) gpa.free(chunk);
    const f: []f32 = if (plain) &.{} else try gpa.alloc(f32, per * cols);
    defer if (f.len > 0) gpa.free(f);

    // Transposed (fused_transposed expert) blocks cannot be sliced by rows: make them resident once.
    var whole: ?stream.Lease = null;
    defer if (whole) |l| store.release(l);
    if (mref.transposed != null) whole = try mref.acquire(store);

    var r0: usize = 0;
    while (r0 < rows) : (r0 += per) {
        if (ctx.model.budget) |b| try b.checkTime();
        const n = @min(per, rows - r0);
        var lease: ?stream.Lease = null;
        defer if (lease) |l| store.release(l);
        var w: tensor.Weight = undefined;
        if (whole) |wl| {
            const rb = src_dtype.rowBytes(cols);
            w = .{ .data = wl.weight.data[r0 * rb ..][0 .. n * rb], .dtype = src_dtype, .rows = n, .cols = cols };
        } else {
            lease = try store.acquire(mref.ref.rowSlice(r0, n));
            w = lease.?.weight;
        }
        if (plain) {
            if (n_head == null) {
                try ctx.out.writeAll(w.data);
            } else {
                for (0..n) |j| try ctx.out.writeAll(w.rowBytes(hfRowOf(r0 + j, hd) - r0));
            }
            continue;
        }
        for (0..n) |j| {
            const hf_row = if (n_head != null) hfRowOf(r0 + j, hd) else r0 + j;
            const row = f[j * cols ..][0..cols];
            w.row(hf_row - r0, row);
            if (delta) |d| for (0..d.rank) |k| tensor.axpy(row, d.b[hf_row * d.rank + k], d.a[k * cols ..][0..cols]);
        }
        tensor.convertFromF32(out_dtype, f[0 .. n * cols], chunk);
        try ctx.out.writeAll(chunk[0 .. n * out_rb]);
    }
}

fn writeVector(ctx: Ctx, ref: stream.WeightRef, plus_one: bool, n_head: ?usize) !void {
    const store: *stream.WeightStore = @constCast(&ctx.model.store);
    const vals = try store.readVecF32(ctx.gpa, ref);
    defer ctx.gpa.free(vals);
    if (plus_one) for (vals) |*v| {
        v.* += 1.0;
    };
    if (n_head) |h| {
        const hd = vals.len / h;
        const perm = try ctx.gpa.alloc(f32, vals.len);
        defer ctx.gpa.free(perm);
        for (perm, 0..) |*p, g| p.* = vals[hfRowOf(g, hd)];
        try ctx.out.writeAll(std.mem.sliceAsBytes(perm));
    } else {
        try ctx.out.writeAll(std.mem.sliceAsBytes(vals));
    }
}

fn writeRaw(ctx: Ctx, t: gguf.TensorInfo) !void {
    const src = ctx.model.gguf.?;
    const len = t.byteLen().?;
    const buf = try ctx.gpa.alloc(u8, @min(len, 1 << 20));
    defer ctx.gpa.free(buf);
    var done: usize = 0;
    while (done < len) {
        const n = @min(buf.len, len - done);
        try src.file.readRange(ctx.io, src.file.tensorOffset(t) + done, buf[0..n]);
        try ctx.out.writeAll(buf[0..n]);
        done += n;
    }
}

fn writeEntry(ctx: Ctx, e: Entry) !void {
    switch (e.src) {
        .matrix => |m| try writeMatrix(ctx, m.ref, m.delta, m.n_head, e.dtype),
        .vector => |v| try writeVector(ctx, v.ref, v.plus_one, v.n_head),
        .experts => |x| {
            const m = &ctx.model.layers[x.layer].moe.?;
            for (m.experts) |*ex| {
                const mref = switch (x.which) {
                    .gate => ex.gate_ref,
                    .up => ex.up_ref,
                    .down => ex.down_ref,
                };
                try writeMatrix(ctx, mref, if (x.which == .down) ex.down_delta else null, null, e.dtype);
            }
        },
        .raw => |t| try writeRaw(ctx, t),
        .values => |v| try ctx.out.writeAll(std.mem.sliceAsBytes(v)),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Writes `out_dir/model.gguf` (plus README.md) with the deltas merged. An
/// `.incomplete` marker guards the directory while writing.
pub fn saveGguf(gpa: Allocator, io: Io, model: *const Model, out_dir: []const u8, opts: Options, out: *Io.Writer) !void {
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, out_dir);
    var dir = try cwd.openDir(io, out_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = model_mod.export_incomplete_marker, .data = "export in progress\n" });
    try saveInner(gpa, io, model, dir, opts, out);
    try dir.deleteFile(io, model_mod.export_incomplete_marker);
}

fn saveInner(gpa: Allocator, io: Io, model: *const Model, dir: Io.Dir, opts: Options, out: *Io.Writer) !void {
    // Metadata (the plan, vocabulary arrays, embedded JSON copies) is not
    // budgeted, like the model's own metadata; `gpa` pays for tensor chunks only.
    var arena = std.heap.ArenaAllocator.init(model.meta_gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (opts.dtype) |d| {
        if (d.isQuantized() and !quant.canQuantize(d)) {
            std.log.err("ditch cannot quantise to {s}; use f16, bf16, f32, q8_0, q4_0, q4_1, q5_0 or q5_1", .{d.safetensorsName()});
            return error.UnsupportedDType;
        }
    }
    if (!gguf_model.ggufSupported(model.config.arch)) {
        std.log.err("GGUF export is not implemented for the {s} family (export in Hugging Face format instead)", .{model.config.arch.model_type});
        return error.UnsupportedArchitecture;
    }
    var planner = Planner{ .a = a, .model = model, .opts = opts };
    try planner.plan();

    var w = gguf.Writer.init(model.meta_gpa);
    defer w.deinit();
    try addArchKeys(&w, a, model, planner.fileType());
    if (opts.name.len > 0) try w.addString("general.name", opts.name);
    try addVocab(&w, a, model);
    // Copies of the Hugging Face configuration files, for an exact HF re-export.
    try w.addString(gguf_model.key_hf_config, model.config_json);
    try w.addString(gguf_model.key_hf_tokenizer, model.tokenizer_json);
    if (model.tokenizer_config_json) |t| try w.addString(gguf_model.key_hf_tokenizer_config, t);
    if (model.generation_config_json) |g| try w.addString(gguf_model.key_hf_generation_config, g);
    for (planner.entries.items) |e| try w.addTensor(e.name, e.shape, e.dtype);

    try out.print("* Writing {s} ({d} tensors)...\n", .{ file_name, planner.entries.items.len });
    try out.flush();
    const file = try dir.createFile(io, file_name, .{});
    defer file.close(io);
    var buf: [1 << 18]u8 = undefined;
    var fw = file.writer(io, &buf);
    const fo = &fw.interface;
    try w.writeHeader(fo);
    const ctx = Ctx{ .gpa = gpa, .io = io, .model = model, .out = fo, .w = &w };
    for (planner.entries.items, 0..) |e, i| {
        try writeEntry(ctx, e);
        try fo.splatByteAll(0, w.padding(w.tensors.items[i].byte_len));
    }
    try fo.flush();
    if (opts.readme_body) |body| try dir.writeFile(io, .{ .sub_path = "README.md", .data = body });
}

test "hfRowOf inverts the llama permutation" {
    for ([_]usize{ 4, 8, 128 }) |hd| {
        for (0..2 * hd) |hf| {
            try std.testing.expectEqual(hf, hfRowOf(safetensors.llamaPermutedRow(hf, hd), hd));
        }
    }
}
