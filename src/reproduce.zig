//! Reproducibility manifests. When a model is exported, `ditch-reproduce.lua`
//! records everything that determined the result: the source model files and
//! their SHA-256 hashes, the settings that affect abliteration, the prompt
//! datasets with a hash of the exact prompt lists, the selected trial's
//! parameter vector and the scores. `ditch --reproduce <manifest>` reads it
//! back, verifies the hashes, re-derives the directions and applies the
//! recorded trial without running a search.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const compute = @import("compute.zig");
const lua = @import("lua.zig");
const tree = @import("tree.zig");
const hf = @import("hf.zig");
const abliterate = @import("abliterate.zig");
const directions = @import("directions.zig");
const model_mod = @import("model.zig");
const search = @import("search.zig");
const study_mod = @import("study.zig");
const scorers = @import("scorers.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const Component = model_mod.Component;
const Prompt = hf.Prompt;

pub const file_name = "ditch-reproduce.lua";

pub const FileHash = struct {
    name: []const u8,
    sha256: []const u8,
};

pub const Dataset = struct {
    spec: config.DatasetSpec,
    count: usize,
    prompts_sha256: []const u8,
};

pub const Param = struct {
    name: []const u8,
    value: f64,
};

pub const Manifest = struct {
    ditch_version: []const u8,
    model: []const u8,
    model_commit: ?[]const u8,
    config_sha256: []const u8,
    safetensors: []const FileHash,

    seed: u64,
    chat_template: []const u8,
    response_prefix: []const u8,
    system_prompt: []const u8,
    max_response_length: usize,
    orthogonalize_direction: bool,
    row_normalization: abliterate.RowNormalization,
    full_normalization_lora_rank: usize,
    winsorization_quantile: f32,
    expert_selection: abliterate.ExpertSelection,
    n_directions: usize,
    direction_method: directions.Method,
    direction_token_window: usize,
    direction_shrinkage: f32,
    /// As configured: "auto" or "<low>:<high>".
    direction_range: []const u8,
    ablate_inputs: bool,
    kl_tokens: usize,
    fast_search: bool,
    /// The compute backend the run used ("cpu", or a GPU device name). The CPU
    /// path is the reference; a GPU run reproduces it within the tolerance
    /// documented in README's "GPU acceleration", so the device is recorded
    /// but never re-applied on reproduction.
    device: []const u8,
    n_trials: usize,
    n_startup_trials: usize,
    scorers: []const config.ScorerConfig,
    keyword_markers: []const []const u8,
    keyword_rate_score_name: []const u8,

    good_prompts: Dataset,
    bad_prompts: Dataset,
    keyword_rate_prompts: ?Dataset,
    kl_divergence_prompts: ?Dataset,

    trial_index: usize,
    /// Search space names and the sampled values, in search space order.
    params: []const Param,
    direction_index: ?f32,
    parameters: std.EnumMap(Component, abliterate.Params),
    experts: ?search.ExpertSelection,
    scores: []const study_mod.ScoreRecord,
    baseline: []const study_mod.ScoreRecord,

    pub fn param(self: *const Manifest, name: []const u8) ?f64 {
        for (self.params) |p| if (std.mem.eql(u8, p.name, name)) return p.value;
        return null;
    }
};

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

fn hexDigest(a: Allocator, h: *Sha256) ![]const u8 {
    var digest: [Sha256.digest_length]u8 = undefined;
    h.final(&digest);
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
}

/// SHA-256 of a file, read in 64 KiB chunks.
pub fn hashFile(a: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8) ![]const u8 {
    const file = try dir.openFile(io, sub_path, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fr = file.reader(io, &buf);
    var chunk: [1 << 16]u8 = undefined;
    var h = Sha256.init(.{});
    while (true) {
        const n = try fr.interface.readSliceShort(&chunk);
        h.update(chunk[0..n]);
        if (n < chunk.len) break;
    }
    return hexDigest(a, &h);
}

/// Recorded instead of a SHA-256 for shards read from a remote source (`hf://`).
pub const remote_hash_placeholder = "not hashed (remote source)";

/// SHA-256 over the exact prompt list (system prompt and user text of every prompt, in order).
/// SHA-256 of the model's config.json; for a GGUF source (no config.json on
/// disk) the configuration reconstructed from the metadata is hashed instead.
fn hashConfig(a: Allocator, io: Io, src: Io.Dir, model: *const model_mod.Model) ![]const u8 {
    if (model.gguf == null) return hashFile(a, io, src, "config.json");
    var h = Sha256.init(.{});
    h.update(model.config_json);
    return hexDigest(a, &h);
}

pub fn hashPrompts(a: Allocator, prompts: []const Prompt) ![]const u8 {
    var h = Sha256.init(.{});
    for (prompts) |p| {
        h.update(p.system);
        h.update(&[_]u8{0});
        h.update(p.user);
        h.update(&[_]u8{'\n'});
    }
    return hexDigest(a, &h);
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

pub const Inputs = struct {
    settings: *const config.Settings,
    model: *const model_mod.Model,
    /// Name of the chat template in use (`@tagName`).
    template: []const u8,
    good_prompts: []const Prompt,
    bad_prompts: []const Prompt,
    keyword_rate_prompts: ?[]const Prompt,
    kl_divergence_prompts: ?[]const Prompt,
    space: *const search.Space,
    trial: *const study_mod.Trial,
    baseline: []const scorers.NamedScore,
};

/// The prompts of the first scorer of `kind`, if one is configured.
pub fn scorerPrompts(ev: *const scorers.Evaluator, kind: config.ScorerKind) ?[]const Prompt {
    for (ev.entries) |e| switch (e.scorer) {
        .keyword_rate => |k| if (kind == .keyword_rate) return k.prompts,
        .kl_divergence => |k| if (kind == .kl_divergence) return k.prompts,
        .refusal_logit => |r| if (kind == .refusal_logit or kind == .keyword_rate) return r.prompts,
    };
    if (ev.deferred) |d| if (kind == .keyword_rate) return d.scorer.keyword_rate.prompts;
    return null;
}

fn dataset(a: Allocator, spec: config.DatasetSpec, prompts: []const Prompt) !Dataset {
    return .{ .spec = spec, .count = prompts.len, .prompts_sha256 = try hashPrompts(a, prompts) };
}

/// Collects the manifest for `in.trial`, hashing the model files. Everything is allocated in `a`.
pub fn build(a: Allocator, io: Io, in: Inputs) !Manifest {
    const s = in.settings;
    const model = in.model;
    var src = try Io.Dir.cwd().openDir(io, model.source_dir, .{});
    defer src.close(io);
    const config_sha = try hashConfig(a, io, src, model);
    const files = try a.alloc(FileHash, model.files.len);
    // Shards of a remote source are not on disk; hashing them would mean downloading the model.
    for (model.files, 0..) |f, i| files[i] = .{ .name = try a.dupe(u8, f.path), .sha256 = if (f.isRemote()) remote_hash_placeholder else try hashFile(a, io, src, f.path) };

    const params = try a.alloc(Param, in.space.space.names.len);
    for (in.space.space.names, 0..) |name, i| params[i] = .{ .name = name, .value = in.trial.params[i] };
    const cfg = search.decode(in.space, in.trial.params);

    const baseline = try a.alloc(study_mod.ScoreRecord, in.baseline.len);
    for (in.baseline, 0..) |b, i| baseline[i] = .{ .name = b.name, .value = b.score.value, .display = b.score.display };
    var range_buf: [64]u8 = undefined;

    return .{
        .ditch_version = config.version,
        .model = s.model,
        .model_commit = s.model_commit,
        .config_sha256 = config_sha,
        .safetensors = files,
        .seed = s.seed orelse 0,
        .chat_template = in.template,
        .response_prefix = s.response_prefix orelse "",
        .system_prompt = s.system_prompt,
        .max_response_length = s.max_response_length,
        .orthogonalize_direction = s.orthogonalize_direction,
        .row_normalization = s.row_normalization,
        .full_normalization_lora_rank = s.full_normalization_lora_rank,
        .winsorization_quantile = s.winsorization_quantile,
        .expert_selection = s.expert_selection,
        .n_directions = s.n_directions,
        .direction_method = s.direction_method,
        .direction_token_window = s.direction_token_window,
        .direction_shrinkage = s.direction_shrinkage,
        .direction_range = try a.dupe(u8, s.direction_range.describe(&range_buf)),
        .ablate_inputs = s.ablate_inputs,
        .kl_tokens = s.kl_tokens,
        .fast_search = s.fast_search,
        .device = try a.dupe(u8, compute.active.name),
        .n_trials = s.n_trials,
        .n_startup_trials = s.n_startup_trials,
        .scorers = s.scorers,
        .keyword_markers = s.keyword_rate.keyword_markers,
        .keyword_rate_score_name = s.keyword_rate.score_name,
        .good_prompts = try dataset(a, s.good_prompts, in.good_prompts),
        .bad_prompts = try dataset(a, s.bad_prompts, in.bad_prompts),
        .keyword_rate_prompts = if (in.keyword_rate_prompts) |p| try dataset(a, s.keyword_rate.prompts, p) else null,
        .kl_divergence_prompts = if (in.kl_divergence_prompts) |p| try dataset(a, s.kl_divergence.prompts, p) else null,
        .trial_index = in.trial.index,
        .params = params,
        .direction_index = cfg.direction_index,
        .parameters = cfg.parameters,
        .experts = cfg.experts,
        .scores = in.trial.scores,
        .baseline = baseline,
    };
}

// ---------------------------------------------------------------------------
// Writing (Lua)
// ---------------------------------------------------------------------------

fn luaString(w: *Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        else => if (c < 0x20 or c == 0x7f) try w.print("\\{d:0>3}", .{c}) else try w.writeByte(c),
    };
    try w.writeByte('"');
}

fn luaField(w: *Io.Writer, indent: []const u8, key: []const u8, value: anytype) !void {
    const T = @TypeOf(value);
    const info = @typeInfo(T);
    if (info == .optional) {
        if (value) |v| try luaField(w, indent, key, v);
        return;
    }
    try w.print("{s}{s} = ", .{ indent, key });
    switch (info) {
        .bool => try w.writeAll(if (value) "true" else "false"),
        .int, .comptime_int => try w.print("{d}", .{value}),
        .float, .comptime_float => if (T == f32) try w.print("{d}", .{value}) else try w.print("{d}", .{@as(f64, @floatCast(value))}),
        .@"enum" => try luaString(w, @tagName(value)),
        else => try luaString(w, value),
    }
    try w.writeAll(",\n");
}

fn writeDataset(w: *Io.Writer, key: []const u8, d: Dataset) !void {
    try w.print("    {s} = {{\n", .{key});
    try luaField(w, "      ", "dataset", d.spec.dataset);
    try luaField(w, "      ", "config", d.spec.config);
    try luaField(w, "      ", "split", d.spec.split);
    try luaField(w, "      ", "column", d.spec.column);
    try luaField(w, "      ", "prefix", d.spec.prefix);
    try luaField(w, "      ", "suffix", d.spec.suffix);
    try luaField(w, "      ", "system_prompt", d.spec.system_prompt);
    try luaField(w, "      ", "count", d.count);
    try luaField(w, "      ", "prompts_sha256", d.prompts_sha256);
    try w.writeAll("    },\n");
}

fn writeScores(w: *Io.Writer, key: []const u8, scores: []const study_mod.ScoreRecord) !void {
    try w.print("  {s} = {{\n", .{key});
    for (scores) |s| {
        try w.writeAll("    { name = ");
        try luaString(w, s.name);
        try w.print(", value = {d}, display = ", .{s.value});
        try luaString(w, s.display);
        try w.writeAll(" },\n");
    }
    try w.writeAll("  },\n");
}

/// Writes the manifest as a Lua chunk returning a table.
pub fn write(m: *const Manifest, w: *Io.Writer) !void {
    try w.writeAll("-- ditch reproducibility manifest. Reproduce this model with:\n");
    try w.print("--   ditch --reproduce {s}\n", .{file_name});
    try w.writeAll("-- The file is a Lua chunk returning a table; it is read in ditch's Lua sandbox.\n");
    try w.writeAll("return {\n");
    try luaField(w, "  ", "ditch_version", m.ditch_version);
    try w.writeAll("  model = {\n");
    try luaField(w, "    ", "id", m.model);
    try luaField(w, "    ", "commit", m.model_commit);
    try luaField(w, "    ", "config_sha256", m.config_sha256);
    try w.writeAll("    safetensors = {\n");
    for (m.safetensors) |f| {
        try w.writeAll("      { name = ");
        try luaString(w, f.name);
        try w.writeAll(", sha256 = ");
        try luaString(w, f.sha256);
        try w.writeAll(" },\n");
    }
    try w.writeAll("    },\n  },\n");

    try w.writeAll("  settings = {\n");
    try luaField(w, "    ", "seed", m.seed);
    try luaField(w, "    ", "chat_template", m.chat_template);
    try luaField(w, "    ", "response_prefix", m.response_prefix);
    try luaField(w, "    ", "system_prompt", m.system_prompt);
    try luaField(w, "    ", "max_response_length", m.max_response_length);
    try luaField(w, "    ", "orthogonalize_direction", m.orthogonalize_direction);
    try luaField(w, "    ", "row_normalization", m.row_normalization);
    try luaField(w, "    ", "full_normalization_lora_rank", m.full_normalization_lora_rank);
    try luaField(w, "    ", "winsorization_quantile", m.winsorization_quantile);
    try luaField(w, "    ", "expert_selection", m.expert_selection);
    try luaField(w, "    ", "n_directions", m.n_directions);
    try luaField(w, "    ", "direction_method", m.direction_method);
    try luaField(w, "    ", "direction_token_window", m.direction_token_window);
    try luaField(w, "    ", "direction_shrinkage", m.direction_shrinkage);
    try luaField(w, "    ", "direction_range", m.direction_range);
    try luaField(w, "    ", "ablate_inputs", m.ablate_inputs);
    try luaField(w, "    ", "kl_tokens", m.kl_tokens);
    try luaField(w, "    ", "fast_search", m.fast_search);
    try luaField(w, "    ", "device", m.device);
    try luaField(w, "    ", "n_trials", m.n_trials);
    try luaField(w, "    ", "n_startup_trials", m.n_startup_trials);
    try w.writeAll("    scorers = {\n");
    for (m.scorers) |sc| {
        try w.print("      {{ plugin = \"{s}\", optimization = \"{s}\"", .{ @tagName(sc.kind), @tagName(sc.optimization) });
        if (sc.instance_name) |n| {
            try w.writeAll(", instance_name = ");
            try luaString(w, n);
        }
        try w.writeAll(" },\n");
    }
    try w.writeAll("    },\n");
    try luaField(w, "    ", "keyword_rate_score_name", m.keyword_rate_score_name);
    try w.writeAll("    keyword_markers = {\n");
    for (m.keyword_markers) |k| {
        try w.writeAll("      ");
        try luaString(w, k);
        try w.writeAll(",\n");
    }
    try w.writeAll("    },\n  },\n");

    try w.writeAll("  datasets = {\n");
    try writeDataset(w, "good_prompts", m.good_prompts);
    try writeDataset(w, "bad_prompts", m.bad_prompts);
    if (m.keyword_rate_prompts) |d| try writeDataset(w, "keyword_rate_prompts", d);
    if (m.kl_divergence_prompts) |d| try writeDataset(w, "kl_divergence_prompts", d);
    try w.writeAll("  },\n");

    try w.writeAll("  trial = {\n");
    try luaField(w, "    ", "index", m.trial_index);
    try w.writeAll("    -- Sampled values of the search space parameters (applied verbatim by --reproduce).\n");
    try w.writeAll("    params = {\n");
    for (m.params) |p| try w.print("      [\"{s}\"] = {d},\n", .{ p.name, p.value });
    try w.writeAll("    },\n");
    try w.writeAll("    -- The same values decoded into the abliteration kernel parameters.\n");
    try w.writeAll("    decoded = {\n");
    try luaField(w, "      ", "direction_scope", @as([]const u8, if (m.direction_index == null) "per layer" else "global"));
    try luaField(w, "      ", "direction_index", m.direction_index);
    for (Component.all) |comp| {
        const p = m.parameters.get(comp) orelse continue;
        try w.print("      [\"{s}\"] = {{ max_weight = {d}, max_weight_position = {d}, min_weight = {d}, min_weight_distance = {d} }},\n", .{ comp.name(), p.max_weight, p.max_weight_position, p.min_weight, p.min_weight_distance });
    }
    if (m.experts) |e| try w.print("      experts = {{ n_experts = {d}, strength = {d} }},\n", .{ e.n_experts, e.strength });
    try w.writeAll("    },\n  },\n");

    try writeScores(w, "scores", m.scores);
    try writeScores(w, "baseline", m.baseline);
    try w.writeAll("}\n");
}

/// Writes `ditch-reproduce.lua` into `dir_path`.
pub fn writeFile(gpa: Allocator, io: Io, m: *const Manifest, dir_path: []const u8) !void {
    var buf: Io.Writer.Allocating = .init(gpa);
    defer buf.deinit();
    try write(m, &buf.writer);
    var dir = try Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = file_name, .data = buf.written() });
}

/// Markdown section for the exported model card.
pub fn markdown(m: *const Manifest, w: *Io.Writer) !void {
    try w.writeAll("\n## Reproduce\n\n");
    try w.print("Reproduce with: `ditch --reproduce {s}` (the manifest is stored next to this file).\n\n", .{file_name});
    try w.writeAll("| Item | Value |\n| :--- | :--- |\n");
    try w.print("| **Source model** | `{s}`", .{m.model});
    if (m.model_commit) |c| try w.print(" @ `{s}`", .{c});
    try w.writeAll(" |\n");
    try w.print("| **config.json SHA-256** | `{s}` |\n", .{m.config_sha256});
    for (m.safetensors) |f| try w.print("| **{s} SHA-256** | `{s}` |\n", .{ f.name, f.sha256 });
    try w.print("| **ditch version** | {s} |\n", .{m.ditch_version});
    try w.print("| **Seed** | {d} |\n", .{m.seed});
    try w.print("| **Chat template** | {s} |\n", .{m.chat_template});
    try w.writeAll("| **Response prefix** | `");
    try w.writeAll(m.response_prefix);
    try w.writeAll("` |\n");
    try w.print("| **Row normalization** | {s} (rank {d}) |\n", .{ @tagName(m.row_normalization), m.full_normalization_lora_rank });
    try w.print("| **Orthogonalize direction** | {s} |\n", .{if (m.orthogonalize_direction) "true" else "false"});
    try w.print("| **Winsorization quantile** | {d} |\n", .{m.winsorization_quantile});
    try w.print("| **Expert selection** | {s} |\n", .{@tagName(m.expert_selection)});
    try w.print("| **Directions** | {d} per layer, method {s}, token window {d}, direction_index range {s}{s} |\n", .{ m.n_directions, @tagName(m.direction_method), m.direction_token_window, m.direction_range, if (m.ablate_inputs) ", input side ablated" else "" });
    try w.print("| **KL divergence positions** | {d} |\n", .{m.kl_tokens});
    if (m.fast_search) try w.writeAll("| **Search** | fast (KL divergence + refusal-logit proxy; keyword scorer on the Pareto candidates) |\n");
    try w.print("| **Max response length** | {d} |\n", .{m.max_response_length});
    try w.print("| **Compute device** | {s} |\n", .{m.device});
    try markdownDataset(w, "Good prompts", m.good_prompts);
    try markdownDataset(w, "Bad prompts", m.bad_prompts);
    if (m.keyword_rate_prompts) |d| try markdownDataset(w, "Refusal scoring prompts", d);
    if (m.kl_divergence_prompts) |d| try markdownDataset(w, "KL divergence prompts", d);
    try w.print("| **Trial** | {d} of {d} ({d} startup trials) |\n", .{ m.trial_index, m.n_trials, m.n_startup_trials });
    try w.writeAll("| **Search parameters** | ");
    for (m.params, 0..) |p, i| try w.print("{s}`{s}` = {d:.4}", .{ if (i == 0) "" else ", ", p.name, p.value });
    try w.writeAll(" |\n");
}

fn markdownDataset(w: *Io.Writer, label: []const u8, d: Dataset) !void {
    try w.print("| **{s}** | `{s}`", .{ label, d.spec.dataset });
    if (d.spec.config) |c| try w.print(" config `{s}`", .{c});
    if (d.spec.split) |s| try w.print(" split `{s}`", .{s});
    if (d.spec.column) |c| try w.print(" column `{s}`", .{c});
    try w.print(", {d} prompts, SHA-256 `{s}` |\n", .{ d.count, d.prompts_sha256 });
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

const ReadError = error{ InvalidManifest, OutOfMemory };

const Reader = struct {
    a: Allocator,

    fn missing(path: []const u8) ReadError {
        std.log.err("manifest is missing or has an invalid value for {s}", .{path});
        return error.InvalidManifest;
    }

    fn table(_: Reader, t: *const tree.Table, key: []const u8) ?*const tree.Table {
        return t.getTable(key);
    }

    fn reqTable(self: Reader, t: *const tree.Table, key: []const u8) ReadError!*const tree.Table {
        return self.table(t, key) orelse missing(key);
    }

    fn str(self: Reader, t: *const tree.Table, key: []const u8) ReadError!?[]const u8 {
        const v = t.get(key) orelse return null;
        if (v != .string) return missing(key);
        return try self.a.dupe(u8, v.string);
    }

    fn reqStr(self: Reader, t: *const tree.Table, key: []const u8) ReadError![]const u8 {
        return (try self.str(t, key)) orelse missing(key);
    }

    fn num(_: Reader, t: *const tree.Table, key: []const u8) ReadError!f64 {
        const v = t.get(key) orelse return missing(key);
        return v.asFloat() orelse missing(key);
    }

    fn int(self: Reader, t: *const tree.Table, key: []const u8) ReadError!usize {
        const f = try self.num(t, key);
        if (f < 0) return missing(key);
        return @intFromFloat(f);
    }

    fn boolean(_: Reader, t: *const tree.Table, key: []const u8) ReadError!bool {
        const v = t.get(key) orelse return missing(key);
        if (v != .boolean) return missing(key);
        return v.boolean;
    }

    /// Array items of `key`; an empty Lua table is an empty array.
    fn array(_: Reader, t: *const tree.Table, key: []const u8) ReadError![]const tree.Value {
        const v = t.get(key) orelse return &.{};
        return switch (v) {
            .array => |arr| arr,
            .table => |tab| if (tab.map.count() == 0) &.{} else missing(key),
            else => missing(key),
        };
    }

    fn scores(self: Reader, t: *const tree.Table, key: []const u8) ReadError![]const study_mod.ScoreRecord {
        const items = try self.array(t, key);
        const out = try self.a.alloc(study_mod.ScoreRecord, items.len);
        for (items, 0..) |v, i| {
            if (v != .table) return missing(key);
            out[i] = .{ .name = try self.reqStr(v.table, "name"), .value = try self.num(v.table, "value"), .display = try self.reqStr(v.table, "display") };
        }
        return out;
    }

    fn dataset(self: Reader, t: *const tree.Table, key: []const u8) ReadError!?Dataset {
        const d = self.table(t, key) orelse return null;
        return .{
            .spec = .{
                .dataset = try self.reqStr(d, "dataset"),
                .config = try self.str(d, "config"),
                .split = try self.str(d, "split"),
                .column = try self.str(d, "column"),
                .prefix = (try self.str(d, "prefix")) orelse "",
                .suffix = (try self.str(d, "suffix")) orelse "",
                .system_prompt = try self.str(d, "system_prompt"),
            },
            .count = try self.int(d, "count"),
            .prompts_sha256 = try self.reqStr(d, "prompts_sha256"),
        };
    }
};

/// Converts a parsed manifest table into a `Manifest` (strings copied into `a`).
pub fn fromTable(a: Allocator, root: *const tree.Table) ReadError!Manifest {
    const r = Reader{ .a = a };
    const model = try r.reqTable(root, "model");
    const settings = try r.reqTable(root, "settings");
    const datasets = try r.reqTable(root, "datasets");
    const trial = try r.reqTable(root, "trial");

    const st_items = try r.array(model, "safetensors");
    const files = try a.alloc(FileHash, st_items.len);
    for (st_items, 0..) |v, i| {
        if (v != .table) return Reader.missing("model.safetensors");
        files[i] = .{ .name = try r.reqStr(v.table, "name"), .sha256 = try r.reqStr(v.table, "sha256") };
    }

    const scorer_items = try r.array(settings, "scorers");
    const scorer_list = try a.alloc(config.ScorerConfig, scorer_items.len);
    for (scorer_items, 0..) |v, i| {
        if (v != .table) return Reader.missing("settings.scorers");
        const plugin = try r.reqStr(v.table, "plugin");
        const opt = try r.reqStr(v.table, "optimization");
        scorer_list[i] = .{
            .kind = config.ScorerKind.fromPlugin(plugin) orelse return Reader.missing("settings.scorers.plugin"),
            .optimization = std.meta.stringToEnum(config.Optimization, opt) orelse return Reader.missing("settings.scorers.optimization"),
            .instance_name = try r.str(v.table, "instance_name"),
        };
    }
    const marker_items = try r.array(settings, "keyword_markers");
    const markers = try a.alloc([]const u8, marker_items.len);
    for (marker_items, 0..) |v, i| {
        if (v != .string) return Reader.missing("settings.keyword_markers");
        markers[i] = try a.dupe(u8, v.string);
    }

    const params_t = try r.reqTable(trial, "params");
    const params = try a.alloc(Param, params_t.map.count());
    var it = params_t.map.iterator();
    var i: usize = 0;
    while (it.next()) |e| : (i += 1) {
        params[i] = .{ .name = try a.dupe(u8, e.key_ptr.*), .value = e.value_ptr.asFloat() orelse return Reader.missing("trial.params") };
    }
    const decoded = try r.reqTable(trial, "decoded");
    const scope = try r.reqStr(decoded, "direction_scope");
    var direction_index: ?f32 = null;
    if (!std.mem.eql(u8, scope, "per layer")) direction_index = @floatCast(try r.num(decoded, "direction_index"));
    var parameters = std.EnumMap(Component, abliterate.Params){};
    for (Component.all) |comp| {
        const p = r.table(decoded, comp.name()) orelse continue;
        parameters.put(comp, .{
            .max_weight = @floatCast(try r.num(p, "max_weight")),
            .max_weight_position = @floatCast(try r.num(p, "max_weight_position")),
            .min_weight = @floatCast(try r.num(p, "min_weight")),
            .min_weight_distance = @floatCast(try r.num(p, "min_weight_distance")),
        });
    }
    var experts: ?search.ExpertSelection = null;
    if (r.table(decoded, "experts")) |e| experts = .{ .n_experts = try r.int(e, "n_experts"), .strength = @floatCast(try r.num(e, "strength")) };

    const row_norm_s = try r.reqStr(settings, "row_normalization");
    const expert_sel_s = try r.reqStr(settings, "expert_selection");
    // Settings added after the first manifests: absent means the (heretic) default.
    const defaults = config.Settings{};
    const method_s = (try r.str(settings, "direction_method")) orelse "mean";
    var range_buf: [64]u8 = undefined;
    const range_s = (try r.str(settings, "direction_range")) orelse try a.dupe(u8, defaults.direction_range.describe(&range_buf));
    if (config.DirectionRange.parse(range_s) == null) return Reader.missing("settings.direction_range");
    return .{
        .ditch_version = (try r.str(root, "ditch_version")) orelse "unknown",
        .model = try r.reqStr(model, "id"),
        .model_commit = try r.str(model, "commit"),
        .config_sha256 = try r.reqStr(model, "config_sha256"),
        .safetensors = files,
        .seed = try r.int(settings, "seed"),
        .chat_template = try r.reqStr(settings, "chat_template"),
        .response_prefix = (try r.str(settings, "response_prefix")) orelse "",
        .system_prompt = try r.reqStr(settings, "system_prompt"),
        .max_response_length = try r.int(settings, "max_response_length"),
        .orthogonalize_direction = try r.boolean(settings, "orthogonalize_direction"),
        .row_normalization = abliterate.RowNormalization.parse(row_norm_s) orelse return Reader.missing("settings.row_normalization"),
        .full_normalization_lora_rank = try r.int(settings, "full_normalization_lora_rank"),
        .winsorization_quantile = @floatCast(try r.num(settings, "winsorization_quantile")),
        .expert_selection = abliterate.ExpertSelection.parse(expert_sel_s) orelse return Reader.missing("settings.expert_selection"),
        .n_directions = if (settings.get("n_directions") != null) try r.int(settings, "n_directions") else defaults.n_directions,
        .direction_method = directions.Method.parse(method_s) orelse return Reader.missing("settings.direction_method"),
        .direction_token_window = if (settings.get("direction_token_window") != null) try r.int(settings, "direction_token_window") else defaults.direction_token_window,
        .direction_shrinkage = if (settings.get("direction_shrinkage") != null) @floatCast(try r.num(settings, "direction_shrinkage")) else defaults.direction_shrinkage,
        .direction_range = range_s,
        .ablate_inputs = if (settings.get("ablate_inputs") != null) try r.boolean(settings, "ablate_inputs") else defaults.ablate_inputs,
        .kl_tokens = if (settings.get("kl_tokens") != null) try r.int(settings, "kl_tokens") else defaults.kl_tokens,
        .fast_search = if (settings.get("fast_search") != null) try r.boolean(settings, "fast_search") else defaults.fast_search,
        .device = (try r.str(settings, "device")) orelse "cpu",
        .n_trials = try r.int(settings, "n_trials"),
        .n_startup_trials = try r.int(settings, "n_startup_trials"),
        .scorers = scorer_list,
        .keyword_markers = markers,
        .keyword_rate_score_name = (try r.str(settings, "keyword_rate_score_name")) orelse "Refusals",
        .good_prompts = (try r.dataset(datasets, "good_prompts")) orelse return Reader.missing("datasets.good_prompts"),
        .bad_prompts = (try r.dataset(datasets, "bad_prompts")) orelse return Reader.missing("datasets.bad_prompts"),
        .keyword_rate_prompts = try r.dataset(datasets, "keyword_rate_prompts"),
        .kl_divergence_prompts = try r.dataset(datasets, "kl_divergence_prompts"),
        .trial_index = try r.int(trial, "index"),
        .params = params,
        .direction_index = direction_index,
        .parameters = parameters,
        .experts = experts,
        .scores = try r.scores(root, "scores"),
        .baseline = try r.scores(root, "baseline"),
    };
}

/// Loads a manifest file through the Lua sandbox.
pub fn load(gpa: Allocator, a: Allocator, io: Io, path: []const u8) !Manifest {
    const text = Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |err| {
        std.log.err("could not read manifest {s}: {s}", .{ path, @errorName(err) });
        return err;
    };
    defer gpa.free(text);
    var result = try lua.parse(gpa, text, path);
    defer result.parsed.deinit();
    if (result.err) |e| {
        std.log.err("could not load manifest {s}: {s}", .{ path, e });
        return error.InvalidManifest;
    }
    return fromTable(a, result.parsed.root);
}

/// Copies the recorded settings that affect the result into `s`.
pub fn applySettings(m: *const Manifest, s: *config.Settings) void {
    s.model = m.model;
    s.model_commit = m.model_commit;
    s.seed = m.seed;
    s.chat_template = m.chat_template;
    s.response_prefix = m.response_prefix;
    s.system_prompt = m.system_prompt;
    s.max_response_length = m.max_response_length;
    s.orthogonalize_direction = m.orthogonalize_direction;
    s.row_normalization = m.row_normalization;
    s.full_normalization_lora_rank = m.full_normalization_lora_rank;
    s.winsorization_quantile = m.winsorization_quantile;
    s.expert_selection = m.expert_selection;
    s.n_directions = m.n_directions;
    s.direction_method = m.direction_method;
    s.direction_token_window = m.direction_token_window;
    s.direction_shrinkage = m.direction_shrinkage;
    s.direction_range = config.DirectionRange.parse(m.direction_range) orelse s.direction_range;
    s.ablate_inputs = m.ablate_inputs;
    s.kl_tokens = m.kl_tokens;
    s.fast_search = m.fast_search;
    s.n_trials = m.n_trials;
    s.n_startup_trials = m.n_startup_trials;
    if (m.scorers.len > 0) s.scorers = m.scorers;
    s.keyword_rate.keyword_markers = m.keyword_markers;
    s.keyword_rate.score_name = m.keyword_rate_score_name;
    s.good_prompts = m.good_prompts.spec;
    s.bad_prompts = m.bad_prompts.spec;
    if (m.keyword_rate_prompts) |d| s.keyword_rate.prompts = d.spec;
    if (m.kl_divergence_prompts) |d| s.kl_divergence.prompts = d.spec;
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn reportMismatch(out: *Io.Writer, what: []const u8, expected: []const u8, actual: []const u8) !void {
    try out.print("* {s}: MISMATCH\n    recorded {s}\n    actual   {s}\n", .{ what, expected, actual });
}

/// Compares the hashes of the loaded model's files with the recorded ones.
/// With `ignore`, mismatches are reported but tolerated.
pub fn verifyModelFiles(a: Allocator, io: Io, m: *const Manifest, model: *const model_mod.Model, ignore: bool, out: *Io.Writer) !void {
    try out.writeAll("\nVerifying model files against the manifest...\n");
    try out.flush();
    var src = try Io.Dir.cwd().openDir(io, model.source_dir, .{});
    defer src.close(io);
    var mismatches: usize = 0;
    const config_sha = try hashConfig(a, io, src, model);
    if (std.mem.eql(u8, config_sha, m.config_sha256)) {
        try out.writeAll("* config.json: ok\n");
    } else {
        mismatches += 1;
        try reportMismatch(out, "config.json", m.config_sha256, config_sha);
    }
    for (m.safetensors) |f| {
        const actual = hashFile(a, io, src, f.name) catch |err| {
            mismatches += 1;
            try reportMismatch(out, f.name, f.sha256, @errorName(err));
            continue;
        };
        if (std.mem.eql(u8, actual, f.sha256)) {
            try out.print("* {s}: ok\n", .{f.name});
        } else {
            mismatches += 1;
            try reportMismatch(out, f.name, f.sha256, actual);
        }
    }
    for (model.files) |f| {
        var recorded = false;
        for (m.safetensors) |r| recorded = recorded or std.mem.eql(u8, r.name, f.path);
        if (!recorded) {
            mismatches += 1;
            try out.print("* {s}: MISMATCH (not part of the recorded model)\n", .{f.path});
        }
    }
    try out.flush();
    if (mismatches == 0) return;
    if (ignore) {
        try out.print("* {d} mismatch(es) ignored (--ignore-mismatches); the result may differ from the recorded one.\n", .{mismatches});
        return;
    }
    try out.print("Error: {d} model file(s) differ from the ones recorded in the manifest (source model {s}); the result would not be reproducible. Pass --ignore-mismatches to proceed anyway.\n", .{ mismatches, m.model });
    try out.flush();
    return error.ModelMismatch;
}

/// Checks that a loaded prompt list is the recorded one.
pub fn verifyPrompts(a: Allocator, d: Dataset, prompts: []const Prompt, label: []const u8, ignore: bool, out: *Io.Writer) !void {
    const actual = try hashPrompts(a, prompts);
    if (std.mem.eql(u8, actual, d.prompts_sha256) and prompts.len == d.count) {
        try out.print("* {s}: {d} prompts, hash ok\n", .{ label, prompts.len });
        return;
    }
    try out.print("* {s}: MISMATCH (recorded {d} prompts with hash {s}, loaded {d} prompts with hash {s})\n", .{ label, d.count, d.prompts_sha256, prompts.len, actual });
    try out.flush();
    if (ignore) return;
    try out.print("Error: the {s} differ from the prompts recorded in the manifest (dataset {s}); pass --ignore-mismatches to proceed anyway.\n", .{ label, d.spec.dataset });
    try out.flush();
    return error.PromptMismatch;
}

/// The recorded parameter vector in the order of `space`.
pub fn vectorFor(a: Allocator, m: *const Manifest, space: *const search.Space) ![]f64 {
    const v = try a.alloc(f64, space.space.names.len);
    for (space.space.names, 0..) |name, i| {
        v[i] = m.param(name) orelse {
            std.log.err("the manifest has no value for search parameter {s}; was it produced for a different model architecture?", .{name});
            return error.InvalidManifest;
        };
    }
    return v;
}

/// Prints the reproduced scores next to the recorded ones. Returns true when all displays match.
pub fn printComparison(m: *const Manifest, scores: []const scorers.NamedScore, out: *Io.Writer) !bool {
    var all_match = true;
    try out.writeAll("\nReproduced scores (recorded scores in parentheses):\n");
    for (scores) |s| {
        var recorded: []const u8 = "not recorded";
        for (m.scores) |r| if (std.mem.eql(u8, r.name, s.name)) {
            recorded = r.display;
        };
        const same = std.mem.eql(u8, recorded, s.score.display);
        if (!same) all_match = false;
        try out.print("  * {s}: {s} (recorded {s}){s}\n", .{ s.name, s.score.display, recorded, if (same) "" else " <- differs" });
    }
    try out.writeAll(if (all_match) "* All scores match the manifest.\n" else "* Some scores differ from the manifest.\n");
    return all_match;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "manifest write and parse round trip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const pool = @import("tensor.zig").Pool.init(io, 1);
    const model = try model_mod.Model.load(gpa, io, &pool, "tests/fixtures/qwen2");
    defer model.deinit();
    var space = try search.buildSpace(gpa, model);
    defer space.deinit();

    var settings = config.Settings{ .model = "tests/fixtures/qwen2", .seed = 7, .response_prefix = "<think>\n</think>", .row_normalization = .pre, .n_directions = 2, .direction_method = .separating, .direction_token_window = 3, .direction_range = .auto, .ablate_inputs = true, .kl_tokens = 2, .fast_search = true };
    settings.good_prompts = .{ .dataset = "good.txt", .split = "[:2]" };
    const good = [_]Prompt{ .{ .system = "sys", .user = "hello \"world\"" }, .{ .system = "sys", .user = "line\nbreak" } };
    const bad = [_]Prompt{.{ .system = "sys", .user = "bad" }};
    const params = [_]f64{ 0, 1.25, 1.0, 1.5, 0.5, 1.0, -0.125, 1.75, 0.25, 1.0 };
    const cfg = search.decode(&space, &params);
    const scores = [_]study_mod.ScoreRecord{ .{ .name = "Refusals", .value = 0.5, .display = "1/2" }, .{ .name = "KL divergence", .value = 0.0625, .display = "0.0625" } };
    const trial = study_mod.Trial{ .index = 3, .params = &params, .losses = &.{ 0.5, 0.0625 }, .direction_index = cfg.direction_index, .parameters = cfg.parameters, .scores = &scores };
    const baseline = [_]scorers.NamedScore{.{ .name = "Refusals", .score = .{ .value = 1, .display = "2/2" } }};
    const m = try build(a, io, .{
        .settings = &settings,
        .model = model,
        .template = "chatml",
        .good_prompts = &good,
        .bad_prompts = &bad,
        .keyword_rate_prompts = &bad,
        .kl_divergence_prompts = null,
        .space = &space,
        .trial = &trial,
        .baseline = &baseline,
    });
    try std.testing.expectEqual(@as(usize, 2), m.safetensors.len);
    try std.testing.expectEqual(@as(usize, 64), m.config_sha256.len);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..n];
    try writeFile(gpa, io, &m, dir_path);
    const path = try std.fs.path.join(a, &.{ dir_path, file_name });
    const back = try load(gpa, a, io, path);

    try std.testing.expectEqualStrings(config.version, back.ditch_version);
    try std.testing.expectEqualStrings("tests/fixtures/qwen2", back.model);
    try std.testing.expectEqualStrings(m.config_sha256, back.config_sha256);
    try std.testing.expectEqualStrings(m.safetensors[1].sha256, back.safetensors[1].sha256);
    try std.testing.expectEqual(@as(u64, 7), back.seed);
    try std.testing.expectEqualStrings("<think>\n</think>", back.response_prefix);
    try std.testing.expectEqual(abliterate.RowNormalization.pre, back.row_normalization);
    try std.testing.expectEqualStrings("[:2]", back.good_prompts.spec.split.?);
    try std.testing.expectEqual(@as(usize, 2), back.good_prompts.count);
    try std.testing.expectEqualStrings(m.good_prompts.prompts_sha256, back.good_prompts.prompts_sha256);
    try std.testing.expectEqualStrings(try hashPrompts(a, &good), back.good_prompts.prompts_sha256);
    try std.testing.expect(back.kl_divergence_prompts == null);
    try std.testing.expectEqual(@as(usize, 3), back.trial_index);
    try std.testing.expectEqual(@as(usize, 2), back.scorers.len);
    try std.testing.expectEqual(config.default_markers.len, back.keyword_markers.len);
    const vector = try vectorFor(a, &back, &space);
    try std.testing.expectEqualSlices(f64, &params, vector);
    try std.testing.expectEqual(@as(?f32, 1.25), back.direction_index);
    try std.testing.expectEqual(@as(f32, 0.0), back.parameters.get(.mlp_down_proj).?.max_weight);
    try std.testing.expectEqualStrings("0.0625", back.scores[1].display);
    try std.testing.expectEqualStrings("2/2", back.baseline[0].display);

    // Verification passes on the untouched fixture and reports a corrupted hash.
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try verifyModelFiles(a, io, &back, model, false, &sink.writer);
    var corrupt = back;
    corrupt.config_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    try std.testing.expectError(error.ModelMismatch, verifyModelFiles(a, io, &corrupt, model, false, &sink.writer));
    try verifyModelFiles(a, io, &corrupt, model, true, &sink.writer);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "config.json: MISMATCH") != null);
    try verifyPrompts(a, back.good_prompts, &good, "good prompts", false, &sink.writer);
    try std.testing.expectError(error.PromptMismatch, verifyPrompts(a, back.good_prompts, &bad, "good prompts", false, &sink.writer));

    // Applying the manifest restores the recorded settings.
    var fresh = config.Settings{};
    applySettings(&back, &fresh);
    try std.testing.expectEqual(@as(?u64, 7), fresh.seed);
    try std.testing.expectEqualStrings("good.txt", fresh.good_prompts.dataset);
    try std.testing.expectEqualStrings("chatml", fresh.chat_template.?);
    try std.testing.expectEqual(@as(usize, 2), fresh.n_directions);
    try std.testing.expectEqual(directions.Method.separating, fresh.direction_method);
    try std.testing.expectEqual(@as(usize, 3), fresh.direction_token_window);
    try std.testing.expectEqual(config.DirectionRange.auto, fresh.direction_range);
    try std.testing.expect(fresh.ablate_inputs and fresh.fast_search);
    try std.testing.expectEqual(@as(usize, 2), fresh.kl_tokens);
    try std.testing.expectEqualStrings("auto", back.direction_range);

    // A manifest without the newer settings reads them as the defaults.
    const text = try Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
    var stripped = std.ArrayList(u8).empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    const newer = [_][]const u8{ "    n_directions = ", "    direction_method = ", "    direction_token_window = ", "    direction_shrinkage = ", "    direction_range = ", "    ablate_inputs = ", "    kl_tokens = ", "    fast_search = " };
    while (lines.next()) |line| {
        var skip = false;
        for (newer) |k| skip = skip or std.mem.startsWith(u8, line, k);
        if (skip) continue;
        try stripped.appendSlice(a, line);
        try stripped.append(a, '\n');
    }
    const old_path = try std.fs.path.join(a, &.{ dir_path, "old.lua" });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = old_path, .data = stripped.items });
    const old = try load(gpa, a, io, old_path);
    try std.testing.expectEqual(@as(usize, 1), old.n_directions);
    try std.testing.expectEqual(directions.Method.mean, old.direction_method);
    try std.testing.expectEqual(@as(usize, 1), old.kl_tokens);
    try std.testing.expect(!old.ablate_inputs and !old.fast_search);
    try std.testing.expectEqualStrings("0.4:0.9", old.direction_range);
}

test "hash file matches a known digest" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "abc.txt", .data = "abc" });
    const h = try hashFile(gpa, io, tmp.dir, "abc.txt");
    defer gpa.free(h);
    try std.testing.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", h);
}
