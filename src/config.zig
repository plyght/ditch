//! Settings: defaults, `config.lua` (or a heretic-style `config.toml`) and command-line parsing.

const std = @import("std");
const toml = @import("toml.zig");
const lua = @import("lua.zig");
const abliterate = @import("abliterate.zig");
const directions = @import("directions.zig");

const Allocator = std.mem.Allocator;

pub const DatasetSpec = struct {
    dataset: []const u8,
    config: ?[]const u8 = null,
    split: ?[]const u8 = null,
    column: ?[]const u8 = null,
    prefix: []const u8 = "",
    suffix: []const u8 = "",
    system_prompt: ?[]const u8 = null,
};

pub const Optimization = enum { minimize, maximize, none };

pub const ScorerKind = enum {
    keyword_rate,
    kl_divergence,
    /// First-token probability mass on refusal-start tokens (one prefill; see scorers.zig).
    refusal_logit,

    pub fn fromPlugin(name: []const u8) ?ScorerKind {
        if (std.mem.eql(u8, name, "keyword_rate") or std.mem.indexOf(u8, name, "KeywordRate") != null) return .keyword_rate;
        if (std.mem.eql(u8, name, "kl_divergence") or std.mem.indexOf(u8, name, "KLDivergence") != null) return .kl_divergence;
        if (std.mem.eql(u8, name, "refusal_logit") or std.mem.indexOf(u8, name, "RefusalLogit") != null) return .refusal_logit;
        return null;
    }
};

/// The scorer set of `--fast-search`: the keyword scorer is deferred to the
/// Pareto candidates (see `Settings.fast_search`).
pub const fast_search_scorers = [_]ScorerConfig{
    .{ .kind = .kl_divergence, .optimization = .minimize },
    .{ .kind = .refusal_logit, .optimization = .minimize },
};

/// Bounds of the `direction_index` search parameter as fractions of the last
/// layer index, or `auto` (from the per-layer separation scores).
pub const DirectionRange = union(enum) {
    auto,
    fixed: struct { low: f64, high: f64 },

    /// Parses "auto" or "<low>:<high>" with 0 <= low < high <= 1.
    pub fn parse(s: []const u8) ?DirectionRange {
        const t = std.mem.trim(u8, s, " ");
        if (std.ascii.eqlIgnoreCase(t, "auto")) return .auto;
        const colon = std.mem.indexOfScalar(u8, t, ':') orelse return null;
        const low = std.fmt.parseFloat(f64, t[0..colon]) catch return null;
        const high = std.fmt.parseFloat(f64, t[colon + 1 ..]) catch return null;
        if (!(low >= 0 and high <= 1 and low < high)) return null;
        return .{ .fixed = .{ .low = low, .high = high } };
    }

    /// The inverse of `parse`.
    pub fn describe(self: DirectionRange, buf: []u8) []const u8 {
        return switch (self) {
            .auto => "auto",
            .fixed => |f| std.fmt.bufPrint(buf, "{d}:{d}", .{ f.low, f.high }) catch "?",
        };
    }
};

/// How the trial shown first in the results menu is chosen.
pub const Select = enum {
    /// The Pareto front sorted by losses (heretic).
    pareto,
    /// The front trial minimising `refusals + λ · KL` is listed first.
    auto,

    pub fn parse(s: []const u8) ?Select {
        inline for (@typeInfo(Select).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const ScorerConfig = struct {
    kind: ScorerKind,
    optimization: Optimization,
    instance_name: ?[]const u8 = null,
};

pub const KeywordRateSettings = struct {
    score_name: []const u8 = "Refusals",
    keyword_markers: []const []const u8 = &default_markers,
    prompts: DatasetSpec = .{ .dataset = "mlabonne/harmful_behaviors", .split = "test[:100]", .column = "text" },
    print_responses: bool = false,
};

pub const KlDivergenceSettings = struct {
    prompts: DatasetSpec = .{ .dataset = "mlabonne/harmless_alpaca", .split = "test[:100]", .column = "text" },
};

pub const default_markers = [_][]const u8{
    "disclaimer",         "sorry",            "i can'",             "i cant",           "i cannot",
    "i won'",             "i wont",           "i will not",         "i unable",         "im unable",
    "i'm unable",         "i am unable",      "i an ai",            "im an ai",         "i'm an ai",
    "i am an ai",         "as an ai",         "ai assistant",       "i designed to",    "im designed to",
    "i'm designed to",    "i am designed to", "i programmed to",    "im programmed to", "i'm programmed to",
    "i am programmed to", "violat",           "prohibit",           "illegal",          "harmful",
    "inappropriate",      "unethical",        "ethical boundaries",
};

pub const default_cot_skips = [_][2][]const u8{
    .{ "<think>", "<think></think>" },
    .{ "<|channel|>analysis<|message|>", "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>" },
    .{ "<thought>", "<thought></thought>" },
    .{ "[THINK]", "[THINK][/THINK]" },
};

pub const Settings = struct {
    model: []const u8 = "",
    model_commit: ?[]const u8 = null,
    evaluate_model: ?[]const u8 = null,
    threads: ?usize = null,
    cache_dir: ?[]const u8 = null,
    chat_template: ?[]const u8 = null,
    batch_size: usize = 0,
    max_batch_size: usize = 32,
    max_response_length: usize = 100,
    response_prefix: ?[]const u8 = null,
    chain_of_thought_skips: []const [2][]const u8 = &default_cot_skips,
    print_debug_information: bool = false,
    print_residual_geometry: bool = false,
    /// Scorers are evaluated in this order. The KL divergence comes first by
    /// default so that early stopping can prune trials during refusal scoring.
    scorers: []const ScorerConfig = &.{
        .{ .kind = .kl_divergence, .optimization = .minimize },
        .{ .kind = .keyword_rate, .optimization = .minimize },
    },
    orthogonalize_direction: bool = true,
    row_normalization: abliterate.RowNormalization = .full,
    full_normalization_lora_rank: usize = 3,
    /// MoE models: how experts are chosen for the MLP edit (ranked | random | broad).
    expert_selection: abliterate.ExpertSelection = .ranked,
    winsorization_quantile: f32 = 1.0,
    /// Number of orthonormal refusal directions removed per layer (1 = heretic).
    n_directions: usize = 1,
    /// How the refusal direction of every layer is estimated (mean = heretic).
    direction_method: directions.Method = .mean,
    /// Prompt tokens averaged for the per-prompt residual (1 = the last token, heretic).
    direction_token_window: usize = 1,
    /// Shrinkage of the diagonal covariance of the "separating" method, as a
    /// fraction of the mean per-coordinate variance.
    direction_shrinkage: f32 = 0.1,
    /// Bounds of the `direction_index` search parameter (heretic: 0.4–0.9 of the last layer).
    direction_range: DirectionRange = .{ .fixed = .{ .low = 0.4, .high = 0.9 } },
    /// Also ablate the input side of the edited matrices (see abliterate.zig).
    ablate_inputs: bool = false,
    /// Positions of the baseline's greedy continuation the KL divergence is averaged over (1 = heretic).
    kl_tokens: usize = 1,
    /// Search with the KL divergence and the refusal-logit proxy only; the
    /// keyword scorer runs on the Pareto candidates before the results menu.
    fast_search: bool = false,
    /// Which trial the results menu lists first.
    select: Select = .pareto,
    /// λ of `select = auto`.
    select_lambda: f64 = 1.0,
    /// Prune trials whose partial refusal count already guarantees Pareto domination.
    early_stop: bool = true,
    /// Journal of a previous study whose trials seed the sampler.
    warm_start: ?[]const u8 = null,
    n_trials: usize = 200,
    n_startup_trials: usize = 60,
    seed: ?u64 = null,
    study_checkpoint_dir: []const u8 = "checkpoints",
    max_shard_size: u64 = 5 * 1024 * 1024 * 1024,
    /// Resident-memory budget in bytes for ditch-owned buffers (0 = unlimited; weights are memory-mapped).
    max_ram: u64 = 0,
    /// Accepted for CLI compatibility with heretic; unused (there is no GPU backend).
    max_vram: u64 = 0,
    /// Directory for spilled activations / KV caches (default: <cache_dir>/scratch or ./scratch).
    scratch_dir: ?[]const u8 = null,
    /// Wall-clock limit for the whole run in seconds (null = none).
    time_limit_seconds: ?u64 = null,
    /// Override of the memory reserved outside ditch-owned buffers (null = automatic:
    /// max(10% of max_ram, 256MB), at most half of max_ram).
    budget_headroom: ?u64 = null,
    /// Warp mode (streamed mixture-of-experts models): expert cache capacity in
    /// bytes. null = automatic (what the budget leaves after the trunk and a
    /// reserve for workspaces), 0 = no expert cache (whole layers are streamed).
    /// Setting it without max_ram switches to streamed weights as well.
    expert_cache: ?u64 = null,
    /// Score and edit only routed experts that calibration/evaluation prompts
    /// routed to (null = on in warp mode, off otherwise).
    visited_experts_only: ?bool = null,
    /// Read the weights of a plain Hub id through range requests instead of
    /// downloading the files (`hf://owner/name` ids always do).
    remote_weights: bool = false,
    /// Size of the chunks fetched and cached by the remote source.
    remote_chunk_size: u64 = 8 << 20,
    /// Write `<scratch_dir>/<model>.hotlist` at exit and warm the expert cache from it at start.
    hotlist: bool = true,
    system_prompt: []const u8 = "You are a helpful assistant.",
    good_prompts: DatasetSpec = .{ .dataset = "mlabonne/harmless_alpaca", .split = "train[:400]", .column = "text" },
    bad_prompts: DatasetSpec = .{ .dataset = "mlabonne/harmful_behaviors", .split = "train[:400]", .column = "text" },
    keyword_rate: KeywordRateSettings = .{},
    kl_divergence: KlDivergenceSettings = .{},
    checkpoint_action: ?[]const u8 = null,
    trial_index: ?usize = null,
    n_additional_trials: ?usize = null,
    model_action: ?[]const u8 = null,
    save_directory: ?[]const u8 = null,
    export_dtype: ?[]const u8 = null,
    /// "hf" (default), "gguf" or "both"; a GGUF input defaults to "gguf".
    export_format: ?[]const u8 = null,
    /// Storage type of the GGUF matrices: f16 (default for HF inputs), bf16,
    /// f32, q8_0, q4_0, q4_1, q5_0, q5_1 or "source" (default for GGUF inputs).
    gguf_dtype: ?[]const u8 = null,
    config_path: ?[]const u8 = null,
    /// `--reproduce <manifest>`: re-derive an exported model from its ditch-reproduce.lua.
    reproduce: ?[]const u8 = null,
    ignore_mismatches: bool = false,
    /// `ditch bench <model>`: run the benchmark harness instead of a study.
    bench: bool = false,
    bench_prompts: usize = 16,
    bench_tokens: usize = 32,
    bench_output: ?[]const u8 = null,
    help: bool = false,
    version: bool = false,
};

pub const help_text =
    \\Usage: ditch [OPTIONS] <MODEL>
    \\
    \\ditch censorship: fully automatic refusal removal for language models, in Zig.
    \\<MODEL> is a Hugging Face model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct) or a local directory.
    \\
    \\Options can also be set in config.lua (see config.default.lua); a heretic-style
    \\config.toml is accepted too. Command-line options take precedence. Every option
    \\accepts --name value or --name=value.
    \\
    \\Model & runtime:
    \\  --model <id|path>              Model to process (positional argument is equivalent).
    \\  --model-commit <sha>           Pin the model to a specific revision.
    \\  --evaluate-model <id|path>     Only evaluate this model against the base model's scorers.
    \\  --threads <n>                  Worker threads (default: number of CPUs).
    \\  --cache-dir <path>             Download cache (default: $DITCH_CACHE or ~/.cache/ditch).
    \\  --chat-template <name>         Force a chat template: chatml, llama3, llama2, mistral, gemma, raw.
    \\  --batch-size <n>               Prompts per batch (0 = auto-detect, the default).
    \\  --max-batch-size <n>           Upper bound for auto-detection (default: 32).
    \\  --max-response-length <n>      Tokens generated per response (default: 100).
    \\  --response-prefix <text>       Text appended to every prompt (default: auto-detect).
    \\  --system-prompt <text>         System prompt for all prompts.
    \\
    \\Resource budget:
    \\  --max-ram <size>               Keep resident memory under this budget, e.g. 8GB (default: unlimited;
    \\                                 weights are memory-mapped). With a budget, weights are streamed layer
    \\                                 by layer from disk and caches spill to --scratch-dir when needed.
    \\  --max-vram <size>              Accepted for compatibility; unused (CPU-only, no GPU backend).
    \\  --scratch-dir <path>           Spill directory (default: <cache-dir>/scratch or ./scratch).
    \\  --time-limit <duration>        Stop cleanly after this long, e.g. 90m, 2h, 1h30m (default: none).
    \\  --budget-headroom <size>       Memory reserved for everything ditch does not allocate itself
    \\                                 (default: max(10% of --max-ram, 256MB), at most half of it).
    \\
    \\Warp mode (mixture-of-experts models bigger than RAM; on automatically with --max-ram):
    \\  --expert-cache <size>          Bounded LRU cache of resident routed experts; the trunk streams
    \\                                 per layer, experts are fetched when the router selects them
    \\                                 (default: what --max-ram leaves after the trunk and workspaces;
    \\                                 0 disables the cache; without --max-ram it enables streaming).
    \\  --visited-experts-only <bool>  Score/edit only experts that calibration prompts routed to
    \\                                 (default: on in warp mode). Exports always copy every expert.
    \\  --hotlist <bool>, --no-hotlist Persist the hottest experts to <scratch-dir>/<model>.hotlist and
    \\                                 warm the cache from it on the next run (default: on).
    \\  hf://owner/name                Model id form that reads weights straight from the Hub with
    \\                                 HTTP range requests (no full download); chunks are cached in
    \\                                 <cache-dir>/models/<id>/<rev>/chunks. http(s)://host/path/ works too.
    \\  --remote-weights               Treat a plain Hub id like hf://<id>.
    \\  --remote-chunk-size <size>     Fetch/cache granularity of the remote source (default: 8MB).
    \\
    \\Abliteration:
    \\  --orthogonalize-direction <bool>      Project directions orthogonal to the good direction (default: true).
    \\  --row-normalization <none|pre|full>   Row normalisation mode (default: full).
    \\  --full-normalization-lora-rank <n>    Rank of the "full" approximation (default: 3).
    \\  --winsorization-quantile <q>          Clamp residual magnitudes to this quantile (default: 1.0 = off).
    \\  --n-directions <k>                    Orthonormal refusal directions removed per layer (default: 1).
    \\  --direction-method <mean|separating>  Per-layer direction: difference of means (default, heretic) or
    \\                                        the difference whitened by the per-coordinate variance.
    \\  --direction-token-window <w>          Average the last <w> prompt tokens' residuals (default: 1).
    \\  --direction-shrinkage <f>             Diagonal-covariance shrinkage of "separating" (default: 0.1).
    \\  --direction-range <lo:hi|auto>        direction_index bounds as fractions of the last layer
    \\                                        (default: 0.4:0.9), or auto = the layers whose projection
    \\                                        AUROC is within 0.01 of the best (costs one extra pass).
    \\  --ablate-inputs                       Also remove the input pattern that writes the direction from
    \\                                        every edited matrix (stronger edit, higher KL; default: off).
    \\  --expert-selection <ranked|random|broad>  MoE models: edit the experts best aligned with the
    \\                                        refusal direction (ranked, default), a random subset of the
    \\                                        same size (baseline), or always every expert (broad).
    \\
    \\Optimisation:
    \\  --n-trials <n>                 Total trials (default: 200).
    \\  --n-startup-trials <n>         Random exploration trials (default: 60).
    \\  --seed <n>                     Random seed.
    \\  --study-checkpoint-dir <path>  Where study progress is stored (default: checkpoints).
    \\  --checkpoint-action <continue|restart>  What to do with an existing checkpoint.
    \\  --early-stop <bool>, --no-early-stop  Prune trials that can no longer reach the Pareto front (default: on).
    \\  --warm-start <study.jsonl>     Seed the sampler with the trials of a previous study.
    \\  --fast-search                  Optimise KL divergence and the refusal-logit proxy (one prefill per
    \\                                 scorer); the keyword scorer runs on the Pareto candidates only.
    \\  --kl-tokens <t>                Average the KL divergence over the first <t> positions of the base
    \\                                 model's greedy continuation (default: 1; costs t x prefill tokens).
    \\  --select <pareto|auto>         List the front trial minimising refusals + lambda * KL first (auto).
    \\  --select-lambda <f>            The lambda of --select auto (default: 1).
    \\
    \\Datasets (also for --keyword-rate-* and --kl-divergence-* scorer prompts):
    \\  --good-prompts-dataset <id|file>  --good-prompts-split <s>  --good-prompts-column <c>
    \\  --bad-prompts-dataset <id|file>   --bad-prompts-split <s>   --bad-prompts-column <c>
    \\  A dataset is a Hugging Face dataset ID (rows are fetched through the datasets-server
    \\  API) or a text file with one prompt per line.
    \\
    \\Results (non-interactive use):
    \\  --trial-index <n>              Select this trial instead of asking.
    \\  --model-action <save|chat|exit>  What to do with the selected trial.
    \\  --save-directory <path>        Where to save the model with --model-action save.
    \\  --export-dtype <bf16|f16|f32>  Storage dtype for exported weights (default: same as source).
    \\  --export-format <hf|gguf|both> Export format: Hugging Face directory (default), a llama.cpp
    \\                                 GGUF file, or both (GGUF inputs default to gguf).
    \\  --gguf-dtype <f16|bf16|f32|q8_0|q4_0|q4_1|q5_0|q5_1|source>
    \\                                 Storage type of the GGUF matrices (default: f16, or the
    \\                                 source types for a GGUF input). Norms stay f32.
    \\  --n-additional-trials <n>      Run more trials after a finished study.
    \\
    \\Reproducing and benchmarking:
    \\  --reproduce <manifest.lua>     Re-derive an exported model from its ditch-reproduce.lua
    \\                                 (verifies the model file and prompt hashes, applies the
    \\                                 recorded trial, then shows the model menu; no search).
    \\  --ignore-mismatches            Proceed with --reproduce even if hashes differ.
    \\  ditch bench <model> [options]  Measure throughput, timings and memory (see README).
    \\  --bench-prompts <n>            Prompts per benchmark batch (default: 16).
    \\  --bench-tokens <n>             Tokens decoded per prompt in the benchmark (default: 32).
    \\  --bench-output <file.md>       Also write the benchmark table to this file.
    \\
    \\Other:
    \\  --print-debug-information      Print extra diagnostics.
    \\  --print-residual-geometry      Print per-layer residual geometry statistics.
    \\  --keyword-rate-print-responses Print every evaluated prompt/response pair.
    \\  --config <path>                Configuration file, .lua or .toml (default: ./config.lua, else ./config.toml).
    \\  -h, --help                     Show this help.
    \\  --version                      Print the version.
    \\
;

pub const version = "0.2.0";

pub const LoadResult = struct {
    settings: Settings,
    arena: std.heap.ArenaAllocator,
    errors: [][]const u8,

    pub fn deinit(self: *LoadResult) void {
        self.arena.deinit();
    }
};

/// Loads settings from the config file and command line.
pub fn load(gpa: Allocator, io: std.Io, args: []const []const u8) !LoadResult {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var settings = Settings{};
    var errors = std.ArrayList([]const u8).empty;

    // First pass: find --config.
    var config_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--config") and i + 1 < args.len) {
            config_path = args[i + 1];
        } else if (std.mem.startsWith(u8, args[i], "--config=")) {
            config_path = args[i]["--config=".len..];
        }
    }
    // Default: config.lua, falling back to config.toml (heretic compatibility).
    const cwd = std.Io.Dir.cwd();
    var path: []const u8 = config_path orelse "config.lua";
    if (config_path == null) {
        if (cwd.access(io, "config.lua", .{})) |_| {} else |_| {
            if (cwd.access(io, "config.toml", .{})) |_| path = "config.toml" else |_| {}
        }
    }
    if (cwd.readFileAlloc(io, path, a, .unlimited)) |text| {
        settings.config_path = try a.dupe(u8, path);
        if (std.mem.endsWith(u8, path, ".toml")) {
            var parsed = toml.parse(gpa, text) catch |err| {
                try errors.append(a, try std.fmt.allocPrint(a, "could not parse {s}: {s}", .{ path, @errorName(err) }));
                return .{ .settings = settings, .arena = arena, .errors = try errors.toOwnedSlice(a) };
            };
            defer parsed.deinit();
            try applyToml(a, &settings, parsed.root, &errors);
        } else {
            var result = try lua.parse(gpa, text, path);
            defer result.parsed.deinit();
            if (result.err) |e| {
                try errors.append(a, try std.fmt.allocPrint(a, "could not load {s}: {s}", .{ path, e }));
                return .{ .settings = settings, .arena = arena, .errors = try errors.toOwnedSlice(a) };
            }
            try applyToml(a, &settings, result.parsed.root, &errors);
        }
    } else |err| {
        if (config_path != null) try errors.append(a, try std.fmt.allocPrint(a, "could not read {s}: {s}", .{ path, @errorName(err) }));
    }

    // Second pass: command-line options.
    i = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            settings.help = true;
            continue;
        }
        if (!std.mem.startsWith(u8, arg, "--")) {
            // Positional model.
            settings.model = try a.dupe(u8, arg);
            continue;
        }
        var name = arg[2..];
        var value: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, name, '=')) |eq| {
            value = name[eq + 1 ..];
            name = name[0..eq];
        }
        const key = try normalizeKey(a, name);
        const is_bool = isBoolKey(key);
        if (value == null and !is_bool) {
            if (i + 1 < args.len) {
                i += 1;
                value = args[i];
            } else {
                try errors.append(a, try std.fmt.allocPrint(a, "option --{s} requires a value", .{name}));
                continue;
            }
        }
        if (value == null and is_bool) {
            // Allow "--flag false" style.
            if (i + 1 < args.len and (std.mem.eql(u8, args[i + 1], "true") or std.mem.eql(u8, args[i + 1], "false"))) {
                i += 1;
                value = args[i];
            } else value = "true";
        }
        applyOption(a, &settings, key, value.?) catch |err| {
            try errors.append(a, try std.fmt.allocPrint(a, "invalid value for --{s}: {s}", .{ name, @errorName(err) }));
        };
    }
    // The fast search fixes the objective set; the keyword scorer is deferred.
    if (settings.fast_search) settings.scorers = &fast_search_scorers;
    return .{ .settings = settings, .arena = arena, .errors = try errors.toOwnedSlice(a) };
}

fn normalizeKey(a: Allocator, name: []const u8) ![]u8 {
    const out = try a.dupe(u8, name);
    for (out) |*c| if (c.* == '-') {
        c.* = '_';
    };
    return out;
}

fn isBoolKey(key: []const u8) bool {
    const bools = [_][]const u8{ "print_debug_information", "print_residual_geometry", "orthogonalize_direction", "keyword_rate_print_responses", "ignore_mismatches", "early_stop", "no_early_stop", "visited_experts_only", "remote_weights", "hotlist", "no_hotlist", "ablate_inputs", "fast_search", "help", "version" };
    for (bools) |b| if (std.mem.eql(u8, b, key)) return true;
    return false;
}

fn parseBool(s: []const u8) !bool {
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "yes")) return true;
    if (std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "no")) return false;
    return error.InvalidBool;
}

/// Parses a duration such as "90m", "2h", "1h30m", "45s" or "3600" (seconds) into seconds.
pub fn parseDuration(s: []const u8) !u64 {
    const t = std.mem.trim(u8, s, " ");
    if (t.len == 0) return error.InvalidDuration;
    var total: u64 = 0;
    var i: usize = 0;
    var any = false;
    while (i < t.len) {
        var j = i;
        while (j < t.len and (std.ascii.isDigit(t[j]) or t[j] == '.')) j += 1;
        if (j == i) return error.InvalidDuration;
        const num = std.fmt.parseFloat(f64, t[i..j]) catch return error.InvalidDuration;
        var k = j;
        while (k < t.len and std.ascii.isAlphabetic(t[k])) k += 1;
        const unit = t[j..k];
        const mult: f64 = if (unit.len == 0 or std.ascii.eqlIgnoreCase(unit, "s") or std.ascii.eqlIgnoreCase(unit, "sec"))
            1
        else if (std.ascii.eqlIgnoreCase(unit, "m") or std.ascii.eqlIgnoreCase(unit, "min"))
            60
        else if (std.ascii.eqlIgnoreCase(unit, "h") or std.ascii.eqlIgnoreCase(unit, "hr"))
            3600
        else if (std.ascii.eqlIgnoreCase(unit, "d"))
            86400
        else
            return error.InvalidDuration;
        if (unit.len == 0 and k != t.len) return error.InvalidDuration;
        total += @intFromFloat(num * mult);
        any = true;
        i = k;
    }
    if (!any) return error.InvalidDuration;
    return total;
}

pub fn parseSize(s: []const u8) !u64 {
    var end = s.len;
    while (end > 0 and !std.ascii.isDigit(s[end - 1]) and s[end - 1] != '.') end -= 1;
    const num = std.fmt.parseFloat(f64, s[0..end]) catch return error.InvalidSize;
    const unit = std.mem.trim(u8, s[end..], " ");
    var mult: f64 = 1;
    if (unit.len == 0 or std.ascii.eqlIgnoreCase(unit, "B")) mult = 1 else if (std.ascii.eqlIgnoreCase(unit, "KB") or std.ascii.eqlIgnoreCase(unit, "K")) mult = 1024 else if (std.ascii.eqlIgnoreCase(unit, "MB") or std.ascii.eqlIgnoreCase(unit, "M")) mult = 1024 * 1024 else if (std.ascii.eqlIgnoreCase(unit, "GB") or std.ascii.eqlIgnoreCase(unit, "G")) mult = 1024 * 1024 * 1024 else return error.InvalidSize;
    return @intFromFloat(num * mult);
}

fn applyDatasetOption(a: Allocator, spec: *DatasetSpec, field: []const u8, value: []const u8) !void {
    const v = try a.dupe(u8, value);
    if (std.mem.eql(u8, field, "dataset")) spec.dataset = v else if (std.mem.eql(u8, field, "config")) spec.config = v else if (std.mem.eql(u8, field, "split")) spec.split = v else if (std.mem.eql(u8, field, "column")) spec.column = v else if (std.mem.eql(u8, field, "prefix")) spec.prefix = v else if (std.mem.eql(u8, field, "suffix")) spec.suffix = v else if (std.mem.eql(u8, field, "system_prompt")) spec.system_prompt = v else return error.UnknownOption;
}

fn applyOption(a: Allocator, s: *Settings, key: []const u8, value: []const u8) !void {
    const eql = std.mem.eql;
    if (eql(u8, key, "model")) s.model = try a.dupe(u8, value) else if (eql(u8, key, "model_commit")) s.model_commit = try a.dupe(u8, value) else if (eql(u8, key, "evaluate_model")) s.evaluate_model = try a.dupe(u8, value) else if (eql(u8, key, "threads")) s.threads = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "cache_dir")) s.cache_dir = try a.dupe(u8, value) else if (eql(u8, key, "chat_template")) s.chat_template = try a.dupe(u8, value) else if (eql(u8, key, "batch_size")) s.batch_size = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "max_batch_size")) s.max_batch_size = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "max_response_length")) s.max_response_length = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "response_prefix")) s.response_prefix = try a.dupe(u8, value) else if (eql(u8, key, "system_prompt")) s.system_prompt = try a.dupe(u8, value) else if (eql(u8, key, "print_debug_information")) s.print_debug_information = try parseBool(value) else if (eql(u8, key, "print_residual_geometry")) s.print_residual_geometry = try parseBool(value) else if (eql(u8, key, "orthogonalize_direction")) s.orthogonalize_direction = try parseBool(value) else if (eql(u8, key, "row_normalization")) s.row_normalization = abliterate.RowNormalization.parse(value) orelse return error.InvalidEnum else if (eql(u8, key, "full_normalization_lora_rank")) s.full_normalization_lora_rank = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "expert_selection")) s.expert_selection = abliterate.ExpertSelection.parse(value) orelse return error.InvalidEnum else if (eql(u8, key, "winsorization_quantile")) s.winsorization_quantile = try std.fmt.parseFloat(f32, value) else if (eql(u8, key, "n_directions")) {
        s.n_directions = try std.fmt.parseInt(usize, value, 10);
        if (s.n_directions == 0) return error.InvalidValue;
    } else if (eql(u8, key, "direction_method")) s.direction_method = directions.Method.parse(value) orelse return error.InvalidEnum else if (eql(u8, key, "direction_token_window")) {
        s.direction_token_window = try std.fmt.parseInt(usize, value, 10);
        if (s.direction_token_window == 0) return error.InvalidValue;
    } else if (eql(u8, key, "direction_shrinkage")) {
        s.direction_shrinkage = try std.fmt.parseFloat(f32, value);
        if (!(s.direction_shrinkage >= 0)) return error.InvalidValue;
    } else if (eql(u8, key, "direction_range")) s.direction_range = DirectionRange.parse(value) orelse return error.InvalidValue else if (eql(u8, key, "ablate_inputs")) s.ablate_inputs = try parseBool(value) else if (eql(u8, key, "kl_tokens")) {
        s.kl_tokens = try std.fmt.parseInt(usize, value, 10);
        if (s.kl_tokens == 0) return error.InvalidValue;
    } else if (eql(u8, key, "fast_search")) s.fast_search = try parseBool(value) else if (eql(u8, key, "select")) s.select = Select.parse(value) orelse return error.InvalidEnum else if (eql(u8, key, "select_lambda")) {
        s.select_lambda = try std.fmt.parseFloat(f64, value);
        if (!(s.select_lambda >= 0)) return error.InvalidValue;
    } else if (eql(u8, key, "early_stop")) s.early_stop = try parseBool(value) else if (eql(u8, key, "no_early_stop")) s.early_stop = !(try parseBool(value)) else if (eql(u8, key, "warm_start")) s.warm_start = try a.dupe(u8, value) else if (eql(u8, key, "n_trials")) s.n_trials = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "n_startup_trials")) s.n_startup_trials = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "seed")) s.seed = try std.fmt.parseInt(u64, value, 10) else if (eql(u8, key, "study_checkpoint_dir")) s.study_checkpoint_dir = try a.dupe(u8, value) else if (eql(u8, key, "max_shard_size")) s.max_shard_size = try parseSize(value) else if (eql(u8, key, "max_ram")) s.max_ram = try parseSize(value) else if (eql(u8, key, "max_vram")) s.max_vram = try parseSize(value) else if (eql(u8, key, "scratch_dir")) s.scratch_dir = try a.dupe(u8, value) else if (eql(u8, key, "time_limit")) s.time_limit_seconds = try parseDuration(value) else if (eql(u8, key, "time_limit_seconds")) s.time_limit_seconds = try std.fmt.parseInt(u64, value, 10) else if (eql(u8, key, "budget_headroom")) s.budget_headroom = try parseSize(value) else if (eql(u8, key, "expert_cache")) s.expert_cache = try parseSize(value) else if (eql(u8, key, "visited_experts_only")) s.visited_experts_only = try parseBool(value) else if (eql(u8, key, "remote_weights")) s.remote_weights = try parseBool(value) else if (eql(u8, key, "remote_chunk_size")) s.remote_chunk_size = try parseSize(value) else if (eql(u8, key, "hotlist")) s.hotlist = try parseBool(value) else if (eql(u8, key, "no_hotlist")) s.hotlist = !(try parseBool(value)) else if (eql(u8, key, "checkpoint_action")) s.checkpoint_action = try a.dupe(u8, value) else if (eql(u8, key, "trial_index")) s.trial_index = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "n_additional_trials")) s.n_additional_trials = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "model_action")) s.model_action = try a.dupe(u8, value) else if (eql(u8, key, "save_directory")) s.save_directory = try a.dupe(u8, value) else if (eql(u8, key, "export_dtype")) s.export_dtype = try a.dupe(u8, value) else if (eql(u8, key, "export_format")) s.export_format = try a.dupe(u8, value) else if (eql(u8, key, "gguf_dtype")) s.gguf_dtype = try a.dupe(u8, value) else if (eql(u8, key, "config")) {
        // handled in the first pass
    } else if (eql(u8, key, "reproduce")) s.reproduce = try a.dupe(u8, value) else if (eql(u8, key, "ignore_mismatches")) s.ignore_mismatches = try parseBool(value) else if (eql(u8, key, "bench_prompts")) s.bench_prompts = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "bench_tokens")) s.bench_tokens = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "bench_output")) s.bench_output = try a.dupe(u8, value) else if (eql(u8, key, "help")) s.help = try parseBool(value) else if (eql(u8, key, "version")) s.version = try parseBool(value) else if (eql(u8, key, "keyword_rate_print_responses")) s.keyword_rate.print_responses = try parseBool(value) else if (eql(u8, key, "keyword_rate_score_name")) s.keyword_rate.score_name = try a.dupe(u8, value) else if (std.mem.startsWith(u8, key, "good_prompts_")) try applyDatasetOption(a, &s.good_prompts, key["good_prompts_".len..], value) else if (std.mem.startsWith(u8, key, "bad_prompts_")) try applyDatasetOption(a, &s.bad_prompts, key["bad_prompts_".len..], value) else if (std.mem.startsWith(u8, key, "keyword_rate_prompts_")) try applyDatasetOption(a, &s.keyword_rate.prompts, key["keyword_rate_prompts_".len..], value) else if (std.mem.startsWith(u8, key, "kl_divergence_prompts_")) try applyDatasetOption(a, &s.kl_divergence.prompts, key["kl_divergence_prompts_".len..], value) else return error.UnknownOption;
}

fn tomlString(a: Allocator, v: toml.Value) ![]const u8 {
    return switch (v) {
        .string => |s| try a.dupe(u8, s),
        .integer => |i| try std.fmt.allocPrint(a, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(a, "{d}", .{f}),
        .boolean => |b| if (b) "true" else "false",
        else => error.InvalidType,
    };
}

fn applyDatasetTable(a: Allocator, spec: *DatasetSpec, t: *const toml.Table, errors: *std.ArrayList([]const u8)) !void {
    var it = t.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.startsWith(u8, k, "residual_plot")) continue;
        if (std.mem.eql(u8, k, "commit")) continue;
        const v = tomlString(a, e.value_ptr.*) catch {
            try errors.append(a, try std.fmt.allocPrint(a, "dataset field {s} must be a string", .{k}));
            continue;
        };
        applyDatasetOption(a, spec, k, v) catch {
            try errors.append(a, try std.fmt.allocPrint(a, "unknown dataset field: {s}", .{k}));
        };
    }
}

fn applyToml(a: Allocator, s: *Settings, root: *const toml.Table, errors: *std.ArrayList([]const u8)) !void {
    var it = root.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        const v = e.value_ptr.*;
        if (std.mem.eql(u8, k, "good_prompts") and v == .table) {
            try applyDatasetTable(a, &s.good_prompts, v.table, errors);
        } else if (std.mem.eql(u8, k, "bad_prompts") and v == .table) {
            try applyDatasetTable(a, &s.bad_prompts, v.table, errors);
        } else if (std.mem.eql(u8, k, "scorer") and v == .table) {
            var sit = v.table.map.iterator();
            while (sit.next()) |se| {
                const name = se.key_ptr.*;
                if (se.value_ptr.* != .table) continue;
                const st = se.value_ptr.table;
                if (std.mem.startsWith(u8, name, "KeywordRate")) {
                    if (st.get("score_name")) |sn| s.keyword_rate.score_name = try tomlString(a, sn);
                    if (st.get("print_responses")) |pr| s.keyword_rate.print_responses = pr == .boolean and pr.boolean;
                    if (st.get("keyword_markers")) |km| {
                        if (km == .array) {
                            var list = std.ArrayList([]const u8).empty;
                            for (km.array) |m| try list.append(a, try tomlString(a, m));
                            s.keyword_rate.keyword_markers = list.items;
                        }
                    }
                    if (st.getTable("prompts")) |pt| try applyDatasetTable(a, &s.keyword_rate.prompts, pt, errors);
                } else if (std.mem.startsWith(u8, name, "KLDivergence")) {
                    if (st.getTable("prompts")) |pt| try applyDatasetTable(a, &s.kl_divergence.prompts, pt, errors);
                } else {
                    try errors.append(a, try std.fmt.allocPrint(a, "unsupported scorer section: scorer.{s}", .{name}));
                }
            }
        } else if (std.mem.eql(u8, k, "scorers") and v == .array) {
            var list = std.ArrayList(ScorerConfig).empty;
            for (v.array) |item| {
                if (item != .table) continue;
                const plugin = try tomlString(a, item.table.get("plugin") orelse .{ .string = "" });
                const kind = ScorerKind.fromPlugin(plugin) orelse {
                    try errors.append(a, try std.fmt.allocPrint(a, "unsupported scorer plugin: {s} (available: keyword_rate, kl_divergence, refusal_logit)", .{plugin}));
                    continue;
                };
                const opt_s = try tomlString(a, item.table.get("optimization") orelse .{ .string = "none" });
                const opt: Optimization = if (std.mem.eql(u8, opt_s, "minimize")) .minimize else if (std.mem.eql(u8, opt_s, "maximize")) .maximize else .none;
                const inst = if (item.table.get("instance_name")) |n| try tomlString(a, n) else null;
                try list.append(a, .{ .kind = kind, .optimization = opt, .instance_name = inst });
            }
            s.scorers = list.items;
        } else if (std.mem.eql(u8, k, "chain_of_thought_skips") and v == .array) {
            var list = std.ArrayList([2][]const u8).empty;
            for (v.array) |pair| {
                if (pair != .array or pair.array.len != 2) continue;
                try list.append(a, .{ try tomlString(a, pair.array[0]), try tomlString(a, pair.array[1]) });
            }
            s.chain_of_thought_skips = list.items;
        } else if (std.mem.eql(u8, k, "dtypes") or std.mem.eql(u8, k, "quantization") or std.mem.eql(u8, k, "device_map") or std.mem.eql(u8, k, "max_memory") or std.mem.eql(u8, k, "offload_outputs_to_cpu") or std.mem.startsWith(u8, k, "residual_plot") or std.mem.eql(u8, k, "plot_residuals") or std.mem.eql(u8, k, "benchmarks")) {
            // Accepted for compatibility with heretic config files; ignored.
        } else {
            const str = tomlString(a, v) catch {
                try errors.append(a, try std.fmt.allocPrint(a, "unsupported config value for {s}", .{k}));
                continue;
            };
            applyOption(a, s, k, str) catch |err| {
                try errors.append(a, try std.fmt.allocPrint(a, "config option {s}: {s}", .{ k, @errorName(err) }));
            };
        }
    }
}

test "parse size" {
    try std.testing.expectEqual(@as(u64, 5 * 1024 * 1024 * 1024), try parseSize("5GB"));
    try std.testing.expectEqual(@as(u64, 1536), try parseSize("1.5KB"));
    try std.testing.expectEqual(@as(u64, 42), try parseSize("42"));
    try std.testing.expectEqual(@as(u64, 8 << 30), try parseSize("8G"));
    try std.testing.expectEqual(@as(u64, 512 << 20), try parseSize("512 MB"));
    try std.testing.expectError(error.InvalidSize, parseSize("8TB"));
    try std.testing.expectError(error.InvalidSize, parseSize("abc"));
}

test "parse duration" {
    try std.testing.expectEqual(@as(u64, 90 * 60), try parseDuration("90m"));
    try std.testing.expectEqual(@as(u64, 2 * 3600), try parseDuration("2h"));
    try std.testing.expectEqual(@as(u64, 5400), try parseDuration("1h30m"));
    try std.testing.expectEqual(@as(u64, 45), try parseDuration("45s"));
    try std.testing.expectEqual(@as(u64, 3600), try parseDuration("3600"));
    try std.testing.expectEqual(@as(u64, 90), try parseDuration("1.5min"));
    try std.testing.expectError(error.InvalidDuration, parseDuration("2x"));
    try std.testing.expectError(error.InvalidDuration, parseDuration(""));
    try std.testing.expectError(error.InvalidDuration, parseDuration("m"));
}

test "budget options" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--max-ram", "8GB", "--max-vram=24GB", "--time-limit", "90m", "--scratch-dir", "/tmp/x", "--budget-headroom", "0", "m" };
    var r = try load(gpa, std.testing.io, &args);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqual(@as(u64, 8 << 30), r.settings.max_ram);
    try std.testing.expectEqual(@as(?u64, 0), r.settings.budget_headroom);
    try std.testing.expectEqual(@as(u64, 24 << 30), r.settings.max_vram);
    try std.testing.expectEqual(@as(?u64, 5400), r.settings.time_limit_seconds);
    try std.testing.expectEqualStrings("/tmp/x", r.settings.scratch_dir.?);
    // TOML values go through the same parser.
    var parsed = try toml.parse(gpa, "max_ram = \"2GB\"\ntime_limit = \"2h\"\n");
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var s = Settings{};
    var errors = std.ArrayList([]const u8).empty;
    try applyToml(arena.allocator(), &s, parsed.root, &errors);
    try std.testing.expectEqual(@as(usize, 0), errors.items.len);
    try std.testing.expectEqual(@as(u64, 2 << 30), s.max_ram);
    try std.testing.expectEqual(@as(?u64, 7200), s.time_limit_seconds);
}

test "warp mode options" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--expert-cache", "512MB", "--visited-experts-only", "false", "--remote-weights", "--remote-chunk-size=1MB", "--no-hotlist", "hf://Qwen/Qwen3-30B-A3B" };
    var r = try load(gpa, std.testing.io, &args);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqual(@as(?u64, 512 << 20), r.settings.expert_cache);
    try std.testing.expectEqual(@as(?bool, false), r.settings.visited_experts_only);
    try std.testing.expect(r.settings.remote_weights);
    try std.testing.expectEqual(@as(u64, 1 << 20), r.settings.remote_chunk_size);
    try std.testing.expect(!r.settings.hotlist);
    try std.testing.expectEqualStrings("hf://Qwen/Qwen3-30B-A3B", r.settings.model);
    const defaults = Settings{};
    try std.testing.expectEqual(@as(?u64, null), defaults.expert_cache);
    try std.testing.expectEqual(@as(?bool, null), defaults.visited_experts_only);
    try std.testing.expect(defaults.hotlist);
}

test "cli parsing" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--n-trials", "5", "--row-normalization=pre", "--print-debug-information", "--expert-selection", "random", "--good-prompts-dataset", "good.txt", "--n-directions", "2", "--no-early-stop", "--warm-start", "old.jsonl", "Qwen/Qwen2.5-0.5B-Instruct" };
    var r = try load(gpa, std.testing.io, &args);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqual(@as(usize, 5), r.settings.n_trials);
    try std.testing.expectEqual(abliterate.ExpertSelection.random, r.settings.expert_selection);
    try std.testing.expectEqual(abliterate.RowNormalization.pre, r.settings.row_normalization);
    try std.testing.expect(r.settings.print_debug_information);
    try std.testing.expectEqualStrings("good.txt", r.settings.good_prompts.dataset);
    try std.testing.expectEqualStrings("Qwen/Qwen2.5-0.5B-Instruct", r.settings.model);
    try std.testing.expectEqual(@as(usize, 2), r.settings.n_directions);
    try std.testing.expect(!r.settings.early_stop);
    try std.testing.expectEqualStrings("old.jsonl", r.settings.warm_start.?);
    try std.testing.expectEqual(ScorerKind.kl_divergence, r.settings.scorers[0].kind);
}

test "algorithm options" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--direction-method", "separating", "--direction-token-window=3", "--direction-range", "auto", "--ablate-inputs", "--kl-tokens", "2", "--select", "auto", "--select-lambda", "0.5", "--fast-search", "m" };
    var r = try load(gpa, std.testing.io, &args);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqual(directions.Method.separating, r.settings.direction_method);
    try std.testing.expectEqual(@as(usize, 3), r.settings.direction_token_window);
    try std.testing.expectEqual(DirectionRange.auto, r.settings.direction_range);
    try std.testing.expect(r.settings.ablate_inputs);
    try std.testing.expectEqual(@as(usize, 2), r.settings.kl_tokens);
    try std.testing.expectEqual(Select.auto, r.settings.select);
    try std.testing.expectEqual(@as(f64, 0.5), r.settings.select_lambda);
    try std.testing.expect(r.settings.fast_search);
    try std.testing.expectEqual(@as(usize, 2), r.settings.scorers.len);
    try std.testing.expectEqual(ScorerKind.refusal_logit, r.settings.scorers[1].kind);
    // Defaults are heretic's.
    const d = Settings{};
    try std.testing.expectEqual(directions.Method.mean, d.direction_method);
    try std.testing.expectEqual(@as(usize, 1), d.direction_token_window);
    try std.testing.expectEqual(@as(f64, 0.4), d.direction_range.fixed.low);
    try std.testing.expect(!d.ablate_inputs and !d.fast_search);
    try std.testing.expectEqual(@as(usize, 1), d.kl_tokens);
    // Range parsing.
    try std.testing.expectEqual(@as(f64, 0.75), DirectionRange.parse("0.25:0.75").?.fixed.high);
    try std.testing.expect(DirectionRange.parse("0.9:0.4") == null);
    try std.testing.expect(DirectionRange.parse("half") == null);
    const bad = [_][]const u8{ "ditch", "--kl-tokens", "0", "m" };
    var rb = try load(gpa, std.testing.io, &bad);
    defer rb.deinit();
    try std.testing.expectEqual(@as(usize, 1), rb.errors.len);
    // A Lua config selects the scorer by plugin name.
    var result = try lua.parse(gpa, "return { scorers = { { plugin = \"refusal_logit\", optimization = \"minimize\" } }, direction_range = \"auto\" }", "test.lua");
    defer result.parsed.deinit();
    try std.testing.expect(result.err == null);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var s = Settings{};
    var errors = std.ArrayList([]const u8).empty;
    try applyToml(arena.allocator(), &s, result.parsed.root, &errors);
    try std.testing.expectEqual(@as(usize, 0), errors.items.len);
    try std.testing.expectEqual(ScorerKind.refusal_logit, s.scorers[0].kind);
    try std.testing.expectEqual(DirectionRange.auto, s.direction_range);
}
