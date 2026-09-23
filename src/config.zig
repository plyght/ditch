//! Settings: defaults, `config.lua` and command-line parsing.

const std = @import("std");
const tree = @import("tree.zig");
const lua = @import("lua.zig");
const abliterate = @import("abliterate.zig");
const directions = @import("directions.zig");
const compute = @import("compute.zig");

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
    /// Writes the refusal directions and the good / bad residual means to this
    /// safetensors file after calibration (for checking an edit independently).
    dump_directions: ?[]const u8 = null,
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
    /// Accepted for CLI compatibility with heretic; unused (the memory budget
    /// of a GPU backend is `gpu_memory`).
    max_vram: u64 = 0,
    /// Compute backend: "cpu" (the default and the reference), "metal", or
    /// "auto" to probe for a GPU and fall back to the CPU. Also DITCH_DEVICE.
    device: []const u8 = "cpu",
    /// Device memory a GPU backend may keep hot weights in (0 = upload every
    /// tile, compute and drop it). Ignored by the CPU backend.
    gpu_memory: u64 = 0,
    /// Matrix products below this many multiply-accumulates stay on the CPU even
    /// with a GPU backend (null = the backend's default). 0 sends every one to
    /// the device, which is what a test needs to be sure the GPU did the work.
    device_min_macs: ?u64 = null,
    /// `ditch selftest`: check the selected backend against the CPU reference
    /// kernels instead of running a study.
    selftest: bool = false,
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
    /// Concurrent range requests of the remote source (0 = its default, 16).
    remote_connections: u32 = 0,
    /// Bound of the remote source's on-disk chunk cache; least recently used
    /// chunks are evicted to stay under it (trunk chunks last). null = half of
    /// the free space on the cache filesystem plus what is already cached, at
    /// most 64GB; 0 = keep nothing on disk (fetch, use, discard).
    remote_cache_size: ?u64 = null,
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
    /// `ditch bench --kernels`: per-kernel throughput only, without a model.
    bench_kernels: bool = false,
    /// Use Apple's Accelerate framework for prefill-shaped matrix products
    /// (macOS builds only; ignored elsewhere). Turn it off to compare the Zig
    /// kernels against it, or for a bit-for-bit reproducibility check.
    accelerate: bool = true,
    /// `ditch probe <model> --prompt TEXT`: show tokens, first-token logits and the greedy reply.
    probe: bool = false,
    probe_prompts: []const []const u8 = &.{},
    /// Feed `--prompt` texts verbatim (no chat template, no BOS).
    probe_raw: bool = false,
    /// Also report the last token's residual at every layer (`--residuals`),
    /// so a reference implementation can be compared layer by layer.
    probe_residuals: bool = false,
    /// `ditch truncate <model> <K> <out>`: write a checkpoint of a few decoder
    /// layers (see truncate.zig) instead of running a study.
    truncate: bool = false,
    /// `--layers 0,1,5`: the layers `ditch truncate` keeps, in this order.
    truncate_layers: ?[]const u8 = null,
    /// `--kinds`: keep the fewest layers that cover every layer kind of the model.
    truncate_kinds: bool = false,
    /// `--drop PREFIX` (repeatable): leave out tensors whose name starts with it.
    truncate_drop: []const []const u8 = &.{},
    /// `--rows NAME=FILE` (repeatable): write only these rows of a table.
    truncate_rows: []const []const u8 = &.{},
    /// Positional arguments after the model (the layer count and output
    /// directory of `ditch truncate`; an error for every other command).
    positionals: []const []const u8 = &.{},
    help: bool = false,
    version: bool = false,
    /// Print only trial results, scores, menus and errors: no banner and no progress lines.
    quiet: bool = false,
    /// Append one JSON object per completed trial to this file.
    json_log: ?[]const u8 = null,
    /// Stop after loading the model and printing the memory estimate.
    dry_run: bool = false,
    /// Never prompt: fail with the flag to pass instead.
    no_input: bool = false,
    /// Prompt even when stdin is not a terminal (scripted answers).
    interactive: bool = false,
    /// Overwrite a non-empty save directory without asking.
    force: bool = false,
    /// Results, evaluation or benchmark as one JSON document on stdout.
    json: bool = false,
    /// No tables and no colour on stdout.
    plain: bool = false,
    /// Never use ANSI colour.
    no_color: bool = false,
    /// File holding the Hugging Face token (default: HF_TOKEN, then ~/.cache/huggingface/token).
    token_file: ?[]const u8 = null,
    /// Connect / stall timeout of downloads in seconds.
    http_timeout_seconds: u64 = 30,
    /// `ditch help <topic>`.
    help_topic: ?[]const u8 = null,
    /// `ditch add-model <model>`: draft a Lua model definition from a checkpoint (docs/models.md).
    add_model: bool = false,
    /// Directory of Lua model definitions, read after $XDG_CONFIG_HOME/ditch/models.
    models_dir: ?[]const u8 = null,

    /// The parsed `--device`, or null when it names no known backend.
    pub fn deviceKind(self: *const Settings) ?compute.Kind {
        return compute.Kind.parse(self.device);
    }
};

/// One heading and its body of the help text; `writeHelp` renders the
/// headings in bold on a terminal.
pub const HelpSection = struct { title: []const u8, body: []const u8 };

pub const tagline = "Fully automatic refusal removal for open-weight language models, on a CPU.";
pub const issues_url = "https://github.com/plyght/ditch/issues";

pub const usage_text =
    \\  ditch [OPTIONS] <MODEL>          run the abliteration study on a model
    \\  ditch bench [OPTIONS] <MODEL>    measure throughput, timings and memory
    \\  ditch probe [OPTIONS] <MODEL> --prompt TEXT   show tokens, first-token logits, greedy reply
    \\  ditch verify [OPTIONS] <MODEL>   check it against the official implementation (see ditch verify --help)
    \\  ditch selftest [--device D]      check a compute backend against the CPU reference kernels
    \\  ditch truncate <MODEL> <K> <OUT> write a checkpoint of the first K decoder layers
    \\  ditch add-model <MODEL>          draft a Lua model definition for a model_type ditch does not know
    \\  ditch help [bench]               this help (or the benchmark options)
    \\
    \\<MODEL> is a Hugging Face model id (Qwen/Qwen2.5-0.5B-Instruct), a local directory, a .gguf
    \\file or hf://owner/name (weights fetched on demand). A local directory named like a
    \\subcommand must be given as a path, e.g. ./bench.
    \\
;

pub const examples_text =
    \\  ditch Qwen/Qwen2.5-0.5B-Instruct
    \\      Full run: download, calibrate, search 200 trials, then choose a trial to save or chat with.
    \\  ditch Qwen/Qwen2.5-0.5B-Instruct --n-trials 40 --trial-index 1 --model-action save -o out/
    \\      Non-interactive: 40 trials, save the best Pareto trial to out/.
    \\  ditch hf://Qwen/Qwen3-30B-A3B --max-ram 12GB --expert-cache 6GB --time-limit 4h
    \\      A model bigger than RAM: stream weights, keep a bounded expert cache, stop cleanly after 4 h.
    \\  ditch bench Qwen/Qwen2.5-0.5B-Instruct --json > bench.json
    \\      Benchmark, machine-readable.
    \\
;

pub const help_sections = [_]HelpSection{
    .{ .title = "Common options", .body =
    \\  --n-trials <n>                 Total trials (default: 200; use far fewer on a CPU).
    \\  --batch-size <n>               Prompts per batch (0 = auto-detect, the default).
    \\  --max-ram <size>               Keep resident memory under this budget, e.g. 8GB.
    \\  -o, --output <path>            Where to save the model (= --save-directory).
    \\  --trial-index <n>, --model-action <save|chat|exit>   Answer the result menus non-interactively.
    \\  -n, --dry-run                  Stop after loading the model and printing the memory estimate.
    \\  -q, --quiet                    Only trial results, scores and errors (no banner, no progress).
    \\  --json                         Results, evaluation or benchmark as one JSON document on stdout.
    \\  --config <path>                Lua config file (default: ~/.config/ditch/config.lua).
    \\
    },
    .{ .title = "Model and runtime", .body =
    \\  --model <id|path>              Model to process (positional argument is equivalent).
    \\  --model-commit <sha>           Pin the model to a specific revision.
    \\  --evaluate-model <id|path>     Only evaluate this model against the base model's scorers.
    \\  --threads <n>                  Worker threads (default: number of CPUs; also DITCH_THREADS).
    \\  --cache-dir <path>             Download cache (default: $DITCH_CACHE or ~/.cache/ditch).
    \\  --chat-template <name>         Use a built-in chat format (chatml, llama3, gemma, raw, ...), not the model's template.
    \\  --device <auto|cpu|metal>      Compute backend (default: cpu, the reference implementation;
    \\                                 auto probes for a GPU and falls back to the CPU with a note;
    \\                                 metal needs a -Dmetal build on Apple silicon). Also DITCH_DEVICE.
    \\  --gpu-memory <size>            Device memory a GPU backend may keep hot weights in, e.g. 4GB
    \\                                 (default: 0 = upload each weight tile, compute, drop it).
    \\                                 Also DITCH_GPU_MEMORY.
    \\  --device-min-macs <n>          Smaller matrix products stay on the CPU (default 1048576; 0 = all
    \\                                 on the device). Also DITCH_DEVICE_MIN_MACS.
    \\  --max-batch-size <n>           Upper bound for batch-size auto-detection (default: 32).
    \\  --max-response-length <n>      Tokens generated per response (default: 100).
    \\  --response-prefix <text>       Text appended to every prompt (default: auto-detect).
    \\  --system-prompt <text>         System prompt for all prompts.
    \\
    },
    .{ .title = "Resource budget", .body =
    \\  --max-ram <size>               Keep resident memory under this budget, e.g. 8GB (default: unlimited;
    \\                                 weights are memory-mapped). With a budget, weights are streamed layer
    \\                                 by layer from disk and caches spill to --scratch-dir when needed.
    \\                                 Also DITCH_MAX_RAM.
    \\  --max-vram <size>              Accepted for compatibility; unused (see --device, --gpu-memory).
    \\  --scratch-dir <path>           Spill directory (default: <cache-dir>/scratch, else $TMPDIR/ditch-scratch).
    \\  --time-limit <duration>        Stop cleanly after this long, e.g. 90m, 2h, 1h30m (default: none).
    \\  --budget-headroom <size>       Memory reserved for everything ditch does not allocate itself
    \\                                 (default: max(10% of --max-ram, 256MB), at most half of it).
    \\
    },
    .{ .title = "Warp mode (mixture-of-experts models bigger than RAM; on automatically with --max-ram)", .body =
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
    \\  --remote-connections <n>       Range requests in flight at once (default: 16).
    \\  --remote-cache-size <size>     Disk bound of the chunk cache; least recently used chunks are
    \\                                 evicted, the trunk last (default: half the free disk space plus
    \\                                 what is cached, at most 64GB; 0 keeps nothing on disk).
    \\                                 Also DITCH_REMOTE_CACHE_SIZE.
    \\
    },
    .{ .title = "Abliteration", .body =
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
    },
    .{ .title = "Optimisation", .body =
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
    },
    .{ .title = "Datasets (also for --keyword-rate-* and --kl-divergence-* scorer prompts)", .body =
    \\  --good-prompts-dataset <id|file>  --good-prompts-split <s>  --good-prompts-column <c>
    \\  --bad-prompts-dataset <id|file>   --bad-prompts-split <s>   --bad-prompts-column <c>
    \\  A dataset is a Hugging Face dataset ID (rows are fetched through the datasets-server
    \\  API) or a text file with one prompt per line.
    \\
    },
    .{ .title = "Results (non-interactive use)", .body =
    \\  --trial-index <n>              Select this trial instead of asking.
    \\  --model-action <save|chat|exit>  What to do with the selected trial.
    \\  --save-directory <path>        Where to save the model with --model-action save (also -o, --output).
    \\  -f, --force                    Overwrite a non-empty save directory without asking.
    \\  --export-dtype <bf16|f16|f32>  Storage dtype for exported weights (default: same as source).
    \\  --export-format <hf|gguf|both> Export format: Hugging Face directory (default), a llama.cpp
    \\                                 GGUF file, or both (GGUF inputs default to gguf).
    \\  --gguf-dtype <f16|bf16|f32|q8_0|q4_0|q4_1|q5_0|q5_1|source>
    \\                                 Storage type of the GGUF matrices (default: f16, or the
    \\                                 source types for a GGUF input). Norms stay f32.
    \\  --n-additional-trials <n>      Run more trials after a finished study.
    \\
    },
    .{ .title = "Reproducing and benchmarking", .body =
    \\  --reproduce <manifest.lua>     Re-derive an exported model from its ditch-reproduce.lua
    \\                                 (verifies the model file and prompt hashes, applies the
    \\                                 recorded trial, then shows the model menu; no search).
    \\  --ignore-mismatches            Proceed with --reproduce even if hashes differ.
    \\  ditch bench <model> [options]  Measure throughput, timings and memory (see README).
    \\  --bench-prompts <n>            Prompts per benchmark batch (default: 16).
    \\  --bench-tokens <n>             Tokens decoded per prompt in the benchmark (default: 32).
    \\  --bench-output <file.md>       Also write the benchmark table to this file.
    \\  --kernels                      Per-kernel throughput only, no model needed: matmul, matvec,
    \\                                 attention, activation and weight conversion.
    \\  --accelerate <bool>, --no-accelerate
    \\                                 Use Apple's Accelerate framework for batched matrix products
    \\                                 on macOS (default: on where it is built in).
    \\  ditch probe <model> --prompt TEXT [--prompt TEXT ...] [--raw] [--residuals]
    \\                                 Print the rendered prompt, token ids, the top first-token
    \\                                 logits and the greedy reply (--json: the full logit vector),
    \\                                 for checking against tools/probe_reference.py. --residuals
    \\                                 adds the last token's residual at every layer (its norm in
    \\                                 text mode), which locates the layer a forward pass diverges at.
    \\  ditch selftest [--device D]    Check every kernel of a compute backend against the CPU
    \\                                 reference on random inputs and a sweep of shapes, printing
    \\                                 the largest absolute and relative error per kernel (--json
    \\                                 for machine-readable output). Exit 1 when a kernel is
    \\                                 outside its tolerance, 2 when the device is unavailable.
    \\  ditch truncate <model> <K> <out> [--layers 0,1,5] [--kinds] [--drop PREFIX] [--rows NAME=FILE]
    \\                                 Write a checkpoint of a few decoder layers of a model into
    \\                                 <out>, range-reading only the tensors it keeps (nothing else
    \\                                 is downloaded; quantisation stays as stored). <K> keeps layers
    \\                                 0..K-1; --layers keeps those layers, renumbered in that order;
    \\                                 --kinds the fewest layers covering every layer kind. config.json
    \\                                 gets the kept count and every per-layer list cut to match.
    \\                                 --drop leaves out tensors whose name starts with PREFIX (e.g.
    \\                                 mtp.); --rows writes only the rows of tensor NAME listed in FILE
    \\                                 (one index per line) into a sparse file of the full size.
    \\
    },
    .{ .title = "Output and interaction", .body =
    \\  -q, --quiet                    Only trial results, scores and errors (no banner, no progress).
    \\  --json                         Print the results, evaluation or benchmark as one JSON document on
    \\                                 stdout; everything else goes to stderr.
    \\  --json-log <file>              Append one JSON object per trial (parameters, scores, losses, timing).
    \\  --plain                        No tables, no colour: one "key: value" or one trial per line.
    \\  --no-color                     Never use ANSI colour (also NO_COLOR, DITCH_NO_COLOR, TERM=dumb;
    \\                                 FORCE_COLOR turns it on).
    \\  --no-input                     Never prompt; fail with the flag to pass instead.
    \\  --interactive                  Prompt even when stdin is not a terminal (scripted answers).
    \\  -d, --debug                    Print extra diagnostics (= --print-debug-information).
    \\  --print-residual-geometry      Print per-layer residual geometry statistics.
    \\  --dump-directions <file>       Write the refusal directions and the good / bad residual means
    \\                                 ([entries][hidden] f32, entry 0 = embedding) to a safetensors file.
    \\  --keyword-rate-print-responses Print every evaluated prompt/response pair.
    \\
    },
    .{ .title = "Network and secrets", .body =
    \\  --token-file <path>            Read the Hugging Face token from this file (default: HF_TOKEN,
    \\                                 then ~/.cache/huggingface/token). A token is never taken as a
    \\                                 flag value and never printed.
    \\  --http-timeout <duration>      Connect / stall timeout of downloads (default: 30s); transient
    \\                                 failures are retried three times and downloads resume .part files.
    \\
    },
    .{ .title = "Configuration", .body =
    \\  --config <path>                Lua config file to use instead of ~/.config/ditch/config.lua
    \\                                 ($XDG_CONFIG_HOME/ditch/config.lua). The installer puts
    \\                                 config.default.lua, every option documented, beside it.
    \\  Per-model settings go in ~/.config/ditch/configs/<org>/<name>.lua (for example
    \\  configs/Qwen/Qwen3-8B.lua; configs/<name>.lua for a local model) and apply on top of the
    \\  global file whenever that model is run.
    \\  --models-dir <dir>             Also read Lua model definitions from this directory (after
    \\                                 $XDG_CONFIG_HOME/ditch/models; see docs/models.md).
    \\  Precedence: flags > DITCH_* environment variables (DITCH_THREADS, DITCH_MAX_RAM, DITCH_CACHE,
    \\  DITCH_DEVICE, DITCH_GPU_MEMORY, DITCH_REMOTE_CACHE_SIZE, DITCH_NO_COLOR) > the model's config file > the global config file.
    \\  Every option accepts --name value or --name=value; flags and subcommands may come in any order.
    \\
    },
    .{ .title = "Exit codes", .body =
    \\  0  success (also --dry-run and a clean stop at --time-limit)
    \\  1  failure
    \\  2  usage error, or the memory budget is too small for the model
    \\
    },
    .{ .title = "Other", .body =
    \\  -h, --help                     Show this help (ditch help bench: the benchmark options).
    \\  --version                      Print the version, the Zig version and the target.
    \\  Supported model families, with their caveats: docs/models.md.
    \\
    },
};

pub const bench_help_text =
    \\  ditch bench [OPTIONS] <MODEL>
    \\
    \\Measures prefill and decode throughput, the residual-mean pass, the abliteration apply time
    \\and one full trial with the same model loading, prompt datasets and scorers as a study.
    \\
    \\  --bench-prompts <n>            Prompts per benchmark batch (default: 16).
    \\  --bench-tokens <n>             Tokens decoded per prompt in the benchmark (default: 32).
    \\  --bench-output <file.md>       Also write the benchmark table to this file.
    \\  --kernels                      Per-kernel throughput only, no model needed: matmul, matvec,
    \\                                 attention, activation and weight conversion.
    \\  --accelerate <bool>, --no-accelerate
    \\                                 Use Apple's Accelerate framework for batched matrix products
    \\                                 on macOS (default: on where it is built in).
    \\  --json                         Print the results as one JSON document on stdout.
    \\  --plain                        One "metric: value" line per row instead of a Markdown table.
    \\  --threads <n>, --batch-size <n>, --max-ram <size>, --expert-cache <size> and the dataset
    \\  options apply as for a study; see ditch --help.
    \\
;

/// The whole help as plain text (the option table for suggestions is scanned from it).
pub const help_text = "Usage:\n" ++ usage_text ++ "\n" ++ tagline ++ "\n\nExamples:\n" ++ examples_text ++ blk: {
    var t: []const u8 = "";
    for (help_sections) |s| t = t ++ "\n" ++ s.title ++ ":\n" ++ s.body;
    break :blk t;
} ++ "\nReport issues at " ++ issues_url ++ "\n";

const ansi_bold = "\x1b[1m";
const ansi_reset = "\x1b[0m";

fn writeHeading(w: *std.Io.Writer, title: []const u8, bold: bool) !void {
    if (bold) try w.writeAll(ansi_bold);
    try w.writeAll(title);
    try w.writeAll(":");
    if (bold) try w.writeAll(ansi_reset);
    try w.writeAll("\n");
}

/// The full help; `bold` renders headings with ANSI bold.
pub fn writeHelp(w: *std.Io.Writer, bold: bool) !void {
    try writeHeading(w, "Usage", bold);
    try w.writeAll(usage_text);
    try w.print("\n{s}\n\n", .{tagline});
    try writeHeading(w, "Examples", bold);
    try w.writeAll(examples_text);
    for (help_sections) |s| {
        try w.writeAll("\n");
        try writeHeading(w, s.title, bold);
        try w.writeAll(s.body);
    }
    try w.print("\nReport issues at {s}\n", .{issues_url});
}

/// The short help printed when ditch is run without arguments.
pub fn writeConciseHelp(w: *std.Io.Writer, bold: bool) !void {
    try w.print("ditch {s}: {c}{s}\n\n", .{ version, std.ascii.toLower(tagline[0]), tagline[1..] });
    try writeHeading(w, "Usage", bold);
    try w.writeAll(usage_text);
    try w.writeAll("\n");
    try writeHeading(w, "Examples", bold);
    try w.writeAll(examples_text);
    try w.writeAll("\n");
    try writeHeading(w, help_sections[0].title, bold);
    try w.writeAll(help_sections[0].body);
    try w.writeAll("\nRun ditch --help for all options.\n");
}

pub fn writeBenchHelp(w: *std.Io.Writer, bold: bool) !void {
    try writeHeading(w, "Usage", bold);
    try w.writeAll(bench_help_text);
}

/// Every `--option` named in the help text, in order of appearance (with repeats).
pub fn knownOptions(a: Allocator) ![][]const u8 {
    var list = std.ArrayList([]const u8).empty;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, help_text, i, "--")) |p| {
        var e = p + 2;
        while (e < help_text.len and (std.ascii.isAlphanumeric(help_text[e]) or help_text[e] == '-')) e += 1;
        if (e > p + 2) try list.append(a, help_text[p + 2 .. e]);
        i = e;
    }
    return list.toOwnedSlice(a);
}

/// Levenshtein distance.
pub fn editDistance(a: []const u8, b: []const u8) usize {
    var prev: [128]usize = undefined;
    var cur: [128]usize = undefined;
    if (a.len >= prev.len or b.len >= prev.len) return @max(a.len, b.len);
    for (0..b.len + 1) |j| prev[j] = j;
    for (a, 0..) |ca, i| {
        cur[0] = i + 1;
        for (b, 0..) |cb, j| {
            const cost: usize = if (ca == cb) 0 else 1;
            cur[j + 1] = @min(@min(prev[j + 1] + 1, cur[j] + 1), prev[j] + cost);
        }
        @memcpy(prev[0 .. b.len + 1], cur[0 .. b.len + 1]);
    }
    return prev[b.len];
}

/// The closest known option to a misspelt `--name` (without dashes), if any is close.
pub fn suggestOption(a: Allocator, name: []const u8) !?[]const u8 {
    const options = try knownOptions(a);
    var best: ?[]const u8 = null;
    var best_d: usize = std.math.maxInt(usize);
    const limit = @max(2, name.len / 3);
    for (options) |o| {
        const d = editDistance(name, o);
        if (d < best_d) {
            best_d = d;
            best = o;
        }
        if (d == 0) break;
    }
    if (best_d <= limit) return best;
    // A prefix of a longer option (--good-prompts for --good-prompts-dataset).
    for (options) |o| if (std.mem.startsWith(u8, o, name)) return o;
    return null;
}

pub const version = "0.5.0";

pub const LoadResult = struct {
    settings: Settings,
    arena: std.heap.ArenaAllocator,
    errors: [][]const u8,

    pub fn deinit(self: *LoadResult) void {
        self.arena.deinit();
    }
};

/// Applies one Lua configuration file; a missing file is only an error when
/// it was named explicitly. Returns false when the file does not load.
fn applyConfigFile(gpa: Allocator, io: std.Io, a: Allocator, settings: *Settings, errors: *std.ArrayList([]const u8), path: []const u8, explicit: bool) !bool {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited) catch |err| {
        if (explicit) try errors.append(a, try std.fmt.allocPrint(a, "could not read {s}: {s}", .{ path, @errorName(err) }));
        return true;
    };
    settings.config_path = if (settings.config_path) |prev| try std.fmt.allocPrint(a, "{s}, {s}", .{ prev, path }) else try a.dupe(u8, path);
    var result = try lua.parse(gpa, text, path);
    defer result.parsed.deinit();
    if (result.err) |e| {
        try errors.append(a, try std.fmt.allocPrint(a, "could not load {s}: {s}", .{ path, e }));
        return false;
    }
    try applyTable(a, settings, result.parsed.root, errors);
    return true;
}

/// The directory of the user's configuration: $XDG_CONFIG_HOME/ditch, else
/// ~/.config/ditch (%USERPROFILE%\.config\ditch on Windows).
pub fn configDir(a: Allocator, env: *const std.process.Environ.Map) !?[]const u8 {
    if (env.get("XDG_CONFIG_HOME")) |x| if (x.len > 0) return try std.fs.path.join(a, &.{ x, "ditch" });
    const home = env.get("HOME") orelse env.get("USERPROFILE") orelse return null;
    return try std.fs.path.join(a, &.{ home, ".config", "ditch" });
}

pub const subcommands = [_][]const u8{ "bench", "probe", "verify", "selftest", "truncate", "help", "add-model" };

/// Parses the configuration: the global config file (--config, else
/// ~/.config/ditch/config.lua), the model's own file in
/// ~/.config/ditch/configs/ (see `modelConfigName`), the DITCH_* environment
/// variables and finally the command line, later sources taking precedence.
/// `-h`/`--help` anywhere wins over every error.
pub fn load(gpa: Allocator, io: std.Io, args: []const []const u8, environ: ?*std.process.Environ.Map) !LoadResult {
    var first = try loadLayers(gpa, io, args, environ, null);
    // The model is only known once every source is read; when it has a
    // config file of its own, read everything again with that file layered in.
    const env = environ orelse return first;
    if (first.settings.model.len == 0) return first;
    const a = first.arena.allocator();
    const dir = try configDir(a, env) orelse return first;
    const name = modelConfigName(first.settings.model) orelse return first;
    const path = try std.fmt.allocPrint(a, "{s}{c}configs{c}{s}.lua", .{ dir, std.fs.path.sep, std.fs.path.sep, name });
    std.Io.Dir.cwd().access(io, path, .{}) catch return first;
    const per_model = try gpa.dupe(u8, path);
    defer gpa.free(per_model);
    first.deinit();
    return loadLayers(gpa, io, args, environ, per_model);
}

/// The name of a model's own config file under ~/.config/ditch/configs,
/// without `.lua`: the Hub id for a Hub model ("Qwen/Qwen3-8B", so the file
/// is configs/Qwen/Qwen3-8B.lua), the last path component for a local
/// directory or file ("./out/my-model" and "x/my-model.gguf" give "my-model").
pub fn modelConfigName(model: []const u8) ?[]const u8 {
    var m = model;
    if (std.mem.startsWith(u8, m, "hf://")) m = m["hf://".len..];
    if (std.mem.indexOfScalar(u8, m, '@')) |at| m = m[0..at];
    m = std.mem.trimEnd(u8, m, "/\\");
    if (std.mem.endsWith(u8, m, ".gguf")) m = std.fs.path.basename(m)[0 .. std.fs.path.basename(m).len - ".gguf".len];
    const local = m.len > 0 and (m[0] == '/' or m[0] == '.' or m[0] == '~' or m[0] == '\\' or
        std.mem.indexOfScalar(u8, m, '\\') != null or std.mem.count(u8, m, "/") != 1);
    if (local) m = std.fs.path.basename(m);
    if (m.len == 0 or std.mem.indexOf(u8, m, "..") != null) return null;
    return m;
}

fn loadLayers(gpa: Allocator, io: std.Io, args: []const []const u8, environ: ?*std.process.Environ.Map, per_model: ?[]const u8) !LoadResult {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var settings = Settings{};
    var errors = std.ArrayList([]const u8).empty;

    // First pass: --config and --help anywhere.
    var config_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--config") and i + 1 < args.len) {
            config_path = args[i + 1];
        } else if (std.mem.startsWith(u8, args[i], "--config=")) {
            config_path = args[i]["--config=".len..];
        } else if (std.mem.eql(u8, args[i], "-h") or std.mem.eql(u8, args[i], "--help")) {
            settings.help = true;
        }
    }
    var path: ?[]const u8 = config_path;
    if (path == null) if (environ) |env| if (try configDir(a, env)) |dir| {
        path = try std.fs.path.join(a, &.{ dir, "config.lua" });
    };
    if (path) |p| if (!try applyConfigFile(gpa, io, a, &settings, &errors, p, config_path != null))
        return .{ .settings = settings, .arena = arena, .errors = try errors.toOwnedSlice(a) };
    if (per_model) |p| if (!try applyConfigFile(gpa, io, a, &settings, &errors, p, true))
        return .{ .settings = settings, .arena = arena, .errors = try errors.toOwnedSlice(a) };

    // Environment variables.
    if (environ) |env| {
        const vars = [_][2][]const u8{ .{ "DITCH_THREADS", "threads" }, .{ "DITCH_MAX_RAM", "max_ram" }, .{ "DITCH_DEVICE", "device" }, .{ "DITCH_GPU_MEMORY", "gpu_memory" }, .{ "DITCH_DEVICE_MIN_MACS", "device_min_macs" }, .{ "DITCH_REMOTE_CACHE_SIZE", "remote_cache_size" } };
        for (vars) |v| if (env.get(v[0])) |value| {
            applyOption(a, &settings, v[1], value) catch |err| {
                try errors.append(a, try std.fmt.allocPrint(a, "invalid value for {s}: {s}", .{ v[0], @errorName(err) }));
            };
        };
        if (env.get("DITCH_NO_COLOR")) |v| if (v.len > 0) {
            settings.no_color = true;
        };
    }

    // Second pass: command-line options.
    var expect_help_topic = false;
    var positionals = std.ArrayList([]const u8).empty;
    i = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            settings.help = true;
            continue;
        }
        if (arg.len == 2 and arg[0] == '-' and arg[1] != '-') {
            // Short aliases.
            switch (arg[1]) {
                'q' => settings.quiet = true,
                'n' => settings.dry_run = true,
                'd' => settings.print_debug_information = true,
                'f' => settings.force = true,
                'o' => {
                    if (i + 1 < args.len) {
                        i += 1;
                        settings.save_directory = try a.dupe(u8, args[i]);
                    } else try errors.append(a, try std.fmt.allocPrint(a, "option -o requires a value", .{}));
                },
                else => try errors.append(a, try std.fmt.allocPrint(a, "unknown option {s} (run ditch --help)", .{arg})),
            }
            continue;
        }
        if (!std.mem.startsWith(u8, arg, "--")) {
            // Subcommands (anywhere), the help topic, or the positional model.
            if (expect_help_topic) {
                expect_help_topic = false;
                settings.help_topic = try a.dupe(u8, arg);
                continue;
            }
            if (std.mem.eql(u8, arg, "bench")) {
                settings.bench = true;
                continue;
            }
            if (std.mem.eql(u8, arg, "probe")) {
                settings.probe = true;
                continue;
            }
            if (std.mem.eql(u8, arg, "selftest")) {
                settings.selftest = true;
                continue;
            }
            if (std.mem.eql(u8, arg, "truncate") and !settings.truncate) {
                settings.truncate = true;
                continue;
            }
            if (std.mem.eql(u8, arg, "add-model")) {
                settings.add_model = true;
                continue;
            }
            if (std.mem.eql(u8, arg, "help")) {
                settings.help = true;
                expect_help_topic = true;
                continue;
            }
            if (settings.model.len > 0) {
                // Kept for `ditch truncate`; an error for every other command (below).
                try positionals.append(a, try a.dupe(u8, arg));
                continue;
            }
            // A near miss of a subcommand that is not a path is a typo, not a model.
            var is_path = false;
            if (std.Io.Dir.cwd().access(io, arg, .{})) |_| is_path = true else |_| {}
            if (!is_path and std.mem.indexOfScalar(u8, arg, '/') == null) {
                for (subcommands) |sc| if (editDistance(arg, sc) <= 2) {
                    try errors.append(a, try std.fmt.allocPrint(a, "unknown command {s}; did you mean {s}? (a local model directory of that name: ./{s})", .{ arg, sc, arg }));
                    break;
                };
                if (errors.items.len > 0 and std.mem.startsWith(u8, errors.items[errors.items.len - 1], "unknown command")) continue;
            }
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
        applyOption(a, &settings, key, value.?) catch |err| switch (err) {
            error.UnknownOption => {
                if (try suggestOption(a, name)) |sug| {
                    try errors.append(a, try std.fmt.allocPrint(a, "unknown option --{s}; did you mean --{s}?", .{ name, sug }));
                } else {
                    try errors.append(a, try std.fmt.allocPrint(a, "unknown option --{s} (run ditch --help)", .{name}));
                }
            },
            error.TokenAsFlag => try errors.append(a, try std.fmt.allocPrint(a, "a token is never taken as a flag value: set HF_TOKEN or use --token-file <path>", .{})),
            else => try errors.append(a, try std.fmt.allocPrint(a, "invalid value for --{s}: {s}", .{ name, @errorName(err) })),
        };
    }
    if (expect_help_topic) settings.help_topic = null;
    if (settings.truncate) {
        settings.positionals = positionals.items;
    } else for (positionals.items) |arg| {
        try errors.append(a, try std.fmt.allocPrint(a, "unexpected argument {s} (the model is already {s}; quote values containing spaces)", .{ arg, settings.model }));
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
    const bools = [_][]const u8{ "print_debug_information", "print_residual_geometry", "orthogonalize_direction", "keyword_rate_print_responses", "ignore_mismatches", "early_stop", "no_early_stop", "visited_experts_only", "remote_weights", "hotlist", "no_hotlist", "ablate_inputs", "fast_search", "selftest", "kinds", "raw", "residuals", "kernels", "bench_kernels", "accelerate", "no_accelerate", "help", "version", "quiet", "dry_run", "no_input", "interactive", "force", "json", "plain", "no_color", "debug", "token" };
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
    if (eql(u8, key, "models_dir")) {
        s.models_dir = try a.dupe(u8, value);
        return;
    }
    if (eql(u8, key, "model")) s.model = try a.dupe(u8, value) else if (eql(u8, key, "model_commit")) s.model_commit = try a.dupe(u8, value) else if (eql(u8, key, "evaluate_model")) s.evaluate_model = try a.dupe(u8, value) else if (eql(u8, key, "dump_directions")) s.dump_directions = try a.dupe(u8, value) else if (eql(u8, key, "threads")) s.threads = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "cache_dir")) s.cache_dir = try a.dupe(u8, value) else if (eql(u8, key, "chat_template")) s.chat_template = try a.dupe(u8, value) else if (eql(u8, key, "batch_size")) s.batch_size = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "max_batch_size")) s.max_batch_size = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "max_response_length")) s.max_response_length = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "response_prefix")) s.response_prefix = try a.dupe(u8, value) else if (eql(u8, key, "system_prompt")) s.system_prompt = try a.dupe(u8, value) else if (eql(u8, key, "print_debug_information")) s.print_debug_information = try parseBool(value) else if (eql(u8, key, "print_residual_geometry")) s.print_residual_geometry = try parseBool(value) else if (eql(u8, key, "orthogonalize_direction")) s.orthogonalize_direction = try parseBool(value) else if (eql(u8, key, "row_normalization")) s.row_normalization = abliterate.RowNormalization.parse(value) orelse return error.InvalidEnum else if (eql(u8, key, "full_normalization_lora_rank")) s.full_normalization_lora_rank = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "expert_selection")) s.expert_selection = abliterate.ExpertSelection.parse(value) orelse return error.InvalidEnum else if (eql(u8, key, "winsorization_quantile")) s.winsorization_quantile = try std.fmt.parseFloat(f32, value) else if (eql(u8, key, "n_directions")) {
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
    } else if (eql(u8, key, "early_stop")) s.early_stop = try parseBool(value) else if (eql(u8, key, "no_early_stop")) s.early_stop = !(try parseBool(value)) else if (eql(u8, key, "warm_start")) s.warm_start = try a.dupe(u8, value) else if (eql(u8, key, "n_trials")) s.n_trials = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "n_startup_trials")) s.n_startup_trials = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "seed")) s.seed = try std.fmt.parseInt(u64, value, 10) else if (eql(u8, key, "study_checkpoint_dir")) s.study_checkpoint_dir = try a.dupe(u8, value) else if (eql(u8, key, "max_shard_size")) s.max_shard_size = try parseSize(value) else if (eql(u8, key, "max_ram")) s.max_ram = try parseSize(value) else if (eql(u8, key, "max_vram")) s.max_vram = try parseSize(value) else if (eql(u8, key, "device")) {
        if (compute.Kind.parse(value) == null) return error.InvalidEnum;
        s.device = try a.dupe(u8, value);
    } else if (eql(u8, key, "gpu_memory")) s.gpu_memory = try parseSize(value) else if (eql(u8, key, "device_min_macs")) s.device_min_macs = try std.fmt.parseInt(u64, value, 10) else if (eql(u8, key, "selftest")) s.selftest = try parseBool(value) else if (eql(u8, key, "scratch_dir")) s.scratch_dir = try a.dupe(u8, value) else if (eql(u8, key, "time_limit")) s.time_limit_seconds = try parseDuration(value) else if (eql(u8, key, "time_limit_seconds")) s.time_limit_seconds = try std.fmt.parseInt(u64, value, 10) else if (eql(u8, key, "budget_headroom")) s.budget_headroom = try parseSize(value) else if (eql(u8, key, "expert_cache")) s.expert_cache = try parseSize(value) else if (eql(u8, key, "visited_experts_only")) s.visited_experts_only = try parseBool(value) else if (eql(u8, key, "remote_weights")) s.remote_weights = try parseBool(value) else if (eql(u8, key, "remote_chunk_size")) s.remote_chunk_size = try parseSize(value) else if (eql(u8, key, "remote_connections")) s.remote_connections = try std.fmt.parseInt(u32, value, 10) else if (eql(u8, key, "remote_cache_size")) s.remote_cache_size = try parseSize(value) else if (eql(u8, key, "hotlist")) s.hotlist = try parseBool(value) else if (eql(u8, key, "no_hotlist")) s.hotlist = !(try parseBool(value)) else if (eql(u8, key, "checkpoint_action")) s.checkpoint_action = try a.dupe(u8, value) else if (eql(u8, key, "trial_index")) s.trial_index = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "n_additional_trials")) s.n_additional_trials = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "model_action")) s.model_action = try a.dupe(u8, value) else if (eql(u8, key, "save_directory")) s.save_directory = try a.dupe(u8, value) else if (eql(u8, key, "export_dtype")) s.export_dtype = try a.dupe(u8, value) else if (eql(u8, key, "export_format")) s.export_format = try a.dupe(u8, value) else if (eql(u8, key, "gguf_dtype")) s.gguf_dtype = try a.dupe(u8, value) else if (eql(u8, key, "config")) {
        // handled in the first pass
    } else if (eql(u8, key, "reproduce")) s.reproduce = try a.dupe(u8, value) else if (eql(u8, key, "ignore_mismatches")) s.ignore_mismatches = try parseBool(value) else if (eql(u8, key, "bench_prompts")) s.bench_prompts = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "bench_tokens")) s.bench_tokens = try std.fmt.parseInt(usize, value, 10) else if (eql(u8, key, "bench_output")) s.bench_output = try a.dupe(u8, value) else if (eql(u8, key, "bench_kernels") or eql(u8, key, "kernels")) s.bench_kernels = try parseBool(value) else if (eql(u8, key, "accelerate")) s.accelerate = try parseBool(value) else if (eql(u8, key, "no_accelerate")) s.accelerate = !(try parseBool(value)) else if (eql(u8, key, "prompt")) {
        const list = try a.alloc([]const u8, s.probe_prompts.len + 1);
        @memcpy(list[0..s.probe_prompts.len], s.probe_prompts);
        list[s.probe_prompts.len] = try a.dupe(u8, value);
        s.probe_prompts = list;
    } else if (eql(u8, key, "raw")) s.probe_raw = try parseBool(value) else if (eql(u8, key, "residuals")) s.probe_residuals = try parseBool(value) else if (eql(u8, key, "help")) s.help = try parseBool(value) else if (eql(u8, key, "version")) s.version = try parseBool(value) else if (eql(u8, key, "quiet")) s.quiet = try parseBool(value) else if (eql(u8, key, "json_log")) s.json_log = try a.dupe(u8, value) else if (eql(u8, key, "dry_run")) s.dry_run = try parseBool(value) else if (eql(u8, key, "no_input")) s.no_input = try parseBool(value) else if (eql(u8, key, "interactive")) s.interactive = try parseBool(value) else if (eql(u8, key, "force")) s.force = try parseBool(value) else if (eql(u8, key, "json")) s.json = try parseBool(value) else if (eql(u8, key, "plain")) s.plain = try parseBool(value) else if (eql(u8, key, "no_color")) s.no_color = try parseBool(value) else if (eql(u8, key, "debug")) s.print_debug_information = try parseBool(value) else if (eql(u8, key, "output")) s.save_directory = try a.dupe(u8, value) else if (eql(u8, key, "token_file")) s.token_file = try a.dupe(u8, value) else if (eql(u8, key, "http_timeout")) s.http_timeout_seconds = try parseDuration(value) else if (eql(u8, key, "token")) return error.TokenAsFlag else if (eql(u8, key, "keyword_rate_print_responses")) s.keyword_rate.print_responses = try parseBool(value) else if (eql(u8, key, "keyword_rate_score_name")) s.keyword_rate.score_name = try a.dupe(u8, value) else if (std.mem.startsWith(u8, key, "good_prompts_")) try applyDatasetOption(a, &s.good_prompts, key["good_prompts_".len..], value) else if (std.mem.startsWith(u8, key, "bad_prompts_")) try applyDatasetOption(a, &s.bad_prompts, key["bad_prompts_".len..], value) else if (std.mem.startsWith(u8, key, "keyword_rate_prompts_")) try applyDatasetOption(a, &s.keyword_rate.prompts, key["keyword_rate_prompts_".len..], value) else if (std.mem.startsWith(u8, key, "kl_divergence_prompts_")) try applyDatasetOption(a, &s.kl_divergence.prompts, key["kl_divergence_prompts_".len..], value) else if (eql(u8, key, "layers")) s.truncate_layers = try a.dupe(u8, value) else if (eql(u8, key, "kinds")) s.truncate_kinds = try parseBool(value) else if (eql(u8, key, "drop")) {
        s.truncate_drop = try appendString(a, s.truncate_drop, value);
    } else if (eql(u8, key, "rows")) {
        s.truncate_rows = try appendString(a, s.truncate_rows, value);
    } else return error.UnknownOption;
}

/// `list` with a copy of `value` appended (for repeatable options).
fn appendString(a: Allocator, list: []const []const u8, value: []const u8) ![]const []const u8 {
    const out = try a.alloc([]const u8, list.len + 1);
    @memcpy(out[0..list.len], list);
    out[list.len] = try a.dupe(u8, value);
    return out;
}

fn valueString(a: Allocator, v: tree.Value) ![]const u8 {
    return switch (v) {
        .string => |s| try a.dupe(u8, s),
        .integer => |i| try std.fmt.allocPrint(a, "{d}", .{i}),
        .float => |f| try std.fmt.allocPrint(a, "{d}", .{f}),
        .boolean => |b| if (b) "true" else "false",
        else => error.InvalidType,
    };
}

fn applyDatasetTable(a: Allocator, spec: *DatasetSpec, t: *const tree.Table, errors: *std.ArrayList([]const u8)) !void {
    var it = t.map.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (std.mem.startsWith(u8, k, "residual_plot")) continue;
        if (std.mem.eql(u8, k, "commit")) continue;
        const v = valueString(a, e.value_ptr.*) catch {
            try errors.append(a, try std.fmt.allocPrint(a, "dataset field {s} must be a string", .{k}));
            continue;
        };
        applyDatasetOption(a, spec, k, v) catch {
            try errors.append(a, try std.fmt.allocPrint(a, "unknown dataset field: {s}", .{k}));
        };
    }
}

fn applyTable(a: Allocator, s: *Settings, root: *const tree.Table, errors: *std.ArrayList([]const u8)) !void {
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
                    if (st.get("score_name")) |sn| s.keyword_rate.score_name = try valueString(a, sn);
                    if (st.get("print_responses")) |pr| s.keyword_rate.print_responses = pr == .boolean and pr.boolean;
                    if (st.get("keyword_markers")) |km| {
                        if (km == .array) {
                            var list = std.ArrayList([]const u8).empty;
                            for (km.array) |m| try list.append(a, try valueString(a, m));
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
                const plugin = try valueString(a, item.table.get("plugin") orelse .{ .string = "" });
                const kind = ScorerKind.fromPlugin(plugin) orelse {
                    try errors.append(a, try std.fmt.allocPrint(a, "unsupported scorer plugin: {s} (available: keyword_rate, kl_divergence, refusal_logit)", .{plugin}));
                    continue;
                };
                const opt_s = try valueString(a, item.table.get("optimization") orelse .{ .string = "none" });
                const opt: Optimization = if (std.mem.eql(u8, opt_s, "minimize")) .minimize else if (std.mem.eql(u8, opt_s, "maximize")) .maximize else .none;
                const inst = if (item.table.get("instance_name")) |n| try valueString(a, n) else null;
                try list.append(a, .{ .kind = kind, .optimization = opt, .instance_name = inst });
            }
            s.scorers = list.items;
        } else if (std.mem.eql(u8, k, "chain_of_thought_skips") and v == .array) {
            var list = std.ArrayList([2][]const u8).empty;
            for (v.array) |pair| {
                if (pair != .array or pair.array.len != 2) continue;
                try list.append(a, .{ try valueString(a, pair.array[0]), try valueString(a, pair.array[1]) });
            }
            s.chain_of_thought_skips = list.items;
        } else if (std.mem.eql(u8, k, "dtypes") or std.mem.eql(u8, k, "quantization") or std.mem.eql(u8, k, "device_map") or std.mem.eql(u8, k, "max_memory") or std.mem.eql(u8, k, "offload_outputs_to_cpu") or std.mem.startsWith(u8, k, "residual_plot") or std.mem.eql(u8, k, "plot_residuals") or std.mem.eql(u8, k, "benchmarks")) {
            // Accepted for compatibility with heretic config files; ignored.
        } else {
            const str = valueString(a, v) catch {
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
    var r = try load(gpa, std.testing.io, &args, null);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqual(@as(u64, 8 << 30), r.settings.max_ram);
    try std.testing.expectEqual(@as(?u64, 0), r.settings.budget_headroom);
    try std.testing.expectEqual(@as(u64, 24 << 30), r.settings.max_vram);
    try std.testing.expectEqual(@as(?u64, 5400), r.settings.time_limit_seconds);
    try std.testing.expectEqualStrings("/tmp/x", r.settings.scratch_dir.?);
    // Lua values go through the same parser.
    var parsed = (try lua.parse(gpa, "return { max_ram = \"2GB\", time_limit = \"2h\" }", "test.lua")).parsed;
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var s = Settings{};
    var errors = std.ArrayList([]const u8).empty;
    try applyTable(arena.allocator(), &s, parsed.root, &errors);
    try std.testing.expectEqual(@as(usize, 0), errors.items.len);
    try std.testing.expectEqual(@as(u64, 2 << 30), s.max_ram);
    try std.testing.expectEqual(@as(?u64, 7200), s.time_limit_seconds);
}

test "per-model config files layer between the global file and the flags" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expectEqualStrings("Qwen/Qwen3-8B", modelConfigName("Qwen/Qwen3-8B").?);
    try std.testing.expectEqualStrings("Qwen/Qwen3-8B", modelConfigName("hf://Qwen/Qwen3-8B@main").?);
    try std.testing.expectEqualStrings("my-model", modelConfigName("./out/my-model/").?);
    try std.testing.expectEqualStrings("my-model", modelConfigName("/models/my-model.gguf").?);
    try std.testing.expectEqualStrings("tiny", modelConfigName("tiny").?);
    try std.testing.expect(modelConfigName("") == null);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "ditch/configs/Org");
    try tmp.dir.writeFile(io, .{ .sub_path = "ditch/config.lua", .data = "return { max_ram = \"2GB\", n_trials = 50 }" });
    try tmp.dir.writeFile(io, .{ .sub_path = "ditch/configs/Org/Model.lua", .data = "return { max_ram = \"4GB\", seed = 7 }" });
    const base = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(base);
    var env: std.process.Environ.Map = .init(gpa);
    defer env.deinit();
    try env.put("XDG_CONFIG_HOME", base);

    const args = [_][]const u8{ "ditch", "Org/Model", "--seed", "9" };
    var r = try load(gpa, io, &args, &env);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqual(@as(u64, 4 << 30), r.settings.max_ram);
    try std.testing.expectEqual(@as(usize, 50), r.settings.n_trials);
    try std.testing.expectEqual(@as(?u64, 9), r.settings.seed);

    const other = [_][]const u8{ "ditch", "Org/Other" };
    var r2 = try load(gpa, io, &other, &env);
    defer r2.deinit();
    try std.testing.expectEqual(@as(u64, 2 << 30), r2.settings.max_ram);
}

test "warp mode options" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--expert-cache", "512MB", "--visited-experts-only", "false", "--remote-weights", "--remote-chunk-size=1MB", "--no-hotlist", "hf://Qwen/Qwen3-30B-A3B" };
    var r = try load(gpa, std.testing.io, &args, null);
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
    try std.testing.expectEqual(@as(?u64, null), defaults.remote_cache_size);
}

test "remote cache size: flag over environment, 0 allowed" {
    const gpa = std.testing.allocator;
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    try env.put("DITCH_REMOTE_CACHE_SIZE", "20GB");
    const from_env = [_][]const u8{ "ditch", "hf://Qwen/Qwen3-30B-A3B" };
    var r1 = try load(gpa, std.testing.io, &from_env, &env);
    defer r1.deinit();
    try std.testing.expectEqual(@as(usize, 0), r1.errors.len);
    try std.testing.expectEqual(@as(?u64, 20 << 30), r1.settings.remote_cache_size);
    const from_flag = [_][]const u8{ "ditch", "--remote-cache-size", "0", "hf://Qwen/Qwen3-30B-A3B" };
    var r2 = try load(gpa, std.testing.io, &from_flag, &env);
    defer r2.deinit();
    try std.testing.expectEqual(@as(?u64, 0), r2.settings.remote_cache_size);
    try env.put("DITCH_REMOTE_CACHE_SIZE", "lots");
    var r3 = try load(gpa, std.testing.io, &from_env, &env);
    defer r3.deinit();
    try std.testing.expectEqual(@as(usize, 1), r3.errors.len);
    try std.testing.expect(std.mem.indexOf(u8, r3.errors[0], "DITCH_REMOTE_CACHE_SIZE") != null);
}

test "cli parsing" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--n-trials", "5", "--row-normalization=pre", "--print-debug-information", "--expert-selection", "random", "--good-prompts-dataset", "good.txt", "--n-directions", "2", "--no-early-stop", "--warm-start", "old.jsonl", "Qwen/Qwen2.5-0.5B-Instruct" };
    var r = try load(gpa, std.testing.io, &args, null);
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
    var r = try load(gpa, std.testing.io, &args, null);
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
    var rb = try load(gpa, std.testing.io, &bad, null);
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
    try applyTable(arena.allocator(), &s, result.parsed.root, &errors);
    try std.testing.expectEqual(@as(usize, 0), errors.items.len);
    try std.testing.expectEqual(ScorerKind.refusal_logit, s.scorers[0].kind);
    try std.testing.expectEqual(DirectionRange.auto, s.direction_range);
}

test "cli aliases, subcommands, order and suggestions" {
    const gpa = std.testing.allocator;
    // Short aliases, =-form, subcommand after the model, -o.
    const args = [_][]const u8{ "ditch", "tests/fixtures/qwen2", "bench", "-q", "-n", "-d", "-f", "-o", "out", "--bench-tokens=3", "--json", "--no-input", "--debug=false", "--http-timeout", "1m" };
    var r = try load(gpa, std.testing.io, &args, null);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expect(r.settings.bench);
    try std.testing.expectEqualStrings("tests/fixtures/qwen2", r.settings.model);
    try std.testing.expect(r.settings.quiet and r.settings.dry_run and r.settings.force and r.settings.json and r.settings.no_input);
    try std.testing.expect(!r.settings.print_debug_information);
    try std.testing.expectEqualStrings("out", r.settings.save_directory.?);
    try std.testing.expectEqual(@as(usize, 3), r.settings.bench_tokens);
    try std.testing.expectEqual(@as(u64, 60), r.settings.http_timeout_seconds);

    // `bench` first is the same command.
    const args2 = [_][]const u8{ "ditch", "bench", "--output", "o2", "m" };
    var r2 = try load(gpa, std.testing.io, &args2, null);
    defer r2.deinit();
    try std.testing.expect(r2.settings.bench);
    try std.testing.expectEqualStrings("m", r2.settings.model);
    try std.testing.expectEqualStrings("o2", r2.settings.save_directory.?);

    // Unknown options are usage errors with a suggestion; --help still wins.
    const args3 = [_][]const u8{ "ditch", "--n-trails", "5", "--token=abc", "-x", "m" };
    var r3 = try load(gpa, std.testing.io, &args3, null);
    defer r3.deinit();
    try std.testing.expectEqual(@as(usize, 3), r3.errors.len);
    try std.testing.expectEqualStrings("unknown option --n-trails; did you mean --n-trials?", r3.errors[0]);
    try std.testing.expect(std.mem.indexOf(u8, r3.errors[1], "never taken as a flag value") != null);
    try std.testing.expect(std.mem.startsWith(u8, r3.errors[2], "unknown option -x"));
    const args4 = [_][]const u8{ "ditch", "--n-trails", "5", "--help" };
    var r4 = try load(gpa, std.testing.io, &args4, null);
    defer r4.deinit();
    try std.testing.expect(r4.settings.help);

    // Help topics and misspelt subcommands.
    const args5 = [_][]const u8{ "ditch", "help", "bench" };
    var r5 = try load(gpa, std.testing.io, &args5, null);
    defer r5.deinit();
    try std.testing.expect(r5.settings.help);
    try std.testing.expectEqualStrings("bench", r5.settings.help_topic.?);
    const args6 = [_][]const u8{ "ditch", "bnech", "m" };
    var r6 = try load(gpa, std.testing.io, &args6, null);
    defer r6.deinit();
    try std.testing.expectEqual(@as(usize, 1), r6.errors.len);
    try std.testing.expect(std.mem.startsWith(u8, r6.errors[0], "unknown command bnech; did you mean bench?"));

    try std.testing.expectEqual(@as(usize, 2), editDistance("bench", "bnech"));
    try std.testing.expectEqualStrings("n-trials", (try suggestOption(r6.arena.allocator(), "ntrials")).?);
    try std.testing.expect(std.mem.indexOf(u8, help_text, "Exit codes:") != null);
}

test "cli: truncate takes the layer count and output directory as positionals" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "truncate", "owner/name", "2", "out", "--drop", "mtp.", "--drop=model.mtp", "--rows", "t=rows.txt" };
    var r = try load(gpa, std.testing.io, &args, null);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expect(r.settings.truncate);
    try std.testing.expectEqualStrings("owner/name", r.settings.model);
    try std.testing.expectEqual(@as(usize, 2), r.settings.positionals.len);
    try std.testing.expectEqualStrings("2", r.settings.positionals[0]);
    try std.testing.expectEqualStrings("out", r.settings.positionals[1]);
    try std.testing.expectEqual(@as(usize, 2), r.settings.truncate_drop.len);
    try std.testing.expectEqualStrings("model.mtp", r.settings.truncate_drop[1]);
    try std.testing.expectEqualStrings("t=rows.txt", r.settings.truncate_rows[0]);

    // The subcommand after the model, --layers and --kinds.
    const args2 = [_][]const u8{ "ditch", "owner/name", "truncate", "--layers", "0,1,20", "--kinds", "out" };
    var r2 = try load(gpa, std.testing.io, &args2, null);
    defer r2.deinit();
    try std.testing.expectEqual(@as(usize, 0), r2.errors.len);
    try std.testing.expect(r2.settings.truncate and r2.settings.truncate_kinds);
    try std.testing.expectEqualStrings("0,1,20", r2.settings.truncate_layers.?);
    try std.testing.expectEqualStrings("out", r2.settings.positionals[0]);

    // Every other command takes one positional, the model.
    const args3 = [_][]const u8{ "ditch", "m", "extra" };
    var r3 = try load(gpa, std.testing.io, &args3, null);
    defer r3.deinit();
    try std.testing.expectEqual(@as(usize, 1), r3.errors.len);
    try std.testing.expect(std.mem.startsWith(u8, r3.errors[0], "unexpected argument extra (the model is already m"));
}

test "dump directions option" {
    const gpa = std.testing.allocator;
    const args = [_][]const u8{ "ditch", "--dump-directions", "dirs.safetensors", "m" };
    var r = try load(gpa, std.testing.io, &args, null);
    defer r.deinit();
    try std.testing.expectEqual(@as(usize, 0), r.errors.len);
    try std.testing.expectEqualStrings("dirs.safetensors", r.settings.dump_directions.?);
    try std.testing.expectEqual(@as(?[]const u8, null), (Settings{}).dump_directions);
}
