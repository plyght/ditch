//! Memory and time budgets: a byte-counting allocator that refuses to exceed a
//! limit, process-level accounting (peak RSS, scratch traffic, elapsed time),
//! an up-front feasibility estimate and end-of-run reports.
//!
//! Ownership model: every large buffer ditch allocates (weight buffers in
//! streamed mode, activations, KV caches, abliteration deltas, scratch I/O
//! buffers) comes from `Budget.allocator()`. Metadata (tokenizer, config,
//! rope tables) and allocations outside ditch's control are covered by the
//! reserved `headroom`.

const std = @import("std");

/// Set by the Ctrl+C handler; long non-trial steps (downloads, exports)
/// poll it and stop with `error.Interrupted`, leaving their markers behind.
pub var interrupt_requested = std.atomic.Value(bool).init(false);

pub fn interrupted() bool {
    return interrupt_requested.load(.seq_cst);
}
const Io = std.Io;
const builtin = @import("builtin");
const config = @import("config.zig");
const model_mod = @import("model.zig");
const tensor = @import("tensor.zig");
const export_mod = @import("export.zig");

const Allocator = std.mem.Allocator;

// ---------------------------------------------------------------------------
// Budgeted allocator
// ---------------------------------------------------------------------------

/// Wraps a child allocator, tracks live and peak bytes and fails any
/// allocation that would push live bytes above `limit` (0 = unlimited).
pub const BudgetedAllocator = struct {
    child: Allocator,
    limit: u64,
    /// Spinlock guarding the counters (the allocator interface has no `Io` for `Io.Mutex`).
    locked: std.atomic.Value(bool) = .init(false),
    current: u64 = 0,
    peak: u64 = 0,
    /// Number of allocations that failed because of the limit (after reclaim).
    refused: u64 = 0,
    /// Number of times the limit was hit and memory had to be reclaimed.
    pressure: u64 = 0,
    /// Cumulative bytes handed out (useful to spot churn).
    total_allocated: u64 = 0,
    /// Called (outside the lock) when an allocation of `len` bytes is
    /// refused; returns the number of bytes it released so the allocation can
    /// be retried. Used by the weight store to drop recycled buffers and by
    /// the expert cache to evict unpinned experts under pressure.
    reclaim: ?Reclaim = null,

    pub const Reclaim = struct {
        ctx: *anyopaque,
        func: *const fn (*anyopaque, usize) usize,
    };

    pub fn init(child: Allocator, limit: u64) BudgetedAllocator {
        return .{ .child = child, .limit = limit };
    }

    /// Reserves `len` bytes, invoking the reclaim hook once when refused.
    fn reserveOrReclaim(self: *BudgetedAllocator, len: usize) bool {
        if (self.reserve(len)) return true;
        self.lock();
        self.pressure += 1;
        self.unlock();
        const ok = blk: {
            const r = self.reclaim orelse break :blk false;
            if (r.func(r.ctx, len) == 0) break :blk false;
            break :blk self.reserve(len);
        };
        if (!ok) {
            self.lock();
            self.refused += 1;
            self.unlock();
        }
        return ok;
    }

    pub fn allocator(self: *BudgetedAllocator) Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn lock(self: *BudgetedAllocator) void {
        while (self.locked.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(self: *BudgetedAllocator) void {
        self.locked.store(false, .release);
    }

    const vtable: Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    /// Reserves `len` bytes of budget; returns false when it would exceed the limit.
    fn reserve(self: *BudgetedAllocator, len: usize) bool {
        self.lock();
        defer self.unlock();
        if (self.limit != 0 and self.current + len > self.limit) return false;
        self.current += len;
        self.total_allocated += len;
        if (self.current > self.peak) self.peak = self.current;
        return true;
    }

    fn unreserve(self: *BudgetedAllocator, len: usize) void {
        self.lock();
        defer self.unlock();
        self.current -= @min(self.current, len);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *BudgetedAllocator = @ptrCast(@alignCast(ctx));
        if (!self.reserveOrReclaim(len)) return null;
        const p = self.child.rawAlloc(len, alignment, ret_addr) orelse {
            self.unreserve(len);
            return null;
        };
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *BudgetedAllocator = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len) {
            if (!self.reserveOrReclaim(new_len - memory.len)) return false;
            if (!self.child.rawResize(memory, alignment, new_len, ret_addr)) {
                self.unreserve(new_len - memory.len);
                return false;
            }
            return true;
        }
        if (!self.child.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.unreserve(memory.len - new_len);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *BudgetedAllocator = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len) {
            if (!self.reserveOrReclaim(new_len - memory.len)) return null;
            const p = self.child.rawRemap(memory, alignment, new_len, ret_addr) orelse {
                self.unreserve(new_len - memory.len);
                return null;
            };
            return p;
        }
        const p = self.child.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;
        self.unreserve(memory.len - new_len);
        return p;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *BudgetedAllocator = @ptrCast(@alignCast(ctx));
        self.child.rawFree(memory, alignment, ret_addr);
        self.unreserve(memory.len);
    }

    pub fn currentBytes(self: *BudgetedAllocator) u64 {
        self.lock();
        defer self.unlock();
        return self.current;
    }

    pub fn peakBytes(self: *BudgetedAllocator) u64 {
        self.lock();
        defer self.unlock();
        return self.peak;
    }

    /// Bytes still available under the limit (`maxInt` when unlimited).
    pub fn available(self: *BudgetedAllocator) u64 {
        self.lock();
        defer self.unlock();
        if (self.limit == 0) return std.math.maxInt(u64);
        return self.limit - @min(self.limit, self.current);
    }
};

// ---------------------------------------------------------------------------
// Process accounting
// ---------------------------------------------------------------------------

pub const Rss = struct {
    /// Current resident set size in bytes (0 if unavailable).
    current: u64 = 0,
    /// Peak resident set size in bytes (0 if unavailable).
    peak: u64 = 0,
    available: bool = false,
};

/// Reads VmRSS / VmHWM from /proc/self/status on Linux.
pub fn readRss(io: Io) Rss {
    if (builtin.os.tag != .linux) return .{};
    var buf: [8192]u8 = undefined;
    const file = Io.Dir.openFileAbsolute(io, "/proc/self/status", .{}) catch return .{};
    defer file.close(io);
    const n = file.readPositionalAll(io, &buf, 0) catch return .{};
    var rss = Rss{ .available = true };
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "VmRSS:")) rss.current = parseKb(line["VmRSS:".len..]);
        if (std.mem.startsWith(u8, line, "VmHWM:")) rss.peak = parseKb(line["VmHWM:".len..]);
    }
    return rss;
}

fn parseKb(s: []const u8) u64 {
    const t = std.mem.trim(u8, s, " \t");
    const end = std.mem.indexOfScalar(u8, t, ' ') orelse t.len;
    const v = std.fmt.parseInt(u64, t[0..end], 10) catch return 0;
    return v * 1024;
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

pub const Options = struct {
    /// Maximum resident bytes for ditch-owned buffers plus headroom (0 = unlimited).
    max_ram: u64 = 0,
    /// Reserved for allocations outside ditch's control. null = max(10% of max_ram, 256MB),
    /// capped at half of max_ram.
    headroom: ?u64 = null,
    /// Wall-clock limit for the whole run (null = none).
    time_limit: ?Io.Duration = null,
    /// Directory for spilled activations / KV caches (null = "scratch" in cwd).
    scratch_dir: ?[]const u8 = null,
};

pub const default_headroom_floor: u64 = 256 * 1024 * 1024;

pub fn defaultHeadroom(max_ram: u64) u64 {
    if (max_ram == 0) return 0;
    const h = @max(max_ram / 10, default_headroom_floor);
    return @min(h, max_ram / 2);
}

pub const Budget = struct {
    gpa: Allocator,
    io: Io,
    max_ram: u64,
    headroom: u64,
    time_limit: ?Io.Duration,
    scratch_dir: []const u8,
    start: Io.Timestamp,
    alloc: BudgetedAllocator,
    scratch_written: std.atomic.Value(u64) = .init(0),
    scratch_read: std.atomic.Value(u64) = .init(0),
    weight_bytes_read: std.atomic.Value(u64) = .init(0),
    /// Set once the time limit has been observed as exceeded.
    expired_flag: std.atomic.Value(bool) = .init(false),

    /// `gpa` is both the child allocator for budgeted allocations and the
    /// allocator for the budget's own bookkeeping.
    pub fn init(gpa: Allocator, io: Io, opts: Options) !Budget {
        const headroom = opts.headroom orelse defaultHeadroom(opts.max_ram);
        if (opts.max_ram != 0 and headroom >= opts.max_ram) return error.BudgetTooSmall;
        const limit = if (opts.max_ram == 0) 0 else opts.max_ram - headroom;
        return .{
            .gpa = gpa,
            .io = io,
            .max_ram = opts.max_ram,
            .headroom = headroom,
            .time_limit = opts.time_limit,
            .scratch_dir = try gpa.dupe(u8, opts.scratch_dir orelse "scratch"),
            .start = Io.Clock.awake.now(io),
            .alloc = BudgetedAllocator.init(gpa, limit),
        };
    }

    /// Builds a budget from settings (`--max-ram`, `--time-limit`, `--scratch-dir`).
    /// The scratch directory defaults to `<cache_dir>/scratch` when `cache_dir` is
    /// set, else `$TMPDIR/ditch-scratch` when TMPDIR is set, otherwise `<cwd>/scratch`.
    pub fn fromSettings(gpa: Allocator, io: Io, settings: *const config.Settings) !Budget {
        return fromSettingsEnv(gpa, io, settings, null);
    }

    pub fn fromSettingsEnv(gpa: Allocator, io: Io, settings: *const config.Settings, environ: ?*std.process.Environ.Map) !Budget {
        var scratch_owned: ?[]u8 = null;
        defer if (scratch_owned) |s| gpa.free(s);
        var scratch: ?[]const u8 = settings.scratch_dir;
        if (scratch == null) {
            if (settings.cache_dir) |c| {
                scratch_owned = try std.fs.path.join(gpa, &.{ c, "scratch" });
                scratch = scratch_owned;
            } else if (environ) |env| if (env.get("TMPDIR")) |t| if (t.len > 0) {
                scratch_owned = try std.fs.path.join(gpa, &.{ t, "ditch-scratch" });
                scratch = scratch_owned;
            };
        }
        return init(gpa, io, .{
            .max_ram = settings.max_ram,
            .headroom = settings.budget_headroom,
            .time_limit = if (settings.time_limit_seconds) |s| Io.Duration.fromSeconds(@intCast(s)) else null,
            .scratch_dir = scratch,
        });
    }

    pub fn deinit(self: *Budget) void {
        self.gpa.free(self.scratch_dir);
    }

    /// The allocator every large buffer must come from.
    pub fn allocator(self: *Budget) Allocator {
        return self.alloc.allocator();
    }

    pub fn limited(self: *const Budget) bool {
        return self.max_ram != 0;
    }

    /// Bytes ditch-owned buffers may occupy in total (0 = unlimited).
    pub fn limitBytes(self: *const Budget) u64 {
        return self.alloc.limit;
    }

    pub fn available(self: *Budget) u64 {
        return self.alloc.available();
    }

    /// True if `bytes` more could be allocated right now.
    pub fn fits(self: *Budget, bytes: u64) bool {
        return self.available() >= bytes;
    }

    pub fn elapsed(self: *const Budget) Io.Duration {
        return self.start.durationTo(Io.Clock.awake.now(self.io));
    }

    pub fn expired(self: *Budget) bool {
        if (self.expired_flag.load(.acquire)) return true;
        const limit = self.time_limit orelse return false;
        if (self.elapsed().nanoseconds >= limit.nanoseconds) {
            self.expired_flag.store(true, .release);
            return true;
        }
        return false;
    }

    /// Returns `error.TimeLimitExceeded` once the wall-clock limit has passed.
    pub fn checkTime(self: *Budget) error{TimeLimitExceeded}!void {
        if (self.expired()) return error.TimeLimitExceeded;
    }

    pub fn noteScratchWrite(self: *Budget, bytes: u64) void {
        _ = self.scratch_written.fetchAdd(bytes, .monotonic);
    }
    pub fn noteScratchRead(self: *Budget, bytes: u64) void {
        _ = self.scratch_read.fetchAdd(bytes, .monotonic);
    }
    pub fn noteWeightRead(self: *Budget, bytes: u64) void {
        _ = self.weight_bytes_read.fetchAdd(bytes, .monotonic);
    }

    pub fn report(self: *Budget) Report {
        const rss = readRss(self.io);
        return .{
            .max_ram = self.max_ram,
            .headroom = self.headroom,
            .limit = self.alloc.limit,
            .current = self.alloc.currentBytes(),
            .peak = self.alloc.peakBytes(),
            .refused = self.alloc.refused,
            .pressure = self.alloc.pressure,
            .rss = rss,
            .scratch_written = self.scratch_written.load(.monotonic),
            .scratch_read = self.scratch_read.load(.monotonic),
            .weight_bytes_read = self.weight_bytes_read.load(.monotonic),
            .elapsed = self.elapsed(),
            .time_limit = self.time_limit,
        };
    }
};

pub const Report = struct {
    max_ram: u64,
    headroom: u64,
    limit: u64,
    current: u64,
    peak: u64,
    refused: u64,
    pressure: u64,
    rss: Rss,
    scratch_written: u64,
    scratch_read: u64,
    weight_bytes_read: u64,
    elapsed: Io.Duration,
    time_limit: ?Io.Duration,

    /// Peak resident bytes: process RSS high-water mark when available, otherwise
    /// the allocator's peak.
    pub fn peakResident(self: Report) u64 {
        return if (self.rss.available and self.rss.peak > 0) self.rss.peak else self.peak;
    }

    pub fn print(self: Report, w: *Io.Writer, label: []const u8) !void {
        try w.print("{s}: ", .{label});
        if (self.max_ram != 0) {
            try w.print("budget {f} (limit {f}, headroom {f}), ", .{ fmtBytes(self.max_ram), fmtBytes(self.limit), fmtBytes(self.headroom) });
        } else {
            try w.writeAll("budget unlimited, ");
        }
        try w.print("budgeted peak {f} (live {f}, pressure events {d}, refused {d})", .{ fmtBytes(self.peak), fmtBytes(self.current), self.pressure, self.refused });
        if (self.rss.available) {
            try w.print(", process RSS {f} (peak {f})", .{ fmtBytes(self.rss.current), fmtBytes(self.rss.peak) });
        }
        try w.print(", weights read {f}, scratch written {f} / read {f}, elapsed {f}", .{ fmtBytes(self.weight_bytes_read), fmtBytes(self.scratch_written), fmtBytes(self.scratch_read), fmtDuration(self.elapsed) });
        if (self.time_limit) |t| try w.print(" of {f}", .{fmtDuration(t)});
        try w.writeAll("\n");
    }
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

pub const FmtBytes = struct {
    bytes: u64,
    pub fn format(self: FmtBytes, w: *Io.Writer) Io.Writer.Error!void {
        const b: f64 = @floatFromInt(self.bytes);
        if (self.bytes >= 1 << 30) return w.print("{d:.2}GB", .{b / (1 << 30)});
        if (self.bytes >= 1 << 20) return w.print("{d:.1}MB", .{b / (1 << 20)});
        if (self.bytes >= 1 << 10) return w.print("{d:.1}KB", .{b / (1 << 10)});
        return w.print("{d}B", .{self.bytes});
    }
};

pub fn fmtBytes(bytes: u64) FmtBytes {
    return .{ .bytes = bytes };
}

pub const FmtDuration = struct {
    d: Io.Duration,
    pub fn format(self: FmtDuration, w: *Io.Writer) Io.Writer.Error!void {
        const ms = self.d.toMilliseconds();
        if (ms < 1000) return w.print("{d}ms", .{ms});
        const s = @divTrunc(ms, 1000);
        if (s < 60) return w.print("{d}.{d}s", .{ s, @divTrunc(@mod(ms, 1000), 100) });
        if (s < 3600) return w.print("{d}m{d}s", .{ @divTrunc(s, 60), @mod(s, 60) });
        return w.print("{d}h{d}m", .{ @divTrunc(s, 3600), @divTrunc(@mod(s, 3600), 60) });
    }
};

pub fn fmtDuration(d: Io.Duration) FmtDuration {
    return .{ .d = d };
}

// ---------------------------------------------------------------------------
// Feasibility estimate
// ---------------------------------------------------------------------------

pub const EstimateParams = struct {
    /// Prompts per batch.
    batch_size: usize,
    /// Longest prompt (tokens) expected in a batch.
    max_prompt_tokens: usize,
    /// Tokens generated per response.
    max_response_length: usize,
    /// Worker threads (per-thread kernel scratch).
    threads: usize,
    /// LoRA rank of abliteration deltas.
    lora_rank: usize = 3,
    /// Export dtype (null = source dtype).
    export_dtype: ?tensor.DType = null,
};

pub const Estimate = struct {
    total_weight_bytes: u64,
    largest_layer_bytes: u64,
    largest_tensor_bytes: u64,
    largest_tensor_name: []const u8,
    num_layers: usize,
    /// Forward workspace for `batch_size * max_prompt_tokens` rows.
    workspace_bytes: u64,
    /// Forward workspace for one row (the minimum to run at all).
    workspace_min_bytes: u64,
    /// KV cache for one batch of `batch_size` sequences of prompt + response length.
    kv_cache_bytes: u64,
    /// Per-thread kernel scratch.
    kernel_scratch_bytes: u64,
    /// Abliteration deltas for every layer/component at `lora_rank`.
    delta_bytes: u64,
    /// Peak of the export writer (one row chunk or expert slice of the largest
    /// tensor: raw + f32 copy + output chunk).
    export_bytes: u64,
    /// Resident weights in streamed mode: one layer + the prefetched next layer.
    streamed_weights_bytes: u64,
    /// The bare minimum streamed mode needs: largest layer (or tensor) + minimal workspace + scratch.
    min_streamed_bytes: u64,
    /// Comfortable streamed mode: prefetch + full workspace + KV cache in RAM + deltas.
    streamed_bytes: u64,
    /// Mapped mode: all weights resident + everything else.
    mapped_bytes: u64,
    /// Warp mode (streamed mixture of experts with an expert cache).
    warp: bool = false,
    /// Largest layer without its routed experts.
    trunk_layer_bytes: u64 = 0,
    /// One routed expert (largest).
    expert_bytes: u64 = 0,
    /// Routed experts per layer and how many a token selects.
    num_experts: usize = 0,
    top_k: usize = 0,
    /// Every routed expert of the model.
    total_expert_bytes: u64 = 0,
    /// Configured expert cache capacity.
    expert_cache_bytes: u64 = 0,

    pub fn print(self: Estimate, w: *Io.Writer) !void {
        try w.print("Memory estimate:\n", .{});
        try w.print("  weights total            {f} ({d} layers)\n", .{ fmtBytes(self.total_weight_bytes), self.num_layers });
        try w.print("  largest layer            {f}\n", .{fmtBytes(self.largest_layer_bytes)});
        if (self.warp) {
            try w.print("  trunk per layer          {f} (largest; routed experts excluded)\n", .{fmtBytes(self.trunk_layer_bytes)});
            try w.print("  routed expert            {f} each, {d} per layer, top-{d} per token, {f} in total\n", .{ fmtBytes(self.expert_bytes), self.num_experts, self.top_k, fmtBytes(self.total_expert_bytes) });
            const fits: u64 = if (self.expert_bytes == 0) 0 else self.expert_cache_bytes / self.expert_bytes;
            try w.print("  expert cache             {f} (holds {d} of {d} experts)\n", .{ fmtBytes(self.expert_cache_bytes), fits, self.numExpertsTotal() });
        }
        try w.print("  largest tensor           {f} ({s})\n", .{ fmtBytes(self.largest_tensor_bytes), self.largest_tensor_name });
        try w.print("  forward workspace        {f} (min {f})\n", .{ fmtBytes(self.workspace_bytes), fmtBytes(self.workspace_min_bytes) });
        try w.print("  KV cache per batch       {f}\n", .{fmtBytes(self.kv_cache_bytes)});
        try w.print("  kernel scratch           {f}\n", .{fmtBytes(self.kernel_scratch_bytes)});
        try w.print("  abliteration deltas      {f}\n", .{fmtBytes(self.delta_bytes)});
        try w.print("  export peak              {f}\n", .{fmtBytes(self.export_bytes)});
        if (self.warp) {
            try w.print("  warp mode:     min {f} (trunk + top-{d} experts + workspace), with prefetch + expert cache + RAM caches {f}\n", .{ fmtBytes(self.min_streamed_bytes), self.top_k, fmtBytes(self.streamed_bytes) });
        } else {
            try w.print("  streamed mode: min {f}, with prefetch + RAM caches {f}\n", .{ fmtBytes(self.min_streamed_bytes), fmtBytes(self.streamed_bytes) });
        }
        try w.print("  mapped mode:   {f}\n", .{fmtBytes(self.mapped_bytes)});
    }

    /// Routed experts in the whole model (0 for dense models).
    pub fn numExpertsTotal(self: Estimate) u64 {
        if (self.expert_bytes == 0) return 0;
        return self.total_expert_bytes / self.expert_bytes;
    }
};

fn workspaceBytes(c: *const model_mod.Config, rows: usize, logit_rows: usize) u64 {
    return @as(u64, rows) * model_mod.Workspace.bytesPerRow(c) + @as(u64, logit_rows) * c.vocab_size * 4;
}

/// Computes the estimate from the model's tensor index (works before any weight is read).
pub fn estimate(model: *const model_mod.Model, p: EstimateParams) Estimate {
    const c = &model.config;
    var total: u64 = 0;
    var largest: u64 = 0;
    var largest_name: []const u8 = "";
    var layer_bytes = [_]u64{0} ** 4096;
    var max_cols: u64 = 0;
    for (model.files) |f| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| {
            const info = kv.value_ptr.*;
            total += info.byte_len;
            if (info.byte_len > largest) {
                largest = info.byte_len;
                largest_name = info.name;
            }
            max_cols = @max(max_cols, info.cols());
            if (model.layerIndex(info.name)) |li| {
                if (li < layer_bytes.len) layer_bytes[li] += info.byte_len;
            }
        }
    }
    var largest_layer: u64 = 0;
    for (layer_bytes) |b| largest_layer = @max(largest_layer, b);
    // The model knows about transient buffers (e.g. transposing fused expert blocks).
    largest_layer = @max(largest_layer, model.largest_layer_bytes);

    const rows = @max(1, p.batch_size * p.max_prompt_tokens);
    const ws = workspaceBytes(c, rows, @max(1, p.batch_size));
    const ws_min = workspaceBytes(c, 1, 1);
    const kvd: u64 = c.num_kv_heads * c.head_dim;
    const kv = @as(u64, c.num_layers) * @max(1, p.batch_size) * (p.max_prompt_tokens + p.max_response_length + 1) * kvd * 4 * 2;
    const scratch = @as(u64, @max(1, p.threads)) * (max_cols + 1) * 4 * 2;
    const hidden: u64 = c.hidden_size;
    const inter: u64 = c.intermediate_size;
    const delta = @as(u64, c.num_layers) * @as(u64, p.lora_rank) * ((hidden + hidden) + (hidden + inter)) * 4;
    const export_peak = export_mod.peakBytes(model, p.export_dtype);
    const warp = model.warp();
    var top_k: usize = 0;
    var num_experts: usize = 0;
    for (model.layers) |l| if (l.moe) |m| {
        top_k = @max(top_k, m.top_k);
        num_experts = @max(num_experts, m.experts.len);
    };
    const cache_bytes: u64 = if (model.expert_cache) |ec| ec.capacity else 0;
    // In warp mode the "largest layer" is the trunk plus the experts one
    // token selects; the comfortable working set adds the expert cache.
    const resident_weights = if (warp) 2 * largest_layer + cache_bytes else 2 * largest_layer;
    const min_streamed = @max(largest_layer, largest) + ws_min + scratch + delta;
    const streamed = @max(resident_weights, largest) + ws + kv + scratch + delta;
    const mapped = total + ws + kv + scratch + delta;
    return .{
        .total_weight_bytes = total,
        .largest_layer_bytes = largest_layer,
        .largest_tensor_bytes = largest,
        .largest_tensor_name = largest_name,
        .num_layers = c.num_layers,
        .workspace_bytes = ws,
        .workspace_min_bytes = ws_min,
        .kv_cache_bytes = kv,
        .kernel_scratch_bytes = scratch,
        .delta_bytes = delta,
        .export_bytes = export_peak,
        .streamed_weights_bytes = resident_weights,
        .min_streamed_bytes = min_streamed,
        .streamed_bytes = streamed,
        .mapped_bytes = mapped,
        .warp = warp,
        .trunk_layer_bytes = model.largest_trunk_layer_bytes,
        .expert_bytes = model.largest_expert_bytes,
        .num_experts = num_experts,
        .top_k = top_k,
        .total_expert_bytes = model.total_expert_bytes,
        .expert_cache_bytes = cache_bytes,
    };
}

pub const CheckError = error{ BudgetTooSmall, WriteFailed };

/// Verifies that the budget can hold the minimum resident set. On failure an
/// explanation of what does not fit is written to `w` and `error.BudgetTooSmall`
/// is returned (the caller should exit non-zero). Notes about degraded modes
/// (no prefetch, KV cache spilled to scratch) are also written.
pub fn check(est: Estimate, budget: *const Budget, w: *Io.Writer) CheckError!void {
    if (!budget.limited()) return;
    const limit = budget.limitBytes();
    const need = est.min_streamed_bytes;
    if (limit < need) {
        w.print("Memory budget too small: --max-ram {f} leaves {f} after {f} headroom, but the minimum resident set is {f}:\n", .{ fmtBytes(budget.max_ram), fmtBytes(limit), fmtBytes(budget.headroom), fmtBytes(need) }) catch return error.WriteFailed;
        if (est.warp) {
            w.print("  largest layer {f} = trunk {f} + top-{d} routed experts of {f}, largest tensor {f} (both must be resident one at a time)\n", .{ fmtBytes(est.largest_layer_bytes), fmtBytes(est.trunk_layer_bytes), est.top_k, fmtBytes(est.expert_bytes), fmtBytes(est.largest_tensor_bytes) }) catch return error.WriteFailed;
        } else {
            w.print("  largest layer {f}, largest tensor {f} (both must be resident one at a time)\n", .{ fmtBytes(est.largest_layer_bytes), fmtBytes(est.largest_tensor_bytes) }) catch return error.WriteFailed;
        }
        w.print("  minimal forward workspace {f}, kernel scratch {f}, deltas {f}\n", .{ fmtBytes(est.workspace_min_bytes), fmtBytes(est.kernel_scratch_bytes), fmtBytes(est.delta_bytes) }) catch return error.WriteFailed;
        if (@max(est.largest_layer_bytes, est.largest_tensor_bytes) > limit) {
            w.print("  the largest layer/tensor alone ({f}) does not fit\n", .{fmtBytes(@max(est.largest_layer_bytes, est.largest_tensor_bytes))}) catch return error.WriteFailed;
        } else {
            w.print("  short by {f}\n", .{fmtBytes(need - limit)}) catch return error.WriteFailed;
        }
        return error.BudgetTooSmall;
    }
    if (limit < est.export_bytes) {
        w.print("Warning: exporting the largest tensor needs {f}; export may fail under this budget.\n", .{fmtBytes(est.export_bytes)}) catch return error.WriteFailed;
    }
    if (limit < est.streamed_bytes) {
        w.print("Note: budget {f} < {f}; prefetch may be disabled and KV caches / activations may spill to scratch ({s}).\n", .{ fmtBytes(limit), fmtBytes(est.streamed_bytes), budget.scratch_dir }) catch return error.WriteFailed;
    }
    if (est.warp and est.expert_cache_bytes < @as(u64, est.num_experts) * est.expert_bytes) {
        w.print("Note: the expert cache ({f}) holds fewer experts than one layer has ({d}); a prefill that routes to every expert re-reads experts from the store.\n", .{ fmtBytes(est.expert_cache_bytes), est.num_experts }) catch return error.WriteFailed;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "budgeted allocator enforces limit and tracks peak" {
    var ba = BudgetedAllocator.init(std.testing.allocator, 1000);
    const a = ba.allocator();
    const x = try a.alloc(u8, 600);
    try std.testing.expectEqual(@as(u64, 600), ba.currentBytes());
    try std.testing.expectError(error.OutOfMemory, a.alloc(u8, 500));
    try std.testing.expectEqual(@as(u64, 1), ba.refused);
    const y = try a.alloc(u8, 400);
    try std.testing.expectEqual(@as(u64, 1000), ba.peakBytes());
    a.free(x);
    try std.testing.expectEqual(@as(u64, 400), ba.currentBytes());
    a.free(y);
    try std.testing.expectEqual(@as(u64, 0), ba.currentBytes());
    try std.testing.expectEqual(@as(u64, 1000), ba.peakBytes());
    // Reclaim hook: a refused allocation asks the hook to free memory, then retries.
    const Hook = struct {
        alloc_: Allocator,
        held: ?[]u8,
        fn release(ctx: *anyopaque, _: usize) usize {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            const held = h.held orelse return 0;
            h.alloc_.free(held);
            h.held = null;
            return held.len;
        }
    };
    var hook = Hook{ .alloc_ = a, .held = try a.alloc(u8, 900) };
    ba.reclaim = .{ .ctx = &hook, .func = Hook.release };
    const z2 = try a.alloc(u8, 500);
    try std.testing.expect(hook.held == null);
    a.free(z2);
    ba.reclaim = null;
    // Unlimited.
    var ub = BudgetedAllocator.init(std.testing.allocator, 0);
    const z = try ub.allocator().alloc(u8, 5000);
    ub.allocator().free(z);
    try std.testing.expectEqual(@as(u64, 5000), ub.peakBytes());
}

test "budget time limit and rss" {
    var b = try Budget.init(std.testing.allocator, std.testing.io, .{ .max_ram = 1 << 20, .headroom = 1 << 10, .time_limit = Io.Duration.fromNanoseconds(0) });
    defer b.deinit();
    try std.testing.expectError(error.TimeLimitExceeded, b.checkTime());
    try std.testing.expectEqual(@as(u64, (1 << 20) - (1 << 10)), b.limitBytes());
    const r = b.report();
    if (builtin.os.tag == .linux) try std.testing.expect(r.rss.available and r.rss.peak > 0);
    var sink: Io.Writer.Allocating = .init(std.testing.allocator);
    defer sink.deinit();
    try r.print(&sink.writer, "test");
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "budget 1.0MB") != null);
    var ok = try Budget.init(std.testing.allocator, std.testing.io, .{ .time_limit = Io.Duration.fromSeconds(3600) });
    defer ok.deinit();
    try ok.checkTime();
    try std.testing.expectError(error.BudgetTooSmall, Budget.init(std.testing.allocator, std.testing.io, .{ .max_ram = 100, .headroom = 100 }));
    try std.testing.expectEqual(@as(u64, (8 << 30) / 10), defaultHeadroom(8 << 30));
    try std.testing.expectEqual(default_headroom_floor, defaultHeadroom(1 << 30));
    try std.testing.expectEqual(@as(u64, 100 << 20), defaultHeadroom(200 << 20));
}
