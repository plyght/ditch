//! Bounded tensor access: a `WeightStore` hands out `tensor.Weight` views over
//! either a memory mapping (`mapped`, resident memory unbounded) or budgeted
//! heap buffers filled with positional reads (`streamed`). Also provides the
//! scratch-file primitives used to spill activations and KV caches when they do
//! not fit the budget, and post-export reload validation.
//!
//! Streamed mode trades disk traffic for memory: every forward pass re-reads
//! each layer's weights (and the LM head) once, so a decode step costs one full
//! read of the model. Decode throughput is therefore bounded by the storage
//! read bandwidth, not by compute; see `WeightStore.weight_bytes_read`.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const budget_mod = @import("budget.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Budget = budget_mod.Budget;

pub const Mode = enum { mapped, streamed };

/// Location and shape of a matrix (or a 2-D slice of a bigger tensor, e.g. one
/// expert of a stacked expert tensor) inside a safetensors file.
pub const WeightRef = struct {
    name: []const u8,
    /// Index into `WeightStore.files`.
    file: u32,
    /// Absolute byte offset of the first element within the file.
    offset: u64,
    rows: usize,
    cols: usize,
    dtype: tensor.DType,

    pub fn byteLen(self: WeightRef) usize {
        return self.rows * self.dtype.rowBytes(self.cols);
    }

    /// A ref covering rows `[r0, r0 + n)` of this matrix.
    pub fn rowSlice(self: WeightRef, r0: usize, n: usize) WeightRef {
        std.debug.assert(r0 + n <= self.rows);
        var out = self;
        out.offset += @as(u64, r0) * self.dtype.rowBytes(self.cols);
        out.rows = n;
        return out;
    }

    /// Placeholder view (no data) carrying the shape; valid for shape queries only.
    pub fn shapeOnly(self: WeightRef) Weight {
        return .{ .data = &.{}, .dtype = self.dtype, .rows = self.rows, .cols = self.cols };
    }
};

/// A strided column range `lo, lo + stride, ... < hi` of a block (see `WeightStore.acquireColumns`).
pub const ColumnSpec = struct { lo: usize, hi: usize, stride: usize = 1 };

/// A weight acquired from the store. Must be returned with `WeightStore.release`.
pub const Lease = struct {
    weight: Weight,
    /// Owned buffer in streamed mode; empty in mapped mode.
    buf: []u8 = &.{},
};

pub const WeightStore = struct {
    mode: Mode,
    /// Allocator for weight buffers (the budget's allocator when budgeted).
    gpa: Allocator,
    io: Io,
    files: []const *safetensors.File,
    budget: ?*Budget,
    /// Recycled buffers. Bounded by `pool_cap` bytes; dropped under budget pressure.
    free: std.ArrayList([]u8) = .empty,
    pooled_bytes: u64 = 0,
    pool_lock: std.atomic.Value(bool) = .init(false),
    pool_cap: u64,
    /// Whether the next layer is read on a background task while the current one computes.
    prefetch_enabled: bool,
    pending: ?Pending = null,
    row_buf: []u8 = &.{},
    /// Reclaim hook that was installed on the budget before ours (another
    /// store of the same budget, e.g. a reloaded export); chained on reclaim.
    prev_reclaim: ?budget_mod.BudgetedAllocator.Reclaim = null,
    /// Disk traffic accounting (bytes of weights read from files, streamed mode only).
    weight_bytes_read: std.atomic.Value(u64) = .init(0),
    read_failed: std.atomic.Value(bool) = .init(false),

    const Pending = struct {
        key: usize,
        leases: []Lease,
        group: Io.Group,
    };

    pub const InitOptions = struct {
        budget: ?*Budget = null,
        prefetch: bool = true,
        /// Bytes of released buffers kept for reuse (null = derived from the budget).
        pool_cap: ?u64 = null,
    };

    pub fn init(gpa: Allocator, io: Io, files: []const *safetensors.File, mode: Mode, opts: InitOptions) WeightStore {
        const cap: u64 = opts.pool_cap orelse blk: {
            if (opts.budget) |b| {
                if (b.limited()) break :blk b.limitBytes() / 4;
            }
            break :blk 512 * 1024 * 1024;
        };
        return .{
            .mode = mode,
            .gpa = gpa,
            .io = io,
            .files = files,
            .budget = opts.budget,
            .pool_cap = cap,
            .prefetch_enabled = opts.prefetch and mode == .streamed,
        };
    }

    pub fn deinit(self: *WeightStore) void {
        self.unregisterReclaim();
        self.discardPending();
        self.drainPool();
        self.free.deinit(self.gpa);
        if (self.row_buf.len > 0) self.gpa.free(self.row_buf);
    }

    /// Lets the budget's allocator drop this store's recycled buffers when an
    /// allocation is refused. `self` must have a stable address from now on.
    /// Several stores may share one budget (e.g. while an export is validated
    /// against the source model); their hooks form a chain.
    pub fn registerReclaim(self: *WeightStore) void {
        const b = self.budget orelse return;
        self.prev_reclaim = b.alloc.reclaim;
        b.alloc.reclaim = .{ .ctx = self, .func = reclaimCb };
    }

    pub fn unregisterReclaim(self: *WeightStore) void {
        const b = self.budget orelse return;
        if (b.alloc.reclaim) |r| {
            if (r.ctx == @as(*anyopaque, @ptrCast(self))) b.alloc.reclaim = self.prev_reclaim;
        }
    }

    fn reclaimCb(ctx: *anyopaque, len: usize) usize {
        const self: *WeightStore = @ptrCast(@alignCast(ctx));
        var freed: usize = 0;
        {
            self.pool_lock_acquire();
            defer self.pool_lock_release();
            freed = @intCast(self.pooled_bytes);
            self.drainPoolLocked();
        }
        if (self.prev_reclaim) |r| freed += r.func(r.ctx, len);
        return freed;
    }

    /// Drops every recycled buffer and any pending prefetch, returning their
    /// memory to the budget (e.g. before another model is loaded beside this one).
    pub fn trim(self: *WeightStore) void {
        self.discardPending();
        self.drainPool();
    }

    fn pool_lock_acquire(self: *WeightStore) void {
        while (self.pool_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn pool_lock_release(self: *WeightStore) void {
        self.pool_lock.store(false, .release);
    }

    pub fn refFor(self: *const WeightStore, file_index: usize, info: safetensors.TensorInfo) WeightRef {
        _ = self;
        return .{ .name = info.name, .file = @intCast(file_index), .offset = info.offset, .rows = info.rows(), .cols = info.cols(), .dtype = info.dtype };
    }

    pub fn lookup(self: *const WeightStore, name: []const u8) ?WeightRef {
        for (self.files, 0..) |f, i| {
            if (f.get(name)) |t| return self.refFor(i, t);
        }
        return null;
    }

    // -- buffers ------------------------------------------------------------

    fn takeFromPool(self: *WeightStore, len: usize) ?[]u8 {
        self.pool_lock_acquire();
        defer self.pool_lock_release();
        var best: ?usize = null;
        for (self.free.items, 0..) |b, i| {
            if (b.len >= len and b.len <= len + len / 4) {
                if (best == null or b.len < self.free.items[best.?].len) best = i;
            }
        }
        const i = best orelse return null;
        const b = self.free.swapRemove(i);
        self.pooled_bytes -= b.len;
        return b;
    }

    fn drainPool(self: *WeightStore) void {
        self.pool_lock_acquire();
        defer self.pool_lock_release();
        self.drainPoolLocked();
    }

    fn drainPoolLocked(self: *WeightStore) void {
        for (self.free.items) |b| self.gpa.free(b);
        self.free.clearRetainingCapacity();
        self.pooled_bytes = 0;
    }

    /// Allocates (or recycles) a buffer of exactly `len` usable bytes. When the
    /// budget refuses, pooled buffers are dropped and the allocation retried.
    fn allocBuf(self: *WeightStore, len: usize) ![]u8 {
        if (self.takeFromPool(len)) |b| return b;
        return self.gpa.alloc(u8, len) catch |err| switch (err) {
            error.OutOfMemory => {
                if (self.pooled_bytes == 0) return err;
                self.drainPool();
                return self.gpa.alloc(u8, len);
            },
        };
    }

    fn freeBuf(self: *WeightStore, b: []u8) void {
        self.pool_lock_acquire();
        defer self.pool_lock_release();
        if (self.pooled_bytes + b.len <= self.pool_cap and self.free.items.len < 64) {
            self.free.append(self.gpa, b) catch {
                self.gpa.free(b);
                return;
            };
            self.pooled_bytes += b.len;
        } else {
            self.gpa.free(b);
        }
    }

    // -- reading --------------------------------------------------------------

    fn readInto(self: *WeightStore, ref: WeightRef, buf: []u8) !void {
        const f = self.files[ref.file];
        try f.readRange(self.io, ref.offset, buf[0..ref.byteLen()]);
        _ = self.weight_bytes_read.fetchAdd(ref.byteLen(), .monotonic);
        if (self.budget) |b| b.noteWeightRead(ref.byteLen());
    }

    fn mappedView(self: *const WeightStore, ref: WeightRef) Weight {
        const f = self.files[ref.file];
        return .{ .data = f.mappedSlice(ref.offset, ref.byteLen()), .dtype = ref.dtype, .rows = ref.rows, .cols = ref.cols };
    }

    /// Makes `ref` resident. In mapped mode this is a view; in streamed mode a
    /// budgeted buffer is filled with one positional read.
    pub fn acquire(self: *WeightStore, ref: WeightRef) !Lease {
        switch (self.mode) {
            .mapped => return .{ .weight = self.mappedView(ref) },
            .streamed => {
                const buf = try self.allocBuf(ref.byteLen());
                errdefer self.freeBuf(buf);
                try self.readInto(ref, buf);
                return .{ .weight = .{ .data = buf[0..ref.byteLen()], .dtype = ref.dtype, .rows = ref.rows, .cols = ref.cols }, .buf = buf };
            },
        }
    }

    pub fn release(self: *WeightStore, lease: Lease) void {
        if (lease.buf.len > 0) self.freeBuf(lease.buf);
    }

    /// Makes column ranges of the `[rows][cols]` block `ref` resident
    /// transposed: `out[i]` is the `[hi - lo][rows]` matrix holding columns
    /// `[lo, hi)` of `ranges[i]`. Used for expert tensors stored as
    /// `[hidden][2 * inter]` where the kernels need `[inter][hidden]`. The block
    /// is read once (a transient buffer of `ref.byteLen()` in streamed mode) and
    /// every result is an owned buffer released with `release`.
    pub fn acquireTransposed(self: *WeightStore, ref: WeightRef, ranges: []const [2]usize, out: []Lease) !void {
        var specs: [8]ColumnSpec = undefined;
        std.debug.assert(ranges.len <= specs.len);
        for (ranges, 0..) |r, i| specs[i] = .{ .lo = r[0], .hi = r[1] };
        return self.acquireColumns(ref, specs[0..ranges.len], out);
    }

    /// `acquireTransposed` with a column stride: `out[i]` holds columns
    /// `lo, lo + stride, ...` below `hi` of `specs[i]` (gpt-oss stores gate and
    /// up columns interleaved). Floating-point blocks only.
    pub fn acquireColumns(self: *WeightStore, ref: WeightRef, specs: []const ColumnSpec, out: []Lease) !void {
        std.debug.assert(out.len >= specs.len);
        if (ref.dtype.isQuantized()) return error.UnsupportedDType;
        const block = try self.acquire(ref);
        defer self.release(block);
        const es = ref.dtype.size();
        const src = block.weight.data;
        var done: usize = 0;
        errdefer for (out[0..done]) |l| self.release(l);
        for (specs) |spec| {
            std.debug.assert(spec.lo <= spec.hi and spec.hi <= ref.cols and spec.stride > 0);
            const n = (spec.hi - spec.lo + spec.stride - 1) / spec.stride;
            const buf = try self.allocBuf(n * ref.rows * es);
            var r: usize = 0;
            while (r < ref.rows) : (r += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    const c = spec.lo + j * spec.stride;
                    @memcpy(buf[(j * ref.rows + r) * es ..][0..es], src[(r * ref.cols + c) * es ..][0..es]);
                }
            }
            out[done] = .{ .weight = .{ .data = buf[0 .. n * ref.rows * es], .dtype = ref.dtype, .rows = n, .cols = ref.rows }, .buf = buf };
            done += 1;
        }
    }

    /// Acquires several refs at once. If a prefetch for `key` is pending its
    /// buffers are used; otherwise the refs are read synchronously.
    pub fn acquireSet(self: *WeightStore, key: usize, refs: []const WeightRef, out: []Lease) !void {
        std.debug.assert(out.len >= refs.len);
        if (self.pending) |*p| {
            if (p.key == key and p.leases.len == refs.len) {
                p.group.await(self.io) catch {};
                const failed = self.read_failed.load(.acquire);
                const leases = p.leases;
                self.pending = null;
                if (failed) {
                    for (leases) |l| self.release(l);
                    self.gpa.free(leases);
                    self.read_failed.store(false, .release);
                    return error.ReadFailed;
                }
                @memcpy(out[0..refs.len], leases);
                self.gpa.free(leases);
                return;
            }
            self.discardPending();
        }
        var done: usize = 0;
        errdefer for (out[0..done]) |l| self.release(l);
        for (refs) |r| {
            out[done] = try self.acquire(r);
            done += 1;
        }
    }

    pub fn releaseSet(self: *WeightStore, leases: []const Lease) void {
        for (leases) |l| self.release(l);
    }

    fn readTask(self: *WeightStore, ref: WeightRef, buf: []u8) Io.Cancelable!void {
        self.readInto(ref, buf) catch {
            self.read_failed.store(true, .release);
        };
    }

    /// Starts reading `refs` on background tasks so that a later
    /// `acquireSet(key, ...)` finds them resident. Silently does nothing when
    /// prefetching is disabled or the buffers do not fit the budget (the
    /// later acquire then reads synchronously). Only one prefetch is pending
    /// at a time; the previous one is discarded.
    pub fn prefetch(self: *WeightStore, key: usize, refs: []const WeightRef) void {
        if (!self.prefetch_enabled or refs.len == 0) return;
        self.discardPending();
        // Prefetching must not starve the small runtime allocations (kernel
        // scratch, logits, residual chunks) that happen while the next layer is
        // in flight: require the buffers plus a margin of 1/8 of the budget to
        // be available, otherwise skip this prefetch (the acquire then reads
        // synchronously).
        if (self.budget) |b| {
            if (b.limited()) {
                var need: u64 = b.limitBytes() / 8;
                for (refs) |r| need += r.byteLen();
                if (b.available() < need) return;
            }
        }
        const leases = self.gpa.alloc(Lease, refs.len) catch return;
        var n: usize = 0;
        for (refs) |r| {
            const buf = self.allocBuf(r.byteLen()) catch {
                // Not enough budget for two layers: give up on prefetching for good.
                for (leases[0..n]) |l| self.release(l);
                self.gpa.free(leases);
                self.prefetch_enabled = false;
                return;
            };
            leases[n] = .{ .weight = .{ .data = buf[0..r.byteLen()], .dtype = r.dtype, .rows = r.rows, .cols = r.cols }, .buf = buf };
            n += 1;
        }
        self.pending = .{ .key = key, .leases = leases, .group = .init };
        const p = &self.pending.?;
        for (refs, 0..) |r, i| p.group.async(self.io, readTask, .{ self, r, leases[i].buf });
    }

    fn discardPending(self: *WeightStore) void {
        const p = &(self.pending orelse return);
        p.group.await(self.io) catch {};
        for (p.leases) |l| self.release(l);
        self.gpa.free(p.leases);
        self.read_failed.store(false, .release);
        self.pending = null;
    }

    /// Reads one row of `ref` as f32 without making the whole matrix resident.
    pub fn readRow(self: *WeightStore, ref: WeightRef, r: usize, out: []f32) !void {
        const row_bytes = ref.dtype.rowBytes(ref.cols);
        switch (self.mode) {
            .mapped => self.mappedView(ref).row(r, out),
            .streamed => {
                if (self.row_buf.len < row_bytes) {
                    if (self.row_buf.len > 0) self.gpa.free(self.row_buf);
                    self.row_buf = &.{};
                    self.row_buf = try self.gpa.alloc(u8, row_bytes);
                }
                const f = self.files[ref.file];
                try f.readRange(self.io, ref.offset + @as(u64, r) * row_bytes, self.row_buf[0..row_bytes]);
                _ = self.weight_bytes_read.fetchAdd(row_bytes, .monotonic);
                if (self.budget) |b| b.noteWeightRead(row_bytes);
                tensor.convertToF32(ref.dtype, self.row_buf[0..row_bytes], out[0..ref.cols]);
            },
        }
    }

    /// Reads a whole tensor as a freshly allocated f32 vector (for norms/biases).
    /// The transient read buffer is budgeted when possible; under a budget too
    /// small even for that, an unbudgeted buffer is used so the model still
    /// loads and the feasibility check can explain the problem.
    pub fn readVecF32(self: *WeightStore, gpa: Allocator, ref: WeightRef) ![]f32 {
        const n = ref.rows * ref.cols;
        const out = try gpa.alloc(f32, n);
        errdefer gpa.free(out);
        switch (self.mode) {
            .mapped => tensor.convertToF32(ref.dtype, self.mappedView(ref).data, out),
            .streamed => {
                const fallback = std.heap.page_allocator;
                var budgeted = true;
                const buf = self.gpa.alloc(u8, ref.byteLen()) catch blk: {
                    budgeted = false;
                    break :blk try fallback.alloc(u8, ref.byteLen());
                };
                defer if (budgeted) self.gpa.free(buf) else fallback.free(buf);
                try self.readInto(ref, buf);
                tensor.convertToF32(ref.dtype, buf, out);
            },
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// Scratch files
// ---------------------------------------------------------------------------

/// A temporary file in the scratch directory, deleted on deinit. All I/O is
/// positional and accounted in the budget.
pub const ScratchFile = struct {
    io: Io,
    dir: Io.Dir,
    name: [26]u8,
    file: Io.File,
    budget: ?*Budget,
    bytes_written: u64 = 0,
    bytes_read: u64 = 0,

    pub fn create(io: Io, scratch_dir: []const u8, budget: ?*Budget) !ScratchFile {
        const cwd = Io.Dir.cwd();
        try cwd.createDirPath(io, scratch_dir);
        var dir = try cwd.openDir(io, scratch_dir, .{});
        errdefer dir.close(io);
        var rnd: [8]u8 = undefined;
        io.random(&rnd);
        var name: [26]u8 = undefined;
        _ = std.fmt.bufPrint(&name, "ditch-{x:0>16}.tmp", .{std.mem.readInt(u64, &rnd, .little)}) catch unreachable;
        const file = try dir.createFile(io, &name, .{ .read = true, .truncate = true });
        return .{ .io = io, .dir = dir, .name = name, .file = file, .budget = budget };
    }

    pub fn deinit(self: *ScratchFile) void {
        self.file.close(self.io);
        self.dir.deleteFile(self.io, &self.name) catch {};
        self.dir.close(self.io);
    }

    pub fn write(self: *ScratchFile, offset: u64, bytes: []const u8) !void {
        try self.file.writePositionalAll(self.io, bytes, offset);
        self.bytes_written += bytes.len;
        if (self.budget) |b| b.noteScratchWrite(bytes.len);
    }

    pub fn read(self: *ScratchFile, offset: u64, bytes: []u8) !void {
        const n = try self.file.readPositionalAll(self.io, bytes, offset);
        if (n != bytes.len) return error.UnexpectedEndOfFile;
        self.bytes_read += bytes.len;
        if (self.budget) |b| b.noteScratchRead(bytes.len);
    }

    pub fn writeF32(self: *ScratchFile, index: u64, data: []const f32) !void {
        return self.write(index * 4, std.mem.sliceAsBytes(data));
    }

    pub fn readF32(self: *ScratchFile, index: u64, data: []f32) !void {
        return self.read(index * 4, std.mem.sliceAsBytes(data));
    }
};

// ---------------------------------------------------------------------------
// Spillable activations
// ---------------------------------------------------------------------------

/// The residual stream `[rows][hidden]` of one forward call. Lives in RAM when
/// the budget allows, otherwise in a scratch file accessed chunk by chunk.
pub const Activations = struct {
    gpa: Allocator,
    rows: usize,
    hidden: usize,
    backing: union(enum) {
        ram: []f32,
        scratch: ScratchFile,
    },

    pub const InitOptions = struct {
        budget: ?*Budget = null,
        scratch_dir: []const u8 = "scratch",
        /// Bytes that must remain available after allocating in RAM (e.g. for weights).
        reserve: u64 = 0,
        /// Bytes the budget could reclaim on demand (unpinned expert-cache
        /// entries), counted as available.
        evictable: u64 = 0,
        force_scratch: bool = false,
    };

    pub fn init(gpa: Allocator, io: Io, rows: usize, hidden: usize, opts: InitOptions) !Activations {
        const bytes: u64 = @as(u64, rows) * hidden * 4;
        var use_ram = !opts.force_scratch;
        if (opts.budget) |b| {
            if (b.limited() and b.available() + opts.evictable < bytes + opts.reserve) use_ram = false;
        }
        if (use_ram) {
            if (gpa.alloc(f32, rows * hidden)) |buf| {
                return .{ .gpa = gpa, .rows = rows, .hidden = hidden, .backing = .{ .ram = buf } };
            } else |_| {}
        }
        const sf = try ScratchFile.create(io, opts.scratch_dir, opts.budget);
        return .{ .gpa = gpa, .rows = rows, .hidden = hidden, .backing = .{ .scratch = sf } };
    }

    pub fn deinit(self: *Activations) void {
        switch (self.backing) {
            .ram => |b| self.gpa.free(b),
            .scratch => |*s| s.deinit(),
        }
    }

    pub fn spilled(self: *const Activations) bool {
        return self.backing == .scratch;
    }

    /// Returns rows `[start, start+n)` for in-place processing. In RAM mode this
    /// is a slice of the buffer; in scratch mode the rows are read into `tmp`.
    pub fn chunk(self: *Activations, start: usize, n: usize, tmp: []f32) ![]f32 {
        const h = self.hidden;
        switch (self.backing) {
            .ram => |b| return b[start * h ..][0 .. n * h],
            .scratch => |*s| {
                const dst = tmp[0 .. n * h];
                try s.readF32(@as(u64, start) * h, dst);
                return dst;
            },
        }
    }

    /// Same as `chunk` but without reading (the caller overwrites every row).
    pub fn chunkUninit(self: *Activations, start: usize, n: usize, tmp: []f32) []f32 {
        const h = self.hidden;
        return switch (self.backing) {
            .ram => |b| b[start * h ..][0 .. n * h],
            .scratch => tmp[0 .. n * h],
        };
    }

    /// Writes back the chunk returned by `chunk`/`chunkUninit` (no-op in RAM mode).
    pub fn commit(self: *Activations, start: usize, n: usize, xs: []const f32) !void {
        const h = self.hidden;
        switch (self.backing) {
            .ram => |b| std.debug.assert(xs.ptr == b[start * h ..].ptr),
            .scratch => |*s| try s.writeF32(@as(u64, start) * h, xs[0 .. n * h]),
        }
    }

    pub fn readRow(self: *Activations, r: usize, out: []f32) !void {
        const h = self.hidden;
        switch (self.backing) {
            .ram => |b| @memcpy(out[0..h], b[r * h ..][0..h]),
            .scratch => |*s| try s.readF32(@as(u64, r) * h, out[0..h]),
        }
    }
};

// ---------------------------------------------------------------------------
// Budget-aware sizing
// ---------------------------------------------------------------------------

/// Rows a forward workspace may hold under the model's budget, given that a
/// KV cache of `kv_bytes`, the logits for `logit_rows` rows and the resident
/// weights must also fit. `forward` chunks longer inputs per layer (spilling
/// the residual stream when needed), so a smaller workspace never changes
/// results. Returns `wanted` when there is no budget.
pub fn workspaceRows(model: *const model_mod.Model, wanted: usize, logit_rows: usize, kv_bytes: u64) usize {
    const b = model.budget orelse return wanted;
    if (!b.limited()) return wanted;
    const c = &model.config;
    const per_row: u64 = model_mod.Workspace.bytesPerRow(c);
    // Unpinned expert-cache entries are given back on demand, so they count as available.
    const avail = b.available() + model.expertCacheEvictable();
    // Weights, logits, KV cache, the residual-stream buffer for all rows and a
    // margin for kernel scratch / residual chunks / token arrays.
    const reserve = model.residentWeightNeed() + @as(u64, logit_rows) * c.vocab_size * 4 + kv_bytes + @as(u64, wanted) * c.hidden_size * 4 + b.limitBytes() / 8;
    if (avail <= reserve) return 1;
    const cap: u64 = (avail - reserve) / per_row;
    return @intCast(@max(1, @min(@as(u64, wanted), cap)));
}

// ---------------------------------------------------------------------------
// Export validation
// ---------------------------------------------------------------------------

pub const Validation = struct {
    prompts: usize,
    /// Largest absolute first-token logit difference over all prompts.
    max_abs_diff: f32,
    /// Fraction of prompts whose argmax token matches.
    argmax_match: f32,
};

/// Reloads the model exported to `out_dir` through the streamed path (under
/// `budget` when given) and compares first-token logits for `prompts`
/// (already tokenised) against `model` (with its current deltas applied).
pub fn validateExport(gpa: Allocator, io: Io, pool: *const tensor.Pool, model: *const model_mod.Model, out_dir: []const u8, prompts: []const []const u32, budget: ?*Budget) !Validation {
    const c = &model.config;
    const vocab = c.vocab_size;
    const ref = try firstTokenLogits(gpa, model, prompts);
    defer gpa.free(ref);
    // Two models share the budget for a while: give back what the source model's store and expert cache hold.
    @constCast(&model.store).trim();
    if (model.expert_cache) |ec| ec.trim();
    const reloaded = try model_mod.Model.loadWithOptions(gpa, io, pool, out_dir, .{
        .store = .streamed,
        .budget = budget,
        .scratch_dir = if (budget) |b| b.scratch_dir else null,
    });
    defer reloaded.deinit();
    if (reloaded.config.vocab_size != vocab or reloaded.config.num_layers != c.num_layers) return error.ExportMismatch;
    const got = try firstTokenLogits(gpa, reloaded, prompts);
    defer gpa.free(got);
    var max_diff: f32 = 0;
    var matches: usize = 0;
    for (0..prompts.len) |b| {
        const a = ref[b * vocab ..][0..vocab];
        const g = got[b * vocab ..][0..vocab];
        var ia: usize = 0;
        var ig: usize = 0;
        for (0..vocab) |i| {
            max_diff = @max(max_diff, @abs(a[i] - g[i]));
            if (a[i] > a[ia]) ia = i;
            if (g[i] > g[ig]) ig = i;
        }
        if (ia == ig) matches += 1;
    }
    return .{ .prompts = prompts.len, .max_abs_diff = max_diff, .argmax_match = if (prompts.len == 0) 1 else @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(prompts.len)) };
}

/// First-token logits `[prompts][vocab]`, one prompt at a time so that the
/// KV cache (one layer slice of it when spilled) stays as small as possible;
/// validation runs while two models share the budget.
fn firstTokenLogits(gpa: Allocator, model: *const model_mod.Model, prompts: []const []const u32) ![]f32 {
    const c = &model.config;
    const out = try gpa.alloc(f32, prompts.len * c.vocab_size);
    errdefer gpa.free(out);
    for (prompts, 0..) |p, i| {
        const kv_bytes = model_mod.KvCache.bytesFor(c.num_layers, 1, p.len + 1, c.num_kv_heads * c.head_dim);
        const rows = workspaceRows(model, @max(1, p.len), 1, kv_bytes);
        var ws = try model_mod.Workspace.init(model.gpa, c, rows, 1);
        defer ws.deinit();
        var cache = try model_mod.KvCache.initFor(model, model.gpa, 1, p.len + 1);
        defer cache.deinit();
        try model_mod.prefill(model, &ws, &cache, &.{p}, out[i * c.vocab_size ..][0..c.vocab_size], null);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "scratch file and activations spill" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const scratch_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "scratch" });
    defer gpa.free(scratch_dir);

    var act = try Activations.init(gpa, io, 5, 4, .{ .scratch_dir = scratch_dir, .force_scratch = true });
    defer act.deinit();
    try std.testing.expect(act.spilled());
    var tmpbuf: [8]f32 = undefined;
    // Write rows 0..2 then 2..5.
    var xs = act.chunkUninit(0, 2, &tmpbuf);
    for (xs, 0..) |*x, i| x.* = @floatFromInt(i);
    try act.commit(0, 2, xs);
    var tmpbuf2: [12]f32 = undefined;
    xs = act.chunkUninit(2, 3, &tmpbuf2);
    for (xs, 0..) |*x, i| x.* = @floatFromInt(8 + i);
    try act.commit(2, 3, xs);
    var row: [4]f32 = undefined;
    try act.readRow(1, &row);
    try std.testing.expectEqual(@as(f32, 5), row[1]);
    try act.readRow(4, &row);
    try std.testing.expectEqual(@as(f32, 19), row[3]);
    const back = try act.chunk(1, 2, &tmpbuf);
    try std.testing.expectEqual(@as(f32, 4), back[0]);
    try std.testing.expectEqual(@as(f32, 8), back[4]);
    try std.testing.expectEqual(@as(u64, 80), act.backing.scratch.bytes_written);

    // RAM mode when it fits.
    var ram = try Activations.init(gpa, io, 5, 4, .{ .scratch_dir = scratch_dir });
    defer ram.deinit();
    try std.testing.expect(!ram.spilled());

    // Unwritable scratch path (a regular file where a directory is required).
    try tmp.dir.writeFile(io, .{ .sub_path = "notadir", .data = "x" });
    const bad = try std.fs.path.join(gpa, &.{ path_buf[0..n], "notadir", "scratch" });
    defer gpa.free(bad);
    try std.testing.expectError(error.NotDir, Activations.init(gpa, io, 5, 4, .{ .scratch_dir = bad, .force_scratch = true }));
}
