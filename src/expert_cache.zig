//! Bounded LRU cache of resident routed experts ("warp mode").
//!
//! In streamed mode the trunk of a mixture-of-experts model (attention,
//! norms, router, shared experts, embeddings, LM head) is acquired per layer
//! as usual, while the routed experts are made resident on demand through
//! this cache: after the router has picked the top-k experts of a token
//! batch, only the union of the selected experts of that layer is needed,
//! and the ones that are already resident cost nothing. Entries are keyed by
//! (layer, expert) and hold all three matrices of an expert (gate, up, down,
//! or the slices of a fused block). Misses of one layer are read as a group
//! on background tasks (`prefetch`), evictions are least-recently-used and
//! never touch a pinned entry (one that is being computed with), and every
//! access is counted per expert so that the hottest experts can be written
//! to a "hotlist" at the end of a run and loaded again at the start of the
//! next one.
//!
//! The capacity is a soft bound: a layer whose selected experts do not fit
//! still runs, one expert resident at a time, and the budget's allocator can
//! ask the cache to give memory back (`registerReclaim`) when another
//! allocation is refused. The hard minimum is therefore the trunk plus one
//! expert, and the comfortable working set the trunk plus every expert a
//! layer routes to.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const stream = @import("stream.zig");
const moe = @import("moe.zig");
const model_mod = @import("model.zig");
const budget_mod = @import("budget.zig");
const lua = @import("lua.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Lease = stream.Lease;

pub const Key = struct {
    layer: u32,
    expert: u32,

    pub fn init(layer: usize, expert: usize) Key {
        return .{ .layer = @intCast(layer), .expert = @intCast(expert) };
    }

    fn int(self: Key) u64 {
        return (@as(u64, self.layer) << 32) | self.expert;
    }

    fn fromInt(v: u64) Key {
        return .{ .layer = @intCast(v >> 32), .expert = @truncate(v) };
    }
};

/// A resident expert. Valid (and never evicted) while `pins > 0`.
pub const Entry = struct {
    key: Key,
    gate: Weight = empty_weight,
    up: Weight = empty_weight,
    down: Weight = empty_weight,
    leases: [3]Lease = undefined,
    n_leases: usize = 0,
    bytes: u64,
    pins: u32 = 0,
    /// Loaded by `prefetch`/`warm` and not used since: the first `acquire` counts as a miss.
    prefetched: bool = false,
    prev: ?*Entry = null,
    next: ?*Entry = null,
    /// Set by a background load that failed; the entry is dropped when the group finishes.
    failed: std.atomic.Value(bool) = .init(false),
};

const empty_weight = Weight{ .data = &.{}, .dtype = .f32, .rows = 0, .cols = 0 };

pub const Stats = struct {
    capacity: u64 = 0,
    resident: u64 = 0,
    resident_bytes: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    bytes_read: u64 = 0,
    evictions: u64 = 0,
    warmed: u64 = 0,
    decode_steps: u64 = 0,
    decode_misses: u64 = 0,
    decode_misses_max: u64 = 0,
    /// Distinct (layer, expert) pairs used at least once.
    visited: u64 = 0,

    pub fn hitRate(self: Stats) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn print(self: Stats, w: *Io.Writer, label: []const u8) !void {
        try w.print("{s}: capacity {f}, resident {d} experts ({f}), hits {d}, misses {d} ({d:.1}% hit rate), bytes read {f}, evictions {d}, warmed {d}, visited {d}", .{
            label,                  budget_mod.fmtBytes(self.capacity),   self.resident,  budget_mod.fmtBytes(self.resident_bytes), self.hits,    self.misses,
            100.0 * self.hitRate(), budget_mod.fmtBytes(self.bytes_read), self.evictions, self.warmed,                              self.visited,
        });
        if (self.decode_steps > 0) {
            const avg = @as(f64, @floatFromInt(self.decode_misses)) / @as(f64, @floatFromInt(self.decode_steps));
            try w.print("; decode: {d} steps, {d:.2} misses/step (max {d})", .{ self.decode_steps, avg, self.decode_misses_max });
        }
        try w.writeAll("\n");
    }
};

pub const ExpertCache = struct {
    gpa: Allocator,
    io: Io,
    store: *stream.WeightStore,
    capacity: u64,
    used: u64 = 0,
    map: std.AutoHashMapUnmanaged(u64, *Entry) = .empty,
    /// LRU order: `head` is the least recently used entry, `tail` the most recent.
    head: ?*Entry = null,
    tail: ?*Entry = null,
    /// Access counts per (layer, expert); survive eviction.
    uses: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    lock: std.atomic.Value(bool) = .init(false),
    hits: u64 = 0,
    misses: u64 = 0,
    bytes_read: u64 = 0,
    evictions: u64 = 0,
    warmed: u64 = 0,
    decode_steps: u64 = 0,
    decode_misses: u64 = 0,
    decode_misses_max: u64 = 0,
    step_misses: u64 = 0,
    pending: ?Pending = null,
    budget: ?*budget_mod.Budget = null,
    prev_reclaim: ?budget_mod.BudgetedAllocator.Reclaim = null,

    const Pending = struct {
        group: Io.Group,
        entries: std.ArrayList(*Entry),
    };

    pub fn init(gpa: Allocator, io: Io, store: *stream.WeightStore, capacity: u64) ExpertCache {
        return .{ .gpa = gpa, .io = io, .store = store, .capacity = capacity };
    }

    pub fn deinit(self: *ExpertCache) void {
        self.finishPending();
        self.unregisterReclaim();
        var e = self.head;
        while (e) |cur| {
            e = cur.next;
            self.store.releaseSet(cur.leases[0..cur.n_leases]);
            self.gpa.destroy(cur);
        }
        self.map.deinit(self.gpa);
        self.uses.deinit(self.gpa);
    }

    // -- locking --------------------------------------------------------------

    fn lockAcquire(self: *ExpertCache) void {
        while (self.lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn tryLock(self: *ExpertCache) bool {
        return self.lock.cmpxchgStrong(false, true, .acquire, .monotonic) == null;
    }

    fn lockRelease(self: *ExpertCache) void {
        self.lock.store(false, .release);
    }

    // -- budget integration -------------------------------------------------

    /// Lets the budget's allocator evict unpinned experts when an allocation
    /// is refused (chained with any hook installed before, e.g. the store's).
    pub fn registerReclaim(self: *ExpertCache, budget: *budget_mod.Budget) void {
        self.budget = budget;
        self.prev_reclaim = budget.alloc.reclaim;
        budget.alloc.reclaim = .{ .ctx = self, .func = reclaimCb };
    }

    pub fn unregisterReclaim(self: *ExpertCache) void {
        const b = self.budget orelse return;
        if (b.alloc.reclaim) |r| {
            if (r.ctx == @as(*anyopaque, @ptrCast(self))) b.alloc.reclaim = self.prev_reclaim;
        }
        self.budget = null;
    }

    fn reclaimCb(ctx: *anyopaque, len: usize) usize {
        const self: *ExpertCache = @ptrCast(@alignCast(ctx));
        var freed: usize = 0;
        // The hook can run on any thread that allocates; if this thread holds
        // the cache lock (an allocation inside the cache itself), give up
        // rather than deadlock.
        if (self.tryLock()) {
            defer self.lockRelease();
            freed = @intCast(self.evictLocked(len));
        }
        if (self.prev_reclaim) |r| freed += r.func(r.ctx, len);
        return freed;
    }

    // -- LRU list -----------------------------------------------------------------

    fn unlink(self: *ExpertCache, e: *Entry) void {
        if (e.prev) |p| p.next = e.next else self.head = e.next;
        if (e.next) |n| n.prev = e.prev else self.tail = e.prev;
        e.prev = null;
        e.next = null;
    }

    fn linkTail(self: *ExpertCache, e: *Entry) void {
        e.prev = self.tail;
        e.next = null;
        if (self.tail) |t| t.next = e else self.head = e;
        self.tail = e;
    }

    fn touch(self: *ExpertCache, e: *Entry) void {
        self.unlink(e);
        self.linkTail(e);
    }

    /// Evicts unpinned entries in LRU order until at least `need` bytes were
    /// freed (or nothing evictable is left). Returns the bytes freed. Lock
    /// must be held. Entries being loaded or computed with are pinned and
    /// the hits of the current layer were just touched (most recent), so
    /// the victims are the oldest entries of other layers first.
    fn evictLocked(self: *ExpertCache, need: u64) u64 {
        var freed: u64 = 0;
        var e = self.head;
        while (e) |cur| {
            if (freed >= need) break;
            e = cur.next;
            if (cur.pins > 0) continue;
            self.unlink(cur);
            _ = self.map.remove(cur.key.int());
            self.used -= cur.bytes;
            freed += cur.bytes;
            self.evictions += 1;
            self.store.releaseSet(cur.leases[0..cur.n_leases]);
            self.gpa.destroy(cur);
        }
        return freed;
    }

    /// Makes room for `bytes` more, evicting LRU entries. Returns false if
    /// the entry would still exceed the capacity; the caller decides whether
    /// to proceed anyway.
    fn makeRoom(self: *ExpertCache, bytes: u64) bool {
        self.lockAcquire();
        defer self.lockRelease();
        if (self.used + bytes <= self.capacity) return true;
        _ = self.evictLocked(self.used + bytes - self.capacity);
        return self.used + bytes <= self.capacity;
    }

    /// Creates a pinned, not yet loaded entry for `key`.
    fn insertPinned(self: *ExpertCache, key: Key, bytes: u64) !*Entry {
        const e = try self.gpa.create(Entry);
        errdefer self.gpa.destroy(e);
        e.* = .{ .key = key, .bytes = bytes, .pins = 1 };
        try self.map.ensureUnusedCapacity(self.gpa, 1);
        self.lockAcquire();
        defer self.lockRelease();
        self.map.putAssumeCapacity(key.int(), e);
        self.linkTail(e);
        self.used += bytes;
        return e;
    }

    /// Removes an entry that failed to load (or is otherwise unusable).
    fn dropEntry(self: *ExpertCache, e: *Entry) void {
        self.lockAcquire();
        self.unlink(e);
        _ = self.map.remove(e.key.int());
        self.used -= e.bytes;
        self.lockRelease();
        self.store.releaseSet(e.leases[0..e.n_leases]);
        self.gpa.destroy(e);
    }

    // -- loading --------------------------------------------------------------------

    /// Reads the matrices of `ex` (three, or two for a non-gated expert)
    /// into `e` through the store (synchronously).
    fn loadEntry(self: *ExpertCache, e: *Entry, ex: *const moe.Expert) !void {
        const store = self.store;
        var n: usize = 0;
        errdefer store.releaseSet(e.leases[0..n]);
        if (ex.sharesGateUpBlock()) {
            // Fused, transposed layout: gate and up come out of one block read.
            try store.acquireColumns(ex.gate_ref.ref, &.{ ex.gate_ref.columns(), ex.up_ref.columns() }, e.leases[0..2]);
            n = 2;
        } else {
            e.leases[0] = try ex.gate_ref.acquire(store);
            n = 1;
            if (ex.gated) {
                e.leases[1] = try ex.up_ref.acquire(store);
                n = 2;
            }
        }
        e.leases[n] = try ex.down_ref.acquire(store);
        n += 1;
        e.n_leases = n;
        e.gate = e.leases[0].weight;
        e.up = if (ex.gated) e.leases[1].weight else e.gate;
        e.down = e.leases[n - 1].weight;
        _ = @atomicRmw(u64, &self.bytes_read, .Add, e.bytes, .monotonic);
    }

    fn loadTask(self: *ExpertCache, e: *Entry, ex: *const moe.Expert) Io.Cancelable!void {
        self.loadEntry(e, ex) catch {
            e.failed.store(true, .release);
        };
    }

    /// Waits for the background loads started by `prefetch`/`warm`, unpins
    /// their entries and drops the ones that failed.
    fn finishPending(self: *ExpertCache) void {
        const p = &(self.pending orelse return);
        p.group.await(self.io) catch {};
        for (p.entries.items) |e| {
            self.lockAcquire();
            e.pins -= 1;
            self.lockRelease();
            if (e.failed.load(.acquire)) self.dropEntry(e);
        }
        p.entries.deinit(self.gpa);
        self.pending = null;
    }

    fn noteUse(self: *ExpertCache, key: Key) void {
        const gop = self.uses.getOrPut(self.gpa, key.int()) catch return;
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    // -- public API --------------------------------------------------------------

    /// Starts background reads for the experts of `experts` (indices into
    /// `m.experts` of layer `layer`) that are not resident, as far as they
    /// fit; hits are marked most recently used so the group's own members
    /// are not evicted to make room for each other. The reads are awaited by
    /// the next `acquire`.
    pub fn prefetch(self: *ExpertCache, layer: usize, m: *const moe.MoeLayer, experts: []const usize) void {
        if (experts.len == 0) return;
        self.finishPending();
        // Touch the hits first.
        self.lockAcquire();
        for (experts) |x| {
            if (self.map.get(Key.init(layer, x).int())) |e| self.touch(e);
        }
        self.lockRelease();
        // The group must not move once tasks were started on it: it lives in `self.pending`.
        self.pending = .{ .group = .init, .entries = .empty };
        const pend = &self.pending.?;
        for (experts) |x| {
            const key = Key.init(layer, x);
            self.lockAcquire();
            const present = self.map.get(key.int()) != null;
            self.lockRelease();
            if (present) continue;
            const ex = &m.experts[x];
            const bytes = moe.expertBytes(ex);
            if (!self.makeRoom(bytes)) break;
            const e = self.insertPinned(key, bytes) catch break;
            e.prefetched = true;
            pend.entries.append(self.gpa, e) catch {
                self.dropEntry(e);
                break;
            };
            pend.group.async(self.io, loadTask, .{ self, e, ex });
        }
        if (pend.entries.items.len == 0) self.finishPending();
    }

    /// Makes expert `x` of layer `layer` resident and pins it; `release` it
    /// when done. Hits are free; misses read the expert synchronously
    /// (evicting LRU entries of other layers first).
    pub fn acquire(self: *ExpertCache, layer: usize, m: *const moe.MoeLayer, x: usize) !*Entry {
        self.finishPending();
        const key = Key.init(layer, x);
        self.noteUse(key);
        self.lockAcquire();
        if (self.map.get(key.int())) |e| {
            e.pins += 1;
            self.touch(e);
            const was_prefetched = e.prefetched;
            e.prefetched = false;
            self.lockRelease();
            if (was_prefetched) {
                self.misses += 1;
                self.step_misses += 1;
            } else {
                self.hits += 1;
            }
            return e;
        }
        self.lockRelease();
        const ex = &m.experts[x];
        const bytes = moe.expertBytes(ex);
        _ = self.makeRoom(bytes);
        const e = try self.insertPinned(key, bytes);
        self.loadEntry(e, ex) catch |err| {
            self.dropEntry(e);
            return err;
        };
        self.misses += 1;
        self.step_misses += 1;
        return e;
    }

    pub fn release(self: *ExpertCache, e: *Entry) void {
        self.lockAcquire();
        defer self.lockRelease();
        std.debug.assert(e.pins > 0);
        e.pins -= 1;
    }

    /// Bytes held by unpinned entries (what `trim` or a reclaim could free).
    pub fn evictable(self: *ExpertCache) u64 {
        self.lockAcquire();
        defer self.lockRelease();
        var total: u64 = 0;
        var e = self.head;
        while (e) |cur| : (e = cur.next) {
            if (cur.pins == 0) total += cur.bytes;
        }
        return total;
    }

    /// Evicts every unpinned entry (e.g. before another model shares the budget).
    pub fn trim(self: *ExpertCache) void {
        self.finishPending();
        self.lockAcquire();
        defer self.lockRelease();
        _ = self.evictLocked(std.math.maxInt(u64));
    }

    /// True if expert `x` of `layer` was used at least once in this run.
    pub fn visited(self: *const ExpertCache, layer: usize, x: usize) bool {
        return (self.uses.get(Key.init(layer, x).int()) orelse 0) > 0;
    }

    /// True once any expert was used (before that "visited only" would edit nothing).
    pub fn anyVisited(self: *const ExpertCache) bool {
        return self.uses.count() > 0;
    }

    /// Closes one decode step: records this step's misses (one token per sequence).
    pub fn endDecodeStep(self: *ExpertCache) void {
        self.decode_steps += 1;
        self.decode_misses += self.step_misses;
        self.decode_misses_max = @max(self.decode_misses_max, self.step_misses);
        self.step_misses = 0;
    }

    /// Misses since the last `endDecodeStep` (prefill resets it too).
    pub fn resetStep(self: *ExpertCache) void {
        self.step_misses = 0;
    }

    pub fn stats(self: *ExpertCache) Stats {
        self.lockAcquire();
        defer self.lockRelease();
        return .{
            .capacity = self.capacity,
            .resident = self.map.count(),
            .resident_bytes = self.used,
            .hits = self.hits,
            .misses = self.misses,
            .bytes_read = @atomicLoad(u64, &self.bytes_read, .monotonic),
            .evictions = self.evictions,
            .warmed = self.warmed,
            .decode_steps = self.decode_steps,
            .decode_misses = self.decode_misses,
            .decode_misses_max = self.decode_misses_max,
            .visited = self.uses.count(),
        };
    }

    // -- hotlist --------------------------------------------------------------------

    pub const Hot = struct { layer: u32, expert: u32, uses: u64 };

    /// Access counts sorted by descending use (ties by layer, expert). Caller frees.
    pub fn hotExperts(self: *ExpertCache, gpa: Allocator) ![]Hot {
        const out = try gpa.alloc(Hot, self.uses.count());
        var it = self.uses.iterator();
        var i: usize = 0;
        while (it.next()) |kv| : (i += 1) {
            const key = Key.fromInt(kv.key_ptr.*);
            out[i] = .{ .layer = key.layer, .expert = key.expert, .uses = kv.value_ptr.* };
        }
        std.mem.sort(Hot, out, {}, hotLessThan);
        return out;
    }

    fn hotLessThan(_: void, a: Hot, b: Hot) bool {
        if (a.uses != b.uses) return a.uses > b.uses;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.expert < b.expert;
    }

    /// Writes the hotlist (a Lua table, hottest expert first) to `path`.
    pub fn writeHotlist(self: *ExpertCache, gpa: Allocator, io: Io, path: []const u8, model_id: []const u8) !void {
        const hot = try self.hotExperts(gpa);
        defer gpa.free(hot);
        var text: Io.Writer.Allocating = .init(gpa);
        defer text.deinit();
        const w = &text.writer;
        try w.writeAll("-- ditch expert hotlist: routed experts by number of uses in the last run.\n");
        try w.writeAll("-- Each entry is { layer, expert, uses }; the hottest experts that fit are\n");
        try w.writeAll("-- loaded into the expert cache before the next run on this model starts.\n");
        try w.writeAll("return {\n  model = ");
        try writeLuaString(w, model_id);
        try w.print(",\n  capacity = {d},\n  experts = {{\n", .{self.capacity});
        for (hot) |h| try w.print("    {{ {d}, {d}, {d} }},\n", .{ h.layer, h.expert, h.uses });
        try w.writeAll("  },\n}\n");
        if (std.fs.path.dirname(path)) |d| try Io.Dir.cwd().createDirPath(io, d);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
    }

    /// Reads a hotlist written by `writeHotlist` (hottest first). Caller frees.
    pub fn readHotlist(gpa: Allocator, io: Io, path: []const u8) ![]Hot {
        const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(text);
        var result = try lua.parse(gpa, text, path);
        defer result.parsed.deinit();
        if (result.err != null) return error.InvalidHotlist;
        const experts = result.parsed.root.get("experts") orelse return error.InvalidHotlist;
        if (experts != .array) return error.InvalidHotlist;
        var out = std.ArrayList(Hot).empty;
        errdefer out.deinit(gpa);
        for (experts.array) |item| {
            if (item != .array or item.array.len != 3) return error.InvalidHotlist;
            var v: [3]i64 = undefined;
            for (item.array, 0..) |x, i| v[i] = if (x == .integer) x.integer else return error.InvalidHotlist;
            if (v[0] < 0 or v[1] < 0 or v[2] < 0) return error.InvalidHotlist;
            try out.append(gpa, .{ .layer = @intCast(v[0]), .expert = @intCast(v[1]), .uses = @intCast(v[2]) });
        }
        std.mem.sort(Hot, out.items, {}, hotLessThan);
        return out.toOwnedSlice(gpa);
    }

    /// Loads the hottest experts of `hot` that exist in `model` and fit the
    /// capacity (read concurrently). Returns the number warmed.
    pub fn warm(self: *ExpertCache, model: *const model_mod.Model, hot: []const Hot) !usize {
        self.finishPending();
        self.pending = .{ .group = .init, .entries = .empty };
        const pend = &self.pending.?;
        var n: usize = 0;
        for (hot) |h| {
            if (h.uses == 0) break;
            if (h.layer >= model.layers.len) continue;
            const m = &(model.layers[h.layer].moe orelse continue);
            if (h.expert >= m.experts.len) continue;
            const key = Key.init(h.layer, h.expert);
            self.lockAcquire();
            const present = self.map.get(key.int()) != null;
            self.lockRelease();
            if (present) continue;
            const ex = &m.experts[h.expert];
            const bytes = moe.expertBytes(ex);
            if (self.used + bytes > self.capacity) break;
            const e = self.insertPinned(key, bytes) catch |err| {
                self.finishPending();
                return err;
            };
            e.prefetched = true;
            pend.entries.append(self.gpa, e) catch |err| {
                self.dropEntry(e);
                self.finishPending();
                return err;
            };
            pend.group.async(self.io, loadTask, .{ self, e, ex });
            n += 1;
        }
        self.finishPending();
        // Count what actually made it.
        var ok: usize = 0;
        for (hot) |h| {
            if (h.layer >= model.layers.len) continue;
            if (model.layers[h.layer].moe == null) continue;
            self.lockAcquire();
            const present = self.map.get(Key.init(h.layer, h.expert).int()) != null;
            self.lockRelease();
            if (present) ok += 1;
        }
        self.warmed = @min(ok, n);
        return self.warmed;
    }
};

fn writeLuaString(w: *Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

/// `<scratch_dir>/<model-id>.hotlist`, with the model id sanitised like the study checkpoint name.
pub fn hotlistPath(a: Allocator, scratch_dir: []const u8, model: []const u8) ![]u8 {
    var name = std.ArrayList(u8).empty;
    defer name.deinit(a);
    for (model) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '_' or c == '-') try name.append(a, c) else try name.appendSlice(a, "--");
    }
    try name.appendSlice(a, ".hotlist");
    return std.fs.path.join(a, &.{ scratch_dir, name.items });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const Model = model_mod.Model;

fn tmpScratch(gpa: Allocator, io: Io, tmp: *std.testing.TmpDir) ![]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &buf);
    return std.fs.path.join(gpa, &.{ buf[0..n], "scratch" });
}

test "LRU semantics: hits, misses, eviction order and pinning" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const scratch = try tmpScratch(gpa, io, &tmp);
    defer gpa.free(scratch);
    // Streamed load without an automatic cache: the test drives its own. The
    // budget is unlimited for now (its limit is lowered at the end).
    var budget = try budget_mod.Budget.init(gpa, io, .{ .scratch_dir = scratch });
    defer budget.deinit();
    const model = try Model.loadWithOptions(gpa, io, &pool, "tests/fixtures/qwen3_moe_big", .{ .store = .streamed, .budget = &budget, .expert_cache = 0 });
    defer model.deinit();
    try std.testing.expect(model.expert_cache == null);
    const m = &model.layers[0].moe.?;
    const per = moe.expertBytes(&m.experts[0]);
    try std.testing.expect(per > 0);
    var cache = ExpertCache.init(gpa, io, @constCast(&model.store), 3 * per);
    defer cache.deinit();

    // Three misses fill the cache; a fourth evicts the least recently used (expert 0).
    const e0 = try cache.acquire(0, m, 0);
    cache.release(e0);
    const e1 = try cache.acquire(0, m, 1);
    cache.release(e1);
    const e2 = try cache.acquire(0, m, 2);
    cache.release(e2);
    try std.testing.expectEqual(@as(u64, 3), cache.stats().misses);
    try std.testing.expectEqual(@as(u64, 3 * per), cache.stats().resident_bytes);
    // Touch expert 0 so that expert 1 becomes the LRU victim.
    const h0 = try cache.acquire(0, m, 0);
    cache.release(h0);
    try std.testing.expectEqual(@as(u64, 1), cache.stats().hits);
    const e3 = try cache.acquire(0, m, 3);
    cache.release(e3);
    try std.testing.expectEqual(@as(u64, 1), cache.stats().evictions);
    try std.testing.expect(cache.map.get(Key.init(0, 1).int()) == null);
    try std.testing.expect(cache.map.get(Key.init(0, 0).int()) != null);
    try std.testing.expectEqual(@as(u64, 3), cache.stats().resident);

    // A pinned entry is never evicted, even when it is the LRU one.
    const pinned = try cache.acquire(0, m, 2);
    // Order now: 0, 3, 2(pinned). Load two more (kept pinned): 0 and 3 go, 2 stays.
    const a = try cache.acquire(1, m, 5);
    const b = try cache.acquire(1, m, 6);
    try std.testing.expect(cache.map.get(Key.init(0, 2).int()) != null);
    try std.testing.expect(cache.map.get(Key.init(0, 0).int()) == null);
    try std.testing.expect(cache.map.get(Key.init(0, 3).int()) == null);
    try std.testing.expectEqual(@as(u64, 0), cache.evictable());
    // Pinned set larger than the capacity: still served (soft bound).
    const c = try cache.acquire(1, m, 7);
    try std.testing.expect(cache.stats().resident_bytes > cache.capacity);
    cache.release(c);
    cache.release(b);
    cache.release(a);
    cache.release(pinned);
    try std.testing.expectEqual(@as(u64, 4 * per), cache.evictable());
    // The weights of an entry are the same bytes the store reads directly.
    const direct = try m.experts[2].down_ref.acquire(@constCast(&model.store));
    defer @constCast(&model.store).release(direct);
    const again = try cache.acquire(0, m, 2);
    try std.testing.expectEqualSlices(u8, direct.weight.data, again.down.data);
    cache.release(again);

    // Prefetch loads the misses of a set concurrently; the first acquire counts them as misses.
    cache.trim();
    cache.resetStep();
    try std.testing.expectEqual(@as(u64, 0), cache.stats().resident);
    const set = [_]usize{ 1, 4, 9 };
    cache.prefetch(2, &model.layers[2].moe.?, &set);
    const before = cache.stats();
    for (set) |x| {
        const e = try cache.acquire(2, &model.layers[2].moe.?, x);
        cache.release(e);
    }
    const after = cache.stats();
    try std.testing.expectEqual(before.misses + 3, after.misses);
    try std.testing.expectEqual(before.hits, after.hits);
    // Decode-step accounting.
    cache.endDecodeStep();
    try std.testing.expectEqual(@as(u64, 1), cache.stats().decode_steps);
    try std.testing.expectEqual(@as(u64, 3), cache.stats().decode_misses);
    // Visited bookkeeping.
    try std.testing.expect(cache.visited(2, 9));
    try std.testing.expect(!cache.visited(3, 0));
    // Budget reclaim: an allocation refused by the budget evicts unpinned
    // entries (whose buffers were budgeted) and then succeeds.
    cache.registerReclaim(&budget);
    defer cache.unregisterReclaim();
    budget.max_ram = budget.alloc.currentBytes() + per / 2;
    budget.alloc.limit = budget.max_ram;
    const big = try budget.allocator().alloc(u8, 2 * per);
    defer budget.allocator().free(big);
    try std.testing.expect(cache.stats().evictions > after.evictions);
    try std.testing.expect(cache.stats().resident < 3);
}

test "hotlist round trip warms the hottest experts that fit" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const scratch = try tmpScratch(gpa, io, &tmp);
    defer gpa.free(scratch);
    const model = try Model.loadWithOptions(gpa, io, &pool, "tests/fixtures/qwen3_moe_big", .{ .store = .streamed, .scratch_dir = scratch, .expert_cache = 0 });
    defer model.deinit();
    const per = moe.expertBytes(&model.layers[0].moe.?.experts[0]);
    var cache = ExpertCache.init(gpa, io, @constCast(&model.store), 4 * per);
    defer cache.deinit();
    // Use pattern: (1, 7) three times, (0, 2) twice, (3, 15) once.
    for ([_][2]usize{ .{ 1, 7 }, .{ 0, 2 }, .{ 1, 7 }, .{ 3, 15 }, .{ 0, 2 }, .{ 1, 7 } }) |k| {
        const e = try cache.acquire(k[0], &model.layers[k[0]].moe.?, k[1]);
        cache.release(e);
    }
    const path = try hotlistPath(gpa, scratch, "tests/fixtures/qwen3_moe_big");
    defer gpa.free(path);
    try std.testing.expect(std.mem.endsWith(u8, path, "tests--fixtures--qwen3_moe_big.hotlist"));
    try cache.writeHotlist(gpa, io, path, "tests/fixtures/qwen3_moe_big");
    const hot = try ExpertCache.readHotlist(gpa, io, path);
    defer gpa.free(hot);
    try std.testing.expectEqual(@as(usize, 3), hot.len);
    try std.testing.expectEqual(ExpertCache.Hot{ .layer = 1, .expert = 7, .uses = 3 }, hot[0]);
    try std.testing.expectEqual(ExpertCache.Hot{ .layer = 0, .expert = 2, .uses = 2 }, hot[1]);
    try std.testing.expectEqual(ExpertCache.Hot{ .layer = 3, .expert = 15, .uses = 1 }, hot[2]);
    // A fresh cache with room for two experts warms the two hottest; using them is free.
    var fresh = ExpertCache.init(gpa, io, @constCast(&model.store), 2 * per);
    defer fresh.deinit();
    try std.testing.expectEqual(@as(usize, 2), try fresh.warm(model, hot));
    try std.testing.expect(fresh.map.get(Key.init(1, 7).int()) != null);
    try std.testing.expect(fresh.map.get(Key.init(0, 2).int()) != null);
    try std.testing.expect(fresh.map.get(Key.init(3, 15).int()) == null);
    const st = fresh.stats();
    try std.testing.expectEqual(@as(u64, 2), st.warmed);
    try std.testing.expectEqual(@as(u64, 2 * per), st.resident_bytes);
    // Model-level warm start from a scratch directory.
    const warmed_model = try Model.loadWithOptions(gpa, io, &pool, "tests/fixtures/qwen3_moe_big", .{ .store = .streamed, .scratch_dir = scratch, .expert_cache = 2 * per });
    defer warmed_model.deinit();
    try std.testing.expectEqual(@as(usize, 2), try warmed_model.warmExpertCache(gpa, "tests/fixtures/qwen3_moe_big"));
    // A missing or malformed hotlist is not an error for the warm start, but is for the reader.
    try std.testing.expectEqual(@as(usize, 0), try warmed_model.warmExpertCache(gpa, "no/such/model"));
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "return { experts = { { 1, 2 } } }" });
    try std.testing.expectError(error.InvalidHotlist, ExpertCache.readHotlist(gpa, io, path));
}
