//! Remote safetensors source: run a model straight from the Hugging Face Hub
//! (or any HTTP server that supports `Range` requests) without downloading it
//! first. Only the small files (config, tokenizer, the shard index) and the
//! safetensors headers are fetched up front; tensor bytes are fetched on
//! demand with HTTP range requests in aligned chunks (8 MiB by default) that
//! are cached on disk under `<cache_dir>/models/<id>/chunks/<shard>/<index>`,
//! so nothing is fetched twice across runs and a full download can later
//! reuse them. Up to `max_in_flight` chunks are fetched concurrently (the
//! weight store and the expert cache prefetch on background tasks).
//!
//! The chunk cache is bounded (`--remote-cache-size`): an in-memory index of
//! the chunks on disk (built by scanning the chunk directories at open, in
//! file-time order, then kept up to date on every read and write) tracks their
//! total size and recency, and a write that would exceed the bound first
//! evicts least-recently-used chunks. Chunks of the trunk (every tensor that
//! is not a routed expert, re-read by every forward pass) are evicted only
//! when no expert chunk is left to evict, so a bound that holds the trunk
//! keeps it cached across trials. A chunk that is being read or written is
//! never evicted. A chunk that cannot be kept on disk (bound 0, or everything
//! else is in use) is served from a small in-RAM ring of recent chunks and
//! then dropped. Chunks are written to `<index>.part` and renamed, leftover
//! `.part` files are removed at open, and a chunk whose length does not fit
//! the shard is detected on first use and fetched again.
//!
//! Model ids: `hf://owner/name` (the Hub, revision `main` or `--model-commit`),
//! a plain `owner/name` when `--remote-weights` is set, or an `http(s)://.../`
//! base URL that serves the model files.

const std = @import("std");
const Io = std.Io;
const builtin = @import("builtin");
const hf = @import("hf.zig");
const budget_mod = @import("budget.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;
const fmtBytes = budget_mod.fmtBytes;

pub const default_chunk_size: u64 = 8 << 20;
/// Concurrent range requests to the remote source (`--remote-connections`):
/// one HTTP stream from the Hub runs at a fraction of the link (~40 MB/s
/// measured against ~110 MB/s on 16 streams), so reads fetch their missing
/// chunks in parallel up to this many at a time.
pub const default_connections: u32 = 16;

/// Upper end of the default chunk cache bound. Large enough for the trunk
/// of today's biggest open mixture-of-experts checkpoints plus their hottest
/// experts, small enough that a laptop's disk is never filled by default.
pub const default_cache_cap: u64 = 64 << 30;

/// Chunks that could not be kept on disk are held here for the reads that
/// follow right away (neighbouring tensors, row reads), then dropped.
const ram_slots = 4;

/// The default bound: half of what the cache could use on its filesystem
/// (free space plus what it already holds), at most `default_cache_cap`.
/// Without a free-space figure (unsupported platform) the cap itself.
pub fn defaultCacheSize(free: ?u64, cached: u64) u64 {
    const f = free orelse return default_cache_cap;
    return @min(default_cache_cap, (f +| cached) / 2);
}

/// Bytes available to an unprivileged user on the filesystem holding `path`
/// (null when it cannot be determined).
pub fn diskFree(path: []const u8) ?u64 {
    var zbuf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= zbuf.len) return null;
    @memcpy(zbuf[0..path.len], path);
    zbuf[path.len] = 0;
    const zpath: [*:0]const u8 = @ptrCast(&zbuf);
    // Raw struct buffers: only the leading fields are read, so the exact
    // size of the platform's struct does not matter.
    var buf: [4096]u8 align(8) = undefined;
    switch (builtin.os.tag) {
        .linux => {
            if (@sizeOf(usize) != 8) return null;
            const linux = std.os.linux;
            const rc = linux.syscall2(.statfs, @intFromPtr(zpath), @intFromPtr(&buf));
            if (linux.errno(rc) != .SUCCESS) return null;
            // struct statfs (64-bit): f_type, f_bsize, f_blocks, f_bfree, f_bavail, ..., f_frsize at 72.
            const bsize = std.mem.readInt(u64, buf[8..16], .little);
            const bavail = std.mem.readInt(u64, buf[32..40], .little);
            const frsize = std.mem.readInt(u64, buf[72..80], .little);
            return bavail *| (if (frsize != 0) frsize else bsize);
        },
        .macos => {
            if (!builtin.link_libc) return null;
            // struct statfs (64-bit inodes): u32 f_bsize, i32 f_iosize, u64 f_blocks, u64 f_bfree, u64 f_bavail.
            const statfs = @extern(*const fn ([*:0]const u8, *anyopaque) callconv(.c) c_int, .{ .name = if (builtin.cpu.arch == .x86_64) "statfs$INODE64" else "statfs" });
            if (statfs(zpath, &buf) != 0) return null;
            const bsize = std.mem.readInt(u32, buf[0..4], builtin.cpu.arch.endian());
            const bavail = std.mem.readInt(u64, buf[24..32], builtin.cpu.arch.endian());
            return bavail *| bsize;
        },
        else => return null,
    }
}

/// True for ids that select this source: `hf://`, `http://` and `https://`.
pub fn isRemoteId(model: []const u8) bool {
    return std.mem.startsWith(u8, model, "hf://") or std.mem.startsWith(u8, model, "http://") or std.mem.startsWith(u8, model, "https://");
}

/// The Hub id behind `hf://owner/name` (or the id itself).
pub fn hubId(model: []const u8) []const u8 {
    if (std.mem.startsWith(u8, model, "hf://")) return model["hf://".len..];
    return model;
}

pub const Stats = struct {
    /// Range requests sent to the server.
    ranges_fetched: u64,
    bytes_fetched: u64,
    /// Chunk reads served from the disk cache (chunks an earlier run left).
    chunks_from_disk: u64,
    /// Bytes of chunks on disk now, the most there ever were in this run, and the bound.
    cache_bytes: u64,
    peak_cache_bytes: u64,
    cache_limit: u64,
    /// Chunks evicted to stay under the bound (at open and while running).
    chunks_evicted: u64,
    bytes_evicted: u64,
    /// Fetched chunks served from RAM and dropped instead of being kept on disk.
    chunks_unpersisted: u64,
    /// Chunks found short or missing on disk and fetched again.
    chunks_invalid: u64,
};

pub const Source = struct {
    gpa: Allocator,
    io: Io,
    http: *hf.Http,
    arena: std.heap.ArenaAllocator,
    /// Base URL of the model files, ending with '/'.
    base_url: []const u8,
    /// Local directory with the small files and the `chunks/` cache.
    dir_path: []const u8,
    chunk_size: u64,
    /// Safetensors shard names, sorted.
    shards: []const []const u8,
    files: std.ArrayList(*RemoteFile) = .empty,
    max_in_flight: u32 = default_connections,
    in_flight: std.atomic.Value(u32) = .init(0),
    ranges_fetched: std.atomic.Value(u64) = .init(0),
    bytes_fetched: std.atomic.Value(u64) = .init(0),
    chunks_from_disk: std.atomic.Value(u64) = .init(0),
    chunks_evicted: std.atomic.Value(u64) = .init(0),
    bytes_evicted: std.atomic.Value(u64) = .init(0),
    chunks_unpersisted: std.atomic.Value(u64) = .init(0),
    chunks_invalid: std.atomic.Value(u64) = .init(0),
    write_failure_reported: std.atomic.Value(bool) = .init(false),

    /// Bound of the chunk cache in bytes (0 = keep nothing on disk).
    cache_limit: u64 = 0,
    /// True when `cache_limit` is the default rather than a setting.
    cache_limit_default: bool = false,
    /// Free space on the cache filesystem when the source was opened.
    disk_free: ?u64 = null,
    /// Chunks on disk when the source was opened (before trimming to the bound).
    cached_at_open: u64 = 0,
    /// Guards every chunk index, the byte counters below and the RAM ring.
    /// Held only for map lookups and updates, never across I/O.
    lock: std.atomic.Value(bool) = .init(false),
    /// Bytes of chunks on disk or reserved for a chunk being written.
    cache_bytes: u64 = 0,
    peak_cache_bytes: u64 = 0,
    /// Recency clock. Starts above any file time in nanoseconds, so every
    /// chunk used in this run is more recent than the ones scanned at open.
    tick: u64 = 1 << 63,
    ram: [ram_slots]RamSlot = @splat(.{}),
    /// Chunks queued by `RemoteFile.prefetchRange`, fetched in order by up to
    /// `max_in_flight` workers into the chunk cache (guarded by `lock`).
    prefetch_jobs: std.ArrayList(PrefetchJob) = .empty,
    prefetch_head: usize = 0,
    prefetch_workers: u32 = 0,
    prefetch_group: Io.Group = .init,
    chunks_prefetched: std.atomic.Value(u64) = .init(0),

    const PrefetchJob = struct { file: *RemoteFile, index: u64 };

    pub const OpenOptions = struct {
        revision: ?[]const u8 = null,
        chunk_size: u64 = default_chunk_size,
        /// Bound of the chunk cache (null = `defaultCacheSize`, 0 = nothing on disk).
        cache_size: ?u64 = null,
        /// Concurrent range requests (0 = `default_connections`).
        connections: u32 = 0,
    };

    /// Resolves `model` to a base URL and cache directory, fetches the small
    /// files that are not cached yet, reads the shard index and indexes the
    /// chunks already on disk (trimming them to the bound).
    pub fn open(gpa: Allocator, io: Io, http: *hf.Http, cache_root: []const u8, model: []const u8, opts: OpenOptions, out: *Io.Writer) !*Source {
        const self = try gpa.create(Source);
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .io = io,
            .http = http,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .base_url = undefined,
            .dir_path = undefined,
            .chunk_size = @max(opts.chunk_size, 4096),
            .shards = &.{},
            .max_in_flight = if (opts.connections == 0) default_connections else opts.connections,
        };
        errdefer self.arena.deinit();
        errdefer {
            for (self.files.items) |f| f.deinit();
            self.files.deinit(gpa);
        }
        const a = self.arena.allocator();
        const rev = opts.revision orelse "main";
        if (std.mem.startsWith(u8, model, "http://") or std.mem.startsWith(u8, model, "https://")) {
            self.base_url = if (std.mem.endsWith(u8, model, "/")) try a.dupe(u8, model) else try std.fmt.allocPrint(a, "{s}/", .{model});
            self.dir_path = try std.fs.path.join(a, &.{ cache_root, "models", try sanitizeUrl(a, self.base_url), "main" });
        } else {
            const id = hubId(model);
            if (std.mem.indexOfScalar(u8, id, '/') == null) {
                std.log.err("remote model id must be owner/name: {s}", .{model});
                return error.ModelNotFound;
            }
            self.base_url = try std.fmt.allocPrint(a, "https://huggingface.co/{s}/resolve/{s}/", .{ id, rev });
            // The same directory `hf.resolveModel` uses, so the small files are shared.
            self.dir_path = try std.fs.path.join(a, &.{ cache_root, "models", try hf.sanitizeRepoId(a, id), rev });
        }
        const cwd = Io.Dir.cwd();
        try cwd.createDirPath(io, self.dir_path);
        var dir = try cwd.openDir(io, self.dir_path, .{});
        defer dir.close(io);
        try out.print("* Remote weights from {s} (chunk cache {s})\n", .{ self.base_url, self.dir_path });
        try out.flush();
        try self.fetchSmall(dir, "config.json", true, out);
        // tokenizer.json, else the tiktoken vocabularies of Kimi (tiktoken.model) and Llama 3 (tokenizer.model).
        const tokenizer_files = [_][]const u8{ "tokenizer.json", "tiktoken.model", "tokenizer.model" };
        var have_tokenizer = false;
        for (tokenizer_files) |name| {
            try self.fetchSmall(dir, name, false, out);
            if (dir.access(io, name, .{})) |_| {
                have_tokenizer = true;
                break;
            } else |_| {}
        }
        if (!have_tokenizer) {
            std.log.err("no tokenizer.json, tiktoken.model or tokenizer.model at {s}", .{self.base_url});
            return error.ModelNotFound;
        }
        const optional = [_][]const u8{ "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json" };
        for (optional) |name| try self.fetchSmall(dir, name, false, out);

        // Shard list from the index (or the single-file layout).
        var names = std.ArrayList([]const u8).empty;
        if (dir.readFileAlloc(io, "model.safetensors.index.json", gpa, .limited(256 << 20))) |text| {
            defer gpa.free(text);
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
            defer parsed.deinit();
            const wm = (parsed.value.object.get("weight_map") orelse return error.InvalidResponse);
            if (wm != .object) return error.InvalidResponse;
            var it = wm.object.iterator();
            while (it.next()) |kv| {
                if (kv.value_ptr.* != .string) continue;
                const shard = kv.value_ptr.string;
                var seen = false;
                for (names.items) |n| seen = seen or std.mem.eql(u8, n, shard);
                if (!seen) try names.append(a, try a.dupe(u8, shard));
            }
        } else |_| {
            try names.append(a, "model.safetensors");
        }
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        self.shards = names.items;
        try out.print("* {d} safetensors shard(s); headers are fetched now, tensors on demand in {f} chunks\n", .{ self.shards.len, fmtBytes(self.chunk_size) });

        // Index the chunks earlier runs left, then apply the bound.
        var cached_chunks: u64 = 0;
        for (self.shards) |name| cached_chunks += try self.addFile(name);
        self.cached_at_open = self.cache_bytes;
        self.disk_free = diskFree(self.dir_path);
        self.cache_limit_default = opts.cache_size == null;
        self.cache_limit = opts.cache_size orelse defaultCacheSize(self.disk_free, self.cache_bytes);
        var trimmed: u64 = 0;
        var trimmed_bytes: u64 = 0;
        while (self.cache_bytes > self.cache_limit) {
            const freed = self.evictOne() orelse break;
            trimmed += 1;
            trimmed_bytes += freed;
        }
        try out.print("* Chunk cache: {f} in {d} chunks on disk, bound {f}", .{ fmtBytes(self.cache_bytes), cached_chunks - trimmed, fmtBytes(self.cache_limit) });
        if (self.cache_limit == 0) {
            try out.writeAll(" (nothing is kept on disk)");
        } else if (self.cache_limit_default) {
            try out.print(" (default: half of the free disk space plus what is cached, at most {f}; see --remote-cache-size)", .{fmtBytes(default_cache_cap)});
        }
        try out.writeAll("\n");
        if (trimmed > 0) try out.print("* Chunk cache: evicted {d} least recently used chunks ({f}) to fit the bound\n", .{ trimmed, fmtBytes(trimmed_bytes) });
        self.peak_cache_bytes = self.cache_bytes;
        return self;
    }

    pub fn deinit(self: *Source) void {
        // Drop the queued prefetches and wait for the ones in flight.
        self.lockAcquire();
        self.prefetch_head = self.prefetch_jobs.items.len;
        self.lockRelease();
        self.prefetch_group.await(self.io) catch {};
        self.prefetch_jobs.deinit(self.gpa);
        for (self.ram) |slot| if (slot.body.len > 0) self.gpa.free(slot.body);
        for (self.files.items) |f| f.deinit();
        self.files.deinit(self.gpa);
        self.arena.deinit();
        self.gpa.destroy(self);
    }

    fn fetchSmall(self: *Source, dir: Io.Dir, name: []const u8, required: bool, out: *Io.Writer) !void {
        if (dir.access(self.io, name, .{})) |_| return else |_| {}
        // An optional file that was not there is remembered so later runs do not ask again.
        const missing_marker = try std.fmt.allocPrint(self.gpa, "{s}.missing", .{name});
        defer self.gpa.free(missing_marker);
        if (!required) {
            if (dir.access(self.io, missing_marker, .{})) |_| return else |_| {}
        }
        const url = try std.fmt.allocPrint(self.gpa, "{s}{s}", .{ self.base_url, name });
        defer self.gpa.free(url);
        try out.print("* Fetching {s}...\n", .{name});
        try out.flush();
        self.http.download(dir, name, url, null) catch |err| switch (err) {
            error.NotFound => if (required) {
                std.log.warn("{s} not found at {s}; check the model id (owner/name) and revision", .{ name, self.base_url });
                return error.ModelNotFound;
            } else {
                dir.writeFile(self.io, .{ .sub_path = missing_marker, .data = "" }) catch {};
                return;
            },
            error.Forbidden => {
                std.log.err("access to {s} denied; gated models require HF_TOKEN to be set", .{self.base_url});
                return error.ModelNotFound;
            },
            else => return err,
        };
    }

    /// Creates the shard's RemoteFile and indexes its chunk directory:
    /// leftover `.part` files are deleted, empty or oversized chunks too,
    /// and the rest are ordered by their file times. Returns the number of
    /// chunks indexed.
    fn addFile(self: *Source, name: []const u8) !u64 {
        const io = self.io;
        const f = try self.gpa.create(RemoteFile);
        errdefer self.gpa.destroy(f);
        f.* = .{
            .src = self,
            .name = try self.gpa.dupe(u8, name),
            .url = try std.fmt.allocPrint(self.gpa, "{s}{s}", .{ self.base_url, name }),
            .chunk_dir = try std.fs.path.join(self.gpa, &.{ self.dir_path, "chunks", name }),
        };
        {
            errdefer {
                self.gpa.free(f.name);
                self.gpa.free(f.url);
                self.gpa.free(f.chunk_dir);
            }
            try self.files.append(self.gpa, f);
        }
        const cwd = Io.Dir.cwd();
        try cwd.createDirPath(io, f.chunk_dir);
        var dir = try cwd.openDir(io, f.chunk_dir, .{ .iterate = true });
        defer dir.close(io);
        var doomed = std.ArrayList([]u8).empty;
        defer {
            for (doomed.items) |d| self.gpa.free(d);
            doomed.deinit(self.gpa);
        }
        var count: u64 = 0;
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            const index = std.fmt.parseInt(u64, entry.name, 10) catch {
                // An interrupted write: never valid.
                if (std.mem.endsWith(u8, entry.name, ".part")) try doomed.append(self.gpa, try self.gpa.dupe(u8, entry.name));
                continue;
            };
            const st = dir.statFile(io, entry.name, .{}) catch continue;
            if (st.size == 0 or st.size > self.chunk_size) {
                try doomed.append(self.gpa, try self.gpa.dupe(u8, entry.name));
                continue;
            }
            var t: i96 = st.mtime.nanoseconds;
            if (st.atime) |at| t = @max(t, at.nanoseconds);
            const tick: u64 = if (t <= 0) 0 else @intCast(@min(t, (1 << 63) - 1));
            try f.chunks.put(self.gpa, index, .{ .state = .present, .size = st.size, .tick = tick, .from_disk = true });
            self.cache_bytes += st.size;
            count += 1;
        }
        for (doomed.items) |d| dir.deleteFile(io, d) catch {};
        return count;
    }

    /// Returns the RemoteFile of shard `name` for range reads (the header is
    /// read by the safetensors reader).
    pub fn openFile(self: *Source, name: []const u8) !*RemoteFile {
        for (self.files.items) |f| if (std.mem.eql(u8, f.name, name)) return f;
        _ = try self.addFile(name);
        return self.files.items[self.files.items.len - 1];
    }

    pub fn stats(self: *Source) Stats {
        self.lockAcquire();
        const cache_bytes = self.cache_bytes;
        const peak = self.peak_cache_bytes;
        self.lockRelease();
        return .{
            .ranges_fetched = self.ranges_fetched.load(.monotonic),
            .bytes_fetched = self.bytes_fetched.load(.monotonic),
            .chunks_from_disk = self.chunks_from_disk.load(.monotonic),
            .cache_bytes = cache_bytes,
            .peak_cache_bytes = peak,
            .cache_limit = self.cache_limit,
            .chunks_evicted = self.chunks_evicted.load(.monotonic),
            .bytes_evicted = self.bytes_evicted.load(.monotonic),
            .chunks_unpersisted = self.chunks_unpersisted.load(.monotonic),
            .chunks_invalid = self.chunks_invalid.load(.monotonic),
        };
    }

    /// Waits until the chunks queued by `RemoteFile.prefetchRange` are fetched.
    pub fn awaitPrefetch(self: *Source) void {
        self.prefetch_group.await(self.io) catch {};
    }

    fn prefetchWorker(self: *Source) Io.Cancelable!void {
        while (true) {
            self.lockAcquire();
            if (self.prefetch_head >= self.prefetch_jobs.items.len) {
                self.prefetch_jobs.clearRetainingCapacity();
                self.prefetch_head = 0;
                self.prefetch_workers -= 1;
                self.lockRelease();
                return;
            }
            const job = self.prefetch_jobs.items[self.prefetch_head];
            self.prefetch_head += 1;
            self.lockRelease();
            job.file.prefetchChunk(job.index);
        }
    }

    fn lockAcquire(self: *Source) void {
        while (self.lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn lockRelease(self: *Source) void {
        self.lock.store(false, .release);
    }

    /// Under the lock.
    fn nextTick(self: *Source) u64 {
        self.tick += 1;
        return self.tick;
    }

    const Victim = struct { file: *RemoteFile, index: u64 };

    /// Under the lock: the least recently used chunk that is on disk and not
    /// in use, expert chunks before trunk chunks.
    fn pickVictim(self: *Source) ?Victim {
        var best: ?Victim = null;
        var best_trunk = true;
        var best_tick: u64 = std.math.maxInt(u64);
        for (self.files.items) |f| {
            var it = f.chunks.iterator();
            while (it.next()) |kv| {
                const c = kv.value_ptr;
                if (c.state != .present or c.pins > 0) continue;
                const trunk = f.trunk.contains(kv.key_ptr.*);
                const better = if (trunk != best_trunk) !trunk else c.tick < best_tick;
                if (best == null or better) {
                    best = .{ .file = f, .index = kv.key_ptr.* };
                    best_trunk = trunk;
                    best_tick = c.tick;
                }
            }
        }
        return best;
    }

    /// Evicts one chunk (see `pickVictim`); returns its size, or null when
    /// nothing can be evicted right now. The chunk's bytes stay counted until
    /// its file is gone, so the disk never holds more than the bound.
    fn evictOne(self: *Source) ?u64 {
        self.lockAcquire();
        const v = self.pickVictim() orelse {
            self.lockRelease();
            return null;
        };
        v.file.chunks.getPtr(v.index).?.state = .evicting;
        self.lockRelease();
        v.file.deleteChunkFile(v.index);
        self.lockAcquire();
        const size = if (v.file.chunks.fetchRemove(v.index)) |kv| kv.value.size else 0;
        self.cache_bytes -= size;
        self.lockRelease();
        _ = self.chunks_evicted.fetchAdd(1, .monotonic);
        _ = self.bytes_evicted.fetchAdd(size, .monotonic);
        return size;
    }

    /// Reserves `size` bytes of the bound for chunk `index` of `f` (which the
    /// caller is fetching), evicting as needed. False when the chunk cannot
    /// be kept on disk: the bound is smaller than the chunk, or every other
    /// chunk is in use.
    fn reserve(self: *Source, f: *RemoteFile, index: u64, size: u64) bool {
        if (size > self.cache_limit) return false;
        while (true) {
            self.lockAcquire();
            if (self.cache_bytes + size <= self.cache_limit) {
                self.cache_bytes += size;
                self.peak_cache_bytes = @max(self.peak_cache_bytes, self.cache_bytes);
                f.chunks.getPtr(index).?.size = size;
                self.lockRelease();
                return true;
            }
            self.lockRelease();
            _ = self.evictOne() orelse return false;
        }
    }

    /// Under the lock.
    fn ramFind(self: *Source, f: *RemoteFile, index: u64) ?usize {
        for (&self.ram, 0..) |*s, i| {
            if (s.file == f and s.index == index and s.body.len > 0) return i;
        }
        return null;
    }

    /// Under the lock: puts `body` in the RAM ring, replacing the least
    /// recently used slot not in use. Returns what the caller must free
    /// (the replaced body, or `body` itself when every slot is in use).
    fn ramInsert(self: *Source, f: *RemoteFile, index: u64, body: []u8) []u8 {
        var pick: ?usize = null;
        for (&self.ram, 0..) |*s, i| {
            if (s.pins > 0) continue;
            if (s.body.len == 0) {
                pick = i;
                break;
            }
            if (pick == null or s.tick < self.ram[pick.?].tick) pick = i;
        }
        const i = pick orelse return body;
        const old = self.ram[i].body;
        self.ram[i] = .{ .file = f, .index = index, .body = body, .tick = self.nextTick() };
        return old;
    }

    fn ramRelease(self: *Source, slot: usize) void {
        self.lockAcquire();
        self.ram[slot].pins -= 1;
        self.lockRelease();
    }
};

const RamSlot = struct {
    file: ?*RemoteFile = null,
    index: u64 = 0,
    body: []u8 = &.{},
    pins: u32 = 0,
    tick: u64 = 0,
};

fn sanitizeUrl(a: Allocator, url: []const u8) ![]u8 {
    var s = url;
    if (std.mem.indexOf(u8, s, "://")) |i| s = s[i + 3 ..];
    s = std.mem.trimEnd(u8, s, "/");
    const out = try a.dupe(u8, s);
    for (out) |*c| {
        if (!(std.ascii.isAlphanumeric(c.*) or c.* == '_' or c.* == '-' or c.* == '.')) c.* = '_';
    }
    return out;
}

/// One shard: positional reads are served from the chunk cache, fetching
/// missing chunks with range requests.
pub const RemoteFile = struct {
    src: *Source,
    name: []const u8,
    url: []const u8,
    chunk_dir: []const u8,
    /// Chunks on disk, being fetched or being evicted (guarded by `src.lock`).
    chunks: std.AutoHashMapUnmanaged(u64, Chunk) = .empty,
    /// Chunks that hold trunk tensors (see `Source.pickVictim`).
    trunk: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// Length of the shard's data from its header (0 until known).
    len: u64 = 0,

    const Chunk = struct {
        state: enum { fetching, present, evicting },
        /// Bytes on disk (or reserved while `fetching`).
        size: u64 = 0,
        tick: u64 = 0,
        /// Readers of the file right now; a pinned chunk is never evicted.
        pins: u32 = 0,
        /// The length was checked against the shard length.
        verified: bool = false,
        /// Found too short: fetched again once no reader holds it.
        bad: bool = false,
        /// Left by an earlier run and not read yet in this one.
        from_disk: bool = false,
    };

    const Acquired = union(enum) { disk, ram: usize, fetch };

    pub fn deinit(self: *RemoteFile) void {
        const gpa = self.src.gpa;
        self.chunks.deinit(gpa);
        self.trunk.deinit(gpa);
        gpa.free(self.name);
        gpa.free(self.url);
        gpa.free(self.chunk_dir);
        gpa.destroy(self);
    }

    fn chunkName(buf: []u8, index: u64) []const u8 {
        return std.fmt.bufPrint(buf, "{d}", .{index}) catch unreachable;
    }

    /// Records the shard length (known once the header is parsed); chunks
    /// read before are checked against it again on their next use.
    pub fn setLength(self: *RemoteFile, len: u64) void {
        self.src.lockAcquire();
        defer self.src.lockRelease();
        self.len = len;
        var it = self.chunks.valueIterator();
        while (it.next()) |c| c.verified = false;
    }

    /// Marks the chunks overlapping `[offset, offset + len)` as trunk chunks.
    pub fn markTrunk(self: *RemoteFile, offset: u64, len: u64) !void {
        if (len == 0) return;
        const cs = self.src.chunk_size;
        self.src.lockAcquire();
        defer self.src.lockRelease();
        var i = offset / cs;
        while (i <= (offset + len - 1) / cs) : (i += 1) try self.trunk.put(self.src.gpa, i, {});
    }

    /// Whether a chunk file of `size` bytes can be chunk `index` of this
    /// shard: full-sized, or the shard's last chunk reaching its end.
    fn validSize(self: *const RemoteFile, index: u64, size: u64) bool {
        const cs = self.src.chunk_size;
        if (size == 0 or size > cs) return false;
        if (size == cs or self.len == 0) return true;
        return index * cs + size >= self.len;
    }

    fn deleteChunkFile(self: *RemoteFile, index: u64) void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{d}", .{ self.chunk_dir, index }) catch return;
        Io.Dir.cwd().deleteFile(self.src.io, path) catch {};
    }

    /// Queues the chunks of `[offset, offset + len)` that are not on disk
    /// yet to be fetched in the background into the chunk cache, so that a
    /// later `readRange` finds them there: the routed experts of a layer are
    /// all known before the first of them runs, and fetching them over
    /// `max_in_flight` connections while the others compute is what makes a
    /// model bigger than RAM run at the link's speed. At most half the cache
    /// bound is queued ahead (further hints are dropped, and the chunks are
    /// fetched on demand); nothing is queued without a disk cache.
    pub fn prefetchRange(self: *RemoteFile, offset: u64, len: u64) void {
        if (len == 0) return;
        const src = self.src;
        const cs = src.chunk_size;
        if (src.cache_limit == 0) return;
        const ahead_max = @max(1, src.cache_limit / 2 / cs);
        var spawn: u32 = 0;
        src.lockAcquire();
        var index = offset / cs;
        const last = (offset + len - 1) / cs;
        while (index <= last) : (index += 1) {
            if (src.prefetch_jobs.items.len - src.prefetch_head >= ahead_max) break;
            if (self.chunks.contains(index) or src.ramFind(self, index) != null) continue;
            src.prefetch_jobs.append(src.gpa, .{ .file = self, .index = index }) catch break;
        }
        const queued = src.prefetch_jobs.items.len - src.prefetch_head;
        while (src.prefetch_workers < src.max_in_flight and src.prefetch_workers < queued) {
            src.prefetch_workers += 1;
            spawn += 1;
        }
        src.lockRelease();
        for (0..spawn) |_| src.prefetch_group.concurrent(src.io, Source.prefetchWorker, .{src}) catch {
            src.lockAcquire();
            src.prefetch_workers -= 1;
            src.lockRelease();
        };
    }

    /// Fetches chunk `index` into the cache unless it is there or on its way.
    fn prefetchChunk(self: *RemoteFile, index: u64) void {
        const src = self.src;
        const io = src.io;
        src.lockAcquire();
        if (self.chunks.contains(index) or src.ramFind(self, index) != null) {
            src.lockRelease();
            return;
        }
        self.chunks.put(src.gpa, index, .{ .state = .fetching }) catch {
            src.lockRelease();
            return;
        };
        src.lockRelease();
        var dir = Io.Dir.cwd().openDir(io, self.chunk_dir, .{}) catch {
            self.abandon(index);
            return;
        };
        defer dir.close(io);
        const body = self.fetchChunk(io, index) catch {
            self.abandon(index);
            return;
        };
        if (body.len > src.chunk_size or body.len == 0) {
            src.gpa.free(body);
            self.abandon(index);
            return;
        }
        var name_buf: [32]u8 = undefined;
        self.keep(io, dir, index, chunkName(&name_buf, index), body);
        _ = src.chunks_prefetched.fetchAdd(1, .monotonic);
    }

    /// Reads `out.len` bytes at `offset`. Fails with `UnexpectedEndOfFile`
    /// past the end of the shard.
    pub fn readRange(self: *RemoteFile, io: Io, offset: u64, out: []u8) !void {
        if (out.len == 0) return;
        const cs = self.src.chunk_size;
        var dir = try Io.Dir.cwd().openDir(io, self.chunk_dir, .{});
        defer dir.close(io);
        const end = offset + out.len;
        if (offset / cs == (end - 1) / cs or self.src.max_in_flight <= 1) {
            var pos = offset;
            while (pos < end) {
                const index = pos / cs;
                const in_chunk = pos - index * cs;
                const n: usize = @intCast(@min(cs - in_chunk, end - pos));
                try self.readChunk(io, dir, index, in_chunk, out[@intCast(pos - offset)..][0..n]);
                pos += n;
            }
            return;
        }
        // Several chunks: each on a task of its own, so the missing ones are
        // fetched concurrently (`fetchChunk` bounds the requests in flight).
        var group: Io.Group = .init;
        var failed: std.atomic.Value(u16) = .init(0);
        var pos = offset;
        while (pos < end) {
            const index = pos / cs;
            const in_chunk = pos - index * cs;
            const n: usize = @intCast(@min(cs - in_chunk, end - pos));
            const dest = out[@intCast(pos - offset)..][0..n];
            group.concurrent(io, chunkTask, .{ self, io, dir, index, in_chunk, dest, &failed }) catch
                try chunkTask(self, io, dir, index, in_chunk, dest, &failed);
            pos += n;
        }
        group.await(io) catch {};
        const code = failed.load(.acquire);
        if (code != 0) return @errorFromInt(code);
    }

    fn chunkTask(self: *RemoteFile, io: Io, dir: Io.Dir, index: u64, in_chunk: u64, dest: []u8, failed: *std.atomic.Value(u16)) Io.Cancelable!void {
        self.readChunk(io, dir, index, in_chunk, dest) catch |err| {
            _ = failed.cmpxchgStrong(0, @intFromError(err), .acq_rel, .monotonic);
        };
    }

    /// Copies `dest.len` bytes at `in_chunk` of chunk `index` into `dest`:
    /// from the disk cache, the RAM ring, or a fetch.
    fn readChunk(self: *RemoteFile, io: Io, dir: Io.Dir, index: u64, in_chunk: u64, dest: []u8) !void {
        const src = self.src;
        var name_buf: [32]u8 = undefined;
        const name = chunkName(&name_buf, index);
        var retries: usize = 0;
        while (true) switch (try self.acquire(io, index)) {
            .disk => {
                if (readFile(io, dir, name, in_chunk, dest)) {
                    self.unpin(index, false);
                    return;
                }
                // Missing or too short (an earlier run was interrupted, or
                // another process shares the cache): fetch it again (the
                // next `acquire` drops it once no reader holds it).
                self.unpin(index, true);
                retries += 1;
                if (retries > 3) return error.UnexpectedEndOfFile;
            },
            .ram => |slot| {
                defer src.ramRelease(slot);
                const body = src.ram[slot].body;
                if (body.len < in_chunk + dest.len) return error.UnexpectedEndOfFile;
                @memcpy(dest, body[@intCast(in_chunk)..][0..dest.len]);
                return;
            },
            .fetch => {
                const body = self.fetchChunk(io, index) catch |err| {
                    self.abandon(index);
                    return err;
                };
                if (body.len > src.chunk_size or body.len < in_chunk + dest.len) {
                    src.gpa.free(body);
                    self.abandon(index);
                    return error.UnexpectedEndOfFile;
                }
                @memcpy(dest, body[@intCast(in_chunk)..][0..dest.len]);
                self.keep(io, dir, index, name, body);
                return;
            },
        };
    }

    fn readFile(io: Io, dir: Io.Dir, name: []const u8, in_chunk: u64, dest: []u8) bool {
        const file = dir.openFile(io, name, .{}) catch return false;
        defer file.close(io);
        const got = file.readPositionalAll(io, dest, in_chunk) catch return false;
        return got == dest.len;
    }

    /// Looks chunk `index` up. `.disk`: on disk and pinned (call `unpin`);
    /// `.ram`: in the RAM ring and pinned (call `Source.ramRelease`);
    /// `.fetch`: the caller now owns fetching it (call `keep` or `abandon`).
    /// Waits while another task fetches or evicts it.
    fn acquire(self: *RemoteFile, io: Io, index: u64) !Acquired {
        const src = self.src;
        while (true) {
            src.lockAcquire();
            if (self.chunks.getPtr(index)) |c| {
                if (c.state == .present) {
                    if (!c.verified) {
                        c.verified = true;
                        if (!self.validSize(index, c.size)) c.bad = true;
                    }
                    if (!c.bad) {
                        c.pins += 1;
                        c.tick = src.nextTick();
                        const first = c.from_disk;
                        c.from_disk = false;
                        src.lockRelease();
                        if (first) _ = src.chunks_from_disk.fetchAdd(1, .monotonic);
                        return .disk;
                    }
                    if (c.pins == 0) {
                        // Drop the bad file and take over fetching the chunk.
                        c.state = .evicting;
                        const size = c.size;
                        src.lockRelease();
                        self.deleteChunkFile(index);
                        src.lockAcquire();
                        src.cache_bytes -= size;
                        self.chunks.getPtr(index).?.* = .{ .state = .fetching };
                        src.lockRelease();
                        _ = src.chunks_invalid.fetchAdd(1, .monotonic);
                        return .fetch;
                    }
                }
                // Being fetched, evicted, or a bad chunk still being read.
                src.lockRelease();
                try io.sleep(Io.Duration.fromNanoseconds(std.time.ns_per_ms), .awake);
                continue;
            }
            if (src.ramFind(self, index)) |slot| {
                src.ram[slot].pins += 1;
                src.ram[slot].tick = src.nextTick();
                src.lockRelease();
                return .{ .ram = slot };
            }
            self.chunks.put(src.gpa, index, .{ .state = .fetching }) catch {
                src.lockRelease();
                return error.OutOfMemory;
            };
            src.lockRelease();
            return .fetch;
        }
    }

    fn unpin(self: *RemoteFile, index: u64, bad: bool) void {
        self.src.lockAcquire();
        defer self.src.lockRelease();
        const c = self.chunks.getPtr(index) orelse return;
        c.pins -= 1;
        if (bad) c.bad = true;
    }

    /// Gives up fetching chunk `index` (after a failed fetch).
    fn abandon(self: *RemoteFile, index: u64) void {
        self.src.lockAcquire();
        defer self.src.lockRelease();
        if (self.chunks.fetchRemove(index)) |kv| self.src.cache_bytes -= kv.value.size;
    }

    /// Stores a fetched chunk: on disk when the bound allows (written to a
    /// temporary name and renamed, so a chunk file is always complete),
    /// otherwise in the RAM ring. Takes ownership of `body`.
    fn keep(self: *RemoteFile, io: Io, dir: Io.Dir, index: u64, name: []const u8, body: []u8) void {
        const src = self.src;
        if (src.reserve(self, index, body.len)) {
            if (writeChunk(io, dir, name, body)) {
                src.lockAcquire();
                const c = self.chunks.getPtr(index).?;
                c.state = .present;
                c.tick = src.nextTick();
                c.verified = true;
                src.lockRelease();
                src.gpa.free(body);
                return;
            } else |err| {
                if (!src.write_failure_reported.swap(true, .monotonic))
                    std.log.warn("could not write to the chunk cache {s} ({s}); chunks that do not fit are fetched again when needed", .{ self.chunk_dir, @errorName(err) });
                src.lockAcquire();
                src.cache_bytes -= body.len;
                self.chunks.getPtr(index).?.size = 0;
                src.lockRelease();
            }
        }
        src.lockAcquire();
        _ = self.chunks.remove(index);
        const drop = src.ramInsert(self, index, body);
        src.lockRelease();
        if (drop.len > 0) src.gpa.free(drop);
        _ = src.chunks_unpersisted.fetchAdd(1, .monotonic);
    }

    fn writeChunk(io: Io, dir: Io.Dir, name: []const u8, body: []const u8) !void {
        var tmp_buf: [48]u8 = undefined;
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}.part", .{name}) catch unreachable;
        errdefer dir.deleteFile(io, tmp) catch {};
        try dir.writeFile(io, .{ .sub_path = tmp, .data = body });
        try dir.rename(tmp, dir, name, io);
    }

    fn fetchChunk(self: *RemoteFile, io: Io, index: u64) ![]u8 {
        const src = self.src;
        // Bounded concurrency across all shards of the source.
        while (true) {
            const cur = src.in_flight.load(.monotonic);
            if (cur < src.max_in_flight) {
                if (src.in_flight.cmpxchgWeak(cur, cur + 1, .acquire, .monotonic) == null) break;
                continue;
            }
            try io.sleep(Io.Duration.fromNanoseconds(2 * std.time.ns_per_ms), .awake);
        }
        defer _ = src.in_flight.fetchSub(1, .release);
        const start = index * src.chunk_size;
        const last = start + src.chunk_size - 1;
        const body = try src.http.getRange(self.url, start, last);
        _ = src.ranges_fetched.fetchAdd(1, .monotonic);
        _ = src.bytes_fetched.fetchAdd(body.len, .monotonic);
        return body;
    }
};

/// What the remote source needs on disk for a model: the trunk (every stored
/// tensor that is not a routed expert, re-read by every forward pass), the
/// routed experts, and the configured chunk cache bound.
pub const Footprint = struct {
    chunk_size: u64,
    /// Stored bytes of the trunk tensors, and of the chunks they span.
    trunk_bytes: u64 = 0,
    trunk_chunk_bytes: u64 = 0,
    /// Stored bytes of the tables read a row at a time (never fetched whole).
    table_bytes: u64 = 0,
    /// Stored bytes of the largest routed expert and of all of them.
    expert_bytes: u64 = 0,
    total_expert_bytes: u64 = 0,
    num_experts: u64 = 0,
    /// Routed experts one token runs, over all its MoE layers (sum of top-k).
    experts_per_token: u64 = 0,
    cache_limit: u64,
    cache_limit_default: bool,
    cache_bytes: u64,
    disk_free: ?u64,

    pub fn trunkFits(self: Footprint) bool {
        return self.trunk_chunk_bytes <= self.cache_limit;
    }

    /// The smallest round bound that holds the trunk's chunks.
    pub fn suggestedLimit(self: Footprint) u64 {
        const t = self.trunk_chunk_bytes;
        const unit: u64 = if (t >= 1 << 30) 1 << 30 else if (t >= 1 << 20) 1 << 20 else 1 << 10;
        return (std.math.divCeil(u64, self.trunk_chunk_bytes, unit) catch unreachable) * unit;
    }

    pub fn print(self: Footprint, w: *Io.Writer) !void {
        try w.print("Disk estimate (remote chunk cache, {f} chunks):\n", .{fmtBytes(self.chunk_size)});
        try w.print("  trunk                    {f} stored, {f} of chunks (re-read by every forward pass)\n", .{ fmtBytes(self.trunk_bytes), fmtBytes(self.trunk_chunk_bytes) });
        if (self.num_experts > 0)
            try w.print("  routed expert            {f} each stored, {d} experts, {f} in total\n", .{ fmtBytes(self.expert_bytes), self.num_experts, fmtBytes(self.total_expert_bytes) });
        if (self.table_bytes > 0)
            try w.print("  row-read tables          {f} stored (only the rows a prompt hashes to are fetched)\n", .{fmtBytes(self.table_bytes)});
        try w.print("  chunk cache bound        {f}{s}, {f} cached now", .{ fmtBytes(self.cache_limit), if (self.cache_limit_default) " (default)" else "", fmtBytes(self.cache_bytes) });
        if (self.disk_free) |f| try w.print(", {f} free on its filesystem", .{fmtBytes(f)});
        try w.writeAll("\n");
        if (self.cache_limit == 0) {
            try w.writeAll("  nothing is kept on disk: every forward pass fetches the trunk again\n");
        } else if (!self.trunkFits()) {
            try w.print("  too small for the trunk: every forward pass fetches it again (--remote-cache-size {f} holds it)\n", .{fmtBytes(self.suggestedLimit())});
        } else if (self.num_experts > 0) {
            const room = self.cache_limit - self.trunk_chunk_bytes;
            const experts = if (self.expert_bytes == 0) 0 else room / self.expert_bytes;
            try w.print("  the trunk stays cached; room for {d} of {d} experts besides\n", .{ @min(experts, self.num_experts), self.num_experts });
        }
        try self.printPerToken(w);
    }

    /// Link rate the per-token time estimate assumes: about what the Hub
    /// serves to `default_connections` range requests at once.
    pub const assumed_rate: u64 = 100 * 1000 * 1000;

    /// Bytes one decoded token fetches in the steady state, when every
    /// expert is equally likely: the trunk again unless the cache holds it,
    /// plus the share of the routed experts the cache cannot hold.
    pub fn bytesPerToken(self: Footprint) u64 {
        var bytes: u64 = if (self.trunkFits()) 0 else self.trunk_chunk_bytes;
        if (self.num_experts > 0 and self.expert_bytes > 0) {
            const room = if (self.trunkFits()) self.cache_limit - self.trunk_chunk_bytes else 0;
            const held = @min(self.num_experts, room / self.expert_bytes);
            const routed = self.experts_per_token * self.expert_bytes;
            bytes += @intCast(@as(u128, routed) * (self.num_experts - held) / self.num_experts);
        }
        return bytes;
    }

    fn printPerToken(self: Footprint, w: *Io.Writer) !void {
        if (self.experts_per_token == 0 and self.trunkFits()) return;
        const b = self.bytesPerToken();
        const ms = b * 1000 / assumed_rate;
        try w.print("  per decoded token        {f} fetched once warm (~{d}.{d:0>1} s at {f}/s)", .{ fmtBytes(b), ms / 1000, ms % 1000 / 100, fmtBytes(assumed_rate) });
        if (self.experts_per_token > 0)
            try w.print("; the first ones up to {f} ({d} routed experts)", .{ fmtBytes(self.experts_per_token * self.expert_bytes + if (self.trunkFits()) 0 else self.trunk_chunk_bytes), self.experts_per_token });
        try w.writeAll("\n");
    }

    /// The startup note when the bound cannot hold the trunk.
    pub fn warn(self: Footprint, w: *Io.Writer) !void {
        if (self.trunkFits()) return;
        if (self.cache_limit == 0) {
            try w.print("Note: --remote-cache-size 0 keeps nothing on disk, so every forward pass fetches the trunk ({f}) again.\n", .{fmtBytes(self.trunk_chunk_bytes)});
            return;
        }
        try w.print("Note: the trunk spans {f} of remote chunks but the chunk cache is bounded at {f}{s}; every forward pass will fetch it again. --remote-cache-size {f} would keep it cached.\n", .{ fmtBytes(self.trunk_chunk_bytes), fmtBytes(self.cache_limit), if (self.cache_limit_default) " (the default)" else "", fmtBytes(self.suggestedLimit()) });
    }
};

fn modulePrefix(name: []const u8) []const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return name;
    return name[0..dot];
}

/// Measures `model`'s stored bytes by class and marks the chunks of the
/// trunk, so that eviction keeps them longest. Tensors are classified by
/// module (the name up to its last '.'), so the packed codes, scales and
/// zero points of a quantised expert count with the expert.
pub fn planModel(src: *Source, gpa: Allocator, model: *const model_mod.Model) !Footprint {
    src.lockAcquire();
    var fp: Footprint = .{
        .chunk_size = src.chunk_size,
        .cache_limit = src.cache_limit,
        .cache_limit_default = src.cache_limit_default,
        .cache_bytes = src.cache_bytes,
        .disk_free = src.disk_free,
    };
    src.lockRelease();
    const Group = struct { bytes: u64 = 0, experts: u32 = 0 };
    var groups = std.StringHashMapUnmanaged(Group).empty;
    defer groups.deinit(gpa);
    for (model.layers) |*l| if (l.moe) |*m| for (m.experts) |*ex| {
        var seen: [3][]const u8 = undefined;
        var n_seen: usize = 0;
        for ([_][]const u8{ ex.gate_ref.ref.name, ex.up_ref.ref.name, ex.down_ref.ref.name }) |name| {
            const p = modulePrefix(name);
            var dup = false;
            for (seen[0..n_seen]) |s| dup = dup or std.mem.eql(u8, s, p);
            if (dup) continue;
            seen[n_seen] = p;
            n_seen += 1;
            const gop = try groups.getOrPut(gpa, p);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            gop.value_ptr.experts += 1;
        }
        fp.num_experts += 1;
    };
    for (model.layers) |*l| if (l.moe) |*m| {
        fp.experts_per_token += m.top_k;
    };
    for (model.files) |f| {
        const rf = switch (f.source) {
            .remote => |r| r,
            .local => continue,
        };
        var it = f.tensors.iterator();
        while (it.next()) |kv| {
            const info = kv.value_ptr.*;
            // Decoded views of quantised tensors live above the file's bytes.
            if (info.offset >= f.len) continue;
            try classify(model, &groups, rf, &fp, info.name, info.offset, info.byte_len);
        }
        var rit = f.raw.iterator();
        while (rit.next()) |kv| try classify(model, &groups, rf, &fp, kv.value_ptr.name, kv.value_ptr.offset, kv.value_ptr.byte_len);
        // A dequantised tensor is such a view, its stored codes and scales out
        // of both indexes: count them by the view's name (gpt-oss's MXFP4
        // expert blocks, fp8 trunks), in whichever file holds each.
        for (f.dequants.items) |dq| {
            for ([_]?@TypeOf(dq.data){ dq.data, dq.scale, dq.zero }) |maybe| if (maybe) |p| {
                if (p.byte_len == 0) continue;
                const prf = switch (p.file.source) {
                    .remote => |r| r,
                    .local => continue,
                };
                try classify(model, &groups, prf, &fp, dq.name, p.offset, p.byte_len);
            };
        }
    }
    // Then the chunks the trunk spans, once every file's trunk is marked.
    for (model.files) |f| {
        const rf = switch (f.source) {
            .remote => |r| r,
            .local => continue,
        };
        const cs = src.chunk_size;
        src.lockAcquire();
        var tit = rf.trunk.keyIterator();
        while (tit.next()) |idx| {
            const start = idx.* * cs;
            fp.trunk_chunk_bytes += if (f.len > start) @min(cs, f.len - start) else 0;
        }
        src.lockRelease();
    }
    var git = groups.valueIterator();
    while (git.next()) |g| fp.total_expert_bytes += g.bytes;
    for (model.layers) |*l| if (l.moe) |*m| for (m.experts) |*ex| {
        var seen: [3][]const u8 = undefined;
        var n_seen: usize = 0;
        var bytes: u64 = 0;
        for ([_][]const u8{ ex.gate_ref.ref.name, ex.up_ref.ref.name, ex.down_ref.ref.name }) |name| {
            const p = modulePrefix(name);
            var dup = false;
            for (seen[0..n_seen]) |s| dup = dup or std.mem.eql(u8, s, p);
            if (dup) continue;
            seen[n_seen] = p;
            n_seen += 1;
            const g = groups.get(p).?;
            bytes += g.bytes / @max(g.experts, 1);
        }
        fp.expert_bytes = @max(fp.expert_bytes, bytes);
    };
    return fp;
}

fn classify(model: *const model_mod.Model, groups: anytype, rf: *RemoteFile, fp: *Footprint, name: []const u8, offset: u64, len: u64) !void {
    if (model.isRowTable(name)) {
        fp.table_bytes += len;
        return;
    }
    if (groups.getPtr(modulePrefix(name))) |g| {
        g.bytes += len;
        return;
    }
    fp.trunk_bytes += len;
    try rf.markTrunk(offset, len);
}

/// Number of chunk files a full copy of a shard of `len` bytes needs.
pub fn chunkCount(len: u64, chunk_size: u64) u64 {
    return (len + chunk_size - 1) / chunk_size;
}

test "remote id forms" {
    try std.testing.expect(isRemoteId("hf://Qwen/Qwen3-30B-A3B"));
    try std.testing.expect(isRemoteId("http://127.0.0.1:8000/"));
    try std.testing.expect(!isRemoteId("Qwen/Qwen3-30B-A3B"));
    try std.testing.expectEqualStrings("Qwen/Qwen3-30B-A3B", hubId("hf://Qwen/Qwen3-30B-A3B"));
    const gpa = std.testing.allocator;
    const s = try sanitizeUrl(gpa, "http://127.0.0.1:8000/models/x/");
    defer gpa.free(s);
    try std.testing.expectEqualStrings("127.0.0.1_8000_models_x", s);
    try std.testing.expectEqual(@as(u64, 3), chunkCount(20, 8));
}

test "default chunk cache bound" {
    try std.testing.expectEqual(default_cache_cap, defaultCacheSize(null, 0));
    try std.testing.expectEqual(@as(u64, 50 << 30), defaultCacheSize(80 << 30, 20 << 30));
    try std.testing.expectEqual(default_cache_cap, defaultCacheSize(1 << 40, 0));
    try std.testing.expectEqual(@as(u64, 0), defaultCacheSize(0, 0));
    if (builtin.os.tag == .linux) try std.testing.expect(diskFree(".") != null);
    try std.testing.expectEqualStrings("model.layers.0.mlp.experts.3.up_proj", modulePrefix("model.layers.0.mlp.experts.3.up_proj.weight_packed"));
}
