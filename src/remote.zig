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
//! Model ids: `hf://owner/name` (the Hub, revision `main` or `--model-commit`),
//! a plain `owner/name` when `--remote-weights` is set, or an `http(s)://.../`
//! base URL that serves the model files.

const std = @import("std");
const Io = std.Io;
const hf = @import("hf.zig");

const Allocator = std.mem.Allocator;

pub const default_chunk_size: u64 = 8 << 20;

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
    /// Chunk reads served from the disk cache.
    chunks_from_disk: u64,
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
    max_in_flight: u32 = 4,
    in_flight: std.atomic.Value(u32) = .init(0),
    ranges_fetched: std.atomic.Value(u64) = .init(0),
    bytes_fetched: std.atomic.Value(u64) = .init(0),
    chunks_from_disk: std.atomic.Value(u64) = .init(0),

    pub const OpenOptions = struct {
        revision: ?[]const u8 = null,
        chunk_size: u64 = default_chunk_size,
    };

    /// Resolves `model` to a base URL and cache directory, fetches the small
    /// files that are not cached yet and reads the shard index.
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
        };
        errdefer self.arena.deinit();
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
        const required = [_][]const u8{ "config.json", "tokenizer.json" };
        const optional = [_][]const u8{ "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json" };
        for (required) |name| try self.fetchSmall(dir, name, true, out);
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
        try out.print("* {d} safetensors shard(s); headers are fetched now, tensors on demand in {f} chunks\n", .{ self.shards.len, @import("budget.zig").fmtBytes(self.chunk_size) });
        return self;
    }

    pub fn deinit(self: *Source) void {
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

    /// Opens shard `name` for range reads (the header is read by the safetensors reader).
    pub fn openFile(self: *Source, name: []const u8) !*RemoteFile {
        const f = try self.gpa.create(RemoteFile);
        errdefer self.gpa.destroy(f);
        f.* = .{
            .src = self,
            .name = try self.gpa.dupe(u8, name),
            .url = try std.fmt.allocPrint(self.gpa, "{s}{s}", .{ self.base_url, name }),
            .chunk_dir = try std.fs.path.join(self.gpa, &.{ self.dir_path, "chunks", name }),
        };
        errdefer {
            self.gpa.free(f.name);
            self.gpa.free(f.url);
            self.gpa.free(f.chunk_dir);
        }
        try Io.Dir.cwd().createDirPath(self.io, f.chunk_dir);
        try self.files.append(self.gpa, f);
        return f;
    }

    pub fn stats(self: *const Source) Stats {
        return .{
            .ranges_fetched = self.ranges_fetched.load(.monotonic),
            .bytes_fetched = self.bytes_fetched.load(.monotonic),
            .chunks_from_disk = self.chunks_from_disk.load(.monotonic),
        };
    }
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
    /// Chunks known to be on disk or being fetched right now.
    states: std.AutoHashMapUnmanaged(u64, State) = .empty,
    lock: std.atomic.Value(bool) = .init(false),

    const State = enum { fetching, present };

    pub fn deinit(self: *RemoteFile) void {
        const gpa = self.src.gpa;
        self.states.deinit(gpa);
        gpa.free(self.name);
        gpa.free(self.url);
        gpa.free(self.chunk_dir);
        gpa.destroy(self);
    }

    fn lockAcquire(self: *RemoteFile) void {
        while (self.lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn lockRelease(self: *RemoteFile) void {
        self.lock.store(false, .release);
    }

    fn chunkName(buf: []u8, index: u64) []const u8 {
        return std.fmt.bufPrint(buf, "{d}", .{index}) catch unreachable;
    }

    /// Reads `out.len` bytes at `offset`. Fails with `UnexpectedEndOfFile`
    /// past the end of the shard.
    pub fn readRange(self: *RemoteFile, io: Io, offset: u64, out: []u8) !void {
        if (out.len == 0) return;
        const cs = self.src.chunk_size;
        var dir = try Io.Dir.cwd().openDir(io, self.chunk_dir, .{});
        defer dir.close(io);
        var pos = offset;
        const end = offset + out.len;
        while (pos < end) {
            const index = pos / cs;
            try self.ensureChunk(io, dir, index);
            const in_chunk = pos - index * cs;
            const n: usize = @intCast(@min(cs - in_chunk, end - pos));
            var name_buf: [32]u8 = undefined;
            const file = try dir.openFile(io, chunkName(&name_buf, index), .{});
            defer file.close(io);
            const got = try file.readPositionalAll(io, out[@intCast(pos - offset)..][0..n], in_chunk);
            if (got != n) return error.UnexpectedEndOfFile;
            pos += n;
        }
    }

    /// Makes chunk `index` present on disk, fetching it unless another task
    /// is already doing so (then waits for it).
    fn ensureChunk(self: *RemoteFile, io: Io, dir: Io.Dir, index: u64) !void {
        var name_buf: [32]u8 = undefined;
        const name = chunkName(&name_buf, index);
        while (true) {
            self.lockAcquire();
            const state = self.states.get(index);
            if (state == .present) {
                self.lockRelease();
                return;
            }
            if (state == null) {
                // Not seen in this run: on disk from an earlier run?
                if (dir.access(io, name, .{})) |_| {
                    self.states.put(self.src.gpa, index, .present) catch {};
                    self.lockRelease();
                    _ = self.src.chunks_from_disk.fetchAdd(1, .monotonic);
                    return;
                } else |_| {}
                self.states.put(self.src.gpa, index, .fetching) catch {
                    self.lockRelease();
                    return error.OutOfMemory;
                };
                self.lockRelease();
                break;
            }
            // Someone else is fetching it.
            self.lockRelease();
            try io.sleep(Io.Duration.fromNanoseconds(2 * std.time.ns_per_ms), .awake);
        }
        self.fetchChunk(io, dir, index, name) catch |err| {
            self.lockAcquire();
            _ = self.states.remove(index);
            self.lockRelease();
            return err;
        };
        self.lockAcquire();
        self.states.put(self.src.gpa, index, .present) catch {};
        self.lockRelease();
    }

    fn fetchChunk(self: *RemoteFile, io: Io, dir: Io.Dir, index: u64, name: []const u8) !void {
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
        defer src.gpa.free(body);
        _ = src.ranges_fetched.fetchAdd(1, .monotonic);
        _ = src.bytes_fetched.fetchAdd(body.len, .monotonic);
        var tmp_buf: [48]u8 = undefined;
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}.part", .{name}) catch unreachable;
        try dir.writeFile(io, .{ .sub_path = tmp, .data = body });
        try dir.rename(tmp, dir, name, io);
    }
};

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
