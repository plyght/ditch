//! Tests for the remote weight source (remote.zig) over a local HTTP server
//! that honours Range requests (tools/range_server.py): loading fetches only
//! the small files, headers and norms; a prefill fetches the trunk plus the
//! routed experts; chunks are cached on disk and a second load fetches nothing.
//! The chunk cache stays under its bound (`--remote-cache-size`) through a
//! forward pass, under concurrent fetches and at size 0, keeps the trunk
//! over the experts, and detects and repairs truncated chunk files.

const std = @import("std");
const Io = std.Io;
const hf = @import("hf.zig");
const remote = @import("remote.zig");
const model_mod = @import("model.zig");
const tensor = @import("tensor.zig");

const Model = model_mod.Model;
const fixture = "tests/fixtures/qwen3_moe_big";

const Server = struct {
    child: std.process.Child,
    port: u16,

    fn start(io: Io, dir: []const u8, log_path: []const u8) !Server {
        var child = try std.process.spawn(io, .{
            .argv = &.{ "python3", "tools/range_server.py", dir, log_path },
            .stdin = .ignore,
            .stdout = .pipe,
            .stderr = .inherit,
        });
        errdefer child.kill(io);
        var buf: [256]u8 = undefined;
        var fr = child.stdout.?.readerStreaming(io, &buf);
        const line = try fr.interface.takeDelimiterExclusive('\n');
        if (!std.mem.startsWith(u8, line, "PORT ")) return error.ServerDidNotStart;
        const port = try std.fmt.parseInt(u16, std.mem.trim(u8, line["PORT ".len..], " \r"), 10);
        return .{ .child = child, .port = port };
    }

    fn stop(self: *Server, io: Io) void {
        self.child.kill(io);
    }
};

fn countLines(gpa: std.mem.Allocator, io: Io, path: []const u8) !usize {
    const text = Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer gpa.free(text);
    return std.mem.count(u8, text, "\n");
}

fn firstTokenLogits(gpa: std.mem.Allocator, model: *Model, ids: []const u32) ![]f32 {
    const c = &model.config;
    var ws = try model_mod.Workspace.init(model.gpa, c, 16, 1);
    defer ws.deinit();
    var cache = try model_mod.KvCache.initFor(model, model.gpa, 1, 16);
    defer cache.deinit();
    const logits = try gpa.alloc(f32, c.vocab_size);
    errdefer gpa.free(logits);
    try model_mod.prefill(model, &ws, &cache, &.{ids}, logits, null);
    return logits;
}

test "remote source: headers up front, tensors on demand, chunks cached on disk" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..n];
    const log_path = try std.fs.path.join(gpa, &.{ root, "requests.log" });
    defer gpa.free(log_path);
    const cache_root = try std.fs.path.join(gpa, &.{ root, "cache" });
    defer gpa.free(cache_root);
    const scratch = try std.fs.path.join(gpa, &.{ root, "scratch" });
    defer gpa.free(scratch);

    var server = try Server.start(io, fixture, log_path);
    defer server.stop(io);
    var url_buf: [64]u8 = undefined;
    const base_url = try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/", .{server.port});

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    // An empty environment: no proxy settings, no token.
    var environ = std.process.Environ.Map.init(gpa);
    defer environ.deinit();
    var http = try hf.Http.init(gpa, io, arena.allocator(), &environ);
    defer http.deinit();
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();

    // The reference: the same fixture memory-mapped from disk.
    const local = try Model.load(gpa, io, &pool, fixture);
    defer local.deinit();
    const total_bytes = blk: {
        var t: u64 = 0;
        for (local.files) |f| {
            var it = f.tensors.iterator();
            while (it.next()) |kv| t += kv.value_ptr.byte_len;
        }
        break :blk t;
    };
    const ids = [_]u32{ 40, 100 };
    const want = try firstTokenLogits(gpa, local, &ids);
    defer gpa.free(want);

    // First run: small files, then the shard headers (and the norms the loader reads).
    {
        const src = try remote.Source.open(gpa, io, &http, cache_root, base_url, .{ .chunk_size = chunk }, &sink.writer);
        defer src.deinit();
        try std.testing.expectEqual(@as(usize, 2), src.shards.len);
        try std.testing.expect(std.mem.indexOf(u8, sink.written(), "Fetching config.json") != null);
        const model = try Model.loadWithOptions(gpa, io, &pool, src.dir_path, .{ .store = .streamed, .remote = src, .scratch_dir = scratch });
        defer model.deinit();
        try std.testing.expect(model.streamed() and model.warp());
        for (model.files) |f| try std.testing.expect(f.isRemote());
        const after_load = src.stats();
        try std.testing.expect(after_load.ranges_fetched > 0);
        // Headers and norms only: a small fraction of the model.
        try std.testing.expect(after_load.bytes_fetched < total_bytes / 4);
        // A two-token prefill: the trunk of every layer plus at most 4 routed
        // experts per layer, far less than the whole model, and bitwise the
        // same logits as the local model.
        const got = try firstTokenLogits(gpa, model, &ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        const after_prefill = src.stats();
        try std.testing.expect(after_prefill.ranges_fetched > after_load.ranges_fetched);
        try std.testing.expect(after_prefill.bytes_fetched > after_load.bytes_fetched);
        try std.testing.expect(after_prefill.bytes_fetched < total_bytes * 6 / 10);
        // Every fetched range is one aligned chunk.
        try std.testing.expect(after_prefill.bytes_fetched <= after_prefill.ranges_fetched * chunk);
        // The expert cache saw the routed experts and nothing else.
        const st = model.expert_cache.?.stats();
        try std.testing.expect(st.misses > 0 and st.misses <= 4 * model.config.num_layers);
        // Chunks are on disk.
        const chunk_file = try std.fs.path.join(gpa, &.{ src.dir_path, "chunks", "model-00001-of-00002.safetensors", "0" });
        defer gpa.free(chunk_file);
        const stat = try Io.Dir.cwd().statFile(io, chunk_file, .{});
        try std.testing.expectEqual(chunk, stat.size);
        std.debug.print("\n[remote qwen3_moe_big, {d}B chunks] load: {d} ranges / {d} B; after 2-token prefill: {d} ranges / {d} B of {d} B total\n", .{
            chunk, after_load.ranges_fetched, after_load.bytes_fetched, after_prefill.ranges_fetched, after_prefill.bytes_fetched, total_bytes,
        });
    }
    const requests_after_first = try countLines(gpa, io, log_path);
    try std.testing.expect(requests_after_first > 0);

    // Second run over the same cache: nothing is fetched, the same prefill is served from disk.
    {
        const src = try remote.Source.open(gpa, io, &http, cache_root, base_url, .{ .chunk_size = chunk }, &sink.writer);
        defer src.deinit();
        const model = try Model.loadWithOptions(gpa, io, &pool, src.dir_path, .{ .store = .streamed, .remote = src, .scratch_dir = scratch });
        defer model.deinit();
        const got = try firstTokenLogits(gpa, model, &ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        const st = src.stats();
        try std.testing.expectEqual(@as(u64, 0), st.ranges_fetched);
        try std.testing.expect(st.chunks_from_disk > 0);
    }
    try std.testing.expectEqual(requests_after_first, try countLines(gpa, io, log_path));

    // Reading the rest of a shard (an export does that) fills the chunk
    // cache, after which the full-download path assembles the file from it.
    {
        const src = try remote.Source.open(gpa, io, &http, cache_root, base_url, .{ .chunk_size = chunk }, &sink.writer);
        defer src.deinit();
        const model = try Model.loadWithOptions(gpa, io, &pool, src.dir_path, .{ .store = .streamed, .remote = src, .scratch_dir = scratch });
        defer model.deinit();
        const out_dir = try std.fs.path.join(gpa, &.{ root, "exported" });
        defer gpa.free(out_dir);
        try @import("export.zig").saveModel(gpa, io, model, out_dir, .{}, &sink.writer);
        const st = src.stats();
        try std.testing.expect(st.bytes_fetched > 0);
        // The export re-read every tensor: the cached chunks now cover both shards.
        var mdir = try Io.Dir.cwd().openDir(io, src.dir_path, .{});
        defer mdir.close(io);
        for (src.shards) |shard| {
            try std.testing.expectError(error.FileNotFound, mdir.access(io, shard, .{}));
            try std.testing.expect(try hf.assembleFromChunks(gpa, io, mdir, shard));
            const assembled = try mdir.readFileAlloc(io, shard, gpa, .unlimited);
            defer gpa.free(assembled);
            const original_path = try std.fs.path.join(gpa, &.{ fixture, shard });
            defer gpa.free(original_path);
            const original = try Io.Dir.cwd().readFileAlloc(io, original_path, gpa, .unlimited);
            defer gpa.free(original);
            try std.testing.expectEqualSlices(u8, original, assembled);
        }
        // The exported model is complete (every expert copied) and reloads locally.
        const reloaded = try Model.load(gpa, io, &pool, out_dir);
        defer reloaded.deinit();
        const got = try firstTokenLogits(gpa, reloaded, &ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
    }
    // The 404 path is reported as a missing model.
    var missing_buf: [96]u8 = undefined;
    const missing_url = try std.fmt.bufPrint(&missing_buf, "http://127.0.0.1:{d}/nope/", .{server.port});
    try std.testing.expectError(error.ModelNotFound, remote.Source.open(gpa, io, &http, cache_root, missing_url, .{}, &sink.writer));
}

// ---------------------------------------------------------------------------
// Bounded chunk cache
// ---------------------------------------------------------------------------

const chunk: u64 = 4096;
/// Eight tokens: the router picks more experts than the two-token prefill does.
const long_ids = [_]u32{ 40, 100, 7, 250, 3, 199, 61, 120 };
const shard_names = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };

/// A temporary directory with a range server over the fixture and an HTTP client.
const Env = struct {
    gpa: std.mem.Allocator,
    io: Io,
    tmp: std.testing.TmpDir,
    root: []const u8,
    server: Server,
    base_url: []const u8,
    arena: std.heap.ArenaAllocator,
    environ: std.process.Environ.Map,
    http: hf.Http,
    sink: Io.Writer.Allocating,

    fn init(self: *Env, gpa: std.mem.Allocator, io: Io) !void {
        return self.initDir(gpa, io, fixture);
    }

    fn initDir(self: *Env, gpa: std.mem.Allocator, io: Io, dir: []const u8) !void {
        self.gpa = gpa;
        self.io = io;
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const n = try self.tmp.dir.realPath(io, &path_buf);
        self.root = try gpa.dupe(u8, path_buf[0..n]);
        errdefer gpa.free(self.root);
        const log_path = try std.fs.path.join(gpa, &.{ self.root, "requests.log" });
        defer gpa.free(log_path);
        self.server = try Server.start(io, dir, log_path);
        errdefer self.server.stop(io);
        self.base_url = try std.fmt.allocPrint(gpa, "http://127.0.0.1:{d}/", .{self.server.port});
        self.arena = std.heap.ArenaAllocator.init(gpa);
        self.environ = std.process.Environ.Map.init(gpa);
        self.http = try hf.Http.init(gpa, io, self.arena.allocator(), &self.environ);
        self.sink = .init(gpa);
    }

    fn deinit(self: *Env) void {
        self.sink.deinit();
        self.http.deinit();
        self.environ.deinit();
        self.arena.deinit();
        self.gpa.free(self.base_url);
        self.server.stop(self.io);
        self.gpa.free(self.root);
        self.tmp.cleanup();
    }

    fn path(self: *Env, sub: []const u8) ![]u8 {
        return std.fs.path.join(self.gpa, &.{ self.root, sub });
    }

    fn open(self: *Env, cache: []const u8, cache_size: ?u64) !*remote.Source {
        return self.openWith(cache, cache_size, 0);
    }

    fn openWith(self: *Env, cache: []const u8, cache_size: ?u64, connections: u32) !*remote.Source {
        return remote.Source.open(self.gpa, self.io, &self.http, cache, self.base_url, .{ .chunk_size = chunk, .cache_size = cache_size, .connections = connections }, &self.sink.writer);
    }
};

const Usage = struct { files: u64, bytes: u64, parts: u64 };

/// Files and total bytes under `src`'s chunk directories (`.part` files included).
fn chunkDirUsage(io: Io, src: *remote.Source) !Usage {
    var u: Usage = .{ .files = 0, .bytes = 0, .parts = 0 };
    for (src.files.items) |f| {
        var dir = Io.Dir.cwd().openDir(io, f.chunk_dir, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file) continue;
            // A file can be evicted between listing and stat.
            const st = dir.statFile(io, entry.name, .{}) catch continue;
            u.files += 1;
            u.bytes += st.size;
            if (std.mem.endsWith(u8, entry.name, ".part")) u.parts += 1;
        }
    }
    return u;
}

fn loadRemote(gpa: std.mem.Allocator, io: Io, pool: *const tensor.Pool, src: *remote.Source, scratch: []const u8) !*Model {
    return Model.loadWithOptions(gpa, io, pool, src.dir_path, .{ .store = .streamed, .remote = src, .scratch_dir = scratch });
}

test "remote chunk cache: bounded below the model, LRU with the trunk kept, bit-identical results" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var env: Env = undefined;
    try env.init(gpa, io);
    defer env.deinit();
    const scratch = try env.path("scratch");
    defer gpa.free(scratch);

    const local = try Model.load(gpa, io, &pool, fixture);
    defer local.deinit();
    const want = try firstTokenLogits(gpa, local, &long_ids);
    defer gpa.free(want);
    var stored_total: u64 = 0;
    for (local.files) |f| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| stored_total += kv.value_ptr.byte_len;
    }

    // Unbounded reference run over the remote source.
    const cache_a = try env.path("cache_a");
    defer gpa.free(cache_a);
    var footprint: remote.Footprint = undefined;
    var unbounded: remote.Stats = undefined;
    {
        const src = try env.open(cache_a, std.math.maxInt(u64));
        defer src.deinit();
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        footprint = try remote.planModel(src, gpa, model);
        const got = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        unbounded = src.stats();
        try std.testing.expectEqual(@as(u64, 0), unbounded.chunks_evicted);
        try std.testing.expectEqual(@as(u64, 0), unbounded.chunks_unpersisted);
    }
    // The footprint splits the stored bytes into trunk and experts (bf16
    // fixture: the stored experts are exactly the model's routed experts).
    try std.testing.expectEqual(stored_total, footprint.trunk_bytes + footprint.total_expert_bytes);
    try std.testing.expectEqual(local.total_expert_bytes, footprint.total_expert_bytes);
    try std.testing.expectEqual(@as(u64, 4 * 16), footprint.num_experts);
    try std.testing.expectEqual(local.largest_expert_bytes, footprint.expert_bytes);
    try std.testing.expect(footprint.trunk_chunk_bytes >= footprint.trunk_bytes);
    try std.testing.expect(footprint.trunkFits());

    // Bounded run: room for the trunk and a few chunks of experts, far less
    // than the model and less than one pass touches.
    const bound = footprint.trunk_chunk_bytes + 24 * chunk;
    try std.testing.expect(bound < stored_total * 6 / 10);
    try std.testing.expect(bound < unbounded.peak_cache_bytes);
    const cache_b = try env.path("cache_b");
    defer gpa.free(cache_b);
    {
        const src = try env.open(cache_b, bound);
        defer src.deinit();
        try std.testing.expectEqual(bound, src.cache_limit);
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        const fp = try remote.planModel(src, gpa, model);
        try std.testing.expect(fp.trunkFits());
        var dry: Io.Writer.Allocating = .init(gpa);
        defer dry.deinit();
        try fp.print(&dry.writer);
        try std.testing.expect(std.mem.indexOf(u8, dry.written(), "the trunk stays cached") != null);
        // Per decoded token: at most the routed experts, none of the trunk.
        try std.testing.expect(std.mem.indexOf(u8, dry.written(), "per decoded token") != null);
        var routed: u64 = 0;
        for (model.layers) |*l| if (l.moe) |*m| {
            routed += m.top_k;
        };
        try std.testing.expect(routed > 0);
        try std.testing.expectEqual(routed, fp.experts_per_token);
        try std.testing.expect(fp.bytesPerToken() <= fp.experts_per_token * fp.expert_bytes);
        const got = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        const first = src.stats();
        try std.testing.expect(first.chunks_evicted > 0);
        try std.testing.expect(first.peak_cache_bytes <= bound);
        try std.testing.expect((try chunkDirUsage(io, src)).bytes <= bound);
        // A second pass (the next trial) re-reads the trunk. It stayed
        // cached, so only evicted expert chunks are fetched again.
        model.expert_cache.?.trim();
        const again = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(again);
        try std.testing.expectEqualSlices(f32, want, again);
        const second = src.stats();
        const refetched = second.ranges_fetched - first.ranges_fetched;
        // Every trunk chunk a pass reads (the ones the unbounded run cached)
        // is still on disk: evictions only took expert chunks.
        var trunk_kept: u64 = 0;
        for (src.files.items) |f| {
            const a_dir = try std.mem.replaceOwned(u8, gpa, f.chunk_dir, cache_b, cache_a);
            defer gpa.free(a_dir);
            var dir_a = try Io.Dir.cwd().openDir(io, a_dir, .{});
            defer dir_a.close(io);
            var it = f.trunk.keyIterator();
            while (it.next()) |idx| {
                var nb: [32]u8 = undefined;
                if (dir_a.access(io, try std.fmt.bufPrint(&nb, "{d}", .{idx.*}), .{})) |_| {
                    const c = f.chunks.get(idx.*) orelse return error.TrunkChunkEvicted;
                    try std.testing.expect(c.state == .present);
                    trunk_kept += 1;
                } else |_| {}
            }
        }
        try std.testing.expect(trunk_kept > 0);
        // So the next pass fetches only expert chunks again.
        try std.testing.expect(refetched > 0);
        try std.testing.expect(refetched < first.ranges_fetched);
        try std.testing.expect(second.peak_cache_bytes <= bound);
        try std.testing.expect((try chunkDirUsage(io, src)).bytes <= bound);
        std.debug.print("\n[remote bounded cache] bound {d} B (trunk {d} B of chunks), stored model {d} B, unbounded pass cached {d} B: {d} evictions, {d} trunk chunks kept, pass 1 fetched {d} ranges, pass 2 {d}\n", .{ bound, fp.trunk_chunk_bytes, stored_total, unbounded.peak_cache_bytes, second.chunks_evicted, trunk_kept, first.ranges_fetched, refetched });
    }
    // Reopening with a smaller bound trims the cache to it first.
    {
        const src = try env.open(cache_b, bound / 2);
        defer src.deinit();
        try std.testing.expect(src.stats().chunks_evicted > 0);
        try std.testing.expect((try chunkDirUsage(io, src)).bytes <= bound / 2);
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        const fp = try remote.planModel(src, gpa, model);
        // The bound no longer holds the trunk: said once, with the size that would.
        try std.testing.expect(!fp.trunkFits());
        var note: Io.Writer.Allocating = .init(gpa);
        defer note.deinit();
        try fp.warn(&note.writer);
        try std.testing.expect(std.mem.indexOf(u8, note.written(), "every forward pass will fetch it again. --remote-cache-size") != null);
        try fp.print(&note.writer);
        try std.testing.expect(std.mem.indexOf(u8, note.written(), "too small for the trunk") != null);
        try std.testing.expect(fp.suggestedLimit() >= fp.trunk_chunk_bytes);
        // Without the trunk cached, every token fetches it and all its experts.
        try std.testing.expectEqual(fp.trunk_chunk_bytes + fp.experts_per_token * fp.expert_bytes, fp.bytesPerToken());
        const got = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        try std.testing.expect(src.stats().peak_cache_bytes <= bound / 2);
        try std.testing.expect((try chunkDirUsage(io, src)).bytes <= bound / 2);
    }
}

test "remote footprint: dequantised tensors' codes and scales are counted (gpt-oss MXFP4 experts, fp8 trunk)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    for ([_][]const u8{ "tests/fixtures/gpt_oss_mxfp4", "tests/fixtures/qwen2_fp8" }) |dir| {
        var env: Env = undefined;
        try env.initDir(gpa, io, dir);
        defer env.deinit();
        const scratch = try env.path("scratch");
        defer gpa.free(scratch);
        const cache = try env.path("cache");
        defer gpa.free(cache);
        const src = try env.open(cache, std.math.maxInt(u64));
        defer src.deinit();
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        const fp = try remote.planModel(src, gpa, model);
        // Every stored byte is trunk or expert, the quantised ones included.
        var stored: u64 = 0;
        for (model.files) |f| stored += f.len - 8 - f.header_len;
        try std.testing.expectEqual(stored, fp.trunk_bytes + fp.total_expert_bytes);
        if (fp.num_experts > 0) {
            // gpt-oss: the experts are their MXFP4 blocks and scales, not only their biases.
            try std.testing.expect(fp.total_expert_bytes > stored / 2);
            try std.testing.expectEqual(fp.total_expert_bytes, fp.expert_bytes * fp.num_experts);
        }
    }
}

test "remote readRange: a span of many chunks, fetched one at a time or concurrently, reads the same bytes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var env: Env = undefined;
    try env.init(gpa, io);
    defer env.deinit();
    const shard = "model-00001-of-00002.safetensors";
    // The reference bytes, read straight from the fixture.
    var dir = try Io.Dir.cwd().openDir(io, fixture, .{});
    defer dir.close(io);
    const file = try dir.openFile(io, shard, .{});
    defer file.close(io);
    const len: usize = 40 * chunk + 123;
    const offset: u64 = chunk / 2 + 7; // straddles chunk boundaries at both ends
    const want = try gpa.alloc(u8, len);
    defer gpa.free(want);
    try std.testing.expectEqual(len, try file.readPositionalAll(io, want, offset));
    for ([_]u32{ 1, 16 }) |connections| {
        const cache = try env.path(if (connections == 1) "cache_1" else "cache_16");
        defer gpa.free(cache);
        const src = try env.openWith(cache, std.math.maxInt(u64), connections);
        defer src.deinit();
        const rf = try src.openFile(shard);
        const got = try gpa.alloc(u8, len);
        defer gpa.free(got);
        try rf.readRange(io, offset, got);
        try std.testing.expectEqualSlices(u8, want, got);
        // Each chunk of the span fetched once: 41 of them (plus the header's).
        const st = src.stats();
        try std.testing.expect(st.ranges_fetched >= 41 and st.ranges_fetched <= 43);
        // And read again, all from the disk cache.
        try rf.readRange(io, offset, got);
        try std.testing.expectEqualSlices(u8, want, got);
        try std.testing.expectEqual(st.ranges_fetched, src.stats().ranges_fetched);
    }
}

test "remote prefetchRange: queued chunks are fetched in the background, then read from disk" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var env: Env = undefined;
    try env.init(gpa, io);
    defer env.deinit();
    const shard = "model-00001-of-00002.safetensors";
    var dir = try Io.Dir.cwd().openDir(io, fixture, .{});
    defer dir.close(io);
    const file = try dir.openFile(io, shard, .{});
    defer file.close(io);
    const len: usize = 24 * chunk + 5;
    const offset: u64 = 3 * chunk + 11;
    const want = try gpa.alloc(u8, len);
    defer gpa.free(want);
    try std.testing.expectEqual(len, try file.readPositionalAll(io, want, offset));
    const cache = try env.path("cache");
    defer gpa.free(cache);
    const src = try env.openWith(cache, std.math.maxInt(u64), 16);
    defer src.deinit();
    const rf = try src.openFile(shard);
    const before = src.stats().ranges_fetched;
    rf.prefetchRange(offset, len);
    rf.prefetchRange(offset, len); // queued twice, fetched once
    src.awaitPrefetch();
    const fetched = src.stats().ranges_fetched - before;
    try std.testing.expect(fetched >= 25 and fetched <= 26);
    const got = try gpa.alloc(u8, len);
    defer gpa.free(got);
    try rf.readRange(io, offset, got);
    try std.testing.expectEqualSlices(u8, want, got);
    try std.testing.expectEqual(before + fetched, src.stats().ranges_fetched);
}

test "remote chunk cache: size 0 keeps nothing on disk" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var env: Env = undefined;
    try env.init(gpa, io);
    defer env.deinit();
    const scratch = try env.path("scratch");
    defer gpa.free(scratch);
    const local = try Model.load(gpa, io, &pool, fixture);
    defer local.deinit();
    const want = try firstTokenLogits(gpa, local, &long_ids);
    defer gpa.free(want);

    const cache = try env.path("cache");
    defer gpa.free(cache);
    const src = try env.open(cache, 0);
    defer src.deinit();
    try std.testing.expect(std.mem.indexOf(u8, env.sink.written(), "nothing is kept on disk") != null);
    const model = try loadRemote(gpa, io, &pool, src, scratch);
    defer model.deinit();
    const fp = try remote.planModel(src, gpa, model);
    try std.testing.expect(!fp.trunkFits());
    const got = try firstTokenLogits(gpa, model, &long_ids);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(f32, want, got);
    // And again: every chunk is fetched again, the result does not change.
    model.expert_cache.?.trim();
    const again = try firstTokenLogits(gpa, model, &long_ids);
    defer gpa.free(again);
    try std.testing.expectEqualSlices(f32, want, again);
    const st = src.stats();
    try std.testing.expect(st.ranges_fetched > 0);
    try std.testing.expect(st.chunks_unpersisted > 0);
    try std.testing.expectEqual(@as(u64, 0), st.peak_cache_bytes);
    try std.testing.expectEqual(@as(u64, 0), st.cache_bytes);
    try std.testing.expectEqual(@as(u64, 0), (try chunkDirUsage(io, src)).files);
}

const reads_per_worker = 80;

const Worker = struct {
    src: *remote.Source,
    original: [2][]const u8,
    seed: u64,
    failures: u32 = 0,
    reads: u32 = 0,

    fn run(self: *Worker, io: Io) void {
        var prng = std.Random.DefaultPrng.init(self.seed);
        const r = prng.random();
        var buf: [3 * chunk]u8 = undefined;
        var i: usize = 0;
        while (i < reads_per_worker) : (i += 1) {
            const fi = r.uintLessThan(usize, 2);
            const data = self.original[fi];
            const len = r.intRangeAtMost(usize, 1, buf.len);
            const off = r.uintLessThan(usize, data.len - len);
            self.src.files.items[fi].readRange(io, off, buf[0..len]) catch {
                self.failures += 1;
                continue;
            };
            if (!std.mem.eql(u8, buf[0..len], data[off..][0..len])) self.failures += 1;
            self.reads += 1;
        }
    }
};

/// Samples the bytes on disk while the workers run.
const Monitor = struct {
    src: *remote.Source,
    stop: std.atomic.Value(bool) = .init(false),
    max_bytes: u64 = 0,
    scans: u64 = 0,

    fn run(self: *Monitor, io: Io) void {
        // A directory listing is not a snapshot. Scanning under the source's
        // lock makes it exact: no bytes can be reserved (so no new chunk file
        // can appear) and no evicted chunk leaves the count until the scan
        // ends, so every file seen is covered by the bound at once. A chunk
        // renamed from `<index>.part` during the scan shows up under both
        // names, so files are counted once per inode.
        var seen = std.AutoHashMap(Io.File.INode, void).init(std.heap.page_allocator);
        defer seen.deinit();
        const lock = &self.src.lock;
        while (!self.stop.load(.acquire)) {
            io.sleep(Io.Duration.fromNanoseconds(200 * std.time.ns_per_us), .awake) catch {};
            while (lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
            defer lock.store(false, .release);
            seen.clearRetainingCapacity();
            var bytes: u64 = 0;
            for (self.src.files.items) |f| {
                var dir = Io.Dir.cwd().openDir(io, f.chunk_dir, .{ .iterate = true }) catch continue;
                defer dir.close(io);
                var it = dir.iterate();
                while (it.next(io) catch null) |entry| {
                    const st = dir.statFile(io, entry.name, .{}) catch continue;
                    const gop = seen.getOrPut(st.inode) catch continue;
                    if (!gop.found_existing) bytes += st.size;
                }
            }
            self.max_bytes = @max(self.max_bytes, bytes);
            self.scans += 1;
        }
    }
};

test "remote chunk cache: eviction under concurrent fetches" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var env: Env = undefined;
    try env.init(gpa, io);
    defer env.deinit();
    var original: [2][]u8 = undefined;
    for (shard_names, 0..) |n, i| {
        const p = try std.fs.path.join(gpa, &.{ fixture, n });
        defer gpa.free(p);
        original[i] = try Io.Dir.cwd().readFileAlloc(io, p, gpa, .unlimited);
    }
    defer for (original) |o| gpa.free(o);

    const cache = try env.path("cache");
    defer gpa.free(cache);
    const bound = 12 * chunk;
    const src = try env.open(cache, bound);
    defer src.deinit();
    try std.testing.expectEqual(@as(usize, 2), src.files.items.len);
    for (src.files.items, 0..) |f, i| f.setLength(original[i].len);

    var workers: [8]Worker = undefined;
    var threads: [8]std.Thread = undefined;
    for (&workers, 0..) |*w, i| w.* = .{ .src = src, .original = .{ original[0], original[1] }, .seed = 1000 + i };
    var monitor: Monitor = .{ .src = src };
    const mt = try std.Thread.spawn(.{}, Monitor.run, .{ &monitor, io });
    for (&threads, &workers) |*t, *w| t.* = try std.Thread.spawn(.{}, Worker.run, .{ w, io });
    for (threads) |t| t.join();
    monitor.stop.store(true, .release);
    mt.join();

    var reads: u64 = 0;
    for (workers) |w| {
        try std.testing.expectEqual(@as(u32, 0), w.failures);
        reads += w.reads;
    }
    try std.testing.expectEqual(@as(u64, 8 * reads_per_worker), reads);
    const st = src.stats();
    try std.testing.expect(st.chunks_evicted > 0);
    try std.testing.expect(st.peak_cache_bytes <= bound);
    try std.testing.expect(monitor.max_bytes <= bound);
    const usage = try chunkDirUsage(io, src);
    try std.testing.expect(usage.bytes <= bound);
    try std.testing.expectEqual(@as(u64, 0), usage.parts);
    // The index agrees with the disk, and what is on disk is the right bytes.
    try std.testing.expectEqual(st.cache_bytes, usage.bytes);
    for (src.files.items, 0..) |f, fi| {
        var dir = try Io.Dir.cwd().openDir(io, f.chunk_dir, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            const idx = try std.fmt.parseInt(u64, entry.name, 10);
            const bytes = try dir.readFileAlloc(io, entry.name, gpa, .unlimited);
            defer gpa.free(bytes);
            const start: usize = @intCast(idx * chunk);
            try std.testing.expectEqualSlices(u8, original[fi][start..@min(start + chunk, original[fi].len)], bytes);
        }
    }
    std.debug.print("\n[remote concurrent eviction] {d} reads on 8 threads, bound {d} B: {d} fetches, {d} evictions, {d} served from RAM, peak {d} B, max seen on disk {d} B over {d} scans\n", .{ reads, bound, st.ranges_fetched, st.chunks_evicted, st.chunks_unpersisted, st.peak_cache_bytes, monitor.max_bytes, monitor.scans });
}

test "remote chunk cache: truncated chunks and leftover .part files are detected and fetched again" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var env: Env = undefined;
    try env.init(gpa, io);
    defer env.deinit();
    const scratch = try env.path("scratch");
    defer gpa.free(scratch);
    const local = try Model.load(gpa, io, &pool, fixture);
    defer local.deinit();
    const want = try firstTokenLogits(gpa, local, &long_ids);
    defer gpa.free(want);
    const shard_path = try std.fs.path.join(gpa, &.{ fixture, shard_names[0] });
    defer gpa.free(shard_path);
    const shard_len = (try Io.Dir.cwd().statFile(io, shard_path, .{})).size;

    const cache = try env.path("cache");
    defer gpa.free(cache);
    {
        const src = try env.open(cache, null);
        defer src.deinit();
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        const got = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
    }
    // What an interrupted run, or a crash before the data reached the disk,
    // could leave behind: short chunk files (chunk 0, read before the shard
    // length is known, the shard's last chunk and every third one) and a
    // stray `.part` file.
    var truncated: usize = 0;
    {
        const src = try env.open(cache, null);
        defer src.deinit();
        var dir = try Io.Dir.cwd().openDir(io, src.files.items[0].chunk_dir, .{ .iterate = true });
        defer dir.close(io);
        const last = (shard_len - 1) / chunk;
        var victims = std.ArrayList(u64).empty;
        defer victims.deinit(gpa);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            const idx = std.fmt.parseInt(u64, entry.name, 10) catch continue;
            if (idx == 0 or idx == last or idx % 3 == 1) try victims.append(gpa, idx);
        }
        for (victims.items) |idx| {
            var nb: [32]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{d}", .{idx});
            const file = try dir.openFile(io, name, .{ .mode = .read_write });
            defer file.close(io);
            const st = try file.stat(io);
            try file.setLength(io, if (idx == 0) 100 else st.size / 2);
            truncated += 1;
        }
        try dir.writeFile(io, .{ .sub_path = "5.part", .data = "partial" });
    }
    try std.testing.expect(truncated >= 3);
    {
        const src = try env.open(cache, null);
        defer src.deinit();
        // The stray .part file is gone before anything is read.
        try std.testing.expectEqual(@as(u64, 0), (try chunkDirUsage(io, src)).parts);
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        const got = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        const st = src.stats();
        try std.testing.expect(st.chunks_invalid >= truncated);
        try std.testing.expect(st.ranges_fetched >= truncated);
        // Every chunk file of the first shard is whole again.
        var dir = try Io.Dir.cwd().openDir(io, src.files.items[0].chunk_dir, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            const idx = try std.fmt.parseInt(u64, entry.name, 10);
            const stat = try dir.statFile(io, entry.name, .{});
            try std.testing.expectEqual(@min(chunk, shard_len - idx * chunk), stat.size);
        }
    }
    // A later run reads the repaired chunks without fetching anything.
    {
        const src = try env.open(cache, null);
        defer src.deinit();
        const model = try loadRemote(gpa, io, &pool, src, scratch);
        defer model.deinit();
        const got = try firstTokenLogits(gpa, model, &long_ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
        try std.testing.expectEqual(@as(u64, 0), src.stats().chunks_invalid);
        try std.testing.expectEqual(@as(u64, 0), src.stats().ranges_fetched);
    }
}
