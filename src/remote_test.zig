//! Tests for the remote weight source (remote.zig) over a local HTTP server
//! that honours Range requests (tools/range_server.py): loading fetches only
//! the small files, headers and norms; a prefill fetches the trunk plus the
//! routed experts; chunks are cached on disk and a second load fetches nothing.

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
    const chunk: u64 = 4096;
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
