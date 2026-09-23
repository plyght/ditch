//! `ditch push` against a local stand-in for the Hub (tools/hub_server.py):
//! the repository is created, a large file goes up in parts (one of them
//! refused once with a 503), a small binary in one PUT, text files inline in
//! the commit; a second push finds every object on the Hub and uploads none.

const std = @import("std");
const Io = std.Io;
const hf = @import("hf.zig");
const push = @import("push.zig");

const Server = struct {
    child: std.process.Child,
    port: u16,

    fn start(io: Io, out_dir: []const u8) !Server {
        var child = try std.process.spawn(io, .{
            .argv = &.{ "python3", "tools/hub_server.py", out_dir },
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

fn expectSameFile(gpa: std.mem.Allocator, io: Io, a: Io.Dir, b: Io.Dir, path: []const u8) !void {
    const x = try a.readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(x);
    const y = try b.readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(y);
    try std.testing.expectEqualSlices(u8, x, y);
}

test "push: create, multipart and basic LFS uploads, inline files, one commit; a rerun uploads nothing" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..n];

    // An exported model: a 200 KiB shard (4 parts of 64 KiB), a small binary,
    // text files, a nested file and hidden leftovers that stay local.
    try tmp.dir.createDirPath(io, "model/sub");
    try tmp.dir.createDirPath(io, "model/.cache");
    try tmp.dir.createDirPath(io, "hub");
    const shard = try gpa.alloc(u8, 200 * 1024 + 17);
    defer gpa.free(shard);
    for (shard, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 7);
    try tmp.dir.writeFile(io, .{ .sub_path = "model/model.safetensors", .data = shard });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/extra.bin", .data = shard[0..3000] });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/config.json", .data = "{\"model_type\":\"qwen2\"}\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/README.md", .data = "---\ntags:\n- ditch\n---\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/ditch-reproduce.lua", .data = "return {}\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/sub/tokenizer.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/.cache/junk", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model/old.safetensors.part", .data = "x" });
    const model_dir = try std.fs.path.join(gpa, &.{ root, "model" });
    defer gpa.free(model_dir);
    const hub_dir = try std.fs.path.join(gpa, &.{ root, "hub" });
    defer gpa.free(hub_dir);

    var server = try Server.start(io, hub_dir);
    defer server.stop(io);
    var url_buf: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/hub", .{server.port});

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    // No proxy settings; the token and the endpoint only.
    var environ = std.process.Environ.Map.init(gpa);
    defer environ.deinit();
    try environ.put("HF_ENDPOINT", endpoint);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();

    try environ.put("HF_TOKEN", "hf_test");
    var http = try hf.Http.init(gpa, io, arena.allocator(), &environ);
    defer http.deinit();

    try push.push(gpa, &http, model_dir, "someone/model", .{ .private = true }, &sink.writer);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "Created someone/model (private)") != null);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "Model pushed to ") != null);
    var got = try tmp.dir.openDir(io, "hub/someone/model", .{});
    defer got.close(io);
    var src = try tmp.dir.openDir(io, "model", .{});
    defer src.close(io);
    for ([_][]const u8{ "model.safetensors", "extra.bin", "config.json", "README.md", "ditch-reproduce.lua", "sub/tokenizer.json" }) |p|
        try expectSameFile(gpa, io, src, got, p);
    try std.testing.expectError(error.FileNotFound, got.access(io, ".cache/junk", .{}));
    try std.testing.expectError(error.FileNotFound, got.access(io, "old.safetensors.part", .{}));

    const log1 = try tmp.dir.readFileAlloc(io, "hub/requests.log", gpa, .unlimited);
    defer gpa.free(log1);
    // Four parts, the first retried after its 503, then the completion; one basic PUT.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, log1, " 503\n"));
    try std.testing.expectEqual(@as(usize, 5 + 1), std.mem.count(u8, log1, "PUT /storage/"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, log1, "POST /hub/complete/"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, log1, "/commit/main 200"));

    // A second push: the repository exists and the Hub has every object.
    sink.clearRetainingCapacity();
    try push.push(gpa, &http, model_dir, "someone/model", .{}, &sink.writer);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "exists") != null);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "2 large file(s) already on the Hub") != null);
    const log2 = try tmp.dir.readFileAlloc(io, "hub/requests.log", gpa, .unlimited);
    defer gpa.free(log2);
    try std.testing.expectEqual(@as(usize, 6), std.mem.count(u8, log2, "PUT /storage/"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, log2, "/commit/main 200"));
}
