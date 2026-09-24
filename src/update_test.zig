//! `ditch update` against a local stand-in for GitHub (python3 -m
//! http.server over a directory): a release JSON, a tar.gz for this build
//! and its SHA256SUMS. The update replaces a stand-in executable in place;
//! a checksum mismatch leaves it untouched.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const config = @import("config.zig");
const update = @import("update.zig");

const Server = struct {
    child: std.process.Child,
    port: u16,

    fn start(io: Io, dir: []const u8) !Server {
        var child = try std.process.spawn(io, .{
            .argv = &.{ "python3", "-u", "-m", "http.server", "0", "--bind", "127.0.0.1", "--directory", dir },
            .stdin = .ignore,
            .stdout = .pipe,
            .stderr = .ignore,
        });
        errdefer child.kill(io);
        var buf: [256]u8 = undefined;
        var fr = child.stdout.?.readerStreaming(io, &buf);
        const line = try fr.interface.takeDelimiterExclusive('\n');
        const at = std.mem.indexOf(u8, line, " port ") orelse return error.ServerDidNotStart;
        var end = at + " port ".len;
        while (end < line.len and std.ascii.isDigit(line[end])) end += 1;
        const port = try std.fmt.parseInt(u16, line[at + " port ".len .. end], 10);
        return .{ .child = child, .port = port };
    }

    fn stop(self: *Server, io: Io) void {
        self.child.kill(io);
    }
};

test "update: installs a release over a stand-in executable, refuses a bad checksum" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const lbl = update.label(update.this_build) orelse return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    // The release archive: <top>/ditch, a script standing in for the binary.
    const tag = "v9.9.9";
    const asset = try update.assetName(a, tag, lbl, update.this_build.v3);
    const top = asset[0 .. asset.len - ".tar.gz".len];
    const script = "#!/bin/sh\necho ditch 9.9.9 test\n";
    try tmp.dir.createDirPath(io, try std.fs.path.join(a, &.{ "stage", top }));
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ "stage", top, "ditch" }), .data = script });
    try tmp.dir.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ "stage", top, "README.md" }), .data = "readme\n" });
    try tmp.dir.createDirPath(io, "srv/api/releases");
    try tmp.dir.createDirPath(io, "srv/files");
    const archive_path = try std.fs.path.join(a, &.{ root, "srv", "files", asset });
    const tar = try std.process.run(gpa, io, .{ .argv = &.{ "tar", "-C", try std.fs.path.join(a, &.{ root, "stage" }), "-czf", archive_path, top } });
    gpa.free(tar.stdout);
    gpa.free(tar.stderr);
    const archive = try Io.Dir.cwd().readFileAlloc(io, archive_path, a, .unlimited);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(archive, &digest, .{});
    const sums_good = try std.fmt.allocPrint(a, "{s}  {s}\n", .{ std.fmt.bytesToHex(digest, .lower), asset });
    try tmp.dir.writeFile(io, .{ .sub_path = "srv/files/SHA256SUMS", .data = sums_good });

    var server = try Server.start(io, try std.fs.path.join(a, &.{ root, "srv" }));
    defer server.stop(io);
    const base = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}", .{server.port});
    const release_json = try std.fmt.allocPrint(a,
        \\{{"tag_name":"{s}","html_url":"https://example.invalid/releases/{s}","draft":false,
        \\"assets":[{{"name":"{s}","size":{d},"browser_download_url":"{s}/files/{s}"}},
        \\{{"name":"SHA256SUMS","size":{d},"browser_download_url":"{s}/files/SHA256SUMS"}}]}}
    , .{ tag, tag, asset, archive.len, base, asset, sums_good.len, base });
    try tmp.dir.writeFile(io, .{ .sub_path = "srv/api/releases/latest", .data = release_json });

    // The executable to replace.
    try tmp.dir.createDirPath(io, "bin");
    try tmp.dir.writeFile(io, .{ .sub_path = "bin/ditch", .data = "old binary\n" });
    try tmp.dir.setFilePermissions(io, "bin/ditch", .fromMode(0o755), .{});
    const exe = try std.fs.path.join(a, &.{ root, "bin", "ditch" });

    var environ = std.process.Environ.Map.init(gpa);
    defer environ.deinit();
    var settings: config.Settings = .{};
    settings.force = true;
    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var result: Io.Writer.Allocating = .init(gpa);
    defer result.deinit();
    var errs: Io.Writer.Allocating = .init(gpa);
    defer errs.deinit();
    var in: Io.Reader = .fixed("");
    const ctx: update.Ctx = .{ .gpa = gpa, .arena = a, .io = io, .env = &environ, .settings = &settings, .out = &out.writer, .result = &result.writer, .in = &in, .interactive = false, .api = try std.fmt.allocPrint(a, "{s}/api", .{base}), .exe_path = exe, .current = "0.1.0", .errors = &errs.writer };

    // --check only reports.
    settings.update_check = true;
    settings.json = true;
    try std.testing.expectEqual(@as(u8, 0), try update.run(ctx));
    try std.testing.expect(std.mem.indexOf(u8, result.written(), "\"update_available\":true") != null);
    try std.testing.expectEqualStrings("old binary\n", try tmp.dir.readFileAlloc(io, "bin/ditch", a, .unlimited));
    settings.update_check = false;
    settings.json = false;
    result.clearRetainingCapacity();

    // A checksum that does not match: nothing is replaced, nothing is left behind.
    try tmp.dir.writeFile(io, .{ .sub_path = "srv/files/SHA256SUMS", .data = try std.fmt.allocPrint(a, "{s}  {s}\n", .{ "0" ** 64, asset }) });
    try std.testing.expectEqual(@as(u8, 1), try update.run(ctx));
    try std.testing.expect(std.mem.indexOf(u8, errs.written(), "error: checksum mismatch") != null);
    try std.testing.expectEqualStrings("old binary\n", try tmp.dir.readFileAlloc(io, "bin/ditch", a, .unlimited));
    // No entry for the archive: refused before downloading it.
    try tmp.dir.writeFile(io, .{ .sub_path = "srv/files/SHA256SUMS", .data = try std.fmt.allocPrint(a, "{s}  other.tar.gz\n", .{"0" ** 64}) });
    try std.testing.expectEqual(@as(u8, 1), try update.run(ctx));
    try std.testing.expect(std.mem.indexOf(u8, errs.written(), "has no entry for") != null);
    try std.testing.expectEqualStrings("old binary\n", try tmp.dir.readFileAlloc(io, "bin/ditch", a, .unlimited));

    // The real thing.
    try tmp.dir.writeFile(io, .{ .sub_path = "srv/files/SHA256SUMS", .data = sums_good });
    try std.testing.expectEqual(@as(u8, 0), try update.run(ctx));
    try std.testing.expectEqualStrings(script, try tmp.dir.readFileAlloc(io, "bin/ditch", a, .unlimited));
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "ditch 0.1.0 → 9.9.9") != null);
    // It ran: the executable bit was kept.
    try std.testing.expect(std.mem.indexOf(u8, result.written(), "ditch 9.9.9 test") != null);
    var bin = try tmp.dir.openDir(io, "bin", .{ .iterate = true });
    defer bin.close(io);
    var it = bin.iterate();
    var entries: usize = 0;
    while (try it.next(io)) |e| {
        entries += 1;
        try std.testing.expectEqualStrings("ditch", e.name);
    }
    try std.testing.expectEqual(@as(usize, 1), entries);

    // Up to date: nothing to do.
    result.clearRetainingCapacity();
    var current = ctx;
    current.current = "9.9.9";
    try std.testing.expectEqual(@as(u8, 0), try update.run(current));
    try std.testing.expect(std.mem.indexOf(u8, result.written(), "up to date") != null);
}
