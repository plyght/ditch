//! Uploads an exported model directory to a Hugging Face Hub model repository,
//! in pure Zig over the Hub's HTTP API (the protocol huggingface_hub speaks):
//!
//! 1. `POST /api/repos/create` (409 = the repository exists already);
//! 2. `POST /api/models/<repo>/preupload/main` lets the Hub decide which files
//!    are LFS objects and which are stored inline in the commit;
//! 3. `POST /<repo>.git/info/lfs/objects/batch` with the sha256 and size of
//!    every LFS file; objects the Hub already has come back without an upload
//!    action and are skipped, the others are PUT in one request ("basic") or
//!    in parts ("multipart": `chunk_size` and one URL per part in the action's
//!    header, then a POST of the parts' etags to the completion URL);
//! 4. one commit, `POST /api/models/<repo>/commit/main`, whose
//!    application/x-ndjson body lists the LFS pointers and the inline files.
//!
//! The Hub stores new repositories on Xet, but still accepts LFS uploads from
//! clients that do not speak Xet and migrates those objects in the background.
//!
//! Files are streamed from disk for hashing and upload; only inline files
//! (small text the Hub chose not to put in LFS) are held in memory. Network
//! errors, 408/429/5xx answers and dropped connections are retried with a
//! growing pause (honouring Retry-After); a multipart upload retries only the
//! part that failed, and a rerun skips every object the Hub already stored.

const std = @import("std");
const Io = std.Io;
const hf = @import("hf.zig");
const budget_mod = @import("budget.zig");
const engine_mod = @import("engine.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;

pub const default_endpoint = "https://huggingface.co";

pub const Options = struct {
    /// Create the repository as private (an existing one keeps its visibility).
    private: bool = false,
    /// Commit title.
    summary: []const u8 = "Upload model exported with ditch",
    /// The Hub, e.g. a mirror; default: $HF_ENDPOINT, else huggingface.co.
    endpoint: ?[]const u8 = null,
    /// Attempts per request before a transient failure ends the push.
    attempts: usize = 8,
    /// Where the progress lines go (null = the `out` writer).
    progress: ?*Io.Writer = null,
};

/// The Hub the push goes to: `$HF_ENDPOINT` (huggingface_hub's variable) or huggingface.co.
pub fn endpointFrom(environ: *std.process.Environ.Map) []const u8 {
    const e = environ.get("HF_ENDPOINT") orelse return default_endpoint;
    const t = std.mem.trimEnd(u8, e, "/");
    return if (t.len == 0) default_endpoint else t;
}

/// Checks `owner/name`: one slash, and letters, digits, '-', '_' and '.' only.
pub fn validRepoId(id: []const u8) bool {
    const slash = std.mem.indexOfScalar(u8, id, '/') orelse return false;
    const owner = id[0..slash];
    const name = id[slash + 1 ..];
    if (owner.len == 0 or name.len == 0 or name.len > 96) return false;
    for (id, 0..) |c, i| {
        if (i == slash) continue;
        if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.')) return false;
    }
    return !std.mem.startsWith(u8, name, ".") and !std.mem.endsWith(u8, name, ".git");
}

const Mode = enum { lfs, regular };

const File = struct {
    /// Path in the repository ('/'-separated).
    path: []const u8,
    /// Path relative to the pushed directory.
    fs_path: []const u8,
    size: u64,
    mode: Mode = .lfs,
    ignore: bool = false,
    /// Hex sha256 (LFS files only).
    oid: [64]u8 = undefined,
};

/// Uploads every file under `dir_path` (hidden files and `.part` downloads
/// excepted) to the model repository `repo_id`, creating it when missing, in
/// one commit on main. Prints the repository URL at the end.
pub fn push(gpa: Allocator, http: *hf.Http, dir_path: []const u8, repo_id: []const u8, opts: Options, out: *Io.Writer) !void {
    const io = http.io;
    if (!validRepoId(repo_id)) {
        std.log.err("invalid Hub repository {s}: expected owner/name (letters, digits, '-', '_', '.')", .{repo_id});
        return error.InvalidRepoId;
    }
    if (http.token == null) {
        std.log.err("pushing to the Hub needs a token with write access: set HF_TOKEN, run `hf auth login`, or pass --token-file <path>", .{});
        return error.NoToken;
    }
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var c = Client{
        .http = http,
        .arena = arena,
        .endpoint = opts.endpoint orelse endpointFrom(http.environ),
        .attempts = @max(opts.attempts, 1),
    };

    var dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch |err| {
        std.log.err("cannot open {s}: {s}", .{ dir_path, @errorName(err) });
        return err;
    };
    defer dir.close(io);
    if (dir.access(io, model_mod.export_incomplete_marker, .{})) |_| {
        std.log.err("{s} holds an unfinished export ({s}); export again before pushing", .{ dir_path, model_mod.export_incomplete_marker });
        return error.UnfinishedExport;
    } else |_| {}
    const files = try listFiles(arena, io, dir);
    if (files.len == 0) {
        std.log.err("{s} has no files to push", .{dir_path});
        return error.NothingToPush;
    }
    var total: u64 = 0;
    for (files) |f| total += f.size;
    try out.print("Pushing {s} to {s}/{s} ({d} file(s), {f})...\n", .{ dir_path, c.endpoint, repo_id, files.len, budget_mod.fmtBytes(total) });
    try out.flush();

    // 1. The repository.
    try createRepo(&c, repo_id, opts.private, out);

    // 2. LFS or inline, per the Hub.
    try preupload(&c, dir, repo_id, files);

    // 3. Hash and upload the LFS objects.
    var lfs_bytes: u64 = 0;
    for (files) |f| if (!f.ignore and f.mode == .lfs) {
        lfs_bytes += f.size;
    };
    const prog_out = opts.progress orelse out;
    if (lfs_bytes > 0) {
        try out.print("* Hashing {f} of large files...\n", .{budget_mod.fmtBytes(lfs_bytes)});
        try out.flush();
        var prog = engine_mod.Progress.init(prog_out, io, "hashed", @intCast(lfs_bytes));
        prog.bytes = true;
        var hashed: u64 = 0;
        for (files) |*f| if (!f.ignore and f.mode == .lfs) {
            f.oid = try hashFile(gpa, io, dir, f.fs_path, &prog, &hashed);
        };
        prog.finish();
        try uploadLfs(&c, dir, repo_id, files, prog_out, out);
    }

    // 4. The commit.
    try out.writeAll("* Committing...\n");
    try out.flush();
    const body = try commitBody(arena, io, dir, files, opts.summary);
    const url = try std.fmt.allocPrint(arena, "{s}/api/models/{s}/commit/main", .{ c.endpoint, repo_id });
    const reply = try c.send(.{ .what = "commit", .method = .POST, .url = url, .content_type = "application/x-ndjson", .body = .{ .bytes = body } });
    try c.expectOk(reply, "commit");
    const repo_url = try std.fmt.allocPrint(arena, "{s}/{s}", .{ c.endpoint, repo_id });
    try out.print("Model pushed to {s}\n", .{repo_url});
    try out.flush();
}

/// Every regular file under `dir`, sorted by path; hidden files and
/// directories and `.part` leftovers are skipped.
fn listFiles(arena: Allocator, io: Io, dir: Io.Dir) ![]File {
    var list = std.ArrayList(File).empty;
    var walker = try dir.walk(arena);
    defer walker.deinit();
    while (try walker.next(io)) |entry| {
        if (entry.basename.len > 0 and entry.basename[0] == '.') {
            if (entry.kind == .directory) walker.leave(io);
            continue;
        }
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.basename, ".part")) continue;
        const fs_path = try arena.dupe(u8, entry.path);
        const path = try arena.dupe(u8, entry.path);
        std.mem.replaceScalar(u8, path, std.fs.path.sep, '/');
        const st = try dir.statFile(io, fs_path, .{});
        try list.append(arena, .{ .path = path, .fs_path = fs_path, .size = st.size });
    }
    std.mem.sort(File, list.items, {}, struct {
        fn lt(_: void, a: File, b: File) bool {
            return std.mem.lessThan(u8, a.path, b.path);
        }
    }.lt);
    return list.items;
}

fn createRepo(c: *Client, repo_id: []const u8, private: bool, out: *Io.Writer) !void {
    const slash = std.mem.indexOfScalar(u8, repo_id, '/').?;
    const body = try std.json.Stringify.valueAlloc(c.arena, .{
        .type = "model",
        .name = repo_id[slash + 1 ..],
        .organization = repo_id[0..slash],
        .private = private,
    }, .{});
    const url = try std.fmt.allocPrint(c.arena, "{s}/api/repos/create", .{c.endpoint});
    const reply = try c.send(.{ .what = "create repository", .method = .POST, .url = url, .content_type = "application/json", .body = .{ .bytes = body } });
    if (reply.status == .conflict) {
        try out.print("* {s} exists; adding a commit to it\n", .{repo_id});
        return;
    }
    try c.expectOk(reply, "create repository");
    try out.print("* Created {s}{s}\n", .{ repo_id, if (private) " (private)" else "" });
}

fn preupload(c: *Client, dir: Io.Dir, repo_id: []const u8, files: []File) !void {
    const io = c.http.io;
    const url = try std.fmt.allocPrint(c.arena, "{s}/api/models/{s}/preupload/main", .{ c.endpoint, repo_id });
    const Entry = struct { path: []const u8, sample: []const u8, size: u64 };
    var start: usize = 0;
    while (start < files.len) : (start += 256) {
        const chunk = files[start..@min(start + 256, files.len)];
        const entries = try c.arena.alloc(Entry, chunk.len);
        for (chunk, entries) |f, *e| {
            // The Hub looks at the first 512 bytes to tell text from binary.
            var buf: [512]u8 = undefined;
            const file = try dir.openFile(io, f.fs_path, .{});
            defer file.close(io);
            const n = try file.readPositionalAll(io, &buf, 0);
            e.* = .{ .path = f.path, .sample = try base64(c.arena, buf[0..n]), .size = f.size };
        }
        const body = try std.json.Stringify.valueAlloc(c.arena, .{ .files = entries }, .{});
        const reply = try c.send(.{ .what = "preupload", .method = .POST, .url = url, .content_type = "application/json", .body = .{ .bytes = body } });
        try c.expectOk(reply, "preupload");
        try applyPreupload(c.arena, reply.body, chunk);
    }
    // Empty files cannot be LFS objects.
    for (files) |*f| if (f.size == 0) {
        f.mode = .regular;
    };
}

/// Sets the upload mode and ignore flag of `files` from a preupload answer
/// (`{"files":[{"path","uploadMode":"lfs"|"regular","shouldIgnore"}]}`).
fn applyPreupload(arena: Allocator, body: []const u8, files: []File) !void {
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return error.InvalidResponse;
    if (parsed != .object) return error.InvalidResponse;
    const list = parsed.object.get("files") orelse return error.InvalidResponse;
    if (list != .array) return error.InvalidResponse;
    for (list.array.items) |item| {
        if (item != .object) return error.InvalidResponse;
        const path = jsonString(item, "path") orelse return error.InvalidResponse;
        const mode = jsonString(item, "uploadMode") orelse return error.InvalidResponse;
        for (files) |*f| if (std.mem.eql(u8, f.path, path)) {
            f.mode = if (std.mem.eql(u8, mode, "regular")) .regular else .lfs;
            if (item.object.get("shouldIgnore")) |v| f.ignore = v == .bool and v.bool;
        };
    }
}

fn hashFile(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, prog: *engine_mod.Progress, hashed: *u64) ![64]u8 {
    const file = try dir.openFile(io, sub_path, .{});
    defer file.close(io);
    const buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var off: u64 = 0;
    while (true) {
        if (budget_mod.interrupted()) return error.Interrupted;
        const n = try file.readPositionalAll(io, buf, off);
        if (n == 0) break;
        h.update(buf[0..n]);
        off += n;
        hashed.* += n;
        prog.update(@intCast(hashed.*));
        if (n < buf.len) break;
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

const lfs_content_type = "application/vnd.git-lfs+json";

fn uploadLfs(c: *Client, dir: Io.Dir, repo_id: []const u8, files: []File, prog_out: *Io.Writer, out: *Io.Writer) !void {
    const io = c.http.io;
    const url = try std.fmt.allocPrint(c.arena, "{s}/{s}.git/info/lfs/objects/batch", .{ c.endpoint, repo_id });
    // One object per distinct oid.
    var objects = std.ArrayList(*File).empty;
    for (files) |*f| {
        if (f.ignore or f.mode != .lfs) continue;
        const dup = for (objects.items) |o| {
            if (std.mem.eql(u8, &o.oid, &f.oid)) break true;
        } else false;
        if (!dup) try objects.append(c.arena, f);
    }
    var todo = std.ArrayList(struct { file: *File, action: std.json.Value }).empty;
    var todo_bytes: u64 = 0;
    var start: usize = 0;
    while (start < objects.items.len) : (start += 256) {
        const chunk = objects.items[start..@min(start + 256, objects.items.len)];
        const body = try batchRequestBody(c.arena, chunk);
        const reply = try c.send(.{ .what = "LFS batch", .method = .POST, .url = url, .content_type = lfs_content_type, .accept = lfs_content_type, .body = .{ .bytes = body } });
        try c.expectOk(reply, "LFS batch");
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, c.arena, reply.body, .{}) catch return error.InvalidResponse;
        const objs = if (parsed == .object) parsed.object.get("objects") else null;
        if (objs == null or objs.? != .array) return error.InvalidResponse;
        for (objs.?.array.items) |o| {
            if (o != .object) return error.InvalidResponse;
            const oid = jsonString(o, "oid") orelse return error.InvalidResponse;
            const f = for (chunk) |f| {
                if (std.mem.eql(u8, &f.oid, oid)) break f;
            } else return error.InvalidResponse;
            if (o.object.get("error")) |e| {
                const msg = if (e == .object) jsonString(e, "message") orelse "" else "";
                std.log.err("the Hub refused {s}: {s}", .{ f.path, msg });
                return error.LfsRefused;
            }
            const actions = o.object.get("actions") orelse continue; // already on the Hub
            if (actions != .object) return error.InvalidResponse;
            try todo.append(c.arena, .{ .file = f, .action = actions });
            if (actions.object.get("upload") != null) todo_bytes += f.size;
        }
    }
    const skipped = objects.items.len - todo.items.len;
    if (skipped > 0) try out.print("* {d} large file(s) already on the Hub\n", .{skipped});
    if (todo.items.len == 0) return;
    try out.print("* Uploading {d} large file(s), {f}...\n", .{ todo.items.len, budget_mod.fmtBytes(todo_bytes) });
    try out.flush();
    var prog = engine_mod.Progress.init(prog_out, io, "uploaded", @intCast(@max(todo_bytes, 1)));
    prog.bytes = true;
    c.progress = &prog;
    defer c.progress = null;
    for (todo.items) |t| try uploadObject(c, dir, t.file, t.action);
    prog.finish();
}

fn batchRequestBody(arena: Allocator, objects: []const *File) ![]u8 {
    const Obj = struct { oid: []const u8, size: u64 };
    const list = try arena.alloc(Obj, objects.len);
    for (objects, list) |f, *o| o.* = .{ .oid = &f.oid, .size = f.size };
    return std.json.Stringify.valueAlloc(arena, .{
        .operation = "upload",
        .transfers = [_][]const u8{ "basic", "multipart" },
        .objects = list,
        .hash_algo = "sha256",
        .ref = .{ .name = "main" },
    }, .{});
}

fn uploadObject(c: *Client, dir: Io.Dir, f: *const File, actions: std.json.Value) !void {
    const io = c.http.io;
    const file = try dir.openFile(io, f.fs_path, .{});
    defer file.close(io);
    if (actions.object.get("upload")) |up| {
        if (up != .object) return error.InvalidResponse;
        const href = jsonString(up, "href") orelse return error.InvalidResponse;
        const header = up.object.get("header");
        const chunk_size: ?u64 = if (header) |h| if (h == .object) if (h.object.get("chunk_size")) |cs| switch (cs) {
            .string => |s| std.fmt.parseInt(u64, s, 10) catch return error.InvalidResponse,
            .integer => |i| @intCast(@max(i, 1)),
            else => return error.InvalidResponse,
        } else null else null else null;
        if (chunk_size) |cs| {
            const parts = try partUrls(c.arena, header.?, f.size, cs);
            const etags = try c.arena.alloc([]const u8, parts.len);
            for (parts, 0..) |part_url, i| {
                const off = @as(u64, i) * cs;
                const reply = try c.send(.{ .what = f.path, .method = .PUT, .url = part_url, .body = .{ .file = .{ .file = file, .offset = off, .len = @min(cs, f.size - off) } } });
                try c.expectOk(reply, f.path);
                etags[i] = reply.etag orelse {
                    std.log.err("{s}: the storage answered part {d} without an etag", .{ f.path, i + 1 });
                    return error.InvalidResponse;
                };
            }
            const body = try completionBody(c.arena, &f.oid, etags);
            const reply = try c.send(.{ .what = f.path, .method = .POST, .url = href, .content_type = lfs_content_type, .accept = lfs_content_type, .body = .{ .bytes = body } });
            try c.expectOk(reply, f.path);
        } else {
            const reply = try c.send(.{ .what = f.path, .method = .PUT, .url = href, .body = .{ .file = .{ .file = file, .offset = 0, .len = f.size } } });
            try c.expectOk(reply, f.path);
        }
    }
    if (actions.object.get("verify")) |v| {
        const href = (if (v == .object) jsonString(v, "href") else null) orelse return error.InvalidResponse;
        const body = try std.json.Stringify.valueAlloc(c.arena, .{ .oid = @as([]const u8, &f.oid), .size = f.size }, .{});
        const reply = try c.send(.{ .what = f.path, .method = .POST, .url = href, .content_type = lfs_content_type, .accept = lfs_content_type, .body = .{ .bytes = body } });
        try c.expectOk(reply, f.path);
    }
}

/// The part URLs of a multipart action's header (keys "1", "2", ... or
/// zero-padded), in part order; their number must cover `size`.
fn partUrls(arena: Allocator, header: std.json.Value, size: u64, chunk_size: u64) ![]const []const u8 {
    const Part = struct { n: u64, url: []const u8 };
    var parts = std.ArrayList(Part).empty;
    var it = header.object.iterator();
    while (it.next()) |kv| {
        const n = std.fmt.parseInt(u64, kv.key_ptr.*, 10) catch continue;
        if (kv.value_ptr.* != .string) return error.InvalidResponse;
        try parts.append(arena, .{ .n = n, .url = kv.value_ptr.string });
    }
    std.mem.sort(Part, parts.items, {}, struct {
        fn lt(_: void, a: Part, b: Part) bool {
            return a.n < b.n;
        }
    }.lt);
    if (chunk_size == 0 or parts.items.len != std.math.divCeil(u64, size, chunk_size) catch unreachable) return error.InvalidResponse;
    const urls = try arena.alloc([]const u8, parts.items.len);
    for (parts.items, urls) |p, *u| u.* = p.url;
    return urls;
}

fn completionBody(arena: Allocator, oid: []const u8, etags: []const []const u8) ![]u8 {
    const Part = struct { partNumber: usize, etag: []const u8 };
    const parts = try arena.alloc(Part, etags.len);
    for (etags, parts, 1..) |e, *p, n| p.* = .{ .partNumber = n, .etag = e };
    return std.json.Stringify.valueAlloc(arena, .{ .oid = oid, .parts = parts }, .{});
}

/// The ndjson commit: a header line, then one line per file (an LFS pointer,
/// or the base64 content of an inline file).
fn commitBody(arena: Allocator, io: Io, dir: Io.Dir, files: []const File, summary: []const u8) ![]u8 {
    var w: Io.Writer.Allocating = .init(arena);
    const o = &w.writer;
    try std.json.Stringify.value(.{ .key = "header", .value = .{ .summary = summary, .description = "" } }, .{}, o);
    try o.writeByte('\n');
    for (files) |f| {
        if (f.ignore) continue;
        switch (f.mode) {
            .lfs => try std.json.Stringify.value(.{ .key = "lfsFile", .value = .{ .path = f.path, .algo = "sha256", .oid = @as([]const u8, &f.oid), .size = f.size } }, .{}, o),
            .regular => {
                // The Hub only asks for small text files inline.
                if (f.size > 64 << 20) {
                    std.log.err("the Hub asked for {s} ({f}) inline; refusing to load it into memory", .{ f.path, budget_mod.fmtBytes(f.size) });
                    return error.InvalidResponse;
                }
                const content = try dir.readFileAlloc(io, f.fs_path, arena, .limited(64 << 20));
                try std.json.Stringify.value(.{ .key = "file", .value = .{ .path = f.path, .content = try base64(arena, content), .encoding = "base64" } }, .{}, o);
            },
        }
        try o.writeByte('\n');
    }
    return w.written();
}

fn base64(arena: Allocator, bytes: []const u8) ![]u8 {
    const enc = std.base64.standard.Encoder;
    const buf = try arena.alloc(u8, enc.calcSize(bytes.len));
    return @constCast(enc.encode(buf, bytes));
}

fn jsonString(v: std.json.Value, key: []const u8) ?[]const u8 {
    const x = v.object.get(key) orelse return null;
    return if (x == .string) x.string else null;
}

// ---------------------------------------------------------------------------
// HTTP with retries
// ---------------------------------------------------------------------------

const Body = union(enum) {
    none,
    bytes: []const u8,
    /// `len` bytes of `file` from `offset`, streamed.
    file: struct { file: Io.File, offset: u64, len: u64 },
};

const Request = struct {
    /// Names the request in messages (URLs of the storage are signed and never printed).
    what: []const u8,
    method: std.http.Method,
    url: []const u8,
    content_type: ?[]const u8 = null,
    accept: ?[]const u8 = null,
    body: Body = .none,
};

const Reply = struct {
    status: std.http.Status,
    body: []const u8,
    etag: ?[]const u8 = null,
    retry_after: ?u64 = null,
};

const Client = struct {
    http: *hf.Http,
    arena: Allocator,
    endpoint: []const u8,
    attempts: usize,
    /// Byte progress of the uploads under way.
    progress: ?*engine_mod.Progress = null,
    uploaded: u64 = 0,

    fn retryableStatus(s: std.http.Status) bool {
        return switch (s) {
            .request_timeout, .too_many_requests, .internal_server_error, .bad_gateway, .service_unavailable, .gateway_timeout => true,
            else => false,
        };
    }

    /// Seconds to wait before attempt `attempt + 1`: 2, 4, 8, ... up to a minute.
    fn backoff(attempt: usize) u64 {
        return @min(@as(u64, 1) << @intCast(@min(attempt, 6)), 60);
    }

    fn pause(self: *Client, seconds: u64) void {
        if (seconds == 0) return;
        self.http.io.sleep(Io.Duration.fromSeconds(@intCast(seconds)), .awake) catch {};
    }

    /// Sends `req`, retrying transport failures and retryable statuses.
    /// The reply may still be an error status; see `expectOk`.
    fn send(self: *Client, req: Request) !Reply {
        var attempt: usize = 1;
        while (true) : (attempt += 1) {
            const mark = self.uploaded;
            const reply = self.sendOnce(req) catch |err| {
                self.rewind(mark);
                if (budget_mod.interrupted()) return error.Interrupted;
                if (err == error.OutOfMemory or err == error.Interrupted or attempt >= self.attempts) return err;
                const wait = backoff(attempt);
                std.log.warn("{s}: {s}; retrying in {d} s ({d}/{d})", .{ req.what, @errorName(err), wait, attempt + 1, self.attempts });
                self.pause(wait);
                continue;
            };
            if (retryableStatus(reply.status) and attempt < self.attempts) {
                self.rewind(mark);
                const wait = @min(reply.retry_after orelse backoff(attempt), 300);
                std.log.warn("{s}: HTTP {d}; retrying in {d} s ({d}/{d})", .{ req.what, @intFromEnum(reply.status), wait, attempt + 1, self.attempts });
                self.pause(wait);
                if (budget_mod.interrupted()) return error.Interrupted;
                continue;
            }
            return reply;
        }
    }

    fn rewind(self: *Client, mark: u64) void {
        self.uploaded = mark;
        if (self.progress) |p| p.update(@intCast(mark));
    }

    /// Fails on a non-2xx reply, logging the Hub's message.
    fn expectOk(self: *Client, reply: Reply, what: []const u8) !void {
        _ = self;
        if (reply.status.class() == .success) return;
        const msg = std.mem.trim(u8, reply.body[0..@min(reply.body.len, 400)], " \r\n");
        switch (reply.status) {
            .unauthorized, .forbidden => {
                std.log.err("{s}: HTTP {d} {s}; the token needs write access to the repository", .{ what, @intFromEnum(reply.status), msg });
                return error.Forbidden;
            },
            .not_found => {
                std.log.err("{s}: HTTP 404 {s}", .{ what, msg });
                return error.NotFound;
            },
            else => {
                std.log.err("{s}: HTTP {d} {s}", .{ what, @intFromEnum(reply.status), msg });
                return error.HttpError;
            },
        }
    }

    /// The authorization header goes to the Hub only, never to the storage
    /// URLs it hands out.
    fn toHub(self: *Client, url: []const u8) bool {
        if (!std.mem.startsWith(u8, url, self.endpoint)) return false;
        return url.len == self.endpoint.len or url[self.endpoint.len] == '/' or url[self.endpoint.len] == '?';
    }

    fn sendOnce(self: *Client, req: Request) !Reply {
        const gpa = self.http.gpa;
        const io = self.http.io;
        const client = &self.http.client;
        const uri = try std.Uri.parse(req.url);
        var extra: [1]std.http.Header = undefined;
        var n_extra: usize = 0;
        if (req.accept) |a| {
            extra[0] = .{ .name = "accept", .value = a };
            n_extra = 1;
        }
        // Redirects are not followed, so the header cannot leak to another host.
        var auth_buf: [512]u8 = undefined;
        var auth: std.http.Client.Request.Headers.Value = .omit;
        if (self.toHub(req.url)) if (self.http.token) |t| {
            auth = .{ .override = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{t}) };
        };
        var r = try client.request(req.method, uri, .{
            .redirect_behavior = .unhandled,
            .headers = .{
                .authorization = auth,
                .content_type = if (req.content_type) |ct| .{ .override = ct } else .default,
                .user_agent = .{ .override = "ditch/" ++ @import("config.zig").version },
            },
            .extra_headers = extra[0..n_extra],
        });
        defer r.deinit();
        switch (req.body) {
            .none => {
                if (req.method.requestHasBody()) {
                    r.transfer_encoding = .{ .content_length = 0 };
                    var bw = try r.sendBodyUnflushed(&.{});
                    try bw.end();
                    try r.connection.?.flush();
                } else try r.sendBodiless();
            },
            .bytes => |b| {
                r.transfer_encoding = .{ .content_length = b.len };
                var bw = try r.sendBodyUnflushed(&.{});
                try bw.writer.writeAll(b);
                try bw.end();
                try r.connection.?.flush();
            },
            .file => |f| {
                r.transfer_encoding = .{ .content_length = f.len };
                const buf = try gpa.alloc(u8, 1 << 20);
                defer gpa.free(buf);
                var bw = try r.sendBodyUnflushed(&.{});
                var sent: u64 = 0;
                while (sent < f.len) {
                    if (budget_mod.interrupted()) return error.Interrupted;
                    const want: usize = @intCast(@min(buf.len, f.len - sent));
                    const n = try f.file.readPositionalAll(io, buf[0..want], f.offset + sent);
                    if (n != want) return error.FileChanged;
                    try bw.writer.writeAll(buf[0..n]);
                    try bw.flush();
                    sent += n;
                    self.uploaded += n;
                    if (self.progress) |p| p.update(@intCast(self.uploaded));
                }
                try bw.end();
                try r.connection.?.flush();
            },
        }
        var response = try r.receiveHead(&.{});
        var reply: Reply = .{ .status = response.head.status, .body = "" };
        var hit = response.head.iterateHeaders();
        while (hit.next()) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, "etag")) {
                reply.etag = try self.arena.dupe(u8, h.value);
            } else if (std.ascii.eqlIgnoreCase(h.name, "retry-after")) {
                reply.retry_after = std.fmt.parseInt(u64, h.value, 10) catch null;
            }
        }
        var body: Io.Writer.Allocating = .init(self.arena);
        const decompress_buf: []u8 = switch (response.head.content_encoding) {
            .identity => &.{},
            .zstd => try self.arena.alloc(u8, std.compress.zstd.default_window_len),
            .deflate, .gzip => try self.arena.alloc(u8, std.compress.flate.max_window_len),
            .compress => return error.UnsupportedCompressionMethod,
        };
        var transfer_buf: [64]u8 = undefined;
        var decompress: std.http.Decompress = undefined;
        const reader = response.readerDecompressing(&transfer_buf, &decompress, decompress_buf);
        _ = reader.streamRemaining(&body.writer) catch |err| switch (err) {
            error.ReadFailed => return response.bodyErr().?,
            else => |e| return e,
        };
        reply.body = body.written();
        return reply;
    }
};

test "repository ids" {
    try std.testing.expect(validRepoId("plyght/qwen-0.5b-ditched"));
    try std.testing.expect(validRepoId("org_name/Model.v2"));
    try std.testing.expect(!validRepoId("no-slash"));
    try std.testing.expect(!validRepoId("a/b/c"));
    try std.testing.expect(!validRepoId("/name"));
    try std.testing.expect(!validRepoId("owner/"));
    try std.testing.expect(!validRepoId("owner/na me"));
    try std.testing.expect(!validRepoId("owner/.hidden"));
}

test "LFS batch, multipart completion and ndjson commit bodies" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var f = File{ .path = "model.safetensors", .fs_path = "model.safetensors", .size = 42 };
    @memset(&f.oid, 'a');
    const batch = try batchRequestBody(a, &.{&f});
    try std.testing.expectEqualStrings(
        \\{"operation":"upload","transfers":["basic","multipart"],"objects":[{"oid":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":42}],"hash_algo":"sha256","ref":{"name":"main"}}
    , batch);
    const done = try completionBody(a, "abc", &.{ "\"e1\"", "\"e2\"" });
    try std.testing.expectEqualStrings(
        \\{"oid":"abc","parts":[{"partNumber":1,"etag":"\"e1\""},{"partNumber":2,"etag":"\"e2\""}]}
    , done);

    // Parts come back keyed by (zero-padded) number, in any order.
    const header = try std.json.parseFromSliceLeaky(std.json.Value, a,
        \\{"chunk_size":"20","00002":"http://s/2","00001":"http://s/1","00003":"http://s/3"}
    , .{});
    const urls = try partUrls(a, header, 42, 20);
    try std.testing.expectEqual(@as(usize, 3), urls.len);
    try std.testing.expectEqualStrings("http://s/1", urls[0]);
    try std.testing.expectEqualStrings("http://s/3", urls[2]);
    try std.testing.expectError(error.InvalidResponse, partUrls(a, header, 80, 20));

    // Preupload answers set the modes; the commit lists LFS pointers and inline files.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    var files = [_]File{ f, .{ .path = "config.json", .fs_path = "config.json", .size = 2 } };
    try applyPreupload(a,
        \\{"files":[{"path":"config.json","uploadMode":"regular","shouldIgnore":false},{"path":"model.safetensors","uploadMode":"lfs","shouldIgnore":false}]}
    , &files);
    try std.testing.expectEqual(Mode.lfs, files[0].mode);
    try std.testing.expectEqual(Mode.regular, files[1].mode);
    const body = try commitBody(a, io, tmp.dir, &files, "Upload");
    try std.testing.expectEqualStrings(
        \\{"key":"header","value":{"summary":"Upload","description":""}}
        \\{"key":"lfsFile","value":{"path":"model.safetensors","algo":"sha256","oid":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":42}}
        \\{"key":"file","value":{"path":"config.json","content":"e30=","encoding":"base64"}}
        \\
    , body);
}
