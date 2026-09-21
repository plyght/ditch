//! Hugging Face Hub access: model downloads, dataset rows via the
//! datasets-server API, and prompt loading from local text files.
//!
//! HTTP goes through `std.http.Client`; when that fails (for example because
//! of an unusual TLS setup) ditch falls back to invoking `curl`.

const std = @import("std");
const Io = std.Io;
const config = @import("config.zig");

const Allocator = std.mem.Allocator;

pub const Prompt = struct {
    system: []const u8,
    user: []const u8,
};

pub const Http = struct {
    gpa: Allocator,
    io: Io,
    client: std.http.Client,
    token: ?[]const u8,
    environ: *std.process.Environ.Map,
    native_ok: bool = true,

    pub fn init(gpa: Allocator, io: Io, arena: Allocator, environ: *std.process.Environ.Map) !Http {
        var client: std.http.Client = .{ .allocator = gpa, .io = io };
        client.initDefaultProxies(arena, environ) catch {};
        const token = environ.get("HF_TOKEN") orelse environ.get("HUGGING_FACE_HUB_TOKEN");
        return .{ .gpa = gpa, .io = io, .client = client, .token = token, .environ = environ };
    }

    pub fn deinit(self: *Http) void {
        self.client.deinit();
    }

    fn authHeader(self: *Http, buf: []u8) ?std.http.Header {
        const t = self.token orelse return null;
        const v = std.fmt.bufPrint(buf, "Bearer {s}", .{t}) catch return null;
        return .{ .name = "authorization", .value = v };
    }

    /// Fetches `url` into memory. Returns error.NotFound for 404.
    pub fn get(self: *Http, url: []const u8) ![]u8 {
        if (self.native_ok) {
            if (self.getNative(url)) |body| {
                return body;
            } else |err| switch (err) {
                error.NotFound, error.Forbidden, error.OutOfMemory => return err,
                else => {
                    std.log.debug("native http failed for {s}: {s}; trying curl", .{ url, @errorName(err) });
                    self.native_ok = false;
                },
            }
        }
        return self.getCurl(url);
    }

    fn getNative(self: *Http, url: []const u8) ![]u8 {
        var body: Io.Writer.Allocating = .init(self.gpa);
        errdefer body.deinit();
        var auth_buf: [512]u8 = undefined;
        var headers: [1]std.http.Header = undefined;
        var n_headers: usize = 0;
        if (self.authHeader(&auth_buf)) |h| {
            headers[0] = h;
            n_headers = 1;
        }
        const res = try self.client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &body.writer,
            .extra_headers = headers[0..n_headers],
        });
        switch (res.status) {
            .ok => return body.toOwnedSlice(),
            .not_found => return error.NotFound,
            .unauthorized, .forbidden => return error.Forbidden,
            else => return error.HttpError,
        }
    }

    fn getCurl(self: *Http, url: []const u8) ![]u8 {
        var argv = std.ArrayList([]const u8).empty;
        defer argv.deinit(self.gpa);
        try argv.appendSlice(self.gpa, &.{ "curl", "-L", "-sS", "--fail-with-body", "-w", "\n%{http_code}" });
        var auth_buf: [512]u8 = undefined;
        if (self.authHeader(&auth_buf)) |h| {
            try argv.append(self.gpa, "-H");
            try argv.append(self.gpa, try std.fmt.allocPrint(self.gpa, "{s}: {s}", .{ h.name, h.value }));
        }
        defer if (self.token != null and argv.items.len >= 2) self.gpa.free(argv.items[argv.items.len - 2]);
        try argv.append(self.gpa, url);
        const result = std.process.run(self.gpa, self.io, .{ .argv = argv.items, .stdout_limit = .limited(1 << 30) }) catch |err| {
            std.log.err("could not run curl ({s}); install curl or fix the network configuration", .{@errorName(err)});
            return error.HttpError;
        };
        defer self.gpa.free(result.stderr);
        defer self.gpa.free(result.stdout);
        const nl = std.mem.lastIndexOfScalar(u8, result.stdout, '\n') orelse return error.HttpError;
        const code = std.fmt.parseInt(u16, std.mem.trim(u8, result.stdout[nl + 1 ..], " \r\n"), 10) catch 0;
        if (code == 404) return error.NotFound;
        if (code == 401 or code == 403) return error.Forbidden;
        if (code != 200) {
            std.log.err("curl failed for {s}: {s}", .{ url, std.mem.trim(u8, result.stderr, "\n") });
            return error.HttpError;
        }
        return self.gpa.dupe(u8, result.stdout[0..nl]);
    }

    /// Downloads `url` to `dir/sub_path`, writing to a temporary file first.
    pub fn download(self: *Http, dir: Io.Dir, sub_path: []const u8, url: []const u8, expected_size: ?u64) !void {
        const tmp_name = try std.fmt.allocPrint(self.gpa, "{s}.part", .{sub_path});
        defer self.gpa.free(tmp_name);
        var ok = false;
        if (self.native_ok) {
            if (self.downloadNative(dir, tmp_name, url)) {
                ok = true;
            } else |err| switch (err) {
                error.NotFound, error.Forbidden => return err,
                else => {
                    std.log.debug("native download failed ({s}); trying curl", .{@errorName(err)});
                    self.native_ok = false;
                },
            }
        }
        if (!ok) try self.downloadCurl(dir, tmp_name, url);
        if (expected_size) |sz| {
            const st = try dir.statFile(self.io, tmp_name, .{});
            if (st.size != sz) {
                std.log.err("size mismatch for {s}: expected {d} bytes, got {d}", .{ sub_path, sz, st.size });
                return error.SizeMismatch;
            }
        }
        try dir.rename(self.io, tmp_name, dir, sub_path);
    }

    fn downloadNative(self: *Http, dir: Io.Dir, sub_path: []const u8, url: []const u8) !void {
        const file = try dir.createFile(self.io, sub_path, .{});
        defer file.close(self.io);
        var buf: [1 << 16]u8 = undefined;
        var fw = file.writer(self.io, &buf);
        var auth_buf: [512]u8 = undefined;
        var headers: [1]std.http.Header = undefined;
        var n_headers: usize = 0;
        if (self.authHeader(&auth_buf)) |h| {
            headers[0] = h;
            n_headers = 1;
        }
        const res = try self.client.fetch(.{
            .location = .{ .url = url },
            .response_writer = &fw.interface,
            .extra_headers = headers[0..n_headers],
        });
        try fw.interface.flush();
        switch (res.status) {
            .ok => {},
            .not_found => return error.NotFound,
            .unauthorized, .forbidden => return error.Forbidden,
            else => return error.HttpError,
        }
    }

    fn downloadCurl(self: *Http, dir: Io.Dir, sub_path: []const u8, url: []const u8) !void {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = try dir.realPath(self.io, &path_buf);
        const full = try std.fs.path.join(self.gpa, &.{ path_buf[0..dir_path], sub_path });
        defer self.gpa.free(full);
        var argv = std.ArrayList([]const u8).empty;
        defer argv.deinit(self.gpa);
        try argv.appendSlice(self.gpa, &.{ "curl", "-L", "-sS", "--fail", "-o", full });
        var auth_buf: [512]u8 = undefined;
        var auth_line: ?[]u8 = null;
        defer if (auth_line) |l| self.gpa.free(l);
        if (self.authHeader(&auth_buf)) |h| {
            auth_line = try std.fmt.allocPrint(self.gpa, "{s}: {s}", .{ h.name, h.value });
            try argv.append(self.gpa, "-H");
            try argv.append(self.gpa, auth_line.?);
        }
        try argv.append(self.gpa, url);
        const result = std.process.run(self.gpa, self.io, .{ .argv = argv.items }) catch |err| {
            std.log.err("could not run curl ({s})", .{@errorName(err)});
            return error.HttpError;
        };
        defer self.gpa.free(result.stderr);
        defer self.gpa.free(result.stdout);
        switch (result.term) {
            .exited => |code| if (code != 0) {
                std.log.err("curl failed for {s}: {s}", .{ url, std.mem.trim(u8, result.stderr, "\n") });
                if (code == 22) return error.NotFound;
                return error.HttpError;
            },
            else => return error.HttpError,
        }
    }
};

// ---------------------------------------------------------------------------
// Cache directory
// ---------------------------------------------------------------------------

pub fn cacheDir(arena: Allocator, settings: *const config.Settings, environ: *std.process.Environ.Map) ![]const u8 {
    if (settings.cache_dir) |c| return c;
    if (environ.get("DITCH_CACHE")) |c| return c;
    if (environ.get("XDG_CACHE_HOME")) |c| return std.fs.path.join(arena, &.{ c, "ditch" });
    if (environ.get("HOME")) |h| return std.fs.path.join(arena, &.{ h, ".cache", "ditch" });
    if (environ.get("USERPROFILE")) |h| return std.fs.path.join(arena, &.{ h, ".cache", "ditch" });
    return "ditch-cache";
}

fn sanitizeRepoId(arena: Allocator, id: []const u8) ![]u8 {
    const out = try arena.dupe(u8, id);
    for (out) |*c| if (c.* == '/') {
        c.* = '-';
    } else if (!(std.ascii.isAlphanumeric(c.*) or c.* == '_' or c.* == '-' or c.* == '.')) {
        c.* = '_';
    };
    // "owner/name" -> "owner--name" to mirror the HF cache convention.
    if (std.mem.indexOfScalar(u8, id, '/')) |slash| {
        const res = try arena.alloc(u8, out.len + 1);
        @memcpy(res[0..slash], out[0..slash]);
        res[slash] = '-';
        res[slash + 1] = '-';
        @memcpy(res[slash + 2 ..], out[slash + 1 ..]);
        return res;
    }
    return out;
}

pub fn isLocalDir(io: Io, path: []const u8) bool {
    var d = Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    d.close(io);
    return true;
}

pub fn isLocalFile(io: Io, path: []const u8) bool {
    const f = Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

/// Resolves a model id or path to a local directory, downloading if needed.
/// Returns the directory path (allocated in `arena`).
pub fn resolveModel(arena: Allocator, http: *Http, cache_root: []const u8, model: []const u8, revision: ?[]const u8, out: *Io.Writer) ![]const u8 {
    const io = http.io;
    if (isLocalDir(io, model)) return model;
    if (std.mem.indexOfScalar(u8, model, '/') == null or std.mem.startsWith(u8, model, ".") or std.mem.startsWith(u8, model, "/")) {
        std.log.err("model directory not found: {s}", .{model});
        return error.ModelNotFound;
    }
    const rev = revision orelse "main";
    const dir_name = try sanitizeRepoId(arena, model);
    const model_dir = try std.fs.path.join(arena, &.{ cache_root, "models", dir_name, if (revision) |r| r else "main" });
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, model_dir);
    var dir = try cwd.openDir(io, model_dir, .{});
    defer dir.close(io);

    // List repository files.
    const api_url = try std.fmt.allocPrint(arena, "https://huggingface.co/api/models/{s}/revision/{s}", .{ model, rev });
    const info_text = http.get(api_url) catch |err| switch (err) {
        error.NotFound => {
            std.log.err("model {s} not found on Hugging Face (revision {s})", .{ model, rev });
            return error.ModelNotFound;
        },
        error.Forbidden => {
            std.log.err("access to {s} denied; gated models require HF_TOKEN to be set", .{model});
            return error.ModelNotFound;
        },
        else => {
            // Offline: use the cache if it looks complete.
            if (isLocalFile(io, try std.fs.path.join(arena, &.{ model_dir, "config.json" }))) {
                try out.print("* Could not reach Hugging Face; using cached files in {s}\n", .{model_dir});
                return model_dir;
            }
            return err;
        },
    };
    defer http.gpa.free(info_text);
    var parsed = try std.json.parseFromSlice(std.json.Value, http.gpa, info_text, .{});
    defer parsed.deinit();
    const siblings = (parsed.value.object.get("siblings") orelse return error.InvalidResponse).array;
    var wanted = std.ArrayList([]const u8).empty;
    for (siblings.items) |s| {
        const name = s.object.get("rfilename").?.string;
        if (std.mem.indexOfScalar(u8, name, '/') != null) continue; // nested (e.g. original/) files
        const keep = std.mem.eql(u8, name, "config.json") or std.mem.eql(u8, name, "generation_config.json") or
            std.mem.eql(u8, name, "tokenizer.json") or std.mem.eql(u8, name, "tokenizer_config.json") or
            std.mem.eql(u8, name, "chat_template.jinja") or std.mem.eql(u8, name, "special_tokens_map.json") or
            std.mem.eql(u8, name, "model.safetensors.index.json") or
            (std.mem.endsWith(u8, name, ".safetensors") and !std.mem.startsWith(u8, name, "consolidated"));
        if (keep) try wanted.append(arena, try arena.dupe(u8, name));
    }
    if (wanted.items.len == 0) return error.ModelNotFound;
    for (wanted.items) |name| {
        if (isLocalFile(io, try std.fs.path.join(arena, &.{ model_dir, name }))) continue;
        try out.print("* Downloading {s}...\n", .{name});
        try out.flush();
        const url = try std.fmt.allocPrint(arena, "https://huggingface.co/{s}/resolve/{s}/{s}", .{ model, rev, name });
        try http.download(dir, name, url, null);
    }
    return model_dir;
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

const SplitSpec = struct { name: []const u8, start: ?usize, end: ?usize };

/// Parses "train[:400]" style split strings.
pub fn parseSplit(split: []const u8) !SplitSpec {
    const lb = std.mem.indexOfScalar(u8, split, '[') orelse return .{ .name = split, .start = null, .end = null };
    const rb = std.mem.lastIndexOfScalar(u8, split, ']') orelse return error.InvalidSplit;
    const inner = split[lb + 1 .. rb];
    const colon = std.mem.indexOfScalar(u8, inner, ':') orelse return error.InvalidSplit;
    const a = std.mem.trim(u8, inner[0..colon], " ");
    const b = std.mem.trim(u8, inner[colon + 1 ..], " ");
    if (std.mem.indexOfScalar(u8, inner, '%') != null) return error.InvalidSplit;
    return .{
        .name = split[0..lb],
        .start = if (a.len == 0) null else try std.fmt.parseInt(usize, a, 10),
        .end = if (b.len == 0) null else try std.fmt.parseInt(usize, b, 10),
    };
}

fn applySlice(comptime T: type, items: []T, spec: SplitSpec) []T {
    const start = @min(spec.start orelse 0, items.len);
    const end = @min(spec.end orelse items.len, items.len);
    return items[start..@max(start, end)];
}

fn readLines(arena: Allocator, io: Io, path: []const u8) ![][]const u8 {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited);
    var lines = std.ArrayList([]const u8).empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len > 0) try lines.append(arena, t);
    }
    return lines.toOwnedSlice(arena);
}

/// Loads the rows of a Hugging Face dataset column through the datasets-server API.
fn loadHfRows(arena: Allocator, http: *Http, cache_root: []const u8, spec: config.DatasetSpec, split: SplitSpec, out: *Io.Writer) ![][]const u8 {
    const io = http.io;
    const column = spec.column orelse return error.MissingColumn;
    const cache_dir = try std.fs.path.join(arena, &.{ cache_root, "datasets", try sanitizeRepoId(arena, spec.dataset) });
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, cache_dir);

    // Resolve the config name.
    var cfg_name: []const u8 = spec.config orelse "default";
    if (spec.config == null) {
        const splits_path = try std.fs.path.join(arena, &.{ cache_dir, "splits.json" });
        const text = cwd.readFileAlloc(io, splits_path, arena, .unlimited) catch blk: {
            const url = try std.fmt.allocPrint(arena, "https://datasets-server.huggingface.co/splits?dataset={s}", .{spec.dataset});
            const body = http.get(url) catch |err| {
                std.log.err("could not query datasets-server for {s}: {s}", .{ spec.dataset, @errorName(err) });
                return err;
            };
            defer http.gpa.free(body);
            try cwd.writeFile(io, .{ .sub_path = splits_path, .data = body });
            break :blk try arena.dupe(u8, body);
        };
        var parsed = try std.json.parseFromSlice(std.json.Value, http.gpa, text, .{});
        defer parsed.deinit();
        if (parsed.value.object.get("splits")) |splits| {
            for (splits.array.items) |s| {
                if (std.mem.eql(u8, s.object.get("split").?.string, split.name)) {
                    cfg_name = try arena.dupe(u8, s.object.get("config").?.string);
                    break;
                }
            }
        }
    }

    // Fetch rows in pages of 100 (the API maximum) until the slice is satisfied.
    const needed_end = split.end;
    var rows = std.ArrayList([]const u8).empty;
    var offset: usize = 0;
    while (true) {
        if (needed_end) |e| if (offset >= e) break;
        const page_path = try std.fmt.allocPrint(arena, "{s}/{s}-{s}-{d}.json", .{ cache_dir, cfg_name, split.name, offset });
        const text = cwd.readFileAlloc(io, page_path, arena, .unlimited) catch blk: {
            const url = try std.fmt.allocPrint(arena, "https://datasets-server.huggingface.co/rows?dataset={s}&config={s}&split={s}&offset={d}&length=100", .{ spec.dataset, cfg_name, split.name, offset });
            const body = http.get(url) catch |err| {
                std.log.err("could not fetch rows of {s} ({s}/{s}) from datasets-server: {s}", .{ spec.dataset, cfg_name, split.name, @errorName(err) });
                return err;
            };
            defer http.gpa.free(body);
            try cwd.writeFile(io, .{ .sub_path = page_path, .data = body });
            try out.print("* Fetched rows {d}+ of {s}\n", .{ offset, spec.dataset });
            try out.flush();
            break :blk try arena.dupe(u8, body);
        };
        var parsed = try std.json.parseFromSlice(std.json.Value, http.gpa, text, .{});
        defer parsed.deinit();
        const page = (parsed.value.object.get("rows") orelse return error.InvalidResponse).array;
        if (page.items.len == 0) break;
        for (page.items) |r| {
            const row = r.object.get("row").?.object;
            const v = row.get(column) orelse {
                std.log.err("column '{s}' not found in dataset {s}", .{ column, spec.dataset });
                return error.MissingColumn;
            };
            const s = switch (v) {
                .string => |str| str,
                else => "",
            };
            try rows.append(arena, try arena.dupe(u8, s));
        }
        const total: usize = if (parsed.value.object.get("num_rows_total")) |n| @intCast(n.integer) else std.math.maxInt(usize);
        offset += page.items.len;
        if (offset >= total) break;
        if (page.items.len < 100) break;
    }
    return rows.toOwnedSlice(arena);
}

/// Loads prompts as described by a dataset specification.
pub fn loadPrompts(arena: Allocator, http: *Http, cache_root: []const u8, settings: *const config.Settings, spec: config.DatasetSpec, out: *Io.Writer) ![]Prompt {
    var texts: [][]const u8 = undefined;
    if (isLocalFile(http.io, spec.dataset)) {
        texts = try readLines(arena, http.io, spec.dataset);
        if (spec.split) |s| {
            const sp = try parseSplit(if (std.mem.startsWith(u8, s, "[")) s else s);
            texts = applySlice([]const u8, texts, sp);
        }
    } else {
        const split_str = spec.split orelse {
            std.log.err("the \"split\" field is required for dataset {s}", .{spec.dataset});
            return error.MissingSplit;
        };
        const sp = try parseSplit(split_str);
        texts = try loadHfRows(arena, http, cache_root, spec, sp, out);
        texts = applySlice([]const u8, texts, sp);
    }
    const prompts = try arena.alloc(Prompt, texts.len);
    for (texts, 0..) |t, i| {
        var user = t;
        if (spec.prefix.len > 0) user = try std.fmt.allocPrint(arena, "{s} {s}", .{ spec.prefix, user });
        if (spec.suffix.len > 0) user = try std.fmt.allocPrint(arena, "{s} {s}", .{ user, spec.suffix });
        prompts[i] = .{ .system = spec.system_prompt orelse settings.system_prompt, .user = user };
    }
    return prompts;
}

test "parse split" {
    const s = try parseSplit("train[:400]");
    try std.testing.expectEqualStrings("train", s.name);
    try std.testing.expectEqual(@as(?usize, null), s.start);
    try std.testing.expectEqual(@as(?usize, 400), s.end);
    const t = try parseSplit("test");
    try std.testing.expectEqualStrings("test", t.name);
    const u = try parseSplit("[10:20]");
    try std.testing.expectEqual(@as(?usize, 10), u.start);
}
