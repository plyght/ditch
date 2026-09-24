//! `ditch update`: replaces the running binary with a release from GitHub.
//!
//! The archive is the one `.github/workflows/release.yml` publishes for this
//! build (the same OS, architecture, libc and x86-64-v3 variant, so what the
//! installer chose is kept), checked against the release's SHA256SUMS before
//! anything is extracted. The new binary is staged in a temporary directory
//! beside the executable, moved next to it as `<name>.new` and renamed over
//! it; Windows cannot replace a running executable, so there the old one is
//! first renamed to `<name>.old` and deleted on the next start.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const config = @import("config.zig");
const hf = @import("hf.zig");

const Allocator = std.mem.Allocator;

pub const default_api = "https://api.github.com/repos/plyght/ditch";

// ---------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------

fn stripV(v: []const u8) []const u8 {
    const t = std.mem.trim(u8, v, " \t\r\n");
    return if (t.len > 0 and (t[0] == 'v' or t[0] == 'V')) t[1..] else t;
}

fn cmpIdent(a: []const u8, b: []const u8) std.math.Order {
    const na = std.fmt.parseInt(u64, a, 10) catch null;
    const nb = std.fmt.parseInt(u64, b, 10) catch null;
    if (na != null and nb != null) return std.math.order(na.?, nb.?);
    // Numeric identifiers sort before alphanumeric ones (semver).
    if (na != null) return .lt;
    if (nb != null) return .gt;
    return std.mem.order(u8, a, b);
}

/// Semantic-version order of two versions or tags ("v0.6.0", "0.10.1",
/// "1.0.0-rc.1"); a leading v and build metadata (+...) are ignored, missing
/// components count as 0 and a pre-release sorts before its release.
pub fn compareVersions(a: []const u8, b: []const u8) std.math.Order {
    const x = stripV(a);
    const y = stripV(b);
    const xe = std.mem.indexOfScalar(u8, x, '+') orelse x.len;
    const ye = std.mem.indexOfScalar(u8, y, '+') orelse y.len;
    const xd = std.mem.indexOfScalar(u8, x[0..xe], '-') orelse xe;
    const yd = std.mem.indexOfScalar(u8, y[0..ye], '-') orelse ye;
    var xi = std.mem.splitScalar(u8, x[0..xd], '.');
    var yi = std.mem.splitScalar(u8, y[0..yd], '.');
    while (true) {
        const p = xi.next();
        const q = yi.next();
        if (p == null and q == null) break;
        const pn = std.fmt.parseInt(u64, p orelse "0", 10) catch 0;
        const qn = std.fmt.parseInt(u64, q orelse "0", 10) catch 0;
        const o = std.math.order(pn, qn);
        if (o != .eq) return o;
    }
    const xp = if (xd < xe) x[xd + 1 .. xe] else "";
    const yp = if (yd < ye) y[yd + 1 .. ye] else "";
    if (xp.len == 0 and yp.len == 0) return .eq;
    if (xp.len == 0) return .gt;
    if (yp.len == 0) return .lt;
    var pi = std.mem.splitScalar(u8, xp, '.');
    var qi = std.mem.splitScalar(u8, yp, '.');
    while (true) {
        const p = pi.next();
        const q = qi.next();
        if (p == null and q == null) return .eq;
        if (p == null) return .lt;
        if (q == null) return .gt;
        const o = cmpIdent(p.?, q.?);
        if (o != .eq) return o;
    }
}

/// A release tag as the workflow names them: "0.6.0" -> "v0.6.0".
pub fn normalizeTag(a: Allocator, tag: []const u8) ![]const u8 {
    const t = std.mem.trim(u8, tag, " \t\r\n");
    if (t.len > 0 and std.ascii.isDigit(t[0])) return std.fmt.allocPrint(a, "v{s}", .{t});
    return a.dupe(u8, t);
}

// ---------------------------------------------------------------------------
// Assets
// ---------------------------------------------------------------------------

/// What a release archive has to match: the running build.
pub const Build = struct {
    arch: std.Target.Cpu.Arch,
    os: std.Target.Os.Tag,
    abi: std.Target.Abi,
    /// Compiled for x86-64-v3 (AVX2 + FMA): the `-v3` archive.
    v3: bool,
};

pub const this_build: Build = .{
    .arch = builtin.cpu.arch,
    .os = builtin.os.tag,
    .abi = builtin.abi,
    .v3 = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHasAll(builtin.cpu.features, .{ .avx2, .fma }),
};

/// The archive label of a build ("x86_64-linux-musl", "aarch64-macos", ...),
/// or null for a platform the releases do not cover.
pub fn label(b: Build) ?[]const u8 {
    return switch (b.arch) {
        .x86_64 => switch (b.os) {
            .linux => if (b.abi.isMusl()) "x86_64-linux-musl" else if (b.abi.isGnu()) "x86_64-linux-gnu" else null,
            .macos => "x86_64-macos",
            .windows => "x86_64-windows-gnu",
            else => null,
        },
        .aarch64 => switch (b.os) {
            .linux => if (b.abi.isMusl()) "aarch64-linux-musl" else if (b.abi.isGnu()) "aarch64-linux-gnu" else null,
            .macos => "aarch64-macos",
            else => null,
        },
        else => null,
    };
}

/// `ditch-<tag>-<label>[-v3].tar.gz` (`.zip` for Windows).
pub fn assetName(a: Allocator, tag: []const u8, lbl: []const u8, v3: bool) ![]const u8 {
    const ext = if (std.mem.indexOf(u8, lbl, "windows") != null) "zip" else "tar.gz";
    return std.fmt.allocPrint(a, "ditch-{s}-{s}{s}.{s}", .{ tag, lbl, if (v3) "-v3" else "", ext });
}

pub const Asset = struct {
    name: []const u8,
    browser_download_url: []const u8,
    size: u64 = 0,
};

pub const Release = struct {
    tag_name: []const u8,
    html_url: []const u8 = "",
    assets: []const Asset = &.{},

    pub fn find(r: Release, name: []const u8) ?Asset {
        for (r.assets) |x| if (std.mem.eql(u8, x.name, name)) return x;
        return null;
    }
};

pub const Pick = struct {
    asset: Asset,
    /// Why this is not the exact build of this binary (null when it is).
    note: ?[]const u8 = null,
};

fn muslOf(lbl: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, lbl, "x86_64-linux-gnu")) return "x86_64-linux-musl";
    if (std.mem.eql(u8, lbl, "aarch64-linux-gnu")) return "aarch64-linux-musl";
    return null;
}

/// The archive of `release` for build `b`: the same label and variant, else
/// (a release that lacks it) the static musl build for glibc, then the
/// portable build for x86-64-v3, with a note saying so.
pub fn pickAsset(a: Allocator, release: Release, b: Build) !?Pick {
    const lbl = label(b) orelse return null;
    const musl = muslOf(lbl);
    const Candidate = struct { lbl: []const u8, v3: bool, note: ?[]const u8 };
    var cands: [4]Candidate = undefined;
    var n: usize = 0;
    cands[n] = .{ .lbl = lbl, .v3 = b.v3, .note = null };
    n += 1;
    const musl_note = "this release has no glibc build for this platform; installing the static musl build (it cannot load a Vulkan driver)";
    const v3_note = "this release has no x86-64-v3 build for this platform; installing the portable x86-64 build";
    if (musl) |m| {
        cands[n] = .{ .lbl = m, .v3 = b.v3, .note = musl_note };
        n += 1;
    }
    if (b.v3) {
        cands[n] = .{ .lbl = lbl, .v3 = false, .note = v3_note };
        n += 1;
        if (musl) |m| {
            cands[n] = .{ .lbl = m, .v3 = false, .note = "this release has no glibc or x86-64-v3 build for this platform; installing the portable static musl build" };
            n += 1;
        }
    }
    for (cands[0..n]) |c| {
        const name = try assetName(a, release.tag_name, c.lbl, c.v3);
        if (release.find(name)) |asset| return .{ .asset = asset, .note = c.note };
    }
    return null;
}

/// The SHA-256 digest SHA256SUMS lists for `name` (`<hex>  <name>`, or
/// `<hex> *<name>` in binary mode), or null when the file has no such entry.
pub fn findSum(sums: []const u8, name: []const u8) ?[32]u8 {
    var lines = std.mem.splitScalar(u8, sums, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        const sp = std.mem.indexOfAny(u8, line, " \t") orelse continue;
        const hex = line[0..sp];
        var file = std.mem.trimStart(u8, line[sp..], " \t");
        if (file.len > 0 and file[0] == '*') file = file[1..];
        if (file.len > 2 and std.mem.startsWith(u8, file, "./")) file = file[2..];
        if (hex.len != 64 or !std.mem.eql(u8, file, name)) continue;
        var digest: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&digest, hex) catch continue;
        return digest;
    }
    return null;
}

/// The package manager that owns a binary at `path` (resolved), if it looks
/// installed by one.
pub fn packageManager(path: []const u8) ?[]const u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len > buf.len) return null;
    const p = buf[0..path.len];
    for (path, 0..) |c, i| p[i] = if (c == '\\') '/' else std.ascii.toLower(c);
    const contains = [_][2][]const u8{
        .{ "/nix/store/", "Nix" },
        .{ "/gnu/store/", "Guix" },
        .{ "/cellar/", "Homebrew" },
        .{ "/scoop/apps/", "Scoop" },
        .{ "/chocolatey/lib/", "Chocolatey" },
        .{ "/winget/packages/", "winget" },
    };
    for (contains) |e| if (std.mem.indexOf(u8, p, e[0]) != null) return e[1];
    const prefixes = [_][2][]const u8{
        .{ "/snap/", "snap" },
        .{ "/opt/local/", "MacPorts" },
        .{ "/usr/bin/", "your system's package manager" },
        .{ "/usr/sbin/", "your system's package manager" },
        .{ "/usr/lib/", "your system's package manager" },
        .{ "/usr/libexec/", "your system's package manager" },
        .{ "/bin/", "your system's package manager" },
        .{ "/sbin/", "your system's package manager" },
    };
    for (prefixes) |e| if (std.mem.startsWith(u8, p, e[0])) return e[1];
    return null;
}

// ---------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------

pub const Ctx = struct {
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    env: *std.process.Environ.Map,
    settings: *const config.Settings,
    /// Progress and notes (stderr).
    out: *Io.Writer,
    /// The outcome: the --check report, the installed version (stdout).
    result: *Io.Writer,
    /// Answers to the confirmation (stdin).
    in: *Io.Reader,
    /// Whether the confirmation may be asked (a terminal, or --interactive; not --no-input).
    interactive: bool,
    /// The GitHub API base of the repository.
    api: []const u8 = default_api,
    /// The binary to replace (default: the running executable).
    exe_path: ?[]const u8 = null,
    /// The version to compare against (default: this build's).
    current: []const u8 = config.version,
    build: Build = this_build,
    /// Where errors go instead of the log (tests).
    errors: ?*Io.Writer = null,
};

fn fail(c: Ctx, comptime fmt: []const u8, args: anytype) void {
    if (c.errors) |w| {
        w.print("error: " ++ fmt ++ "\n", args) catch {};
    } else std.log.err(fmt, args);
}

fn fetchRelease(c: Ctx, http: *hf.Http, tag: ?[]const u8) !Release {
    const url = if (tag) |t|
        try std.fmt.allocPrint(c.arena, "{s}/releases/tags/{s}", .{ c.api, t })
    else
        try std.fmt.allocPrint(c.arena, "{s}/releases/latest", .{c.api});
    // A GitHub token raises the API rate limit; it is sent to the API only.
    http.token = c.env.get("GITHUB_TOKEN") orelse c.env.get("GH_TOKEN");
    defer http.token = null;
    const body = http.get(url) catch |err| switch (err) {
        error.NotFound => {
            if (tag) |t| fail(c, "no release {s} (see https://github.com/plyght/ditch/releases)", .{t}) else fail(c, "no published release found at {s}", .{url});
            return error.ReleaseNotFound;
        },
        error.Forbidden => {
            fail(c, "GitHub refused the request (rate limited?); set GITHUB_TOKEN or retry later", .{});
            return error.ReleaseNotFound;
        },
        else => |e| {
            fail(c, "could not reach {s}: {s}", .{ url, @errorName(e) });
            return e;
        },
    };
    defer c.gpa.free(body);
    return std.json.parseFromSliceLeaky(Release, c.arena, body, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| {
        fail(c, "unexpected answer from {s}: {s}", .{ url, @errorName(err) });
        return error.ReleaseNotFound;
    };
}

fn writeCheckJson(w: *Io.Writer, current: []const u8, release: Release, available: bool) !void {
    var js: std.json.Stringify = .{ .writer = w };
    try js.beginObject();
    try js.objectField("current");
    try js.write(current);
    try js.objectField("latest");
    try js.write(stripV(release.tag_name));
    try js.objectField("tag");
    try js.write(release.tag_name);
    try js.objectField("update_available");
    try js.write(available);
    try js.objectField("url");
    try js.write(release.html_url);
    try js.endObject();
    try w.writeAll("\n");
}

/// A yes/no question defaulting to yes.
fn confirm(c: Ctx, question: []const u8) !bool {
    try c.out.print("{s} [Y/n] ", .{question});
    try c.out.flush();
    const line = c.in.takeDelimiter('\n') catch return false;
    const t = std.mem.trim(u8, line orelse return false, " \t\r");
    return t.len == 0 or t[0] == 'y' or t[0] == 'Y';
}

/// Runs `ditch update`; returns the exit code.
pub fn run(c: Ctx) !u8 {
    const s = c.settings;
    if (s.model.len > 0) {
        fail(c, "unexpected argument {s} (usage: ditch update [--check] [--version <tag>])", .{s.model});
        return 2;
    }
    var http = try hf.Http.initWithOptions(c.gpa, c.io, c.arena, c.env, .{ .timeout_seconds = s.http_timeout_seconds, .retry_timeout_seconds = s.remote_retry_timeout_seconds orelse 60 });
    defer http.deinit();
    http.token = null;

    const tag = if (s.update_version) |v| try normalizeTag(c.arena, v) else null;
    if (tag != null and s.update_check) {
        fail(c, "--check reports on the latest release; drop --version {s}", .{tag.?});
        return 2;
    }
    const release = try fetchRelease(c, &http, tag);
    const order = compareVersions(release.tag_name, c.current);
    const newer = order == .gt;

    if (s.update_check) {
        if (s.json) {
            try writeCheckJson(c.result, c.current, release, newer);
        } else if (newer) {
            try c.result.print("ditch {s} → {s} is available: {s}\nRun ditch update to install it.\n", .{ c.current, stripV(release.tag_name), release.html_url });
        } else if (order == .eq) {
            try c.result.print("ditch {s} is up to date.\n", .{c.current});
        } else {
            try c.result.print("ditch {s} is newer than the latest release ({s}).\n", .{ c.current, release.tag_name });
        }
        try c.result.flush();
        return 0;
    }
    if (tag == null and !newer) {
        if (order == .eq)
            try c.result.print("ditch {s} is up to date.\n", .{c.current})
        else
            try c.result.print("ditch {s} is newer than the latest release ({s}); nothing to do.\n", .{ c.current, release.tag_name });
        try c.result.flush();
        return 0;
    }

    // The binary to replace.
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe_path = c.exe_path orelse exe_buf[0..try std.process.executablePath(c.io, &exe_buf)];
    if (packageManager(exe_path)) |pm| {
        fail(c, "{s} looks installed by {s}; update it with {s} instead", .{ exe_path, pm, pm });
        return 1;
    }
    const pick = (try pickAsset(c.arena, release, c.build)) orelse {
        const lbl = label(c.build) orelse "this platform";
        fail(c, "release {s} has no archive for {s}{s} (see {s})", .{ release.tag_name, lbl, if (c.build.v3) " (x86-64-v3)" else "", release.html_url });
        return 1;
    };
    const sums = release.find("SHA256SUMS") orelse {
        fail(c, "release {s} has no SHA256SUMS; refusing to install an unverified binary", .{release.tag_name});
        return 1;
    };

    // Stage beside the executable: that proves the directory is writable
    // before asking, and keeps the final rename on one file system.
    var staging = Staging.open(c.io, exe_path) catch |err| switch (err) {
        error.AccessDenied, error.PermissionDenied, error.ReadOnlyFileSystem => {
            const dir = std.fs.path.dirname(exe_path) orelse ".";
            fail(c, "cannot write to {s}, where ditch is installed: rerun with sudo (sudo ditch update), or reinstall with the installer into a directory you own (DITCH_INSTALL_DIR)", .{dir});
            return 1;
        },
        else => |e| return e,
    };
    defer staging.close();

    try c.out.print("ditch {s} → {s}{s}\n", .{ c.current, stripV(release.tag_name), if (order == .eq) " (reinstalling)" else if (order == .lt) " (downgrading)" else "" });
    if (pick.note) |n| try c.out.print("Note: {s}.\n", .{n});
    if (c.interactive and !s.force) {
        const q = try std.fmt.allocPrint(c.arena, "Update to {s}?", .{release.tag_name});
        if (!try confirm(c, q)) {
            try c.out.writeAll("Not updated.\n");
            return 0;
        }
    }

    try c.out.print("Downloading {s}...\n", .{pick.asset.name});
    try c.out.flush();
    const sums_text = http.get(sums.browser_download_url) catch |err| {
        fail(c, "could not download SHA256SUMS: {s}", .{@errorName(err)});
        return 1;
    };
    defer c.gpa.free(sums_text);
    const want = findSum(sums_text, pick.asset.name) orelse {
        fail(c, "SHA256SUMS of {s} has no entry for {s}; refusing to install it", .{ release.tag_name, pick.asset.name });
        return 1;
    };
    http.download(staging.tmp, pick.asset.name, pick.asset.browser_download_url, if (pick.asset.size > 0) pick.asset.size else null) catch |err| {
        fail(c, "could not download {s}: {s}", .{ pick.asset.browser_download_url, @errorName(err) });
        return 1;
    };
    const got = try sha256File(c.io, staging.tmp, pick.asset.name);
    if (!std.mem.eql(u8, &got, &want)) {
        fail(c, "checksum mismatch for {s}: expected {s}, got {s}; not installed", .{ pick.asset.name, std.fmt.bytesToHex(want, .lower), std.fmt.bytesToHex(got, .lower) });
        return 1;
    }

    const bin_name = if (std.mem.endsWith(u8, pick.asset.name, ".zip")) "ditch.exe" else "ditch";
    extract(c.io, staging.tmp, pick.asset.name, bin_name, "ditch.bin") catch |err| {
        fail(c, "could not extract {s} from {s}: {s}", .{ bin_name, pick.asset.name, @errorName(err) });
        return 1;
    };
    try staging.replace("ditch.bin");

    // The installed binary, as it reports itself.
    const ran = std.process.run(c.gpa, c.io, .{ .argv = &.{ exe_path, "--version" }, .stdout_limit = .limited(4096), .stderr_limit = .limited(4096) });
    if (ran) |r| {
        defer c.gpa.free(r.stdout);
        defer c.gpa.free(r.stderr);
        try c.result.print("Installed {s}: {s}\n", .{ exe_path, std.mem.trim(u8, r.stdout, " \r\n") });
    } else |err| {
        std.log.warn("installed {s}, but it did not run ({s})", .{ exe_path, @errorName(err) });
    }
    if (release.html_url.len > 0) try c.result.print("Release notes: {s}\n", .{release.html_url});
    try c.result.flush();
    return 0;
}

fn sha256File(io: Io, dir: Io.Dir, name: []const u8) ![32]u8 {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fr = file.reader(io, &buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    while (true) {
        const chunk = fr.interface.peekGreedy(1) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
        h.update(chunk);
        fr.interface.toss(chunk.len);
    }
    return h.finalResult();
}

/// Extracts the file `bin_name` (at any depth, normally `<top>/<bin_name>`)
/// of the archive `archive` in `dir` to `dir/out_name`.
pub fn extract(io: Io, dir: Io.Dir, archive: []const u8, bin_name: []const u8, out_name: []const u8) !void {
    if (std.mem.endsWith(u8, archive, ".zip")) {
        const file = try dir.openFile(io, archive, .{});
        defer file.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fr = file.reader(io, &buf);
        try dir.createDir(io, "unzipped", .default_dir);
        var dest = try dir.openDir(io, "unzipped", .{ .iterate = true });
        defer dest.close(io);
        try std.zip.extract(dest, &fr, .{ .allow_backslashes = true });
        const top = archive[0 .. archive.len - ".zip".len];
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const inner = try std.fmt.bufPrint(&pbuf, "unzipped/{s}/{s}", .{ top, bin_name });
        try dir.rename(inner, dir, out_name, io);
        return;
    }
    const file = try dir.openFile(io, archive, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fr = file.reader(io, &buf);
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var gz = std.compress.flate.Decompress.init(&fr.interface, .gzip, &window);
    var name_buf: [std.fs.max_path_bytes]u8 = undefined;
    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    var it = std.tar.Iterator.init(&gz.reader, .{ .file_name_buffer = &name_buf, .link_name_buffer = &link_buf });
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.eql(u8, std.fs.path.basenamePosix(entry.name), bin_name)) continue;
        if (std.mem.count(u8, std.mem.trimEnd(u8, entry.name, "/"), "/") > 2) continue;
        const out = try dir.createFile(io, out_name, .{});
        defer out.close(io);
        var wbuf: [1 << 16]u8 = undefined;
        var fw = out.writer(io, &wbuf);
        try it.streamRemaining(entry, &fw.interface);
        try fw.interface.flush();
        return;
    }
    return error.BinaryNotInArchive;
}

/// A temporary directory beside the executable, removed on close.
const Staging = struct {
    io: Io,
    exe_path: []const u8,
    dir: Io.Dir,
    tmp: Io.Dir,
    tmp_name_buf: [64]u8,
    tmp_name_len: usize,

    fn tmpName(self: *const Staging) []const u8 {
        return self.tmp_name_buf[0..self.tmp_name_len];
    }

    fn open(io: Io, exe_path: []const u8) !Staging {
        const dir_path = std.fs.path.dirname(exe_path) orelse ".";
        var dir = try Io.Dir.cwd().openDir(io, dir_path, .{});
        errdefer dir.close(io);
        removeLeftover(io, dir, std.fs.path.basename(exe_path));
        var s: Staging = .{ .io = io, .exe_path = exe_path, .dir = dir, .tmp = undefined, .tmp_name_buf = undefined, .tmp_name_len = 0 };
        var rnd: [6]u8 = undefined;
        io.random(&rnd);
        const name = try std.fmt.bufPrint(&s.tmp_name_buf, ".ditch-update-{s}", .{std.fmt.bytesToHex(rnd, .lower)});
        s.tmp_name_len = name.len;
        try dir.createDir(io, name, .default_dir);
        s.tmp = dir.openDir(io, name, .{ .iterate = true }) catch |err| {
            dir.deleteTree(io, name) catch {};
            return err;
        };
        return s;
    }

    fn close(self: *Staging) void {
        self.tmp.close(self.io);
        self.dir.deleteTree(self.io, self.tmpName()) catch {};
        self.dir.close(self.io);
    }

    /// Moves `tmp/new_name` next to the executable as `<exe>.new` with the
    /// executable's permissions, then over the executable.
    fn replace(self: *Staging, new_name: []const u8) !void {
        const io = self.io;
        const exe_name = std.fs.path.basename(self.exe_path);
        var nbuf: [std.fs.max_path_bytes]u8 = undefined;
        const staged = try std.fmt.bufPrint(&nbuf, "{s}.new", .{exe_name});
        if (Io.File.Permissions.has_executable_bit) {
            const perms = if (self.dir.statFile(io, exe_name, .{})) |st| st.permissions else |_| Io.File.Permissions.executable_file;
            try self.tmp.setFilePermissions(io, new_name, perms, .{});
        }
        try self.tmp.rename(new_name, self.dir, staged, io);
        errdefer self.dir.deleteFile(io, staged) catch {};
        if (builtin.os.tag == .windows) {
            // A running executable can be renamed but not replaced.
            var obuf: [std.fs.max_path_bytes]u8 = undefined;
            const old = try std.fmt.bufPrint(&obuf, "{s}.old", .{exe_name});
            self.dir.deleteFile(io, old) catch {};
            try self.dir.rename(exe_name, self.dir, old, io);
            self.dir.rename(staged, self.dir, exe_name, io) catch |err| {
                self.dir.rename(old, self.dir, exe_name, io) catch {};
                return err;
            };
            // Deletable only once this process has exited; the next start does it.
            self.dir.deleteFile(io, old) catch {};
        } else {
            try self.dir.rename(staged, self.dir, exe_name, io);
        }
    }
};

/// Deletes `<exe>.old`, left by an update on Windows.
fn removeLeftover(io: Io, dir: Io.Dir, exe_name: []const u8) void {
    if (builtin.os.tag != .windows) return;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const old = std.fmt.bufPrint(&buf, "{s}.old", .{exe_name}) catch return;
    dir.deleteFile(io, old) catch {};
}

/// At start-up on Windows: removes the executable an update replaced.
pub fn cleanupAfterUpdate(io: Io) void {
    if (builtin.os.tag != .windows) return;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = std.process.executablePath(io, &buf) catch return;
    const path = buf[0..n];
    var dir = Io.Dir.cwd().openDir(io, std.fs.path.dirname(path) orelse return, .{}) catch return;
    defer dir.close(io);
    removeLeftover(io, dir, std.fs.path.basename(path));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "update: version order" {
    const t = std.testing;
    try t.expectEqual(std.math.Order.eq, compareVersions("v0.6.0", "0.6.0"));
    try t.expectEqual(std.math.Order.gt, compareVersions("v0.10.0", "0.9.9"));
    try t.expectEqual(std.math.Order.lt, compareVersions("0.6.0", "v0.6.1"));
    try t.expectEqual(std.math.Order.gt, compareVersions("1.0", "0.99.99"));
    try t.expectEqual(std.math.Order.eq, compareVersions("v1.2", "1.2.0"));
    try t.expectEqual(std.math.Order.lt, compareVersions("v1.0.0-rc.1", "1.0.0"));
    try t.expectEqual(std.math.Order.lt, compareVersions("1.0.0-rc.2", "1.0.0-rc.10"));
    try t.expectEqual(std.math.Order.lt, compareVersions("1.0.0-alpha", "1.0.0-beta"));
    try t.expectEqual(std.math.Order.eq, compareVersions("1.0.0+abc", "1.0.0"));
    const a = t.allocator;
    const v = try normalizeTag(a, "0.6.0");
    defer a.free(v);
    try t.expectEqualStrings("v0.6.0", v);
}

test "update: asset for each build, with fallbacks" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqualStrings("x86_64-linux-musl", label(.{ .arch = .x86_64, .os = .linux, .abi = .musl, .v3 = false }).?);
    try t.expectEqualStrings("aarch64-linux-gnu", label(.{ .arch = .aarch64, .os = .linux, .abi = .gnu, .v3 = false }).?);
    try t.expectEqualStrings("aarch64-macos", label(.{ .arch = .aarch64, .os = .macos, .abi = .none, .v3 = false }).?);
    try t.expectEqual(@as(?[]const u8, null), label(.{ .arch = .riscv64, .os = .linux, .abi = .musl, .v3 = false }));
    try t.expectEqualStrings("ditch-v0.6.0-x86_64-windows-gnu-v3.zip", try assetName(a, "v0.6.0", "x86_64-windows-gnu", true));
    try t.expectEqualStrings("ditch-v0.6.0-x86_64-macos.tar.gz", try assetName(a, "v0.6.0", "x86_64-macos", false));

    const names = [_][]const u8{ "ditch-v1.0.0-x86_64-linux-musl.tar.gz", "ditch-v1.0.0-x86_64-linux-musl-v3.tar.gz", "ditch-v1.0.0-x86_64-linux-gnu.tar.gz", "ditch-v1.0.0-aarch64-linux-musl.tar.gz", "ditch-v1.0.0-x86_64-windows-gnu-v3.zip", "SHA256SUMS" };
    var assets: [names.len]Asset = undefined;
    for (names, 0..) |n, i| assets[i] = .{ .name = n, .browser_download_url = n };
    const rel: Release = .{ .tag_name = "v1.0.0", .assets = &assets };
    const Case = struct { b: Build, want: []const u8, note: bool };
    const cases = [_]Case{
        .{ .b = .{ .arch = .x86_64, .os = .linux, .abi = .musl, .v3 = true }, .want = "ditch-v1.0.0-x86_64-linux-musl-v3.tar.gz", .note = false },
        .{ .b = .{ .arch = .x86_64, .os = .linux, .abi = .musl, .v3 = false }, .want = "ditch-v1.0.0-x86_64-linux-musl.tar.gz", .note = false },
        .{ .b = .{ .arch = .x86_64, .os = .linux, .abi = .gnu, .v3 = false }, .want = "ditch-v1.0.0-x86_64-linux-gnu.tar.gz", .note = false },
        // No gnu-v3 archive: the musl -v3 one keeps the CPU variant.
        .{ .b = .{ .arch = .x86_64, .os = .linux, .abi = .gnu, .v3 = true }, .want = "ditch-v1.0.0-x86_64-linux-musl-v3.tar.gz", .note = true },
        // An older release without linux-gnu archives.
        .{ .b = .{ .arch = .aarch64, .os = .linux, .abi = .gnu, .v3 = false }, .want = "ditch-v1.0.0-aarch64-linux-musl.tar.gz", .note = true },
        .{ .b = .{ .arch = .x86_64, .os = .windows, .abi = .gnu, .v3 = true }, .want = "ditch-v1.0.0-x86_64-windows-gnu-v3.zip", .note = false },
    };
    for (cases) |c| {
        const p = (try pickAsset(a, rel, c.b)).?;
        try t.expectEqualStrings(c.want, p.asset.name);
        try t.expectEqual(c.note, p.note != null);
    }
    // Nothing for this platform.
    try t.expectEqual(@as(?Pick, null), try pickAsset(a, rel, .{ .arch = .aarch64, .os = .macos, .abi = .none, .v3 = false }));
}

test "update: SHA256SUMS entries" {
    const t = std.testing;
    const sums =
        \\0000000000000000000000000000000000000000000000000000000000000000  ditch-v1.0.0-x86_64-linux-musl.tar.gz
        \\ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789 *ditch-v1.0.0-aarch64-macos.tar.gz
        \\ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff  ./ditch-v1.0.0-x86_64-macos.tar.gz
        \\not-a-hash  ditch-v1.0.0-x86_64-windows-gnu.zip
        \\
    ;
    const z = findSum(sums, "ditch-v1.0.0-x86_64-linux-musl.tar.gz").?;
    try t.expectEqual(@as(u8, 0), z[31]);
    const m = findSum(sums, "ditch-v1.0.0-aarch64-macos.tar.gz").?;
    try t.expectEqual(@as(u8, 0xab), m[0]);
    try t.expectEqual(@as(u8, 0x89), m[31]);
    try t.expect(findSum(sums, "ditch-v1.0.0-x86_64-macos.tar.gz") != null);
    try t.expect(findSum(sums, "ditch-v1.0.0-x86_64-windows-gnu.zip") == null);
    try t.expect(findSum(sums, "ditch-v1.0.0-x86_64-linux-musl") == null);
    try t.expect(findSum(sums, "missing.tar.gz") == null);
}

test "update: package-managed paths" {
    const t = std.testing;
    try t.expectEqualStrings("Homebrew", packageManager("/opt/homebrew/Cellar/ditch/0.6.0/bin/ditch").?);
    try t.expectEqualStrings("Nix", packageManager("/nix/store/abc-ditch-0.6.0/bin/ditch").?);
    try t.expect(packageManager("/usr/bin/ditch") != null);
    try t.expectEqualStrings("Scoop", packageManager("C:\\Users\\me\\scoop\\apps\\ditch\\current\\ditch.exe").?);
    try t.expect(packageManager("/home/me/.local/bin/ditch") == null);
    try t.expect(packageManager("/usr/local/bin/ditch") == null);
    try t.expect(packageManager("C:\\Users\\me\\AppData\\Local\\Programs\\ditch\\ditch.exe") == null);
}
