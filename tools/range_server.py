#!/usr/bin/env python3
"""Static file server that honours HTTP Range requests (206 Partial Content),
used by the tests for ditch's remote weight source. Python's built-in
http.server ignores Range, so a small handler is needed.

Usage: range_server.py <directory> [log-file]

Binds 127.0.0.1 on a free port, prints "PORT <n>" on stdout (flushed) and
serves until killed. Every request is appended to the log file as
"<method> <path> <range-or-'-'> <status> <bytes>".

A path under `/ratelimit-<n>/` serves the same files, but the first <n> Range
requests under that prefix are answered 429 Too Many Requests (a rate-limited
Hub). Under `/failafter-<n>/` the Range requests after the first <n> are
answered 404 (a read that fails once the shard headers are in).
"""
import http.server
import os
import re
import sys
import threading

ROOT = os.path.abspath(sys.argv[1])
LOG = sys.argv[2] if len(sys.argv) > 2 else None


LIMITED = {}
LIMIT_LOCK = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def record(self, rng, status, nbytes):
        if LOG:
            with open(LOG, "a") as f:
                f.write(f"{self.command} {self.path} {rng or '-'} {status} {nbytes}\n")

    def do_GET(self):
        rel = self.path
        m = re.match(r"^/ratelimit-(\d+)(/.*)$", rel)
        if m:
            rel = m.group(2)
            if self.headers.get("Range"):
                with LIMIT_LOCK:
                    LIMITED[m.group(1)] = LIMITED.get(m.group(1), 0) + 1
                    limited = LIMITED[m.group(1)] <= int(m.group(1))
                if limited:
                    self.send_response(429)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    self.record(self.headers.get("Range"), 429, 0)
                    return
        m = re.match(r"^/failafter-(\d+)(/.*)$", rel)
        if m:
            rel = m.group(2)
            if self.headers.get("Range"):
                key = "after-" + m.group(1)
                with LIMIT_LOCK:
                    LIMITED[key] = LIMITED.get(key, 0) + 1
                    failing = LIMITED[key] > int(m.group(1))
                if failing:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    self.record(self.headers.get("Range"), 404, 0)
                    return
        path = os.path.normpath(os.path.join(ROOT, rel.lstrip("/")))
        if not path.startswith(ROOT) or not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.record(self.headers.get("Range"), 404, 0)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng and rng.startswith("bytes="):
            a, _, b = rng[len("bytes="):].partition("-")
            if a:
                start = int(a)
                end = min(int(b), size - 1) if b else size - 1
            else:
                start = max(size - int(b), 0)
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.record(rng, 416, 0)
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(length))
        self.record(rng, status, length)


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"PORT {server.server_address[1]}", flush=True)
    server.serve_forever()
