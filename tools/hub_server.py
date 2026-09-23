#!/usr/bin/env python3
"""A stand-in for the Hugging Face Hub's upload API, used by the tests of
`ditch push` (src/push_test.zig). It speaks just enough of the protocol:

  POST /hub/api/repos/create                    201, then 409 for an existing repo
  POST /hub/api/models/<repo>/preupload/main    .safetensors/.gguf/.bin -> lfs, rest regular
  POST /hub/<repo>.git/info/lfs/objects/batch   multipart for objects >= 64 KiB, basic below,
                                                no actions for objects already stored
  PUT  /storage/<oid>[/<part>]                  stores the bytes (the first part PUT
                                                answers 503 once, to exercise retries)
  POST /hub/complete/<oid>                      assembles the parts after checking etags
  POST /hub/api/models/<repo>/commit/main       checks every LFS oid, writes the tree

Usage: hub_server.py <output-directory>

Binds 127.0.0.1 on a free port and prints "PORT <n>"; the Hub is served under
/hub (HF_ENDPOINT=http://127.0.0.1:<n>/hub) and the storage under /storage, so
a token sent to the storage URLs is caught. Committed files are
written to <output-directory>/<owner>/<name>/<path>; every request is logged
to <output-directory>/requests.log as "<method> <path> <status>". Requests
without "authorization: Bearer ..." to the API are refused with 401.
"""
import base64
import hashlib
import http.server
import json
import os
import sys

OUT = os.path.abspath(sys.argv[1])
CHUNK = 64 * 1024
repos = set()
objects = {}  # oid -> bytes
parts = {}  # oid -> {part: bytes}
failed_once = set()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def reply(self, status, body=b"", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        with open(os.path.join(OUT, "requests.log"), "a") as f:
            f.write(f"{self.command} {self.path} {status}\n")

    def body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n)

    def host(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def base(self):
        return self.host() + "/hub"

    def authorized(self):
        return (self.headers.get("Authorization") or "").startswith("Bearer ")

    def do_PUT(self):
        data = self.body()
        p = self.path.strip("/").split("/")
        if p[0] != "storage" or self.headers.get("Authorization"):
            return self.reply(400, b"storage URLs are signed; no token expected")
        oid = p[1]
        if len(p) == 3:
            if oid not in failed_once:
                failed_once.add(oid)
                return self.reply(503, b"slow down", {"Retry-After": "0"})
            parts.setdefault(oid, {})[int(p[2])] = data
            return self.reply(200, b"", {"ETag": f'"etag-{p[2]}"'})
        if hashlib.sha256(data).hexdigest() != oid:
            return self.reply(400, b"hash mismatch")
        objects[oid] = data
        return self.reply(200)

    def do_POST(self):
        data = self.body()
        path = self.path.split("?")[0]
        if not path.startswith("/hub/"):
            return self.reply(404, {"error": "not found"})
        path = path[len("/hub"):]
        if not self.authorized():
            return self.reply(401, {"error": "Invalid credentials"})
        if path == "/api/repos/create":
            req = json.loads(data)
            if req.get("type") != "model":
                return self.reply(400, {"error": "bad type"})
            rid = f"{req['organization']}/{req['name']}"
            if rid in repos:
                return self.reply(409, {"error": "You already created this model repo"})
            repos.add(rid)
            return self.reply(201, {"url": f"{self.base()}/{rid}"})
        if path.endswith("/preupload/main"):
            req = json.loads(data)
            files = []
            for f in req["files"]:
                base64.b64decode(f["sample"])
                lfs = f["path"].endswith((".safetensors", ".gguf", ".bin"))
                files.append({"path": f["path"], "uploadMode": "lfs" if lfs else "regular", "shouldIgnore": False})
            return self.reply(200, {"files": files})
        if path.endswith(".git/info/lfs/objects/batch"):
            req = json.loads(data)
            if req["operation"] != "upload" or req["hash_algo"] != "sha256":
                return self.reply(400, {"message": "bad batch"})
            out = []
            for o in req["objects"]:
                oid, size = o["oid"], o["size"]
                if oid in objects:
                    out.append({"oid": oid, "size": size})
                elif size >= CHUNK:
                    n = (size + CHUNK - 1) // CHUNK
                    header = {"chunk_size": str(CHUNK)}
                    for i in range(1, n + 1):
                        header[f"{i:05d}"] = f"{self.host()}/storage/{oid}/{i}"
                    out.append({"oid": oid, "size": size, "actions": {"upload": {"href": f"{self.base()}/complete/{oid}", "header": header}}})
                else:
                    out.append({"oid": oid, "size": size, "actions": {"upload": {"href": f"{self.host()}/storage/{oid}"}}})
            return self.reply(200, {"transfer": "basic", "objects": out}, {"Content-Type": "application/vnd.git-lfs+json"})
        if path.startswith("/complete/"):
            oid = path.split("/")[2]
            req = json.loads(data)
            got = parts.get(oid, {})
            for p in req["parts"]:
                if p["etag"] != f'"etag-{p["partNumber"]}"':
                    return self.reply(400, {"error": "bad etag"})
            blob = b"".join(got[i] for i in sorted(got))
            if hashlib.sha256(blob).hexdigest() != req["oid"]:
                return self.reply(400, {"error": "hash mismatch"})
            objects[oid] = blob
            return self.reply(200, {})
        if path.endswith("/commit/main"):
            rid = path[len("/api/models/"):-len("/commit/main")]
            if rid not in repos:
                return self.reply(404, {"error": "Repository not found"})
            if self.headers.get("Content-Type") != "application/x-ndjson":
                return self.reply(400, {"error": "expected ndjson"})
            lines = [json.loads(l) for l in data.decode().splitlines() if l]
            if not lines or lines[0]["key"] != "header" or not lines[0]["value"]["summary"]:
                return self.reply(400, {"error": "missing header"})
            for l in lines[1:]:
                v = l["value"]
                if l["key"] == "lfsFile":
                    if v["oid"] not in objects or len(objects[v["oid"]]) != v["size"]:
                        return self.reply(400, {"error": f"unknown LFS object for {v['path']}"})
                    content = objects[v["oid"]]
                elif l["key"] == "file":
                    content = base64.b64decode(v["content"])
                else:
                    return self.reply(400, {"error": "unexpected key"})
                dest = os.path.join(OUT, rid, v["path"])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(content)
            return self.reply(200, {"commitUrl": f"{self.base()}/{rid}/commit/1", "commitOid": "1"})
        return self.reply(404, {"error": "not found"})


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(f"PORT {server.server_port}", flush=True)
server.serve_forever()
