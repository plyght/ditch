"""Reads a checkpoint written by `truncate_checkpoint.py --lazy`, filling holes on demand.

    store = LazyCheckpoint(model_dir)
    t = store.tensor("layers.3.ffn.experts.17.w1.weight")      # whole tensor, torch
    rows = store.rows("layers.1.engram.embed.weight", idx)      # selected rows only

A tensor that `truncate_checkpoint.py` left as a hole of the sparse file is
fetched from the Hub the first time it is read (a table row by row) and
written into its place in the file, and `lazy.json` records it as filled. So
a reference run leaves behind a file that holds exactly the tensors and rows
it read, which is then what ditch reads. Tensors that are not lazy are read
from the file as they are.
"""
import json
import os
import struct
import time

import numpy as np
import requests
import torch

DTYPES = {
    "F32": (np.float32, torch.float32), "BF16": (np.uint16, torch.bfloat16), "F16": (np.float16, torch.float16),
    "I64": (np.int64, torch.int64), "I32": (np.int32, torch.int32), "I8": (np.int8, torch.int8),
    "U8": (np.uint8, torch.uint8), "F8_E4M3": (np.uint8, torch.float8_e4m3fn), "F8_E8M0": (np.uint8, torch.float8_e8m0fnu),
}


class LazyCheckpoint:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.header, self.where = {}, {}  # name -> info; name -> (fd, base)
        self.fds = {}
        for fn in sorted(os.listdir(model_dir)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(model_dir, fn)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            fd = os.open(path, os.O_RDWR)
            self.fds[fn] = fd
            for k, v in h.items():
                self.header[k] = v
                self.where[k] = (fd, 8 + n)
        lazy_path = os.path.join(model_dir, "lazy.json")
        self.lazy = json.load(open(lazy_path)) if os.path.exists(lazy_path) else {"holes": {}, "filled": {}}
        self.lazy_path = lazy_path
        self.session = requests.Session()
        self.dirty = 0

    def keys(self):
        return self.header.keys()

    def _save(self):
        if self.lazy["holes"]:
            json.dump(self.lazy, open(self.lazy_path, "w"))

    def _fill(self, name, start, length):
        """Fetches bytes [start, start + length) of lazy tensor `name` unless already filled."""
        h = self.lazy["holes"][name]
        done = self.lazy["filled"].setdefault(name, [])
        key = [start, length]
        if key in done or [0, h["bytes"]] in done:
            return
        a = h["src"] + start
        for attempt in range(8):
            try:
                data = self.session.get(h["url"], headers={"Range": f"bytes={a}-{a + length - 1}"}, timeout=300).content
                if len(data) == length:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"could not fetch {name} [{start}, {start + length})")
        fd = self.where[name][0]
        os.pwrite(fd, data, h["dst"] + start)
        done.append(key)
        self.dirty += 1
        if self.dirty % 64 == 0:
            self._save()

    def flush(self):
        self._save()

    def _read(self, name, start, length):
        info = self.header[name]
        if name in self.lazy["holes"]:
            self._fill(name, start, length)
        fd, base = self.where[name]
        return os.pread(fd, length, base + info["data_offsets"][0] + start)

    def tensor(self, name):
        info = self.header[name]
        a, b = info["data_offsets"]
        npdt, tdt = DTYPES[info["dtype"]]
        raw = np.frombuffer(self._read(name, 0, b - a), dtype=npdt).copy()
        return torch.from_numpy(raw).view(tdt).reshape(info["shape"])

    def rows(self, name, idx):
        """Rows `idx` (1-D, int) of a 2-D tensor, fetching only those rows of a lazy one."""
        info = self.header[name]
        a, b = info["data_offsets"]
        npdt, tdt = DTYPES[info["dtype"]]
        row_bytes = (b - a) // info["shape"][0]
        out = np.empty((len(idx), row_bytes), np.uint8)
        for i, r in enumerate(np.asarray(idx).tolist()):
            out[i] = np.frombuffer(self._read(name, r * row_bytes, row_bytes), np.uint8)
        return torch.from_numpy(out.view(npdt).copy()).view(tdt).reshape(len(idx), *info["shape"][1:])
