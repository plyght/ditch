#!/usr/bin/env python3
"""Compare two exported models tensor by tensor.

Used to check that an abliteration run on a GPU backend produces the same
edited weights as the CPU run it is supposed to reproduce (see
tests/metal_e2e.sh), but it is useful for any two safetensors exports with the
same tensor names and shapes.

    python3 tools/compare_exports.py OUT_A OUT_B [--abs 2e-3] [--rel 2e-2]

An element passes when it is within the absolute *or* the relative tolerance.
The report prints, per tensor, the largest absolute and relative deviation and
the number of elements outside both tolerances; the exit code is 1 if any
tensor has one, 2 for a structural difference (missing tensor, different shape
or dtype). Only the standard library is used: F32, F16 and BF16 are decoded
here, other dtypes are compared byte for byte.
"""

import argparse
import json
import os
import struct
import sys

HEADER_LEN = 8


def read_safetensors(path):
    """{name: (dtype, shape, raw bytes)} for one .safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(HEADER_LEN))[0]
        header = json.loads(f.read(n))
        body = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out[name] = (info["dtype"], tuple(info["shape"]), body[start:end])
    return out


def load_dir(path):
    """Every tensor of an export, across shards."""
    tensors = {}
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    if not files:
        sys.exit(f"{path}: no .safetensors files found")
    for name in files:
        tensors.update(read_safetensors(os.path.join(path, name)))
    return tensors


def decode(dtype, raw):
    """Floats of a tensor, or None for a dtype this script does not decode."""
    if dtype == "F32":
        return struct.unpack(f"<{len(raw) // 4}f", raw)
    if dtype == "F16":
        return struct.unpack(f"<{len(raw) // 2}e", raw)
    if dtype == "BF16":
        words = struct.unpack(f"<{len(raw) // 2}H", raw)
        return struct.unpack(f"<{len(words)}f", b"".join(struct.pack("<I", w << 16) for w in words))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--abs", type=float, default=2e-3, dest="abs_tol")
    ap.add_argument("--rel", type=float, default=2e-2, dest="rel_tol")
    ap.add_argument("--quiet", action="store_true", help="only print tensors that differ")
    args = ap.parse_args()

    ta = load_dir(args.a)
    tb = load_dir(args.b)
    if set(ta) != set(tb):
        only_a = sorted(set(ta) - set(tb))
        only_b = sorted(set(tb) - set(ta))
        print(f"tensor sets differ: only in {args.a}: {only_a[:5]}, only in {args.b}: {only_b[:5]}")
        return 2

    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    failed = []
    for name in sorted(ta):
        dta, sha, ra = ta[name]
        dtb, shb, rb = tb[name]
        if dta != dtb or sha != shb:
            print(f"{name}: {dta}{list(sha)} vs {dtb}{list(shb)}")
            return 2
        va, vb = decode(dta, ra), decode(dtb, rb)
        if va is None:
            if ra != rb:
                print(f"{name}: {dta} tensors differ byte for byte")
                failed.append(name)
            continue
        max_abs = max_rel = 0.0
        outside = 0
        for x, y in zip(va, vb):
            d = abs(x - y)
            if d == 0.0:
                continue
            r = d / max(abs(y), 1e-30)
            max_abs = max(max_abs, d)
            if d > args.abs_tol:
                max_rel = max(max_rel, r)
                if r > args.rel_tol:
                    outside += 1
        if max_abs > worst_abs:
            worst_abs, worst_name = max_abs, name
        worst_rel = max(worst_rel, max_rel)
        if outside:
            failed.append(name)
            print(f"{name}: {outside} of {len(va)} elements outside tolerance "
                  f"(max abs {max_abs:.3e}, max rel {max_rel:.3e})")
        elif not args.quiet:
            print(f"{name}: ok (max abs {max_abs:.3e}, max rel {max_rel:.3e})")

    print(f"\n{len(ta)} tensors compared; largest absolute deviation {worst_abs:.3e} "
          f"({worst_name}), largest relative deviation {worst_rel:.3e}")
    print(f"tolerance: {args.abs_tol:.1e} absolute or {args.rel_tol:.1e} relative")
    if failed:
        print(f"OUTSIDE TOLERANCE: {len(failed)} tensor(s): {failed[:10]}")
        return 1
    print("every tensor is within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
