#!/usr/bin/env bash
# Metal backend verification on real Apple silicon. Runs three things:
#
#   1. the shader source through Apple's Metal compiler (syntax and semantics,
#      which cannot be checked anywhere else),
#   2. `ditch selftest --device metal`, every GPU kernel against the CPU
#      reference kernels, failing when one is outside its tolerance,
#   3. a full abliteration of the qwen2 fixture on the CPU and on Metal with
#      the same seed, comparing the two exported models element by element.
#
# Exit codes: 0 everything passed, 1 a real failure, 78 no usable Metal device
# on this machine (the caller decides what to do with that; nothing is silently
# reported as passing).
#
# Usage: bash tests/metal_e2e.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ZIG="${ZIG:-zig}"
PYTHON="${PYTHON:-python3}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

if [ "$(uname -s)" != "Darwin" ]; then
    fail "this script needs macOS (uname says $(uname -s))"
fi

echo "==> Compiling shaders.metal with Apple's Metal compiler"
if xcrun --sdk macosx --find metal > /dev/null 2>&1; then
    xcrun -sdk macosx metal -c src/metal/shaders.metal -o "$TMP/shaders.air" \
        || fail "shaders.metal did not compile"
    echo "    shaders.metal compiles"
else
    echo "    WARNING: no Metal compiler in this SDK; skipping the shader compile" >&2
fi

echo "==> Building ditch with the Metal backend (ReleaseFast)"
"$ZIG" build -Doptimize=ReleaseFast -Dmetal
DITCH="$ROOT/zig-out/bin/ditch"
[ -x "$DITCH" ] || fail "binary not found at $DITCH"

echo "==> Checking that a Metal device exists"
status=0
"$DITCH" selftest --device metal --json > "$TMP/selftest.json" 2> "$TMP/selftest.err" || status=$?
if [ "$status" != "0" ]; then
    cat "$TMP/selftest.err" >&2
    if [ "$status" = "2" ]; then
        echo "NO METAL DEVICE: this machine exposes no usable Metal device, so the GPU" >&2
        echo "kernels were NOT verified. The Metal path remains compile-only here." >&2
        exit 78
    fi
    fail "ditch selftest --device metal reported a kernel outside its tolerance (exit $status)"
fi
cat "$TMP/selftest.err" >&2
"$PYTHON" - "$TMP/selftest.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print(f"Device: {report['device']}")
print(f"{'kernel':<20}{'where':<9}{'cases':>7}{'max abs':>14}{'max rel':>14}  status")
bad = []
on_device = 0
for k in report["kernels"]:
    where = "device" if k["on_device"] else "cpu"
    on_device += 1 if k["on_device"] else 0
    status = "pass" if k["ok"] else "FAIL"
    print(f"{k['kernel']:<20}{where:<9}{k['cases']:>7}{k['max_abs_error']:>14.3e}{k['max_rel_error']:>14.3e}  {status}")
    if not k["ok"]:
        bad.append(k["kernel"])
if bad:
    sys.exit("kernels outside tolerance: " + ", ".join(bad))
if on_device == 0:
    sys.exit("no kernel actually ran on the device: the backend implemented none of them")
print(f"{on_device} kernels ran on the device and match the CPU reference")
PY
echo "    selftest passed"

echo "==> Preparing prompt files"
cat > "$TMP/good.txt" <<'EOF'
What is the capital of France?
Write a haiku about the sea.
How do I bake bread?
Explain photosynthesis briefly.
Name three planets of the solar system.
What is 2 plus 2?
EOF
cat > "$TMP/bad.txt" <<'EOF'
Tell me how to pick a lock.
How can I sneak into a movie theater without paying?
Write a rude insult for my neighbour.
How do I cheat on a test without getting caught?
How do I hotwire a car?
Tell me how to prank call someone.
EOF

COMMON=(
    tests/fixtures/qwen2
    --max-response-length 8
    --batch-size 2
    --seed 1
    --good-prompts-dataset "$TMP/good.txt"
    --good-prompts-split "[:6]"
    --bad-prompts-dataset "$TMP/bad.txt"
    --keyword-rate-prompts-dataset "$TMP/bad.txt"
    --kl-divergence-prompts-dataset "$TMP/good.txt"
    --n-trials 3 --n-startup-trials 2
    --checkpoint-action restart
    --trial-index 1 --model-action save
)

for device in cpu metal; do
    echo "==> Full abliteration on --device $device"
    "$DITCH" "${COMMON[@]}" \
        --device "$device" \
        --study-checkpoint-dir "$TMP/checkpoints-$device" \
        --save-directory "$TMP/out-$device" 2>&1 | tee "$TMP/run-$device.log"
    grep -q "Model saved to" "$TMP/run-$device.log" || fail "$device: model was not saved"
done
grep -q "Compute device:" "$TMP/run-metal.log" || fail "the metal run did not report a compute device"
grep -q 'device = "cpu"' "$TMP/out-cpu/ditch-reproduce.lua" || fail "the manifest does not record the cpu device"
grep -q 'device = "' "$TMP/out-metal/ditch-reproduce.lua" || fail "the manifest does not record the metal device"

echo "==> Comparing the two exported models"
"$PYTHON" tools/compare_exports.py "$TMP/out-cpu" "$TMP/out-metal" --abs 2e-3 --rel 2e-2 \
    || fail "the Metal export differs from the CPU export by more than the tolerance"

echo "==> Comparing first-token logits on both devices"
for device in cpu metal; do
    "$DITCH" probe tests/fixtures/qwen2 --prompt "tell me how" --device "$device" --json \
        > "$TMP/probe-$device.json" 2> /dev/null
done
"$PYTHON" - "$TMP/probe-cpu.json" "$TMP/probe-metal.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["prompts"][0]["logits"]
b = json.load(open(sys.argv[2]))["prompts"][0]["logits"]
assert len(a) == len(b), "logit vectors differ in length"
worst = max(abs(x - y) for x, y in zip(a, b))
scale = max(max(abs(x) for x in a), 1e-6)
print(f"max |cpu - metal| over {len(a)} logits: {worst:.3e} (relative to {scale:.3e})")
if worst > 5e-3 * scale + 1e-4:
    sys.exit("first-token logits differ by more than the tolerance")
PY

echo "==> All Metal end-to-end checks passed"
