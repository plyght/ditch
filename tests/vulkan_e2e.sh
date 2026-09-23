#!/usr/bin/env bash
# Vulkan backend verification. Runs on any machine with a Vulkan driver: a
# real NVIDIA, AMD or Intel GPU, or Mesa's lavapipe (a software Vulkan device;
# apt install mesa-vulkan-drivers libvulkan1), which is how CI runs it without
# a GPU. lavapipe checks correctness only, never speed.
#
#   1. the committed SPIR-V against the GLSL sources (when glslangValidator or
#      glslc is installed),
#   2. `ditch selftest --device vulkan`, every kernel against the CPU reference,
#      on the device's own memory path and again forced through the staging
#      path discrete GPUs use (DITCH_VULKAN_STAGING=1),
#   3. a full abliteration of the qwen2 fixture on the CPU and on Vulkan with
#      the same seed, with and without --gpu-memory, comparing the exported
#      models element by element,
#   4. first-token logits on a subset of fixture architectures, CPU against
#      Vulkan.
#
# Exit codes: 0 everything passed, 1 a real failure, 78 no usable Vulkan
# device on this machine (the caller decides what to do with that; nothing is
# silently reported as passing).
#
# Usage: bash tests/vulkan_e2e.sh

set -euo pipefail
export LC_ALL=C

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

echo "==> Checking the committed SPIR-V against the GLSL sources"
if command -v glslangValidator > /dev/null 2>&1 || command -v glslc > /dev/null 2>&1; then
    bash tools/gen_spirv.sh --check || fail "the committed SPIR-V is stale"
else
    echo "    WARNING: no glslangValidator or glslc; skipping the shader compile" >&2
fi

echo "==> Building ditch (ReleaseFast)"
"$ZIG" build -Doptimize=ReleaseFast
DITCH="$ROOT/zig-out/bin/ditch"
[ -x "$DITCH" ] || fail "binary not found at $DITCH"

selftest() { # <label> [env...]
    local label="$1"
    shift
    echo "==> Selftest: $label"
    local status=0
    env "$@" "$DITCH" selftest --device vulkan --json > "$TMP/selftest-$label.json" 2> "$TMP/selftest-$label.err" || status=$?
    if [ "$status" != "0" ]; then
        cat "$TMP/selftest-$label.err" >&2
        if [ "$status" = "2" ]; then
            echo "NO VULKAN DEVICE: this machine exposes no usable Vulkan device, so the" >&2
            echo "kernels were NOT verified." >&2
            exit 78
        fi
        fail "ditch selftest --device vulkan ($label) reported a kernel outside its tolerance (exit $status)"
    fi
    "$PYTHON" - "$TMP/selftest-$label.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print(f"Device: {report['device']}")
print(f"        {report.get('device_info', '')}")
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
if on_device != len(report["kernels"]):
    sys.exit(f"only {on_device} of {len(report['kernels'])} kernels ran on the device")
print(f"{on_device} kernels ran on the device and match the CPU reference")
PY
}
selftest native
selftest staged DITCH_VULKAN_STAGING=1

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

# The fixture's matrices are far below the size at which ditch hands a product
# to the GPU, so without this the "vulkan" run would do all its arithmetic on
# the CPU and the comparison below would pass trivially. Zero sends every
# weight-tile product to the device.
export DITCH_DEVICE_MIN_MACS=0

run() { # <label> <device> [extra args...]
    local label="$1" device="$2"
    shift 2
    echo "==> Full abliteration: $label"
    "$DITCH" "${COMMON[@]}" \
        --device "$device" "$@" \
        --study-checkpoint-dir "$TMP/checkpoints-$label" \
        --save-directory "$TMP/out-$label" 2>&1 | tee "$TMP/run-$label.log"
    grep -q "Model saved to" "$TMP/run-$label.log" || fail "$label: model was not saved"
    # The export is reloaded and compared with the in-memory model; with
    # --gpu-memory that reload is where a stale resident tile would show.
    if grep -q "does not reproduce the abliterated model" "$TMP/run-$label.log"; then
        fail "$label: the exported model does not reproduce the abliterated model"
    fi
}
run cpu cpu
run vulkan vulkan
run vulkan-resident vulkan --gpu-memory 256MB

for label in vulkan vulkan-resident; do
    grep -q "Compute device: .*Vulkan" "$TMP/run-$label.log" || fail "the $label run did not report a Vulkan compute device"
    served=$(sed -n 's/^\([0-9][0-9]*\) matrix products ran on .*/\1/p' "$TMP/run-$label.log" | tail -1)
    [ -n "$served" ] || fail "the $label run did not report how many products the device served"
    [ "$served" -gt 0 ] || fail "the $label run served 0 matrix products on the device: the GPU did no work"
    echo "    $label: $served matrix products ran on the device"
    grep -q 'device = ".*Vulkan' "$TMP/out-$label/ditch-reproduce.lua" || fail "the $label manifest does not record the Vulkan device"
done
grep -q 'device = "cpu"' "$TMP/out-cpu/ditch-reproduce.lua" || fail "the manifest does not record the cpu device"

echo "==> Comparing the exported models"
for label in vulkan vulkan-resident; do
    "$PYTHON" tools/compare_exports.py "$TMP/out-cpu" "$TMP/out-$label" --abs 2e-3 --rel 2e-2 \
        || fail "the $label export differs from the CPU export by more than the tolerance"
done

echo "==> Comparing first-token logits on both devices"
# One fixture per family of kernels the forward pass exercises: rotary
# attention with RMSNorm, LayerNorm with learned positions, gemma's (1 + w)
# norms and gelu, a mixture of experts, ALiBi and a fused-QKV layout.
PROBE_FIXTURES="${VULKAN_PROBE_FIXTURES:-qwen2 llama gemma3 gpt2 phi3 qwen3_moe mixtral bloom gpt_neox olmo2}"
for fx in $PROBE_FIXTURES; do
    for device in cpu vulkan; do
        "$DITCH" probe "tests/fixtures/$fx" --prompt "tell me how" --device "$device" --json \
            > "$TMP/probe-$fx-$device.json" 2> "$TMP/probe-$fx-$device.err" \
            || { cat "$TMP/probe-$fx-$device.err" >&2; fail "probe of $fx on $device failed"; }
    done
    "$PYTHON" - "$fx" "$TMP/probe-$fx-cpu.json" "$TMP/probe-$fx-vulkan.json" <<'PY'
import json, sys
fx = sys.argv[1]
a = json.load(open(sys.argv[2]))["prompts"][0]["logits"]
b = json.load(open(sys.argv[3]))["prompts"][0]["logits"]
assert len(a) == len(b), "logit vectors differ in length"
worst = max(abs(x - y) for x, y in zip(a, b))
scale = max(max(abs(x) for x in a), 1e-6)
print(f"    {fx:<12} max |cpu - vulkan| over {len(a)} logits: {worst:.3e} (relative to {scale:.3e})")
if worst > 5e-3 * scale + 1e-4:
    sys.exit(f"{fx}: first-token logits differ by more than the tolerance")
PY
done

echo "==> All Vulkan end-to-end checks passed"
