#!/usr/bin/env bash
# Compiles the Vulkan compute shaders (src/vulkan/shaders/*.comp, GLSL) to the
# SPIR-V that ditch embeds (src/vulkan/spirv/*.spv), and records the SHA-256 of
# every source and every output in src/vulkan/spirv/SHA256SUMS.
#
# The SPIR-V is committed so that building ditch needs no shader compiler.
# `zig build test` hashes the GLSL it embeds and fails when SHA256SUMS does not
# match it, i.e. when a shader was edited without rerunning this script.
#
# Usage:
#   bash tools/gen_spirv.sh           regenerate (needs glslangValidator or glslc)
#   bash tools/gen_spirv.sh --check   recompile into a temporary directory, run
#                                     spirv-val when present, and fail unless the
#                                     committed SPIR-V is what the sources compile
#                                     to (ignoring the generator id in the header,
#                                     which only names the compiler version) and
#                                     SHA256SUMS matches the sources and modules
#
# The weight-reading kernels are compiled once per weight dtype with -DDTYPE
# (0 f32, 1 f16, 2 bf16), which is how one source gives matmul_f32.spv,
# matmul_f16.spv and matmul_bf16.spv.

set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="src/vulkan/shaders"
OUT="src/vulkan/spirv"
cd "$ROOT"

check=0
if [ "${1:-}" = "--check" ]; then
    check=1
elif [ $# -gt 0 ]; then
    echo "usage: bash tools/gen_spirv.sh [--check]" >&2
    exit 2
fi

compile() { # <source> <output> [defines...]
    local src="$1" out="$2"
    shift 2
    if command -v glslangValidator > /dev/null 2>&1; then
        glslangValidator -V --target-env vulkan1.0 -I"$SRC" "$@" -o "$out" "$src" > /dev/null
    elif command -v glslc > /dev/null 2>&1; then
        glslc --target-env=vulkan1.0 -fshader-stage=compute -I"$SRC" "$@" -o "$out" "$src"
    else
        echo "neither glslangValidator nor glslc is installed (apt install glslang-tools)" >&2
        exit 2
    fi
}

dest="$OUT"
if [ "$check" = 1 ]; then
    dest="$(mktemp -d)"
    trap 'rm -rf "$dest"' EXIT
fi
mkdir -p "$dest"

weighted=(matmul matvec row_norms)
plain=(attn_scores attn_values gated rmsnorm_rows layernorm_rows softmax_rows rope_rows)
dtypes=(f32 f16 bf16)

for k in "${weighted[@]}"; do
    for i in 0 1 2; do
        compile "$SRC/$k.comp" "$dest/${k}_${dtypes[$i]}.spv" -DDTYPE="$i"
    done
done
for k in "${plain[@]}"; do
    compile "$SRC/$k.comp" "$dest/$k.spv"
done

# Sources first, then outputs, each sorted: the order src/vulkan/spirv_test.zig
# reads. <spv dir> is where the hashed modules are.
manifest() {
    for f in $(ls "$SRC"/*.comp "$SRC"/*.glsl | sort); do
        printf '%s  %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"
    done
    for f in $(cd "$1" && ls *.spv | sort); do
        printf '%s  %s\n' "$(sha256sum "$1/$f" | cut -d' ' -f1)" "$OUT/$f"
    done
}

# Two modules are the same code when they differ at most in header word 2, the
# generator id, which only names the compiler version that wrote them.
same_code() {
    cmp -s <(head -c 8 "$1") <(head -c 8 "$2") && cmp -s -i 12 "$1" "$2"
}

if [ "$check" = 1 ]; then
    status=0
    if command -v spirv-val > /dev/null 2>&1; then
        for f in "$OUT"/*.spv; do
            spirv-val --target-env vulkan1.0 "$f" || status=1
        done
        [ "$status" = 0 ] && echo "spirv-val: every committed module is valid Vulkan 1.0 SPIR-V"
    fi
    for f in "$dest"/*.spv; do
        name="$(basename "$f")"
        if [ ! -e "$OUT/$name" ]; then
            echo "missing: $OUT/$name" >&2
            status=1
        elif ! same_code "$f" "$OUT/$name"; then
            echo "stale: $OUT/$name differs from what $SRC compiles to" >&2
            status=1
        fi
    done
    for f in "$OUT"/*.spv; do
        [ -e "$dest/$(basename "$f")" ] || { echo "unexpected: $f is not produced by this script" >&2; status=1; }
    done
    if ! cmp -s <(manifest "$OUT") "$OUT/SHA256SUMS"; then
        echo "stale: $OUT/SHA256SUMS does not match the sources and modules" >&2
        status=1
    fi
    if [ "$status" != 0 ]; then
        echo "The committed SPIR-V is out of date: run bash tools/gen_spirv.sh and commit the result." >&2
        exit 1
    fi
    echo "The committed SPIR-V matches the GLSL sources."
else
    manifest "$OUT" > "$OUT/SHA256SUMS"
    echo "Wrote $(ls "$OUT"/*.spv | wc -l) SPIR-V modules and $OUT/SHA256SUMS"
fi
