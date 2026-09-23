#!/usr/bin/env bash
# Finds Hugging Face models ditch has not been checked against (model types
# added in the latest transformers release, and trending text-generation
# models), then runs `ditch verify` on each on a truncated checkpoint.
#
# Usage:
#   bash tools/new_models.sh --dry            list the candidates (one Hub id per line) and stop
#   bash tools/new_models.sh [OPTIONS]        list, check support, verify, write the report
#
# Options (each also settable through the environment variable in brackets):
#   --dry                 only print the candidates
#   --models "A B"        check these Hub ids instead of discovering candidates [NEW_MODELS]
#   --max N               at most N candidates (default 12) [NEW_MODELS_MAX]
#   --k N                 layers kept in the truncated checkpoint (default 4) [NEW_MODELS_K]
#   --out DIR             results directory (default results) [NEW_MODELS_OUT]
#   --ditch PATH          ditch binary (default zig-out/bin/ditch) [DITCH]
#   --timeout MIN         per-model time limit in minutes (default 40) [NEW_MODELS_TIMEOUT]
#   --budget MIN          total time for all models in minutes (default 300) [NEW_MODELS_BUDGET]
#   --max-size GB         skip models whose truncated checkpoint would exceed this (default 6) [NEW_MODELS_MAX_SIZE]
#   --add-model           try `ditch add-model` on unsupported models (default: on under GitHub Actions)
#   --no-add-model        never try it
#
# GITHUB_TOKEN (or GH_TOKEN) raises the GitHub API rate limit; HF_TOKEN is sent
# to the Hub when set. Needs bash, curl and jq.
#
# Output of a full run in the results directory:
#   <owner>__<name>.json   ditch verify --json output
#   <owner>__<name>.log    stderr of the dry run and of verify
#   candidates.tsv         id, model_type, source of each candidate
#   summary.tsv            model, model_type, source, status, check, detail
#   summary.md             the same as a markdown table
#   issue.md               failures and unsupported models (absent when there are none)

set -uo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY=0
MODELS="${NEW_MODELS:-}"
MAX="${NEW_MODELS_MAX:-12}"
K="${NEW_MODELS_K:-4}"
OUT="${NEW_MODELS_OUT:-results}"
DITCH="${DITCH:-$ROOT/zig-out/bin/ditch}"
TIMEOUT_MIN="${NEW_MODELS_TIMEOUT:-40}"
BUDGET_MIN="${NEW_MODELS_BUDGET:-300}"
MAX_SIZE_GB="${NEW_MODELS_MAX_SIZE:-6}"
TRENDING_LIMIT="${NEW_MODELS_TRENDING:-30}"
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then ADD_MODEL=1; else ADD_MODEL=0; fi

while [ $# -gt 0 ]; do
    case "$1" in
        --dry) DRY=1 ;;
        --models) MODELS="$2"; shift ;;
        --models=*) MODELS="${1#*=}" ;;
        --max) MAX="$2"; shift ;;
        --max=*) MAX="${1#*=}" ;;
        --k) K="$2"; shift ;;
        --k=*) K="${1#*=}" ;;
        --out) OUT="$2"; shift ;;
        --out=*) OUT="${1#*=}" ;;
        --ditch) DITCH="$2"; shift ;;
        --ditch=*) DITCH="${1#*=}" ;;
        --timeout) TIMEOUT_MIN="$2"; shift ;;
        --timeout=*) TIMEOUT_MIN="${1#*=}" ;;
        --budget) BUDGET_MIN="$2"; shift ;;
        --budget=*) BUDGET_MIN="${1#*=}" ;;
        --max-size) MAX_SIZE_GB="$2"; shift ;;
        --max-size=*) MAX_SIZE_GB="${1#*=}" ;;
        --add-model) ADD_MODEL=1 ;;
        --no-add-model) ADD_MODEL=0 ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

for tool in curl jq; do
    command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 2; }
done

log() { echo "$*" >&2; }

HUB="https://huggingface.co"
GH_API="https://api.github.com"
GH_TOKEN_VALUE="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

gh_get() {
    local args=(-sSfL -g --retry 2 -H "Accept: application/vnd.github+json")
    [ -n "$GH_TOKEN_VALUE" ] && args+=(-H "Authorization: Bearer $GH_TOKEN_VALUE")
    curl "${args[@]}" "$GH_API$1"
}

hub_get() {
    local args=(-sSfL -g --retry 2)
    [ -n "${HF_TOKEN:-}" ] && args+=(-H "Authorization: Bearer $HF_TOKEN")
    curl "${args[@]}" "$HUB$1"
}

# Fields every Hub listing asks for, and the jq filters applied to them.
EXPAND='expand[]=safetensors&expand[]=gated&expand[]=tags&expand[]=library_name&expand[]=config'
JQ_DEFS='
def bytes: (.safetensors.parameters // {}) | to_entries
    | map(.value * ({"F64":8,"I64":8,"U64":8,"F32":4,"I32":4,"U32":4,"F16":2,"BF16":2,"I16":2,"U16":2}[.key] // 1))
    | add // 0;
def reupload:
    ((.library_name // "") | ascii_downcase | test("gguf|mlx|llama\\.cpp|onnx|exllama|ctranslate"))
    or ((.tags // []) | map(ascii_downcase) | any(test("^(gguf|mlx|awq|gptq|exl2|exl3|bitsandbytes|onnx|4-bit|8-bit)$")))
    or (.id | ascii_downcase | test("gguf|mlx|awq|gptq|exl2|exl3|bnb|[-_](int)?[48]-?bit|onnx"));
def usable: (.gated == false) and (.safetensors != null) and (bytes > 0) and (reupload | not);
'

# model_type names in CONFIG_MAPPING_NAMES of the auto mapping sources (the fallback when
# the GitHub API is not reachable).
config_mapping_types() {
    sed -n '/^CONFIG_MAPPING_NAMES = /,/^)/p' |
        sed -n 's/^ *("\([A-Za-z0-9_-]*\)", *"[A-Za-z0-9_]*"),*$/\1/p' | sort -u
}

# configuration_auto.py of a tag, preceded by auto_mappings.py where the tag
# has it (newer releases generate the mapping there).
auto_mapping_sources() {
    local raw="https://raw.githubusercontent.com/huggingface/transformers/$1/src/transformers/models/auto"
    curl -sSfL --retry 2 "$raw/auto_mappings.py" 2>/dev/null
    curl -sSfL --retry 2 "$raw/configuration_auto.py" 2>/dev/null
}

# Prints model_types present in the latest transformers release and absent
# from the release before it.
new_model_types() {
    local releases latest prev a b
    if releases=$(gh_get "/repos/huggingface/transformers/releases?per_page=2" 2>/dev/null) &&
        latest=$(jq -r '.[0].tag_name // empty' <<<"$releases") && [ -n "$latest" ] &&
        prev=$(jq -r '.[1].tag_name // empty' <<<"$releases") && [ -n "$prev" ] &&
        a=$(gh_get "/repos/huggingface/transformers/contents/src/transformers/models?ref=$prev" 2>/dev/null) &&
        b=$(gh_get "/repos/huggingface/transformers/contents/src/transformers/models?ref=$latest" 2>/dev/null); then
        log "transformers $prev -> $latest (models/ directory listing, GitHub API)"
        comm -13 <(jq -r '.[] | select(.type == "dir") | .name' <<<"$a" | sort -u) \
            <(jq -r '.[] | select(.type == "dir") | .name' <<<"$b" | sort -u)
        return 0
    fi
    # Without the GitHub API (rate limited or blocked): the two newest versions
    # on PyPI, and the model_type keys of each tag's auto mappings.
    local versions
    versions=$(curl -sSfL --retry 2 https://pypi.org/pypi/transformers/json 2>/dev/null |
        jq -r '[.releases | to_entries[] | select((.value | length) > 0)
                | select(.key | test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))
                | {k: .key, t: .value[0].upload_time}] | sort_by(.t) | .[-2:] | map(.k) | .[]') || true
    prev=$(sed -n 1p <<<"$versions")
    latest=$(sed -n 2p <<<"$versions")
    if [ -z "$prev" ] || [ -z "$latest" ]; then
        log "warning: could not determine the latest transformers releases"
        return 0
    fi
    a=$(auto_mapping_sources "v$prev")
    b=$(auto_mapping_sources "v$latest")
    if [ -z "$a" ] || [ -z "$b" ]; then
        log "warning: could not fetch the auto mappings for v$prev and v$latest"
        return 0
    fi
    log "transformers v$prev -> v$latest (CONFIG_MAPPING_NAMES from PyPI versions and raw files; GitHub API not reachable)"
    comm -13 <(config_mapping_types <<<"$a") <(config_mapping_types <<<"$b")
}

# The smallest usable text-generation checkpoint of a model_type, as
# "id<TAB>model_type<TAB>bytes".
representative() {
    local mt="$1" list pick
    list=$(hub_get "/api/models?filter=$mt&pipeline_tag=text-generation&sort=downloads&limit=5&$EXPAND" 2>/dev/null) || list="[]"
    pick=$(jq -r "$JQ_DEFS"' [.[] | select(usable)] | sort_by(bytes) | .[0] // empty
        | [.id, (.config.model_type // $mt), (bytes | tostring)] | @tsv' --arg mt "$mt" <<<"$list")
    if [ -z "$pick" ]; then
        list=$(hub_get "/api/models?search=$mt&pipeline_tag=text-generation&sort=downloads&limit=10&$EXPAND" 2>/dev/null) || list="[]"
        pick=$(jq -r "$JQ_DEFS"' [.[] | select(usable and .config.model_type == $mt)] | sort_by(bytes) | .[0] // empty
            | [.id, .config.model_type, (bytes | tostring)] | @tsv' --arg mt "$mt" <<<"$list")
    fi
    printf '%s' "$pick"
}

# Candidates as "id<TAB>model_type<TAB>source", deduplicated, at most $MAX.
collect_candidates() {
    local seen="" n=0 id mt src line
    emit() {
        [ "$n" -ge "$MAX" ] && return 1
        case " $seen " in *" $1 "*) return 0 ;; esac
        seen="$seen $1"
        n=$((n + 1))
        printf '%s\t%s\t%s\n' "$1" "${2:--}" "$3"
    }
    if [ -n "$MODELS" ]; then
        for id in $MODELS; do
            id="${id#hf://}"
            mt=$(hub_get "/api/models/$id?expand[]=config" 2>/dev/null | jq -r '.config.model_type // "-"') || mt="-"
            emit "$id" "$mt" "requested" || break
        done
        return 0
    fi

    local types
    types=$(new_model_types)
    if [ -n "$types" ]; then
        log "new model_types: $(tr '\n' ' ' <<<"$types")"
    else
        log "new model_types: none"
    fi
    for mt in $types; do
        line=$(representative "$mt")
        if [ -z "$line" ]; then
            log "  $mt: no public safetensors text-generation checkpoint on the Hub"
            continue
        fi
        IFS=$'\t' read -r id mt _ <<<"$line"
        log "  $mt: $id"
        emit "$id" "$mt" "new in transformers" || return 0
    done

    local trending
    trending=$(hub_get "/api/models?pipeline_tag=text-generation&sort=trendingScore&limit=$TRENDING_LIMIT&$EXPAND" 2>/dev/null) || {
        log "warning: the Hub trending listing failed"
        return 0
    }
    while IFS=$'\t' read -r id mt; do
        [ -n "$id" ] || continue
        emit "$id" "$mt" "trending" || return 0
    done < <(jq -r "$JQ_DEFS"' .[] | select(usable) | [.id, (.config.model_type // "-")] | @tsv' <<<"$trending")
}

CANDIDATES=$(collect_candidates)

if [ "$DRY" = 1 ]; then
    [ -n "$CANDIDATES" ] && cut -f1 <<<"$CANDIDATES"
    exit 0
fi

# ---------------------------------------------------------------------------
# Full run: support check, size estimate, verify, report.

[ -x "$DITCH" ] || { echo "error: ditch binary not found at $DITCH (zig build -Doptimize=ReleaseFast)" >&2; exit 2; }
DITCH="$(cd "$(dirname "$DITCH")" && pwd)/$(basename "$DITCH")"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
printf '%s\n' "$CANDIDATES" >"$OUT/candidates.tsv"
WORK_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/ditch-new-models"
mkdir -p "$WORK_ROOT"

HELP=$("$DITCH" help 2>&1)
grep -q 'ditch verify' <<<"$HELP" || { echo "error: $DITCH has no verify subcommand" >&2; exit 2; }
HAS_ADD_MODEL=0
grep -q 'add-model' <<<"$HELP" && HAS_ADD_MODEL=1

# verify options beyond the core set are passed only when the binary lists them.
VERIFY_HELP=$("$DITCH" verify --help 2>&1)
VERIFY_OPTS=(--json --kinds --max-ram 5GB)
if grep -q -- '--max-layers' <<<"$VERIFY_HELP"; then
    VERIFY_OPTS+=(--max-layers "$K")
else
    log "warning: ditch verify has no --max-layers; the cut is its --kinds default"
fi
grep -q -- '--no-input' <<<"$VERIFY_HELP" && VERIFY_OPTS+=(--no-input)
grep -q -- '--python' <<<"$VERIFY_HELP" && command -v python3 >/dev/null && VERIFY_OPTS+=(--python "$(command -v python3)")

MAX_BYTES=$(awk -v g="$MAX_SIZE_GB" 'BEGIN { printf "%.0f", g * 1024 * 1024 * 1024 }')
DEADLINE=$(($(date +%s) + BUDGET_MIN * 60))
SUMMARY="$OUT/summary.tsv"
printf 'model\tmodel_type\tsource\tstatus\tcheck\tdetail\n' >"$SUMMARY"

sanitise() {
    local s="${1/\//__}"
    printf '%s\n' "${s//[^A-Za-z0-9._-]/_}"
}

# "942.3MB" -> bytes.
to_bytes() {
    awk -v s="$1" 'BEGIN {
        n = s + 0; u = s; sub(/^[0-9.]+/, "", u)
        m = 1
        if (u == "KB") m = 1024; else if (u == "MB") m = 1024^2
        else if (u == "GB") m = 1024^3; else if (u == "TB") m = 1024^4
        printf "%.0f", n * m
    }'
}

human() { awk -v b="$1" 'BEGIN { printf "%.1fGB", b / 1024^3 }'; }

# The first informative error line of a log, for the report.
reason_line() {
    local r
    r=$(grep -m1 'unsupported model' "$1" 2>/dev/null)
    [ -n "$r" ] || r=$(grep -E 'error:|Error' "$1" 2>/dev/null | grep -v 'curl failed' | head -1)
    [ -n "$r" ] || r=$(grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -1)
    printf '%s' "${r:-no output}" | tr '\t' ' ' | cut -c1-300
}

record() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:--}" "$(printf '%s' "${6:--}" | tr '\t\n' '  ' | cut -c1-300)" >>"$SUMMARY"; }

# Runs the dry run; sets DRY_RC and fills $log. Exit 0 or 2 means the
# architecture loaded (2 is the memory budget refusing the full model).
dry_run() {
    local id="$1" work="$2" log="$3"
    (cd "$work" && DITCH_CACHE="$work/cache" timeout 15m "$DITCH" --dry-run "hf://$id" --max-ram 6GB \
        --remote-cache-size 1MB --no-input) >"$log.dry" 2>&1
    DRY_RC=$?
    cat "$log.dry" >>"$log"
}

supported() { { [ "$DRY_RC" = 0 ] || [ "$DRY_RC" = 2 ]; } && ! grep -q 'unsupported model' "$1"; }

# Truncated checkpoint size from the dry run's memory estimate: k layers of the
# per-layer average plus twice the largest tensor (the embedding and the head).
estimate_bytes() {
    local log="$1" id="$2" total layers largest cfg
    total=$(sed -n 's/^ *weights total *\([0-9.]*[KMGT]\{0,1\}B\) (\([0-9]*\) layers).*/\1 \2/p' "$log" | head -1)
    largest=$(sed -n 's/^ *largest tensor *\([0-9.]*[KMGT]\{0,1\}B\).*/\1/p' "$log" | head -1)
    if [ -n "$total" ]; then
        layers=${total#* }
        total=$(to_bytes "${total% *}")
        largest=$(to_bytes "${largest:-0B}")
    else
        # No estimate printed: safetensors size from the Hub over the layer count in config.json.
        total=$(hub_get "/api/models/$id?expand[]=safetensors" 2>/dev/null | jq -r "$JQ_DEFS"' bytes') || total=0
        cfg=$(hub_get "/$id/resolve/main/config.json" 2>/dev/null) || cfg="{}"
        layers=$(jq -r '.num_hidden_layers // .text_config.num_hidden_layers // .n_layer // 0' <<<"$cfg" 2>/dev/null) || layers=0
        largest=0
    fi
    if [ "${layers:-0}" -le 0 ] 2>/dev/null || [ "${total:-0}" -le 0 ] 2>/dev/null; then
        echo 0
        return
    fi
    local kk=$K
    [ "$kk" -gt "$layers" ] && kk=$layers
    awk -v t="$total" -v l="$layers" -v k="$kk" -v e="$largest" 'BEGIN { printf "%.0f", t / l * k + 2 * e }'
}

while IFS=$'\t' read -r id mt src; do
    [ -n "$id" ] || continue
    san=$(sanitise "$id")
    log="$OUT/$san.log"
    work="$WORK_ROOT/$san"
    rm -rf "$work"
    mkdir -p "$work"
    : >"$log"
    log "==> $id ($mt, $src)"

    now=$(date +%s)
    if [ "$now" -ge $((DEADLINE - 300)) ]; then
        record "$id" "$mt" "$src" "skipped" "-" "time budget of ${BUDGET_MIN} min used up"
        rm -rf "$work"
        continue
    fi

    dry_run "$id" "$work" "$log"
    arch=$(sed -n 's/^\* Architecture: \([^ ]*\).*/\1/p' "$log.dry" | head -1)
    [ -n "$arch" ] && mt="$arch"
    if ! supported "$log.dry" && [ "$HAS_ADD_MODEL" = 1 ] && [ "$ADD_MODEL" = 1 ]; then
        log "    unsupported; trying ditch add-model"
        if timeout 20m "$DITCH" add-model "$id" --no-input >>"$log" 2>&1; then
            if [ -e "$ROOT/.git" ] && [ -n "$(git -C "$ROOT" status --porcelain -- src 2>/dev/null)" ]; then
                (cd "$ROOT" && zig build -Doptimize=ReleaseFast) >>"$log" 2>&1
            fi
            dry_run "$id" "$work" "$log"
            arch=$(sed -n 's/^\* Architecture: \([^ ]*\).*/\1/p' "$log.dry" | head -1)
            [ -n "$arch" ] && mt="$arch"
        fi
    fi
    if ! supported "$log.dry"; then
        record "$id" "$mt" "$src" "unsupported" "-" "$(reason_line "$log.dry")"
        rm -rf "$work" "$log.dry"
        continue
    fi

    est=$(estimate_bytes "$log.dry" "$id")
    rm -f "$log.dry"
    if [ "$est" -gt "$MAX_BYTES" ]; then
        record "$id" "$mt" "$src" "skipped" "-" "truncated checkpoint ~$(human "$est") at k=$K exceeds ${MAX_SIZE_GB}GB"
        rm -rf "$work"
        continue
    fi

    left=$(((DEADLINE - $(date +%s)) / 60))
    limit=$TIMEOUT_MIN
    [ "$left" -lt "$limit" ] && limit=$left
    json="$OUT/$san.json"
    log "    verify: k=$K, estimate $(human "$est"), timeout ${limit}m"
    (cd "$work" && DITCH_CACHE="$work/cache" timeout "${limit}m" "$DITCH" verify "hf://$id" \
        "${VERIFY_OPTS[@]}" --work-dir "$work/verify") >"$json" 2>"$log.verify"
    rc=$?
    cat "$log.verify" >>"$log"

    if [ "$rc" = 124 ]; then
        record "$id" "$mt" "$src" "timeout" "-" "ditch verify did not finish in ${limit} min"
    elif ! jq -e '.checks' "$json" >/dev/null 2>&1; then
        record "$id" "$mt" "$src" "error" "-" "exit $rc, no JSON report: $(reason_line "$log.verify")"
    else
        first=$(jq -r '[.checks[] | select(.status == "fail")][0] // empty | [.name, (.detail // "")] | @tsv' "$json")
        if [ "$rc" = 0 ] && [ -z "$first" ] && [ "$(jq -r '.ok' "$json")" = true ]; then
            record "$id" "$mt" "$src" "pass" "-" "$(jq -r '[.checks[] | select(.status == "pass")] | length' "$json") checks passed"
        elif [ -n "$first" ]; then
            record "$id" "$mt" "$src" "fail" "${first%%$'\t'*}" "${first#*$'\t'}"
        elif [ "$rc" = 2 ]; then
            record "$id" "$mt" "$src" "unsupported" "-" "ditch verify: $(reason_line "$log.verify")"
        else
            record "$id" "$mt" "$src" "error" "-" "exit $rc: $(reason_line "$log.verify")"
        fi
    fi
    rm -rf "$work" "$log.verify"
done <<<"$CANDIDATES"
rm -rf "$WORK_ROOT"

# Markdown: the whole table, and the issue body with only the problems.
md_table() {
    echo "| Model | model_type | Source | Status | First failing check | Detail |"
    echo "|---|---|---|---|---|---|"
    awk -F'\t' -v filter="$1" 'NR > 1 && (filter == "" || $4 ~ filter) {
        for (i = 1; i <= NF; i++) gsub(/\|/, "\\|", $i)
        printf "| [%s](https://huggingface.co/%s) | %s | %s | %s | %s | %s |\n", $1, $1, $2, $3, $4, $5, $6
    }' "$SUMMARY"
}

total=$(($(wc -l <"$SUMMARY") - 1))
count() { awk -F'\t' -v s="$1" 'NR > 1 && $4 == s' "$SUMMARY" | wc -l | tr -d ' '; }
problems=$(awk -F'\t' 'NR > 1 && $4 ~ /^(fail|error|timeout|unsupported)$/' "$SUMMARY" | wc -l | tr -d ' ')
header="$total models, k=$K layers: $(count pass) passed, $(count fail) failed, $(count error) errors, $(count timeout) timed out, $(count unsupported) unsupported, $(count skipped) skipped."

{
    echo "## ditch verify on new and trending models"
    echo
    echo "$header"
    echo
    [ "$total" -gt 0 ] && md_table ""
} >"$OUT/summary.md"

rm -f "$OUT/issue.md"
if [ "$problems" -gt 0 ]; then
    {
        echo "\`ditch verify --kinds --max-layers $K\` on the weekly set of new and trending Hugging Face models."
        echo
        echo "$header"
        echo
        md_table '^(fail|error|timeout|unsupported)$'
        echo
        [ -n "${GITHUB_RUN_ID:-}" ] && echo "Run: ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/actions/runs/$GITHUB_RUN_ID (JSON results in the new-models-results artifact)"
    } >"$OUT/issue.md"
fi

cat "$OUT/summary.md"
exit 0
