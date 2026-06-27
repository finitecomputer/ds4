#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: tests/dflash_runtime_smoke.sh MODEL.gguf DFLASH_DIR [PROMPT]

Runs a tiny greedy baseline decode and a DFlash-enabled decode, then compares
stdout byte-for-byte. Set DS4_BIN, DS4_SMOKE_TOKENS, or DS4_SMOKE_CTX to
override the defaults.

Evidence is preserved in a fresh DS4_SMOKE_EVIDENCE_DIR, or in a fresh temp
directory when unset. Set DS4_SMOKE_MIN_VERIFIED to change the required number
of accepted DFlash draft tokens; the default is 1. The parsed smoke summary
must include attempt, draft, verify, accepted-anchor, rejection, and timing
counts.
USAGE
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=$1
dflash=$2
prompt=${3:-"Reply with one short sentence about local inference."}
bin=${DS4_BIN:-./ds4}
tokens=${DS4_SMOKE_TOKENS:-16}
ctx=${DS4_SMOKE_CTX:-2048}
min_verified=${DS4_SMOKE_MIN_VERIFIED:-1}
evidence_schema=ds4-dflash-runtime-smoke/v1
ds4_commit=unknown
if [[ -f .ds4-dflash-commit ]]; then
    ds4_commit=$(tr -d '\r\n' < .ds4-dflash-commit)
fi

case "$min_verified" in
    ''|*[!0-9]*)
        echo "DFlash smoke failed: DS4_SMOKE_MIN_VERIFIED must be an unsigned integer" >&2
        exit 2
        ;;
esac

if [[ -n "${DS4_SMOKE_EVIDENCE_DIR:-}" ]]; then
    evidence_dir=$DS4_SMOKE_EVIDENCE_DIR
    if [[ -e "$evidence_dir" && ! -d "$evidence_dir" ]]; then
        echo "DFlash smoke failed: DS4_SMOKE_EVIDENCE_DIR exists and is not a directory" >&2
        echo "Evidence: $evidence_dir" >&2
        exit 2
    fi
    mkdir -p "$evidence_dir"
    if [[ -n "$(find "$evidence_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "DFlash smoke failed: DS4_SMOKE_EVIDENCE_DIR must be empty before the smoke starts" >&2
        echo "Evidence: $evidence_dir" >&2
        exit 2
    fi
else
    evidence_dir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dflash-smoke.XXXXXX")
fi

base_out=$evidence_dir/baseline.out
base_err=$evidence_dir/baseline.err
dflash_out=$evidence_dir/dflash.out
dflash_err=$evidence_dir/dflash.err
diff_out=$evidence_dir/stdout.diff
summary=$evidence_dir/dflash-summary.env
metadata=$evidence_dir/metadata.txt

{
    printf 'evidence_schema=%s\n' "$evidence_schema"
    printf 'model=%s\n' "$model"
    printf 'dflash=%s\n' "$dflash"
    printf 'ds4_commit=%s\n' "$ds4_commit"
    printf 'bin=%s\n' "$bin"
    printf 'tokens=%s\n' "$tokens"
    printf 'ctx=%s\n' "$ctx"
    printf 'min_verified=%s\n' "$min_verified"
    printf 'prompt_file=%s\n' "$evidence_dir/prompt.txt"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$metadata"
printf '%s\n' "$prompt" >"$evidence_dir/prompt.txt"

"$bin" \
    -m "$model" \
    --nothink \
    --temp 0 \
    --tokens "$tokens" \
    --ctx "$ctx" \
    -p "$prompt" \
    >"$base_out" 2>"$base_err"

DS4_DFLASH_EXPERIMENTAL_RUN=1 \
DS4_DFLASH_SPEC_LOG=1 \
DS4_DFLASH_TIMING=1 \
"$bin" \
    -m "$model" \
    --dflash "$dflash" \
    --nothink \
    --temp 0 \
    --tokens "$tokens" \
    --ctx "$ctx" \
    -p "$prompt" \
    >"$dflash_out" 2>"$dflash_err"

awk -f "$script_dir/dflash_runtime_summary.awk" "$dflash_err" >"$summary"

summary_field() {
    local key=$1
    awk -F= -v key="$key" '$1 == key { print $2 }' "$summary"
}

require_uint() {
    local key=$1
    local value=$2
    case "$value" in
        ''|*[!0-9]*)
            echo "DFlash smoke failed: summary field $key must be an unsigned integer" >&2
            echo "Evidence: $evidence_dir" >&2
            echo "--- DFlash summary ---" >&2
            cat "$summary" >&2
            echo "--- DFlash stderr ---" >&2
            cat "$dflash_err" >&2
            exit 1
            ;;
    esac
}

attempts=$(awk -F= '$1 == "attempts" { print $2 }' "$summary")
drafted=$(awk -F= '$1 == "drafted" { print $2 }' "$summary")
verified=$(awk -F= '$1 == "verified" { print $2 }' "$summary")
accepted=$(summary_field accepted_including_anchor)
misses=$(summary_field misses)
rejected=$(summary_field rejected_draft_tokens)
timing_lines=$(summary_field timing_lines)

require_uint attempts "$attempts"
require_uint drafted "$drafted"
require_uint verified "$verified"
require_uint accepted_including_anchor "$accepted"
require_uint misses "$misses"
require_uint rejected_draft_tokens "$rejected"
require_uint timing_lines "$timing_lines"

if (( attempts <= 0 || drafted <= 0 )); then
    echo "DFlash smoke failed: DFlash verifier summary was not observed" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if (( timing_lines <= 0 )); then
    echo "DFlash smoke failed: DFlash timing summary was not observed" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if (( accepted < verified || accepted < attempts )); then
    echo "DFlash smoke failed: accepted-anchor count is inconsistent with verifier summary" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- DFlash summary ---" >&2
    cat "$summary" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if (( accepted > drafted + attempts )); then
    echo "DFlash smoke failed: accepted-anchor count exceeds drafted tokens plus verifier attempts" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- DFlash summary ---" >&2
    cat "$summary" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if (( rejected < misses )); then
    echo "DFlash smoke failed: rejection count is inconsistent with verifier misses" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- DFlash summary ---" >&2
    cat "$summary" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if (( verified < min_verified )); then
    echo "DFlash smoke failed: verified DFlash draft tokens $verified < required $min_verified" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- DFlash summary ---" >&2
    cat "$summary" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if ! cmp -s "$base_out" "$dflash_out"; then
    echo "DFlash smoke failed: generated output differs from baseline" >&2
    echo "Evidence: $evidence_dir" >&2
    echo "--- baseline stdout ---" >&2
    cat "$base_out" >&2
    echo >&2
    echo "--- dflash stdout ---" >&2
    cat "$dflash_out" >&2
    echo >&2
    echo "--- diff ---" >&2
    diff -u "$base_out" "$dflash_out" >"$diff_out" || true
    cat "$diff_out" >&2
    exit 1
fi

{
    printf 'result=passed\n'
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$metadata"

echo "DFlash runtime smoke passed: DFlash verifier ran, accepted draft tokens, and stdout matched baseline"
echo "Evidence: $evidence_dir"
cat "$summary"
