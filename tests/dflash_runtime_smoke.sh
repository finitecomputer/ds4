#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: tests/dflash_runtime_smoke.sh MODEL.gguf DFLASH_DIR [PROMPT]

Runs a tiny greedy baseline decode and a DFlash-enabled decode, then compares
stdout byte-for-byte. Set DS4_BIN, DS4_SMOKE_TOKENS, or DS4_SMOKE_CTX to
override the defaults.
USAGE
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 2
fi

model=$1
dflash=$2
prompt=${3:-"Reply with one short sentence about local inference."}
bin=${DS4_BIN:-./ds4}
tokens=${DS4_SMOKE_TOKENS:-16}
ctx=${DS4_SMOKE_CTX:-2048}
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dflash-smoke.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT

base_out=$tmpdir/baseline.out
base_err=$tmpdir/baseline.err
dflash_out=$tmpdir/dflash.out
dflash_err=$tmpdir/dflash.err

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
"$bin" \
    -m "$model" \
    --dflash "$dflash" \
    --nothink \
    --temp 0 \
    --tokens "$tokens" \
    --ctx "$ctx" \
    -p "$prompt" \
    >"$dflash_out" 2>"$dflash_err"

if ! grep -q "ds4: dflash spec" "$dflash_err"; then
    echo "DFlash smoke failed: DFlash verifier log was not observed" >&2
    echo "--- DFlash stderr ---" >&2
    cat "$dflash_err" >&2
    exit 1
fi

if ! cmp -s "$base_out" "$dflash_out"; then
    echo "DFlash smoke failed: generated output differs from baseline" >&2
    echo "--- baseline stdout ---" >&2
    cat "$base_out" >&2
    echo >&2
    echo "--- dflash stdout ---" >&2
    cat "$dflash_out" >&2
    echo >&2
    echo "--- diff ---" >&2
    diff -u "$base_out" "$dflash_out" >&2 || true
    exit 1
fi

echo "DFlash runtime smoke passed: DFlash verifier ran and stdout matched baseline"
