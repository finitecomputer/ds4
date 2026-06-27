#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dflash-summary-test.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT

log=$tmpdir/dflash.err
summary=$tmpdir/dflash-summary.env

cat >"$log" <<'LOG'
ds4: dflash spec miss at=2 draft_token=10 target_token=20 target_top=21 drafted=7 accepted=3
ds4: dflash timing drafted=7 verified=2 draft=1.000 ms verify=2.000 ms total=3.000 ms
ds4: dflash spec drafted=7 verified=2 accepted=3
ds4: dflash timing drafted=4 verified=4 draft=1.500 ms verify=2.500 ms total=4.000 ms
ds4: dflash spec drafted=4 verified=4 accepted=5
LOG

awk -f "$script_dir/dflash_runtime_summary.awk" "$log" >"$summary"

expect_field() {
    local key=$1
    local expected=$2
    local actual
    actual=$(awk -F= -v key="$key" '$1 == key { print $2 }' "$summary")
    if [[ "$actual" != "$expected" ]]; then
        echo "expected $key=$expected, got ${actual:-<missing>}" >&2
        cat "$summary" >&2
        exit 1
    fi
}

expect_field attempts 2
expect_field drafted 11
expect_field verified 6
expect_field accepted_including_anchor 8
expect_field misses 1
expect_field rejected_draft_tokens 5
expect_field timing_lines 2

echo "dflash_runtime_summary_test: OK"
