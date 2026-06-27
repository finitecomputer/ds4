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
ds4: dflash spec drafted=7 verified=2 accepted=3 misses=1 rejected_draft_tokens=5
ds4: dflash timing drafted=4 verified=4 draft=1.500 ms verify=2.500 ms total=4.000 ms
ds4: dflash spec drafted=4 verified=4 accepted=5 misses=0 rejected_draft_tokens=0
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

cat >"$log" <<'LOG'
ds4: dflash spec miss at=1 draft_token=11 target_token=22 target_top=23 drafted=3 accepted=2
ds4: dflash timing drafted=3 verified=1 draft=1.000 ms verify=2.000 ms total=3.000 ms
ds4: dflash spec drafted=3 verified=1 accepted=2
LOG

awk -f "$script_dir/dflash_runtime_summary.awk" "$log" >"$summary"

expect_field attempts 1
expect_field drafted 3
expect_field verified 1
expect_field accepted_including_anchor 2
expect_field misses 1
expect_field rejected_draft_tokens 2
expect_field timing_lines 1

grep -q 'accepted_including_anchor' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'accepted < verified' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'accepted < attempts' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'accepted > drafted + attempts' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'rejected < misses' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'evidence_schema=ds4-dflash-runtime-smoke/v1' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'result=passed' "$script_dir/dflash_runtime_smoke.sh"
grep -q 'completed_utc=' "$script_dir/dflash_runtime_smoke.sh"
bash -n "$script_dir/dflash_runtime_smoke.sh"
python3 -m py_compile "$script_dir/dflash_smoke_evidence_validate.py"

evidence=$tmpdir/evidence
mkdir -p "$evidence"
printf 'same stdout\n' >"$evidence/baseline.out"
cp "$evidence/baseline.out" "$evidence/dflash.out"
printf 'stderr\n' >"$evidence/dflash.err"
printf 'prompt\n' >"$evidence/prompt.txt"
cat >"$evidence/metadata.txt" <<'META'
evidence_schema=ds4-dflash-runtime-smoke/v1
model=/tmp/model.gguf
dflash=/tmp/dflash
ds4_commit=test-commit
started_utc=2026-06-27T00:00:00Z
result=passed
completed_utc=2026-06-27T00:00:01Z
META
cat >"$evidence/dflash-summary.env" <<'SUMMARY'
attempts=2
drafted=11
verified=6
accepted_including_anchor=8
misses=1
rejected_draft_tokens=5
timing_lines=2
SUMMARY

python3 "$script_dir/dflash_smoke_evidence_validate.py" \
    "$evidence" \
    --model /tmp/model.gguf \
    --dflash /tmp/dflash \
    --ds4-commit test-commit \
    --min-verified 2 >/dev/null

if python3 "$script_dir/dflash_smoke_evidence_validate.py" \
    "$evidence" \
    --model /tmp/model.gguf \
    --dflash /tmp/dflash \
    --ds4-commit other-commit >/dev/null 2>&1; then
    echo "expected mismatched ds4_commit evidence to fail" >&2
    exit 1
fi

cp "$evidence/metadata.txt" "$evidence/metadata.good"
awk -F= '$1 != "completed_utc"' "$evidence/metadata.good" >"$evidence/metadata.txt"
if python3 "$script_dir/dflash_smoke_evidence_validate.py" \
    "$evidence" \
    --model /tmp/model.gguf \
    --dflash /tmp/dflash >/dev/null 2>&1; then
    echo "expected missing completed_utc evidence to fail" >&2
    exit 1
fi
cp "$evidence/metadata.good" "$evidence/metadata.txt"

awk -F= '{ if ($1 == "result") print "result=failed"; else print $0 }' "$evidence/metadata.good" >"$evidence/metadata.txt"
if python3 "$script_dir/dflash_smoke_evidence_validate.py" \
    "$evidence" \
    --model /tmp/model.gguf \
    --dflash /tmp/dflash >/dev/null 2>&1; then
    echo "expected non-passed result evidence to fail" >&2
    exit 1
fi
cp "$evidence/metadata.good" "$evidence/metadata.txt"

cat >"$evidence/dflash-summary.env" <<'SUMMARY'
attempts=2
drafted=2
verified=2
accepted_including_anchor=6
misses=0
rejected_draft_tokens=0
timing_lines=1
SUMMARY

if python3 "$script_dir/dflash_smoke_evidence_validate.py" \
    "$evidence" \
    --model /tmp/model.gguf \
    --dflash /tmp/dflash >/dev/null 2>&1; then
    echo "expected impossible accepted-anchor evidence to fail" >&2
    exit 1
fi

stub=$tmpdir/fake-ds4
cat >"$stub" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
is_dflash=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dflash)
            is_dflash=1
            shift 2
            ;;
        -m|--tokens|--ctx|-p|--temp)
            shift 2
            ;;
        --nothink)
            shift
            ;;
        *)
            shift
            ;;
    esac
done
printf 'same stdout\n'
if (( is_dflash )); then
    {
        printf 'ds4: dflash timing drafted=3 verified=2 draft=1.000 ms verify=2.000 ms total=3.000 ms\n'
        printf 'ds4: dflash spec drafted=3 verified=2 accepted=3 misses=0 rejected_draft_tokens=0\n'
    } >&2
fi
STUB
chmod +x "$stub"

runtime_evidence=$tmpdir/runtime-evidence
DS4_BIN="$stub" \
DS4_SMOKE_EVIDENCE_DIR="$runtime_evidence" \
DS4_SMOKE_MIN_VERIFIED=1 \
bash "$script_dir/dflash_runtime_smoke.sh" /tmp/model.gguf /tmp/dflash >/dev/null

grep -q '^evidence_schema=ds4-dflash-runtime-smoke/v1$' "$runtime_evidence/metadata.txt"
grep -q '^result=passed$' "$runtime_evidence/metadata.txt"
grep -q '^completed_utc=' "$runtime_evidence/metadata.txt"
python3 "$script_dir/dflash_smoke_evidence_validate.py" \
    "$runtime_evidence" \
    --model /tmp/model.gguf \
    --dflash /tmp/dflash \
    --min-verified 1 >/dev/null

if DS4_BIN="$stub" \
    DS4_SMOKE_EVIDENCE_DIR="$runtime_evidence" \
    bash "$script_dir/dflash_runtime_smoke.sh" /tmp/model.gguf /tmp/dflash >/dev/null 2>&1; then
    echo "expected reused runtime evidence directory to fail" >&2
    exit 1
fi

echo "dflash_runtime_summary_test: OK"
