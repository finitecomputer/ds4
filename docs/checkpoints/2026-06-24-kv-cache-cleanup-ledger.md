# DS4 KV-cache Cleanup Ledger

Date: 2026-06-24

Branch: `feature/cuda-f16-compressed-kv`

Remote fork: `https://github.com/finitecomputer/ds4.git`

Primary Spark: `spark-123a` / `100.70.210.68`

Latest clean closeout:

```text
docs/checkpoints/2026-06-24-clean-closeout.md
```

Model:
`/home/finite/ds4-data/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

## Current State

The branch is clean and should be treated as the current DS4 GB10 KV-cache
checkpoint. The latest code-bearing commit is:

```text
8ab4381 bench: report decode telemetry and kv policy reason
```

Everything after that on this branch is documentation/checkpoint evidence only:

```text
31a989a docs: add DS4 decode kernel profile checkpoint
5619a0e docs: checkpoint indexed kv ab results
aa13398 docs: checkpoint indexed kv toggle scan
1382fd6 docs: checkpoint indexed topk profile
aeec95d docs: checkpoint indexed rows16 staging
fc095d3 docs: checkpoint indexed topk handoff
d6aa4d0 docs: checkpoint indexed row plan
```

This cleanup ledger is another documentation-only checkpoint.

## What Worked

The real optimization that survived is:

```text
CUDA F16 compressed KV + smarter GB10 device-KV policy
```

What it gives us:

- Target `ctx_alloc=524288` can stay on device KV on GB10 instead of falling
  back to managed KV.
- Allocated KV drops from about `7.08 GiB` to `4.37 GiB`.
- Allocated context buffers drop from about `11.08 GiB` to `8.37 GiB`.
- Prefill is modestly better in the long-allocation ladder.
- The live `ctx=124000`, `prefill_chunk=4096` DS4 frontdoor shape is stable.

Supporting code pieces:

- `aa57650 bench: report context allocation footprint`
- `17ad44c cuda: make managed kv policy tunable`
- `7af2a78 cuda: prefer device kv for moderate footprints`
- `72740b4 cuda: tune kv placement and direct model safety`
- `3a2c1d1 cuda: guard prefill chunk size`
- `8ab4381 bench: report decode telemetry and kv policy reason`

Useful operator profile:

```text
ctx=124000
ctx_alloc=524288 for stress/benchmark validation
prefill_chunk=4096
KV policy=device on the current GB10 target shape
```

Lower-footprint fallback:

```text
prefill_chunk=2048
```

## What Did Not Work

These were tried, benchmarked, and removed or rejected:

| Lane | Result | Decision |
| --- | --- | --- |
| Direct model HMM/ATS | safety probe passed, but prefill was much slower | keep as research/escape hatch only |
| `prefill_chunk=8192` | more memory, no speed win | do not promote |
| `prefill_chunk=16384` | killed by signal 9 in stress probe | do not promote |
| F32 compressed-cache rollback | modest decode win, large memory loss | keep F16 default |
| regular decode heads16/rows8 | local kernel looked faster, end-to-end did not win | removed |
| indexed single-token grouped route | flat overall | removed |
| mask-only indexed host cleanup | slight aggregate loss | removed |
| `DS4_CUDA_INDEXED_TWOPASS=1` | large prefill loss | do not promote |
| `DS4_CUDA_NO_INDEXED_TOPK_SORT=1` | tiny/noisy decode gain, worse indexed attention kernel behavior | keep sort |
| `DS4_CUDA_NO_INDEXED_HEADS8=1` | big prefill loss | keep heads8 path |
| selected-row warp broadcast | `+0.56%` prefill, `-1.92%` generation | removed |
| rows16 selected-row staging | `-1.64%` prefill, `-2.34%` generation | removed |
| top-k ascending handoff | `+0.49%` prefill, `-1.79%` generation | removed |
| indexed row-plan setup sharing | `-1.44%` prefill, `-0.87%` generation | removed |
| F16 selected-row `__ldg` read-only loads | `-0.47%` prefill, `-0.98%` generation | removed |

The latest lesson is important: the standalone indexed-attention setup work is
not expensive enough by itself. Adding a prep kernel or moving row ordering
around costs more than it saves. Future KV-cache work should either change real
selected-row memory locality or fuse into a kernel/stage we already must run.

The follow-up selected-row locality probe added an env-gated
`DS4_CUDA_INDEXED_LDG=1` path that used read-only cached loads in the F16
compressed KV `attention_comp_kv_load4` reader inside the existing grouped
indexed-attention kernel. It built and passed `make cuda-regression`, but the
benchmark lost on aggregate throughput:

| ctx | default prefill | `__ldg` prefill | prefill delta | default gen | `__ldg` gen | gen delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 372.95 | 367.06 | -1.58% | 12.15 | 12.06 | -0.74% |
| 65536 | 320.08 | 320.78 | +0.22% | 11.28 | 11.19 | -0.80% |
| 131072 | 268.41 | 268.25 | -0.06% | 9.94 | 9.80 | -1.41% |

Decision: do not keep a simple read-only-cache hint as a selected-row KV-cache
optimization. The patch was reverted and archived in Spark evidence.

## Rollback Map

Use these as clean anchors:

| Need | Anchor |
| --- | --- |
| Current documented branch state before this cleanup ledger | `d6aa4d0df9b117adca6478e9f454076d986b1021` |
| Latest code-bearing state | `8ab4381` |
| Code state before decode telemetry CSV columns | `3a2c1d1` |
| Device-KV policy PR-candidate branch anchor | `origin/pr-candidate/cuda-f16-compressed-kv-device-kv-2026-06-23` at `7af2a78d08183ca404fc01c4881355368548e060` |
| First complete GB10 optimization checkpoint doc | `532584a` |
| Server concurrency checkpoint doc | `1c7f494` |

Common commands:

```sh
git fetch origin
git switch feature/cuda-f16-compressed-kv
git reset --hard origin/feature/cuda-f16-compressed-kv
```

Inspect code-only state without moving the branch:

```sh
git switch --detach 8ab4381
```

Inspect the narrower PR-candidate branch:

```sh
git switch --detach origin/pr-candidate/cuda-f16-compressed-kv-device-kv-2026-06-23
```

## Cleanup State

The stale Desktop checkout at:

```text
/Users/plebdev/Desktop/Projects/finite/ds4
```

had abandoned top-k handoff prototype edits. Those edits were archived in the
Spark cleanup checkpoint and then reversed. The checkout was fast-forwarded to
`d6aa4d0` and is clean.

Latest abandoned prototype patches archived in Spark evidence:

```text
runs/2026-06-24-ds4-cleanup-checkpoint/stale-desktop-topk-handoff.patch.gz
runs/2026-06-24-ds4-cleanup-checkpoint/abandoned-indexed-row-plan.patch.gz
runs/2026-06-24-ds4-indexed-ldg-ab/abandoned-indexed-ldg.patch.gz
```

Remote scratch source directories for the two latest abandoned prototypes were
removed:

```text
/home/finite/ds4-codex-indexed-topk-asc-handoff
/home/finite/ds4-codex-indexed-row-plan
```

Older remote build directories were left in place because existing historical
run docs reference them. They are not canonical rollback state; Git commits and
run artifacts are canonical.

## Continue From Here

Best next KV-cache directions:

1. Avoid repeating simple selected-row cache-hint changes. If pursuing
   selected-row locality, change layout/staging behavior or collect lower-noise
   kernel profiles first.
2. Explore fusing selected-row metadata into an already-required score/top-k
   stage, only if it avoids extra launches and global reads.
3. Build a longer decode-focused ladder with fixed prefilled contexts if we
   need lower-noise decode measurements.
4. Treat server concurrency separately. The concurrency ladder showed DS4 still
   behaves like a serialized single-worker server; real gains there need
   batching, multiple sessions/workers, or frontdoor routing across processes.

## Upstream PR Hygiene

For a future PR to `antirez/ds4`, trim Finite-specific material:

- Spark runtime slots
- frontdoor routes
- Grafana evidence
- internal host paths
- benchmark run artifacts

The upstreamable code story should stay focused on:

- F16 compressed-KV CUDA path
- smarter managed/device KV policy
- benchmark allocation and decode telemetry reporting
- prefill chunk guardrail
- Linux benchmark file-cache hygiene
