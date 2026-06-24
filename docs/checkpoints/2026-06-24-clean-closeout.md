# DS4 GB10 KV-cache Clean Closeout

Date: 2026-06-24

Branch: `feature/cuda-f16-compressed-kv`

Remote fork: `https://github.com/finitecomputer/ds4.git`

Primary Spark: `spark-123a` / `100.70.210.68`

## Current Verdict

The branch is clean and checkpointed. The upstreamable code story is:

```text
CUDA F16 compressed KV + GB10 device-KV policy + benchmark/operator guardrails
```

The durable result is memory-policy progress, not a broad decode speedup:

- `ctx_alloc=524288` can stay on device KV on the GB10 target shape.
- Allocated KV drops from about `7.08 GiB` to `4.37 GiB`.
- Context buffers drop from about `11.08 GiB` to `8.37 GiB`.
- Prefill improves modestly versus original DS4, roughly `+1.2%` to `+2.8%`
  in the fresh current-vs-original ladder.
- Decode is flat to negative at longer contexts. Keep the F16 cache for memory;
  do not claim it is a decode optimization.
- Server-mode concurrency is still serialized. The DS4 server path did not gain
  batching or true concurrent throughput.

## Clean Anchors

| Purpose | Anchor |
| --- | --- |
| Last experiment/no-go checkpoint before this closeout | `5e1fd49 docs: checkpoint indexed LDG no-go` |
| Current code-bearing tip | `8ab4381 bench: report decode telemetry and kv policy reason` |
| Narrow PR-candidate branch | `origin/pr-candidate/cuda-f16-compressed-kv-device-kv-2026-06-23` at `7af2a78d08183ca404fc01c4881355368548e060` |
| First complete GB10 optimization checkpoint | `532584a docs: checkpoint gb10 cuda kv optimization` |
| Server concurrency checkpoint | `1c7f494 docs: add ds4 server concurrency checkpoint` |
| Initial Spark F16 benchmark result | `379f7ae chore: record f16 spark benchmark result` |

Return to the current checkpoint:

```sh
git fetch origin
git switch feature/cuda-f16-compressed-kv
git reset --hard origin/feature/cuda-f16-compressed-kv
```

Inspect only the code-bearing state:

```sh
git switch --detach 8ab4381
```

Inspect the narrower upstream PR candidate:

```sh
git switch --detach origin/pr-candidate/cuda-f16-compressed-kv-device-kv-2026-06-23
```

## What Worked

| Area | Result | Keep? |
| --- | --- | --- |
| F16 compressed KV storage | Smaller compressed KV footprint | yes |
| F16 compressed KV CUDA decode/prefill readers | Functional with modest prefill upside | yes |
| Heads8 F16 attention path | Main CUDA path that survived benchmarking | yes |
| GB10 device-KV policy | Keeps moderate `ctx_alloc=524288` shape on device KV | yes |
| Prefill chunk guardrail | Prevents unsafe huge prefill chunks by default | yes |
| `prefill_chunk=4096` profile | Best current live/operator default | yes |
| `prefill_chunk=2048` fallback | Lower-footprint fallback profile | yes |
| Benchmark telemetry | Reports allocation, KV policy, first-token, rest-token, avg-token metrics | yes |
| Linux file-cache hygiene | Reduces repeated benchmark/server cold-start pressure | yes |

Current useful operator shape:

```text
ctx=124000
ctx_alloc=524288 for stress/benchmark validation
prefill_chunk=4096
KV policy=device for the current GB10 target shape
```

## What Did Not Work

These lanes were built or probed, benchmarked, and either removed or rejected:

| Lane | Result | Decision |
| --- | --- | --- |
| Direct model HMM/ATS | safety probe passed, but prefill was much slower | research/escape hatch only |
| `prefill_chunk=8192` | more memory, no speed win | do not promote |
| `prefill_chunk=16384` | killed by signal 9 in stress probe | do not promote |
| F32 compressed-cache rollback | small decode win, large memory loss | keep F16 default |
| regular decode heads16/rows8 | local kernel looked faster, end-to-end did not win | removed |
| indexed single-token grouped route | flat overall | removed |
| mask-only indexed host cleanup | slight aggregate loss | removed |
| `DS4_CUDA_INDEXED_TWOPASS=1` | large prefill loss | do not promote |
| `DS4_CUDA_NO_INDEXED_TOPK_SORT=1` | top-k sort removal hurt indexed attention | keep sort |
| `DS4_CUDA_NO_INDEXED_HEADS8=1` | big prefill loss | keep heads8 path |
| selected-row warp broadcast | `+0.56%` prefill, `-1.92%` generation | removed |
| rows16 selected-row staging | `-1.64%` prefill, `-2.34%` generation | removed |
| top-k ascending handoff | `+0.49%` prefill, `-1.79%` generation | removed |
| indexed row-plan setup sharing | `-1.44%` prefill, `-0.87%` generation | removed |
| F16 selected-row `__ldg` read-only loads | `-0.47%` prefill, `-0.98%` generation | removed |

Lesson: the cheap metadata/setup tweaks are not the bottleneck. Do not add
another standalone prep kernel for indexed attention. Do not repeat simple
selected-row cache-hint changes without a kernel profile showing that exact load
instruction is the blocker.

## Benchmarks To Trust

Current-vs-original stress ladder, `ctx_alloc=524288`:

| ctx | original prefill | current prefill | prefill delta | original gen | current gen | gen delta | current KV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32768 | 356.29 | 366.20 | +2.78% | 11.91 | 12.09 | +1.51% | device |
| 65536 | 316.81 | 320.75 | +1.24% | 11.20 | 10.83 | -3.30% | device |
| 131072 | 261.71 | 268.34 | +2.53% | 9.86 | 9.34 | -5.27% | device |
| 262144 | 193.95 | 199.31 | +2.76% | 8.36 | 7.61 | -8.97% | device |

Decode telemetry ladder, 256 decode tokens:

| ctx | original prefill | current F16 prefill | F16 prefill delta | original gen | current F16 gen | F16 gen delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 355.18 | 369.94 | +4.16% | 12.30 | 12.20 | -0.81% |
| 65536 | 315.43 | 323.53 | +2.57% | 11.54 | 11.32 | -1.91% |
| 131072 | 261.22 | 268.62 | +2.83% | 10.16 | 9.96 | -1.97% |

Memory comparison for the current source at `ctx_alloc=524288`:

| Variant | KV policy reason | managed KV | allocated KV | allocated context |
| --- | --- | ---: | ---: | ---: |
| current F16 | `moderate_kv_within_pressure_budget` | 0 | 4.37 GiB | 8.37 GiB |
| current F32 | `moderate_kv_within_pressure_budget` | 0 | 7.08 GiB | 11.08 GiB |
| original default | managed by stderr | managed | 7.08 GiB | 11.08 GiB |

Server-mode concurrency, live `--ctx 124000 --prefill-chunk 4096`:

| concurrency | original success | current success | original p95 | current p95 | original completion t/s | current completion t/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100% | 100% | 2.401s | 2.420s | 13.38 | 13.31 |
| 2 | 100% | 100% | 4.760s | 4.800s | 13.45 | 13.34 |
| 4 | 100% | 100% | 9.525s | 9.577s | 13.44 | 13.36 |
| 8 | 100% | 100% | 19.093s | 19.220s | 13.40 | 13.32 |

Interpretation: concurrency queues through one inference worker. Future
concurrency gains need batching, multiple workers/sessions, or frontdoor
scheduling across more than one DS4 process/Spark.

## Current Cleanliness

- `/tmp/ds4-finite-working` was clean at `5e1fd49` before this closeout doc was
  added.
- `/Users/plebdev/Desktop/Projects/finite/ds4` was fast-forwarded and is clean
  at `5e1fd49`.
- The latest LDG source scratch on `spark-123a` was removed after its patch and
  run artifacts were archived.
- Older remote `ds4-codex-*` scratch directories were intentionally left because
  historical docs reference them. Git commits and run artifacts are canonical.
- Live DS4 on `spark-123a` was restored after every benchmark pass and is the
  current smoke target.

## Next Clean Move

Recommended next work:

1. Prepare an upstream PR candidate from the narrow code-bearing branch, trimming
   Finite-only Spark/frontdoor/Grafana/checkpoint material.
2. If continuing KV-cache research, stop doing tiny selected-row row-id/cache
   hints. Profile first, then target real selected-row layout/staging behavior.
3. If serving throughput matters, work above the kernel: batching, multi-worker
   DS4 sessions, or frontdoor routing across processes/Sparks.

Do not repeat:

- standalone indexed-attention setup kernels
- simple top-k handoff variants
- simple `__ldg` selected-row reader hints
- F32 rollback as the default
