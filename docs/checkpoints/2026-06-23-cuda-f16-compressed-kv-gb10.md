# CUDA F16 Compressed KV GB10 Checkpoint

Date: 2026-06-23

Branch: `feature/cuda-f16-compressed-kv`

Base checkpoint commit: `3a2c1d1 cuda: guard prefill chunk size`

Remote fork: `https://github.com/finitecomputer/ds4.git`

Hardware: `spark-123a`, NVIDIA GB10 / DGX Spark class machine, 128 GB unified memory

Model:
`/home/finite/ds4-data/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

Build used for GB10 validation:

```sh
make cuda-spark DS4_CUDA_ATTN_COMP_CACHE_F16=1
make cuda-regression
```

## Executive Summary

This fork now contains a real DS4 CUDA memory-policy optimization for the GB10
Spark target. The important result is not a large raw throughput jump. The
important result is that the target `ctx_alloc=524288` profile now stays on
device KV instead of falling back to CUDA managed KV.

That changes DS4 on the Spark from "technically works, but long allocations are
managed-memory fragile" to "the 124k frontdoor shape and the 512k allocation
profile have a stable device-KV path." Prefill improves modestly and
consistently. Decode is still the soft spot and remains the next real
optimization target.

Current best operator profile:

```text
ctx=124000
ctx_alloc=524288
prefill_chunk=4096
kv_policy=device for the current 124k frontdoor shape
```

Lower-footprint fallback:

```text
ctx=124000
ctx_alloc=524288
prefill_chunk=2048
kv_policy=device
```

Do not promote `prefill_chunk=8192` as a speed profile; it costs more memory
without improving throughput. Do not promote `prefill_chunk=16384`; it was
killed under the 32k/512k allocation probe.

## What Changed In The Fork

The useful changes landed in several layers.

1. CUDA F16 compressed-KV attention path.

   Commits in the stack include:

   - `ce33a8e feat: carry f16 compressed kv through cuda attention`
   - `41cd5e9 feat: optimize cuda f16 compressed kv heads8 attention`

   This reduced the effective KV/context allocation footprint. The first naive
   F16 path was functionally viable but made long prefill much slower because
   it missed the optimized heads path. The later heads8 path recovered the
   usable prefill shape.

2. Benchmark allocation reporting.

   Commit:

   - `aa57650 bench: report context allocation footprint`

   `ds4-bench` now reports:

   - `allocated_kv_cache_bytes`
   - `allocated_context_bytes`
   - `managed_kv_cache`

   This made the policy result visible in CSVs instead of only in stderr.

3. Tunable and smarter CUDA managed-KV policy.

   Commits:

   - `17ad44c cuda: make managed kv policy tunable`
   - `7af2a78 cuda: prefer device kv for moderate footprints`
   - `72740b4 cuda: tune kv placement and direct model safety`

   The policy can now choose device KV for moderate footprints that fit within
   the GB10 pressure budget. The target `ctx_alloc=524288` case is now device
   backed. A `ctx_alloc=1048576` boundary probe still flips to managed memory,
   which is the correct conservative behavior for that much larger allocation.

   Relevant controls:

   - `DS4_CUDA_MANAGED_KV_CACHE`
   - `DS4_CUDA_NO_MANAGED_KV_CACHE`
   - `DS4_CUDA_MANAGED_KV_RESERVE_MB`
   - `DS4_CUDA_MANAGED_KV_DEVICE_MAX_MB`
   - `DS4_CUDA_MANAGED_KV_MIN_KV_MB`
   - `DS4_CUDA_MANAGED_KV_MIN_CONTEXT_MB`
   - `DS4_CUDA_MANAGED_KV_DEVICE_CONTEXT_PCT`
   - `DS4_CUDA_MANAGED_KV_VERBOSE`

4. Direct model safety.

   Commit:

   - `72740b4 cuda: tune kv placement and direct model safety`

   Direct model HMM/ATS was repaired enough to run safety probes, but it was
   not a speed win. Keep it as an escape hatch and future research path, not as
   the default operator profile.

5. Prefill guardrails and benchmark hygiene.

   Commit:

   - `3a2c1d1 cuda: guard prefill chunk size`

   Added:

   - `DS4_PREFILL_CHUNK_MAX`, default `8192`
   - `DS4_PREFILL_CHUNK_MAX=0` as an explicit uncapped stress mode
   - cap warnings across CLI, server, eval, agent, and bench
   - `ds4-bench --drop-model-file-cache`
   - benchmark prompt-file loading before model open, so bad prompt paths do
     not cold-load the huge model first

## Fresh Current-vs-Original Benchmark

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-23-ds4-current-vs-original-spark123a/
```

Remote run directory:

```text
/home/finite/ds4-runs/2026-06-23-current-vs-original/
```

Benchmark shape:

```sh
--ctx-start 32768 \
--ctx-max 262144 \
--ctx-alloc 524288 \
--step-mul 2 \
--gen-tokens 64 \
--warm-weights \
--prefill-chunk 4096
```

Current optimized binary:

```text
/home/finite/ds4-codex-prefill-guardrail/ds4-bench
```

Original baseline binary:

```text
/home/finite/ds4-direct-f16-ladder-20260622T193000-0500/bin/ds4-bench-default
```

The original baseline leg was interrupted after the 128k row to avoid spending
the slot on the old slow 256k managed-KV row. The 32k, 64k, and 128k rows were
fresh A/B results and matched the prior full checkpoint closely. The 256k old
prefill comparison below uses the prior full original checkpoint.

| ctx | old source | old prefill | current prefill | prefill delta | old gen | current gen | gen delta | current KV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32768 | fresh | 356.29 | 366.20 | +2.78% | 11.91 | 12.09 | +1.51% | device |
| 65536 | fresh | 316.81 | 320.75 | +1.24% | 11.20 | 10.83 | -3.30% | device |
| 131072 | fresh | 261.71 | 268.34 | +2.53% | 9.86 | 9.34 | -5.27% | device |
| 262144 | 2026-06-22 checkpoint | 193.95 | 199.31 | +2.76% | 8.36 | 7.61 | -8.97% | device |

Important caveat: the 256k generation comparison is not apples-to-apples. The
old 256k row came from the full 2026-06-22 ladder with `--gen-tokens 128`; the
current row used `--gen-tokens 64`.

Memory-policy comparison for the same forced `ctx_alloc=524288` shape:

| Metric | Original default | Current optimized | Delta |
| --- | ---: | ---: | ---: |
| KV policy | managed KV | device KV | fixed target fallback |
| KV allocation | 7.08 GiB | 4.37 GiB | -38.3% |
| Context buffers | 11.08 GiB | 8.37 GiB | -24.5% |
| CSV `managed_kv_cache` | unavailable/managed by stderr | 0 | device-backed |

Current optimized policy line:

```text
ds4: CUDA managed KV policy: moderate kv within pressure budget -> device (kv 4.37 GiB, context 8.37 GiB, free 1.79 GiB, reserve 30.42 GiB, total 121.69 GiB, device-max 8.00 GiB, pressure-limit 91.27 GiB)
```

Original baseline policy line:

```text
ds4: CUDA using managed KV cache for ctx=524288 (kv cache 7.08 GiB, context buffers 11.08 GiB); this may degrade performance but is needed for very large contexts
```

## Decode Telemetry Ladder

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-23-ds4-decode-telemetry-ladder/
```

Remote run directory:

```text
/home/finite/ds4-runs/2026-06-23-ds4-decode-telemetry-ladder/
```

Benchmark shape:

```sh
--ctx-start 32768 \
--ctx-max 131072 \
--ctx-alloc 524288 \
--step-mul 2 \
--gen-tokens 256 \
--warm-weights \
--prefill-chunk 4096
```

This run added benchmark columns for:

- `kv_policy_reason`
- `prefill_chunk_requested`
- `prefill_chunk_effective`
- `gen_first_token_ms`
- `gen_rest_tps`
- `gen_avg_token_ms`

It compared three binaries:

- current F16 compressed-KV build
- current F32/default compressed-cache build from the same source
- original DS4 default benchmark binary from the earlier checkpoint

| ctx | original prefill | current F16 prefill | F16 prefill delta | original gen | current F16 gen | F16 gen delta | F16 first token | F16 rest decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 355.18 | 369.94 | +4.16% | 12.30 | 12.20 | -0.81% | 82.420 ms | 12.20 tok/s |
| 65536 | 315.43 | 323.53 | +2.57% | 11.54 | 11.32 | -1.91% | 126.713 ms | 11.34 tok/s |
| 131072 | 261.22 | 268.62 | +2.83% | 10.16 | 9.96 | -1.97% | 151.282 ms | 9.98 tok/s |

Current F32 from the same source kept the wider cache and remained slightly
faster for decode, but it gave back the memory win:

| ctx | F16 prefill | F32 prefill | F16 vs F32 prefill | F16 gen | F32 gen | F16 vs F32 gen |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 369.94 | 372.94 | -0.80% | 12.20 | 12.56 | -2.87% |
| 65536 | 323.53 | 315.91 | +2.41% | 11.32 | 11.60 | -2.41% |
| 131072 | 268.62 | 265.04 | +1.35% | 9.96 | 10.21 | -2.45% |

Memory comparison for the current source at `ctx_alloc=524288`:

| Variant | KV policy reason | managed KV | allocated KV | allocated context |
| --- | --- | ---: | ---: | ---: |
| current F16 | `moderate_kv_within_pressure_budget` | 0 | 4.37 GiB | 8.37 GiB |
| current F32 | `moderate_kv_within_pressure_budget` | 0 | 7.08 GiB | 11.08 GiB |
| original default | managed by stderr | managed | 7.08 GiB | 11.08 GiB |

Result:

- The F16 compressed-KV path is still the right default for the Spark target.
- Versus original default, it is now modestly faster on prefill and only about
  `0.8%` to `2.0%` slower on 256-token decode windows.
- Reverting to F32 would buy only about `2.4%` to `2.9%` decode throughput in
  this ladder, while increasing allocated KV by roughly `62%` and allocated
  context by roughly `32%`.
- The decode tax is steady-state, not just first-token setup. The next kernel
  work should profile and specialize the F16 compressed-cache decode reader
  rather than rolling back to F32 cache storage.

## Earlier Benchmark Checkpoints

### Original Default vs First F16 Full Ladder

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-direct-f16-ladder-spark123a/
```

DS4 commit:

```text
66b7bffa5d0f1e8b6baf442a79c9ecdc16a9e9f0
```

Shape:

```sh
ds4-bench --cuda \
  --ctx-start 32768 \
  --ctx-max 524288 \
  --step-mul 2 \
  --gen-tokens 128 \
  --warm-weights
```

Result:

- F16 compressed KV reduced logged 512k KV from `7.08 GiB` to `4.37 GiB`.
- Context buffers dropped from `11.08 GiB` to `8.38 GiB`.
- Sampled process swap dropped from `1513.9 MiB` to `101.5 MiB`.
- Prefill improved only `+0.07%` to `+2.49%`.
- Generation regressed `-2.65%` to `-6.58%`.
- Both variants still used managed KV at 512k.

This was useful proof of memory headroom, but not yet an operator-ready policy.

### Managed-KV Policy Probe

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-23-ds4-managed-kv-policy-spark123a/
```

DS4 commit:

```text
7af2a78d08183ca404fc01c4881355368548e060
```

32k speed probe with `ctx_alloc=524288`:

| Case | managed KV | prefill | generation |
| --- | ---: | ---: | ---: |
| old auto-managed | 1 | 328.26 | 9.33 |
| new auto-device | 0 | 371.58 | 12.21 |

This was the first proof that avoiding managed KV could materially help both
prefill and decode for the target allocation shape.

### Direct Model And Smarter Policy Probe

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-23-ds4-kv-policy-direct-model-spark123a/
```

DS4 commit:

```text
72740b4389263786a865ed0bbb23ad50a79e0210
```

Result:

- CUDA build passed.
- CUDA regression passed: `cuda long-context regression: OK`.
- Smarter policy preserved device KV for the target F16 compressed-KV case.
- Direct model HMM/ATS safety probe passed.
- Direct model was not a speed win: `99.15` prefill t/s versus `360.97`
  prefill t/s for the new automatic policy in the 32k/512k probe.

Interpretation: keep automatic device KV as the default; keep direct model as an
escape hatch/research path.

### Prefill Chunk And Policy Ladder

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-23-ds4-policy-ladder-prefill-chunk-spark123a/
```

DS4 commit:

```text
72740b4389263786a865ed0bbb23ad50a79e0210
```

32k prompt, `ctx_alloc=524288`, pure prefill:

| prefill chunk | prefill t/s | allocated KV | allocated context | managed KV | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 371.33 | 4514749440 | 6663318528 | false | good fallback |
| 4096 | 375.00 | 4695104512 | 8992238592 | false | best default |
| 8192 | 372.91 | 5033270272 | 13627534336 | false | more memory, no win |
| 16384 | n/a | n/a | n/a | false | killed by signal 9 |

Long-context pure prefill with `prefill_chunk=4096`, `ctx_alloc=524288`:

| ctx | prefill tokens | prefill t/s | managed KV |
| ---: | ---: | ---: | ---: |
| 65536 | 65536 | 347.92 | false |
| 124000 | 124000 | 306.39 | false |
| 131072 | 65536 | 268.78 | false |
| 262144 | 131072 | 199.14 | false |

Boundary probe:

| ctx_alloc | ctx | prefill t/s | managed KV | interpretation |
| ---: | ---: | ---: | ---: | --- |
| 1048576 | 2048 | 274.33 | true | correct fallback at 1M allocation |

### Frontdoor, Toolcall, And Hermes Evidence

Baseline frontdoor artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-toolcall15-hermesagent20-frontdoor-fast/
```

F16 experiment artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/
```

Baseline frontdoor:

- `Toolcall-15`: score `93`, 14 pass, 0 partial, 1 fail, `200.74s`
- `HermesAgent-20`: score `73.75`, 10 pass, 3 partial, 7 fail, `1476.206s`
- Frontdoor/Grafana observability panel added for DS4 Operator Frontdoor (124k)

First F16 frontdoor experiment:

- Operator frontdoor smoke passed.
- `Toolcall-15`: score `93`, 14 pass, 0 partial, 1 fail, `208.416s`
- `HermesAgent-20`: score `73.25`, 10 pass, 2 partial, 8 fail, `1440.474s`
- Decode 256 cap: `15.025 tok/s`, `-1.2%` from baseline.
- Long-prompt prefill wall: `147.953 prompt tok/s`, `-56.8%` from baseline.

Interpretation: the initial F16 frontdoor experiment was functionally viable but
not performance viable. The later heads8/policy work is the relevant
optimization stack; do not use the first F16 long-prefill result as the current
performance claim.

### Server Concurrency Ladder

Artifact:

```text
/Users/plebdev/spark-cluster/runs/2026-06-23-ds4-server-concurrency-ladder/
```

Remote run directory:

```text
/home/finite/ds4-runs/2026-06-23-ds4-server-concurrency-ladder/
```

Shape:

```text
--ctx 124000 --prefill-chunk 4096
Chat Completions
max_tokens=32
concurrency=1,2,4,8
three waves per level
```

This is the live/frontdoor-style server shape, not the `ctx_alloc=524288`
stress allocation shape used to prove the managed-KV policy change.

| concurrency | original success | current success | original p50 | current p50 | original p95 | current p95 | original completion t/s | current completion t/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100% | 100% | 2.384s | 2.402s | 2.401s | 2.420s | 13.38 | 13.31 |
| 2 | 100% | 100% | 3.573s | 3.597s | 4.760s | 4.800s | 13.45 | 13.34 |
| 4 | 100% | 100% | 5.950s | 5.986s | 9.525s | 9.577s | 13.44 | 13.36 |
| 8 | 100% | 100% | 10.736s | 10.816s | 19.093s | 19.220s | 13.40 | 13.32 |

Result:

- Both original and current optimized completed `45/45` requests.
- Throughput did not scale with concurrency. Completion throughput stayed near
  `13.3` to `13.4` tokens/s.
- Latency scaled like a serialized single-worker queue. At concurrency 8, p95
  was about `19s` for both profiles.
- Current optimized was essentially neutral/slightly slower for this short
  124k server-mode path: about `-0.6%` to `-0.8%` completion tokens/s and
  about `+0.6%` to `+0.9%` latency.

Interpretation: this benchmark confirms that the server path still serializes
concurrent HTTP requests through one inference worker. The optimization stack
does not add serving concurrency or batching. To improve this result, the next
work is server batching, multiple sessions/workers, or a frontdoor scheduler
that routes concurrent load across more than one DS4 process/Spark.

## Validation Performed

Local DS4 validation after the guardrail/checkpoint stack:

```sh
make cpu
make ds4_test
./ds4_test --prefill-chunk-max
make
```

Remote isolated GB10 validation on `spark-123a`:

```sh
make cuda-spark DS4_CUDA_ATTN_COMP_CACHE_F16=1
make cuda-regression
```

Observed CUDA regression line:

```text
ds4: CUDA backend initialized on NVIDIA GB10 (sm_121)
cuda long-context regression: OK
```

Runtime restore smoke after benchmark cycles:

```text
./ds4-server --ctx 124000 --kv-disk-dir /home/finite/ds4-data/kv --kv-disk-space-mb 8192 --host 0.0.0.0 --port 8000
```

The restored local endpoint returned `deepseek-v4-flash` and
`deepseek-v4-pro` with `context_length=124000`.

## Current Recommendation

Promote this stack internally as:

```text
DS4 CUDA F16 compressed KV + smart device-KV policy for GB10 Spark
```

Default runtime profile:

```sh
./ds4-server \
  --ctx 124000 \
  --prefill-chunk 4096 \
  --kv-disk-dir /home/finite/ds4-data/kv \
  --kv-disk-space-mb 8192 \
  --host 0.0.0.0 \
  --port 8000
```

Lower-footprint fallback:

```sh
./ds4-server \
  --ctx 124000 \
  --prefill-chunk 2048 \
  --kv-disk-dir /home/finite/ds4-data/kv \
  --kv-disk-space-mb 8192 \
  --host 0.0.0.0 \
  --port 8000
```

For repeated Linux benchmark/server cycles:

```sh
./ds4-bench ... --drop-model-file-cache
```

For unsafe stress tests only:

```sh
DS4_PREFILL_CHUNK_MAX=0 ./ds4-bench ... --prefill-chunk N
```

## Known Limitations

1. Decode remains the next real bottleneck.

   The fresh current-vs-original benchmark shows prefill consistently ahead by
   roughly `+1.2%` to `+2.8%`, but generation is flat to negative after 32k.
   The current optimization stack should be described as memory-policy progress
   with modest prefill improvement, not a broad decoding speedup.

2. The 256k generation comparison is not perfectly apples-to-apples.

   The current run used `--gen-tokens 64`. The old 256k row in the comparison
   came from the prior full checkpoint with `--gen-tokens 128`.

3. `ctx_alloc=1048576` still uses managed KV.

   This is expected and safer on GB10. The policy boundary probe proves the
   fallback remains active. Do not claim 1M allocation is device-backed.

4. Direct model is not defaultable.

   HMM/ATS direct model survived safety probes but was much slower for prefill.

5. Some checkpoint files are Finite/operator evidence, not upstream PR material.

   This document and the Spark run artifacts are useful for us. Before an
   upstream PR to `antirez/ds4`, trim or omit internal paths, Spark runtime
   references, and Finite frontdoor/Grafana evidence. The upstreamable code
   should stay modular, documented through flags/env vars, and independently
   benchmarkable.

## Next Optimization Targets

1. Decode path for F16 compressed KV.

   Prefill is now modestly better. Decode is not. Start by profiling the decode
   step under the device-KV policy at 32k, 64k, 128k, and 256k.

2. Generation-specific benchmark ladder.

   Run a narrow ladder with fixed prefilled contexts and longer decode windows
   to reduce noise from setup/prefill/snapshot work.

3. Device-KV policy telemetry.

   Done in the decode telemetry stack: `ds4-bench` now reports the effective
   policy reason, requested/effective prefill chunk, first-token decode latency,
   steady-state decode throughput, and average decode token latency.

4. Cold-start/file-cache hygiene.

   `--drop-model-file-cache` is now present. Run a repeated-cycle benchmark to
   quantify whether it reduces startup failures after several server/bench
   rotations.

5. Upstream PR cleanup.

   Separate upstreamable pieces:

   - F16 compressed KV CUDA path
   - managed-KV policy controls
   - benchmark allocation reporting
   - prefill chunk cap
   - Linux benchmark file-cache hygiene

   Keep Finite-specific runtime slot, Grafana, and frontdoor evidence outside
   the upstream PR.
