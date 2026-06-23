# DS4 CUDA F16 Compressed KV Optimization Next Steps

Captured: 2026-06-22

## Current Result

The optimized heads8 CUDA F16 compressed-KV path works and is performance
credible at the current 124k DS4 front-door shape.

| Probe | Baseline | Initial F16 | Optimized F16 heads8 |
| --- | ---: | ---: | ---: |
| Decode cap | 15.205 tok/s | 15.025 tok/s | 15.284 tok/s |
| Long prefill wall | 342.585 prompt tok/s | 147.953 prompt tok/s | 338.439 prompt tok/s |
| Toolcall-15 | 93 | 93 | 93 |
| HermesAgent-20 | 73.75 | 73.25 | 73.75 |

The first F16 prototype was correct but too slow because it missed the optimized
heads8 attention path. The heads8 patch recovers the 124k long-prefill
regression, moving from -56.8% versus baseline to about -1.2% versus baseline.

## Highest-Leverage Optimization Questions

1. Does F16 become a real speed win at larger compressed-cache sizes?

   The 124k result is near parity, not a clear win. The best next measurement is
   a same-host direct benchmark ladder at 32k, 128k, 256k/512k, and one stress
   context when stable. Capture prefill, decode, GPU memory, managed-KV fallback
   logs, and server-side progress lines. If F16 is bandwidth-bound, the benefit
   should show up more clearly as compressed rows dominate more of the attention
   window.

2. How much does F16 actually reduce live memory pressure?

   The current front-door speed probe proves throughput but not the exact memory
   win. Record GPU memory before load, after load, after warmup, after long
   prefill, and after repeated decode for both default and F16 builds. Also
   record KV disk usage. This tells us whether F16 creates enough headroom for
   larger context, fewer managed spills, or more concurrent requests.

3. Can the heads8 kernel avoid unpacking F16 into float4 shared memory?

   The current optimized path still converts F16 rows into float4 values before
   dot/accumulate. That preserves the existing math and shared-memory contract,
   which made the patch small and safe, but it also spends conversion work on
   every compressed row load. A second fast path could load `__half2` vectors and
   accumulate with a tighter mixed-precision loop, then compare accuracy and
   speed against the current float4-unpack path.

4. Can the prefill/cuBLAS path avoid materializing all compressed rows as F32?

   The packed prefill path currently materializes F32 rows so cuBLAS can remain
   ignorant of persistent F16 storage. That is a good correctness bridge, but it
   limits upside. Investigate a cuBLASEx or custom mixed-precision path where the
   compressed portion can stay in half longer, while raw KV and output remain
   compatible with the existing DS4 graph.

5. Which remaining attention modes still fall back to generic F16 loaders?

   The important unmasked heads8 paths are now covered. Before upstream PR work,
   audit masked compressed attention, non-512 head dimensions, indexed/two-pass
   variants, and any quality-mode paths. The goal is not to optimize every case,
   but to document which runtime shapes are fast-path-covered and which are
   correct-but-generic.

6. What is the smallest upstreamable surface?

   Keep the feature compile-time opt-in for now. The upstream candidate should
   include only source changes, a short benchmark note, and maybe a targeted
   CUDA regression/smoke recipe. Strip finite fork files, `.scratch`, agent docs,
   and front-door operational metadata before proposing it to `antirez/ds4`.

## Recommended Next Slice

Run a direct DS4 benchmark ladder on `spark-123a` or the next free Spark with the
same model, same branch, same compile pair, and no front-door harness in the
middle:

1. Default CUDA build: 32k, 128k, 256k/512k, stress context if stable.
2. F16 heads8 build: same contexts and prompts.
3. Capture GPU memory and KV disk state around each run.
4. Preserve server logs with prefill progress lines.
5. Only after direct evidence shows a win or a clear neutral result, rerun
   front-door smoke, Toolcall-15, and HermesAgent-20.

This gives us the answer that matters for the next engineering decision: whether
F16 compressed KV is merely parity-at-124k, or whether it opens bigger-context or
lower-memory operating space where baseline DS4 starts to strain.
