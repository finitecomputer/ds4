# Spark F16 Benchmark Checkpoint - 2026-06-22

## Runtime

- Spark: `spark-123a`
- Experimental clone: `/home/finite/ds4-cuda-f16-compressed-kv`
- Branch: `feature/cuda-f16-compressed-kv`
- Initial F16 benchmark commit: `0ba1d0f`
- Optimized F16 heads8 benchmark patch: local patch on top of `0ba1d0f`, to be committed on this branch
- Build: `make cuda-spark DS4_CUDA_ATTN_COMP_CACHE_F16=1`
- Experimental server PID: `364755`
- Experimental server log: `/home/finite/ds4-cuda-f16-compressed-kv/logs/ds4-server-f16-20260622T161001-0500.log`
- Optimized heads8 server PID: `432360`
- Optimized heads8 server log: `/home/finite/ds4-cuda-f16-compressed-kv/logs/ds4-server-f16-heads8-20260622T182030-0500.log`
- Baseline restored after experiment: yes
- Restored baseline PID after initial run: `397745`
- Restored baseline PID after optimized run: `465065`

## Evidence

- Benchmark checkpoint: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/README.md`
- Speed probe: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/speed-probe-f16.json`
- Operator frontdoor smoke: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/operator-frontdoor-smoke-f16.json`
- Toolcall/Hermes matrix: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/cli-matrix-results.jsonl`
- Optimized heads8 checkpoint: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-heads8-kv-spark123a-optimized/README.md`
- Optimized speed probe: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-heads8-kv-spark123a-optimized/speed-probe-f16-heads8.json`
- Optimized operator frontdoor smoke: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-heads8-kv-spark123a-optimized/operator-frontdoor-smoke-f16-heads8.json`
- Optimized Toolcall/Hermes matrix: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-heads8-kv-spark123a-optimized/cli-matrix-results.jsonl`

## Results

| Probe | Baseline | Initial F16 | Optimized F16 heads8 |
| --- | ---: | ---: | ---: |
| Decode cap | 15.205 tok/s | 15.025 tok/s | 15.284 tok/s |
| Long prefill wall | 342.585 prompt tok/s | 147.953 prompt tok/s | 338.439 prompt tok/s |
| Toolcall-15 | score 93; 14 pass, 1 fail | score 93; 14 pass, 1 fail | score 93; 14 pass, 1 fail |
| HermesAgent-20 | score 73.75; 10 pass, 3 partial, 7 fail | score 73.25; 10 pass, 2 partial, 8 fail | score 73.75; 11 pass, 2 partial, 7 fail |

## Interpretation

The first CUDA F16 compressed attention-KV prototype was functionally viable but
not performance viable. It passed operator frontdoor smoke, Toolcall-15, and a
full HermesAgent-20 run, but long-prefill throughput regressed by more than half.

The optimized heads8 slice fixed the main regression for the 124k DS4 front-door
shape. It keeps the optimized heads8 kernels, preserves float math and the
float4 shared-memory layout, and only changes compressed-row source loads to
unpack F16 rows into float4 values. Long-prefill throughput moved from 147.953
prompt tok/s to 338.439 prompt tok/s, within about 1.2% of the 342.585 prompt
tok/s baseline.

This is now performance-credible enough for the next upstreamable iteration, but
it should remain opt-in and experimental until the optimized path has broader
context-size coverage, memory evidence, and a cleaner upstream PR shape.
