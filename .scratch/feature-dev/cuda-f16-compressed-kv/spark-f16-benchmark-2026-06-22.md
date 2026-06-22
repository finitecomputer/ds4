# Spark F16 Benchmark Checkpoint - 2026-06-22

## Runtime

- Spark: `spark-123a`
- Experimental clone: `/home/finite/ds4-cuda-f16-compressed-kv`
- Branch: `feature/cuda-f16-compressed-kv`
- Commit: `0ba1d0f`
- Build: `make cuda-spark DS4_CUDA_ATTN_COMP_CACHE_F16=1`
- Experimental server PID: `364755`
- Experimental server log: `/home/finite/ds4-cuda-f16-compressed-kv/logs/ds4-server-f16-20260622T161001-0500.log`
- Baseline restored after experiment: yes
- Restored baseline PID: `397745`

## Evidence

- Benchmark checkpoint: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/README.md`
- Speed probe: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/speed-probe-f16.json`
- Operator frontdoor smoke: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/operator-frontdoor-smoke-f16.json`
- Toolcall/Hermes matrix: `/Users/plebdev/spark-cluster/runs/2026-06-22-ds4-cuda-f16-kv-spark123a-experiment/cli-matrix-results.jsonl`

## Results

| Probe | Baseline | F16 experimental |
| --- | ---: | ---: |
| Decode 256 cap | 15.205 tok/s | 15.025 tok/s |
| Long prefill wall | 342.585 prompt tok/s | 147.953 prompt tok/s |
| Toolcall-15 | score 93; 14 pass, 1 fail | score 93; 14 pass, 1 fail |
| HermesAgent-20 | score 73.75; 10 pass, 3 partial, 7 fail | score 73.25; 10 pass, 2 partial, 8 fail |

## Interpretation

The first CUDA F16 compressed attention-KV prototype is functionally viable but
not performance viable. It passed operator frontdoor smoke, Toolcall-15, and a
full HermesAgent-20 run, but long-prefill throughput regressed by more than half.

Likely cause: the current F16 compressed-KV read path routes around optimized
F32 heads/static attention kernels and falls through the generic CUDA attention
path. The next useful implementation slice is an optimized F16 compressed-KV
CUDA attention path, not upstream publication of this first prototype.
