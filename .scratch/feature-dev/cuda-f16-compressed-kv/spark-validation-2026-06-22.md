# Spark Validation Checkpoint - 2026-06-22

## Host

- Spark: `spark-123a`
- Validation clone: `/home/finite/ds4-cuda-f16-compressed-kv`
- Branch: `feature/cuda-f16-compressed-kv`
- Live DS4 server left untouched: `/home/finite/ds4/ds4-server --ctx 124000 --kv-disk-dir /home/finite/ds4-data/kv --kv-disk-space-mb 8192 --host 0.0.0.0 --port 8000`

## Passed

- Default CUDA build:
  - `make clean`
  - `make cuda-spark`
- Experimental CUDA build:
  - `make clean`
  - `make cuda-spark DS4_CUDA_ATTN_COMP_CACHE_F16=1`
- CUDA regression:
  - `make cuda-regression`
  - Result: `cuda long-context regression: OK`

## Not Run

- Experimental short CLI smoke.
- Experimental long-context prompt.
- Direct `ds4-bench` before/after comparison.
- Frontdoor smoke through a distinct experimental F16 alias.
- Toolcall-15 and HermesAgent-20.

## Reason

`spark-123a` is actively serving DS4 on port 8000 and the process holds about
105 GB of GPU memory. Other Sparks checked were also occupied:

- `spark-cbee`: VLLM engine using about 96 GB.
- `spark-ee82`: llama-server plus VLLM using about 97 GB combined.
- `spark-2f73`: llama-server plus VLLM processes using about 106 GB combined.

Do not start an experimental DS4 model process until a Spark is freed or the
operator explicitly approves moving or stopping an existing runtime.
