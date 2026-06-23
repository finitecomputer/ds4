# Issue 08: Smarter CUDA KV Placement

## Type

AFK follow-up optimization.

## Goal

Replace the fixed "moderate KV footprint" heuristic with a smarter CUDA policy
that chooses device KV whenever the allocation is plausibly safe, falls back to
managed KV when real pressure says it should, and keeps the existing override
environment variables for experiments.

## Acceptance

- `DS4_CUDA_MANAGED_KV_CACHE` and `DS4_CUDA_NO_MANAGED_KV_CACHE` still force the
  policy.
- The default policy considers KV bytes, graph/context bytes, CUDA free/total
  memory, reserve, and a bounded pressure ratio rather than only a fixed 6 GiB
  cutoff.
- The policy logs enough detail under `DS4_CUDA_MANAGED_KV_VERBOSE=1` to explain
  why it picked managed or device.
- `ds4_session_context_allocation()` still reports `managed_kv_cache`.
- Local build/diff checks pass.
- Spark validation records policy choice and token speed against the previous
  checkpoint.

## Result

Implemented and validated on `spark-123a` on 2026-06-23. The 32k/512k run chose
`moderate kv within pressure budget -> device` with KV 4.37 GiB, context 8.37
GiB, adaptive device max 8.00 GiB, and pressure limit 91.27 GiB.

Speed: 360.97 prefill t/s and 12.23 generation t/s. This is still materially
better than the old managed-KV checkpoint and approximately flat on generation
versus the first device-KV checkpoint.

## Notes

The previous checkpoint proved that a 4.37 GiB KV allocation could safely stay
on device even though the old free-memory check picked managed KV. This slice
should preserve that win without hard-coding only the observed Spark case.
