# Issue 09: Direct-Model Prefill Repair

## Type

AFK follow-up optimization / bug path.

## Goal

Diagnose and repair the `DS4_CUDA_DIRECT_MODEL=1` path that previously left
enough room for device KV but failed during prefill with an illegal CUDA memory
access.

## Acceptance

- Reproduce or narrow the failing path with a command and captured logs.
- Identify whether the failure is in direct model addressability, tensor-span
  fallback, prefetch, or a CUDA attention kernel consuming direct-model weights.
- Implement the smallest fix that makes direct-model prefill usable when the
  host/GPU stack supports it, or gate direct-model away from the broken path with
  a clear diagnostic if the stack cannot support it.
- Preserve default non-direct-model behavior.
- Spark validation either demonstrates a passing direct-model smoke/prefill run
  or captures the exact unsupported capability/remaining fault with stronger
  evidence than the original checkpoint.

## Result

Implemented and validated on `spark-123a` on 2026-06-23. `DS4_CUDA_DIRECT_MODEL=1`
now requires synchronized HMM/ATS prefetch before CUDA kernels receive direct
model pointers; otherwise DS4 falls back to the safe mapped/cached weight path.
`DS4_CUDA_UNSAFE_DIRECT_MODEL=1` preserves the old raw-pointer behavior for
diagnostics.

The old failing 2k prefill shape passed. The full 32k/512k row also passed with
HMM/ATS direct model active, measuring 99.15 prefill t/s and 11.28 generation
t/s. This fixes the illegal-access failure but is not a speed win against the
normal cached-weight auto policy.
