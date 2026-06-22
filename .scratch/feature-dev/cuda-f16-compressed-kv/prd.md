## Problem Statement

DS4 already has the abstraction for storing the compressed attention KV cache as
F16, and Metal uses that storage mode today. CUDA on DGX Spark does not: the
current CUDA backend keeps the compressed attention KV cache in F32 and rejects
the existing `comp_kv_f16` path in multiple attention entry points.

The user wants to investigate whether CUDA F16 compressed attention KV can be
implemented in a minimal, organized, upstreamable way, validated on DGX Spark,
and compared against the current DS4 benchmark checkpoints captured through the
finite Spark frontdoor.

This should be treated as CUDA/Metal parity and a stepping stone toward future
compressed-cache work, not as a CUDA fork, NVIDIA stack patch, or broad
TurboQuant implementation.

## Solution

Add an opt-in CUDA implementation for F16 compressed attention KV storage. The
first implementation must preserve default CUDA behavior and limit scope to the
compressed attention KV cache only. It must not change raw sliding-window KV or
the indexer compressed cache.

The implementation should be correctness-first:

- Make the compressed attention KV storage mode overrideable at compile time.
- Add the CUDA tensor primitive needed to copy F32 staging rows into F16 storage.
- Teach CUDA attention consumers to read compressed attention KV rows from either
  F32 or F16 storage.
- Route experimental F16 mode through generic/correct paths before optimizing.
- Keep optimized F32 paths unchanged.
- Validate on DGX Spark before making any speed or memory claims.

If the experiment succeeds, the finite fork should produce a clean staging PR.
Any later upstream `antirez/ds4` PR should be trimmed to source, tests, and
maintainer-relevant documentation, excluding finite fork workflow scaffolding.

## User Stories

1. As a DS4 contributor, I want CUDA compressed attention KV storage to support
   F16 experimentally, so that CUDA can explore the same storage optimization
   already used by Metal.
2. As a DGX Spark operator, I want to opt into F16 compressed attention KV
   without changing default CUDA behavior, so that I can test the optimization
   safely.
3. As a DS4 maintainer, I want the first patch to be narrow and modular, so that
   I can review correctness risk without also reviewing public API churn.
4. As a DS4 maintainer, I want raw sliding-window KV and indexer compressed cache
   to remain unchanged, so that any drift can be attributed to compressed
   attention KV storage only.
5. As a CUDA backend developer, I want a reusable tensor copy primitive from F32
   to F16, so that the shared graph can stage compressed rows in F32 and commit
   them into F16 storage.
6. As a CUDA backend developer, I want generic attention consumers to load
   compressed KV rows through a narrow helper, so that F32 and F16 storage can be
   handled without duplicating all attention logic immediately.
7. As a performance investigator, I want the first F16 implementation to route
   around optimized F32-only kernels when necessary, so that correctness can be
   established before fast-path restoration.
8. As a performance investigator, I want the prefill packing path to remain
   viable with F16 compressed cache, so that prefill speed is not unnecessarily
   sacrificed.
9. As a DS4 tester, I want the experimental build to pass CUDA regression and
   real prompt smoke tests before speed comparison, so that benchmark results
   are not collected from a broken path.
10. As a DS4 tester, I want long-context behavior tested before speed claims, so
    that compressed-cache correctness is exercised where the cache matters.
11. As a Spark operator, I want before/after `ds4-bench` measurements at short,
    normal-long, and stress contexts, so that I can see whether the optimization
    changes prefill, decode, memory, or managed-KV fallback behavior.
12. As a frontdoor operator, I want a distinct experimental F16 alias before
    rerunning frontdoor checks, so that baseline DS4 and experimental DS4 are
    not confused.
13. As an agent benchmark consumer, I want Toolcall-15 rerun after the direct
    checks pass, so that structured tool-use regressions are caught cheaply.
14. As an agent benchmark consumer, I want HermesAgent-20 rerun only after cheap
    checks pass, so that the expensive suite is used for behavior regression
    confidence rather than early debugging.
15. As a future upstream reviewer, I want finite workflow files excluded from
    the upstream PR, so that the public patch is focused on DS4 itself.

## Implementation Decisions

- The feature is opt-in only for the first implementation.
- The opt-in surface is compile-time only. Do not add CLI, server, or runtime
  flags for the first implementation.
- Existing CUDA defaults must remain unchanged.
- The first implementation scope is compressed attention KV storage only.
- Raw sliding-window KV cache remains unchanged.
- Indexer compressed cache remains unchanged.
- The shared graph storage policy should become overrideable without changing
  the default behavior on existing platforms.
- CUDA needs a real F32-to-F16 tensor copy primitive because the shared graph
  stages compressed rows in F32 and commits them into the persistent cache.
- CUDA attention consumers should read compressed KV through a narrow storage
  loader that can handle F32 and F16.
- Optimized F32 CUDA attention paths should remain available for default CUDA.
- Experimental F16 mode can initially bypass F32-only optimized paths to reach a
  correct baseline.
- Prefill packing should be made F16-aware so the existing packed prefill shape
  can remain viable.
- Any fast-path F16 vectorization belongs after the correctness baseline, unless
  the generic path cannot pass the minimum validation.
- The finite fork may carry workflow scaffolding and issues, but the upstream
  patch should be trimmed before proposing it to `antirez/ds4`.

## Testing Decisions

- Good tests verify observable behavior through DS4's public build, regression,
  CLI, server, and benchmark surfaces rather than only checking internal helper
  implementations.
- Default CUDA behavior must be preserved by building without the experimental
  compile-time define.
- The experimental CUDA build must compile on DGX Spark.
- Minimum correctness gate before speed comparison:
  - `make cuda-regression`
  - short CLI smoke
  - at least one long-context prompt on DGX Spark
- Official vector checks should be run where practical. They are desirable, but
  are best-effort for the first prototype if fixture or backend assumptions make
  them unsuitable as a hard blocker.
- Direct benchmark ladder before publishing:
  - `ds4-bench` before/after on the same Spark, same model, and same commit pair
  - 32K context
  - 128K context
  - one stress context such as 512K or 1M when stable
  - record prefill, decode, memory, and managed-KV fallback logs
- Frontdoor benchmark ladder after direct checks pass:
  - distinct experimental F16 alias in `spark-cluster`
  - existing speed probe
  - `/v1/models`
  - non-streaming chat smoke
  - streaming chat smoke
  - tool-call smoke
- Agent behavior ladder after frontdoor smoke:
  - Toolcall-15 first
  - HermesAgent-20 only after Toolcall-15 and direct/frontdoor checks pass

## Out of Scope

- Making CUDA F16 compressed attention KV the default.
- Adding a CLI, server, or runtime flag for the first implementation.
- Changing raw sliding-window KV storage.
- Changing indexer compressed cache storage.
- Implementing TurboQuant, TQ4, or 4-bit KV cache formats.
- Forking or patching CUDA, NVIDIA drivers, or CUDA toolkit.
- Broad rewrite of CUDA attention kernels before a correctness baseline exists.
- Production deployment or public-beta traffic changes from this feature-dev
  loop.

## Further Notes

Current finite DS4 frontdoor checkpoints to compare against live in
`spark-cluster`:

- Direct speed probe captured decode at `15.205` tokens/sec and long-prompt
  prefill at `342.585` prompt tokens/sec with a 124K context window.
- Toolcall-15 baseline: 14 pass, 1 fail, score 93.
- HermesAgent-20 baseline: 10 pass, 3 partial, 7 fail, score 73.75.

The experiment should avoid overclaiming memory savings. The expected memory
impact is useful but not massive because DS4 already stores long context as
compressed attention rows. The bigger value is CUDA/Metal parity, bandwidth
pressure reduction, and a clean stepping stone for future cache formats.
