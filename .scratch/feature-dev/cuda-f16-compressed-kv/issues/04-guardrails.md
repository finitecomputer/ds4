## Parent

Parent PRD: https://github.com/finitecomputer/ds4/issues/1

## What to build

Add the guardrails and checks that make the experimental path safe to keep in
the fork while preserving the normal release path. The end-to-end behavior is
that default CUDA remains optimized and unchanged, while experimental F16 mode
has explicit dispatch boundaries and clear failure behavior.

This slice should make the implementation reviewable as a narrow DS4 storage
experiment rather than a broad attention rewrite.

## Acceptance criteria

- [ ] Default CUDA build compiles without enabling F16 compressed attention KV.
- [ ] Experimental CUDA build compiles with F16 compressed attention KV enabled.
- [ ] F32-only optimized paths are not silently used with F16 compressed storage.
- [ ] Any fallback from optimized F32-only dispatch to generic F16-capable dispatch is intentional and documented near the code.
- [ ] Existing tests or compile checks cover the default build surface.
- [ ] The implementation has compact comments only where cache lifetime, storage format, or dispatch boundaries are non-obvious.

## Blocked by

- https://github.com/finitecomputer/ds4/issues/4
