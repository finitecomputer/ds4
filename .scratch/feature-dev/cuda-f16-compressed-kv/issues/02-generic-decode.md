## Parent

Parent PRD: https://github.com/finitecomputer/ds4/issues/1

## What to build

Teach generic CUDA mixed decode attention to consume compressed attention KV rows
from either F32 or F16 storage. The end-to-end behavior is that an experimental
F16 compressed-cache build can enter the mixed decode path without immediately
rejecting the existing `comp_kv_f16` mode.

This slice should favor a small storage loader and correctness-first dispatch
over fast-path optimization. Existing optimized F32 decode paths should remain
available to default CUDA builds.

## Acceptance criteria

- [ ] Generic mixed decode attention can read F32 compressed attention KV storage.
- [ ] Generic mixed decode attention can read F16 compressed attention KV storage.
- [ ] Host-side validation accepts F16 compressed attention KV byte sizes when the experimental mode is selected.
- [ ] Default CUDA optimized decode dispatch remains available for F32 storage.
- [ ] Experimental F16 mode routes around F32-only optimized decode paths where needed.
- [ ] No CLI, server, or runtime flag is added.

## Blocked by

- https://github.com/finitecomputer/ds4/issues/2
