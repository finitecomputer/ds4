## Parent

Parent PRD: https://github.com/finitecomputer/ds4/issues/1

## What to build

Create the first opt-in CUDA F16 compressed attention KV storage path without
changing default CUDA behavior. The experimental build should be able to select
F16 compressed attention KV storage at compile time, stage compressed attention
rows in F32 as DS4 already does, and commit those rows into F16 persistent
storage through a CUDA tensor primitive.

This slice is complete when the storage mode is selectable for experiments and
the default build remains unchanged.

## Acceptance criteria

- [ ] Default CUDA build behavior remains unchanged.
- [ ] Experimental CUDA build can select F16 compressed attention KV storage at compile time.
- [ ] CUDA implements the tensor primitive needed to copy F32 compressed attention rows into F16 storage.
- [ ] The implementation returns success/failure through the existing CUDA error-handling style.
- [ ] Raw sliding-window KV and indexer compressed cache remain unchanged.
- [ ] Non-CUDA backends keep their existing behavior.
- [ ] Local compile checks that do not require DGX Spark pass where available.

## Blocked by

None - can start immediately.
