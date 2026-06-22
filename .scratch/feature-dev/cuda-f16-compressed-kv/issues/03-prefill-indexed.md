## Parent

Parent PRD: https://github.com/finitecomputer/ds4/issues/1

## What to build

Extend the experimental F16 compressed attention KV path through CUDA prefill,
masked/static mixed attention, indexed mixed attention, and prefill packing.
The end-to-end behavior is that an experimental F16 compressed-cache build can
exercise both prefill and decode phases without falling off a remaining
F16-reject path.

The implementation should keep the packed prefill shape viable by converting
F16 compressed rows into the existing F32 working representation when needed.

## Acceptance criteria

- [ ] Generic mixed prefill can read F16 compressed attention KV storage.
- [ ] Masked/static mixed attention accepts the experimental F16 storage mode.
- [ ] Indexed mixed attention accepts the experimental F16 storage mode without changing the indexer compressed cache.
- [ ] Prefill packing handles F16 compressed attention rows while keeping its output representation compatible with existing consumers.
- [ ] Host-side byte-size validation uses the selected compressed attention KV element size.
- [ ] Default CUDA prefill behavior remains unchanged.

## Blocked by

- https://github.com/finitecomputer/ds4/issues/3
