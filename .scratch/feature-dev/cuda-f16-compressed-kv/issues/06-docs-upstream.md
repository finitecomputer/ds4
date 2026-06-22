## Parent

Parent PRD: https://github.com/finitecomputer/ds4/issues/1

## What to build

Prepare the result for a clean finite staging PR and a possible upstream
`antirez/ds4` proposal. The end-to-end behavior is a small documentation and
handoff packet that explains how to enable the experimental CUDA F16 compressed
attention KV mode, what was validated, what benchmark deltas were observed, and
what fork-local workflow files must be excluded from any upstream PR.

## Acceptance criteria

- [ ] Experimental mode documentation explains that the first opt-in surface is compile-time only.
- [ ] Documentation states that raw sliding-window KV and indexer compressed cache are unchanged.
- [ ] Validation evidence links to the DGX Spark benchmark packet when available.
- [ ] The finite staging PR body can summarize correctness checks, speed checks, known risks, and out-of-scope items.
- [ ] A future upstream PR checklist identifies fork-local workflow scaffolding to strip.
- [ ] The docs avoid claiming default CUDA support or production readiness unless validation explicitly supports that claim.

## Blocked by

- https://github.com/finitecomputer/ds4/issues/6
