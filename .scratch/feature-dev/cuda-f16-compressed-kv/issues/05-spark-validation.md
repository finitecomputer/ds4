## Parent

Parent PRD: https://github.com/finitecomputer/ds4/issues/1

## What to build

Run the DGX Spark validation and benchmark ladder for the experimental CUDA F16
compressed attention KV build. The end-to-end behavior is an evidence packet
that compares the experimental build against the current DS4 frontdoor
checkpoint and says clearly whether the result is correct enough and fast enough
to keep pursuing.

This slice requires DGX Spark access and may remain HITL until the implementation
commit is ready to test on the Spark host.

## Acceptance criteria

- [ ] Experimental build passes `make cuda-regression` on DGX Spark.
- [ ] Experimental build passes a short CLI smoke on DGX Spark.
- [ ] Experimental build passes at least one long-context prompt on DGX Spark.
- [ ] Direct before/after `ds4-bench` results are captured for 32K, 128K, and one stress context when stable.
- [ ] Benchmark evidence records prefill, decode, memory, and managed-KV fallback logs.
- [ ] A distinct experimental F16 frontdoor alias is used before frontdoor checks.
- [ ] Existing frontdoor speed probe and smoke checks are rerun through `spark-cluster`.
- [ ] Toolcall-15 is rerun if direct and frontdoor checks pass.
- [ ] HermesAgent-20 is rerun only after cheaper checks pass.

## Blocked by

- https://github.com/finitecomputer/ds4/issues/5
