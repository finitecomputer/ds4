# DFlash cutover checkpoint - 2026-06-27

## Judgment

Cut the DS4 fork over to DFlash focus.

Checkpoint refresh at `2026-06-27 17:52 CDT`: this is now the active DS4 fork
development line. The older DS4 fork optimization work should be retired from
the active roadmap and preserved only as evidence, rollback context, and a
baseline for comparison. "Throw away" means stop carrying that line forward, not
delete the evidence that tells us why it was not the best next move.

The DFlash path is going well enough to be the main line of work. It has crossed
the important early risk boundary: the real DeepSeek V4 Flash DFlash artifact is
not just a vague idea anymore. The config shape is understood, the safetensors
layout is validated and mapped, the BF16 weights are readable, target hidden
taps have a DS4-side seam, and the draft transformer now has a CPU reference
path through attention, MLP, block execution, logits, and token mapping.

This is still not DFlash-through-frontdoor. It is executor groundwork. But it is
cleaner and higher leverage than continuing to chase the older DS4 fork
optimization branches. The older fork work produced useful evidence and some
modest wins, but the slope from here is likely in DFlash speculative execution,
not another small KV/cache tweak.

## Active worktree

- Worktree: `/Users/plebdev/Desktop/Projects/finite/ds4-dflash-clean`
- Branch: `codex/ds4-dflash-clean`
- Base: `80ebbc3 Merge pull request #319 from rinaldofesta/fix/eval-grader-false-negatives`
- Current staged executor code head: `7f12901 Add DFlash smoke evidence validator`
- Checkpoint anchor: `3757baf Checkpoint DFlash fork cutover decision`
- Branch state: ahead of `origin/main` with DFlash executor and checkpoint
  commits; use `git log --oneline` for the exact current count.
- Code working tree before this documentation refresh: clean.

The branch head may include documentation-only checkpoint commits above the
staged executable DFlash code. The staged Spark archive remains pinned to the
executable DFlash code at `7f12901` until executable code changes and is
rebuilt/restaged on the Spark.

## Current real artifact shape

The staged DeepSeek V4 Flash DFlash artifact on `spark-123a` is:

`/home/finite/ds4-dflash/deepseek-v4-flash-all-swa-muon-speculators-50k`

Its config shape is:

- `architectures`: `["DFlashDraftModel"]`
- `aux_hidden_state_layer_ids`: `[3, 13, 23, 32, 42]`
- `block_size`: `8`
- `speculative_tokens`: `7`
- `draft_vocab_size`: `32000`
- `mask_token_id`: `1`
- `target_hidden_size`: `null`, defaulting to draft hidden size in DS4
- `hidden_size`: `4096`
- `vocab_size`: `129280`
- `num_hidden_layers`: `5`
- `num_attention_heads`: `64`
- `num_key_value_heads`: `1`
- `head_dim`: `256`
- `intermediate_size`: `2048`
- `hc_mult`: `4`
- `rope_theta`: `10000`
- `sliding_window`: `2048`
- `sliding_window_non_causal`: `false`
- `layer_types`: all five layers are `sliding_attention`

## DFlash commit stack

1. `b7caaef Add DFlash config gate`
   - Added `--dflash` config loading behind fail-closed inspect behavior.
   - Established DFlash as separate from the existing MTP path.

2. `d9ee6af Add DS4 DFlash artifact validation and taps`
   - Parsed the real DeepSeek V4 Flash DFlash config schema.
   - Validated target shape: hidden size, vocab size, target layer ids, mask id,
     draft dimensions.
   - Added `ds4_session_eval_layer_taps` so target hidden states can be captured
     at the DFlash auxiliary layer ids.

3. `4115aa6 Bind DFlash safetensors weights`
   - Mapped `model.safetensors`.
   - Bound the required real artifact tensors:
     `d2t`, `t2d`, embeddings, `fc`, norms, `lm_head`, and all per-layer
     attention and MLP weights.
   - Added BF16-to-f32 tensor reads.
   - Added `ds4_dflash_prepare_block_inputs`: projects captured target taps
     through `fc.weight`, applies `hidden_norm`, and materializes anchor/mask
     embeddings for the draft block.

4. `75b22b5 Add DFlash CPU draft MLP path`
   - Added `ds4_dflash_cpu_eval_mlp`.
   - Implements post-attention RMSNorm, BF16 gate/up/down projections, SiLU
     gating, and residual add.
   - Covered with a tiny safetensors fixture using mapped BF16 weights.

5. `6e55923 Add DFlash CPU attention path`
   - Added `ds4_dflash_cpu_eval_attention`.
   - Implements single-anchor DFlash attention over mapped BF16 weights:
     input RMSNorm, q/k/v/o projections, q/k head RMSNorm, Qwen3-style RoPE
     positions, target+noise KV, softmax, and residual add.
   - The public API explicitly requires caller-filtered target prefix rows and
     one synthetic block.

6. `25ca889 Checkpoint DFlash cutover state`
   - Preserved the cutover recommendation in this checkpoint.
   - Marked DFlash as the active DS4 fork development line while preserving the
     older fork evidence as rollback context.

7. `eb4121e Add DFlash CPU block logits path`
   - Added `ds4_dflash_cpu_eval_layer` and `ds4_dflash_cpu_eval_block`.
   - Added final `norm.weight` + `lm_head.weight` logits evaluation.
   - Added greedy draft token selection plus `d2t`/`t2d` mapping validation.

8. Current checkpoint refresh
   - Adds a bounded `ds4_dflash_hidden_history` primitive for projected target
     hidden rows.
   - Keeps rows in position order, trims to newest visible prefix rows, and
     handles ring overflow without losing chronological copy-out order.

9. Current executor slice
   - Splits target-hidden projection and synthetic noise embedding prep into
     separate DFlash library APIs.
   - Adds session-owned bounded DFlash target history allocation, reset, and
     rewind behavior.
   - Routes DFlash-configured graph sessions through layer-tap sync/eval helpers
     so accepted target tokens can populate projected history.
   - Adds `ds4_session_dflash_propose_argmax`, an internal local proposal seam
     that consumes projected history, runs the CPU draft block/logits path, and
     maps draft tokens back to target tokens.

10. Current verifier slice
   - Adds a correctness-first DFlash speculative verifier loop.
   - The loop commits the normal target token, proposes a DFlash block, compares
     each proposed target token against exact target logits, and commits only
     the matching prefix through normal tapped target eval.
   - Adds `DS4_DFLASH_SPEC_LOG` and `DS4_DFLASH_TIMING` debug output.
   - Extends CLI/server greedy speculative dispatch to call the shared
     speculative entry point for either MTP or DFlash.

11. Current runtime-smoke gate slice
   - Keeps default `--dflash` startup fail-closed for generation.
   - Allows local runtime smoke only when `DS4_DFLASH_EXPERIMENTAL_RUN=1` is
     present and the selected backend can capture target graph taps.
   - Fixes the non-REPL CLI greedy dispatch so DFlash-configured engines use the
     session speculative loop instead of silently falling through to the plain
     argmax helper.
   - Adds `tests/dflash_runtime_smoke.sh`, which compares baseline greedy stdout
     against DFlash-enabled greedy stdout and requires DFlash verifier logs.

12. Real-artifact config compatibility slice
   - Validates against the public
     `inference-optimization/dflash-DeepSeek-V4-Flash-all-swa-muon-speculators-50k`
     config shape.
   - Accepts optional JSON `null` for `target_hidden_size`, matching the real
     artifact, and defaults it to the draft hidden size.

13. Current real-artifact parity slice
   - Parses optional `rope_parameters.rope_theta` from the DFlash config.
   - Uses the parsed RoPE theta in the CPU DFlash attention path instead of the
     previous hardcoded default.
   - Covers the official DeepSeek V4 Flash DFlash fixture with
     `rope_theta: 10000`.
   - Cleans the DFlash tap buffer overflow guards that emitted ARM
     type-limit warnings in the Spark CUDA build.

14. Current runtime evidence slice
   - Strengthens `tests/dflash_runtime_smoke.sh` from a transient stdout compare
     into a persistent evidence-producing smoke gate.
   - Captures baseline/DFlash stdout and stderr, prompt, metadata, stdout diff,
     DFlash verifier/timing summary, and the parsed draft/verify counts.
   - Requires at least one verified DFlash draft token by default, so the first
     real runtime smoke cannot pass on a shallow "log line existed" signal.

15. Current DFlash block-alignment slice
   - Aligns the DS4 proposal path with upstream DFlash block semantics: base
     target hidden rows are strictly before the anchor position, while synthetic
     block row 0 carries the accepted anchor token.
   - Skips synthetic block row 0 when returning generated draft suffix tokens.
   - Caps copied target hidden rows to the configured DFlash sliding window and
     reserves history room for the visible window plus the anchor row.
   - Adds focused unit coverage for both the hidden-history anchor exclusion and
     the generated-draft suffix selection rule.

16. Current sliding-attention parity slice
   - Parses the real artifact's optional `sliding_window_non_causal` flag.
   - Enforces causal same-block synthetic attention when
     `sliding_window > 0 && !sliding_window_non_causal`, matching the public
     DeepSeek V4 Flash DFlash artifact's `sliding_attention` layers.
   - Adds focused unit coverage that compares causal and non-causal synthetic
     block behavior on the tiny BF16 fixture.

17. Current token-map hardening slice
   - Extends the tiny safetensors fixture so tests can create invalid DFlash
     `d2t` and `t2d` mappings.
   - Proves the draft suffix selector rejects a proposed draft token before it
     reaches verifier comparison when the mapped target token is outside the
     target vocab or not marked admissible by `t2d`.

18. Current verifier-evidence slice
   - Factors DFlash runtime-smoke stderr summarization into
     `tests/dflash_runtime_summary.awk`.
   - Persists `misses` and `rejected_draft_tokens` alongside attempts, drafted,
     verified, accepted, and timing counts.
   - Adds a synthetic parser test so accept/reject evidence accounting is covered
     without loading the real target model.

19. Current direct verifier-summary slice
   - Extends the DFlash verifier's final `ds4: dflash spec ...` summary line to
     emit `misses` and `rejected_draft_tokens` directly.
   - Keeps the runtime-smoke parser backward-tolerant for older logs that only
     have `ds4: dflash spec miss ...` lines.
   - Covers both direct-summary and legacy-miss parsing paths in
     `tests/dflash_runtime_summary_test.sh`.

20. Current runtime-smoke evidence-gate slice
   - Makes `tests/dflash_runtime_smoke.sh` fail closed when parsed summary
     fields are missing or non-numeric.
   - Requires `accepted_including_anchor` to cover both verifier attempts and
     verified draft tokens.
   - Requires `rejected_draft_tokens` to cover verifier misses.
   - Adds syntax/static coverage for those smoke-script gates to
     `make dflash-summary-test`, the same hook used by Spark `remote-ready`.

21. Current tap-order config gate slice
   - Rejects non-increasing DFlash `target_layer_ids` during config validation,
     matching the strict layer order required by `ds4_session_eval_layer_taps`.
   - Adds focused coverage for unsorted target-tap layers before any runtime
     proposal or verifier path can run.

22. Current impossible-acceptance evidence gate slice
   - Rejects runtime-smoke summaries where `accepted_including_anchor` exceeds
     `drafted + attempts`.
   - Preserves the invariant that each verifier attempt can accept at most its
     anchor plus the drafted suffix tokens.
   - Adds the same static smoke-script coverage used by Spark `remote-ready`.

23. Current smoke-evidence validator slice
   - Adds `tests/dflash_smoke_evidence_validate.py`, a reusable validator for
     the preserved runtime-smoke evidence directory.
   - Checks required artifacts, exact target GGUF and DFlash metadata, baseline
     stdout equality, numeric verifier summary fields, accepted-anchor bounds,
     rejection accounting, timing presence, and minimum verified draft tokens.
   - Wires the validator into `dflash-summary-test` with pass/fail synthetic
     evidence so Spark `remote-ready` proves the validator is present and
     runnable in the staged archive.

## What is proved

- The DS4 fork can recognize and validate the real DFlash artifact shape for
  DeepSeek V4 Flash.
- The real safetensors layout can be validated before runtime use.
- Required BF16 tensors can be memory-mapped, bound, and read by name.
- Captured target hidden taps can be projected into the DFlash draft hidden
  space.
- Anchor and mask token embeddings can be materialized from the draft artifact.
- CPU reference graph primitives exist for the DFlash decoder layer, block,
  final logits, and target-token proposal path.
- A bounded projected-hidden history can retain the target prefix rows that the
  DFlash draft block needs to attend to.
- DFlash-configured DS4 graph sessions now have a path to capture taps during
  prompt sync and target-token eval, project them, and call the local draft
  proposal helper.
- The shared speculative generation entry point now has a DFlash accept/reject
  path that preserves exact target-token semantics by verifying proposals
  against target logits before committing them.
- The DFlash proposal path now matches the upstream anchor-block mask shape:
  target context excludes the accepted anchor row, and generated drafts begin at
  synthetic block row 1. For the public DeepSeek V4 Flash artifact, the copied
  target rows are also capped to its `sliding_window: 2048`.
- A guarded local smoke command now exists so the real target model plus real
  DFlash artifact can be tested before any Spark deployment work; it now leaves
  an evidence directory and requires an accepted DFlash draft token by default.
- The official DeepSeek V4 Flash DFlash config shape with
  `target_hidden_size: null` and `rope_parameters.rope_theta: 10000` is covered
  by the focused DFlash config test.
- DFlash target tap layers now must be strictly increasing at artifact
  validation time, so malformed configs fail before hidden-state tap capture.
- The real artifact's anchor-block shape is understood: synthetic row 0 is the
  accepted anchor token, generated draft tokens begin at synthetic row 1, and
  the copied target rows are the visible prefix before the anchor.
- The CPU attention path now respects the real artifact's
  `sliding_window_non_causal: false` setting by masking future synthetic rows
  inside a sliding-attention draft block.
- DFlash token selection now has negative coverage for bad vocabulary maps, so
  invalid draft-to-target proposals fail closed before the verifier can compare
  or commit them.
- Runtime-smoke evidence now records verifier rejection behavior directly:
  `misses` counts DFlash verifier misses and `rejected_draft_tokens` estimates
  the uncommitted draft suffix rejected by those misses.
- The verifier itself now emits those rejection fields in the machine-parsed
  summary line, so future runtime smoke does not have to infer accept/reject
  accounting only from per-miss debug lines.
- The runtime-smoke script now requires accepted-anchor and rejection accounting
  to be present and internally consistent before evidence can pass.
- The runtime-smoke script now also rejects impossible accepted-anchor counts
  before Spark launch tooling can validate or consume that evidence.
- The preserved smoke-evidence validator is now a DS4-side checked-in script,
  so Spark launch tooling can delegate to the exact validator staged with the
  executable DS4 DFlash archive.
- The real public DFlash artifact can be inspected on `spark-123a` against the
  live DS4 target GGUF with an isolated inspect lock.
- These primitives are covered by focused C tests with a tiny safetensors
  fixture that exercises actual mapped BF16 bytes rather than synthetic arrays
  only.

## What is not done

- `--dflash` still fails closed for generation by default. It opens and
  validates the artifact, then requires either inspect-only mode or the explicit
  local smoke flag `DS4_DFLASH_EXPERIMENTAL_RUN=1`.
- The DFlash verifier loop is not yet runtime-proven against the real target
  model plus real DFlash artifact, so `--dflash` should stay fail-closed for
  normal generation.
- There is no GPU DFlash executor yet. The CPU path is a correctness/reference
  path, not the production performance target.
- There is no Spark deployment slot or alias for DFlash-through-DS4 yet. A
  read-only fleet audit found the available Sparks currently occupied, so the
  live DS4 frontdoor should remain untouched until a separate test slot is
  explicitly available.

## Verification at this checkpoint

Commands run successfully in `/Users/plebdev/Desktop/Projects/finite/ds4-dflash-clean`:

- `make dflash-config-test`
- `make dflash-summary-test`
- `make cpu`
- `make`
- `git diff --check`
- `./ds4_test --server`
- `/opt/homebrew/opt/llvm/bin/clang -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c99 -I. tests/ds4_dflash_config_test.c ds4_dflash.c -lm -pthread -o /tmp/ds4_dflash_config_test_asan && /tmp/ds4_dflash_config_test_asan`

`make test` progressed through:

- `tests/test_q4k_dot`
- `tests/ds4_dflash_config_test`
- `ds4-eval --self-test-extractors`
- `ds4_agent_test`

Then it stopped at the existing local model fixture issue:

```text
ds4: cannot open model 'ds4flash.gguf': No such file or directory
```

The CPU build still emits existing unused-function warnings in `ds4.c`; those
are not introduced by the DFlash files.

Additional Spark-side evidence:

- A read-only live-fleet check at `2026-06-27 15:44 CDT` found
  `spark-123a` still serving the live DS4 frontdoor on port `8000` with the
  81G DeepSeek V4 Flash target GGUF.
- The public DFlash artifact is staged at:
  `/home/finite/ds4-dflash/deepseek-v4-flash-all-swa-muon-speculators-50k`
- `spark-ee82`, `spark-cbee`, and `spark-2f73` all had active model-serving
  workloads, so there was still no safe free slot for a DFlash runtime smoke or
  test alias.
- A CUDA Spark build of the DFlash branch succeeded.
- After the sliding-attention parity slice, the committed code tree was archived
  to a separate directory on `spark-123a`; `make cuda-spark` and
  `make dflash-config-test` both passed there.
- Inspect-only artifact validation passed with:

```text
ds4: DFlash draft artifact opened: ... (block=8 draft=7 target_layers=5 tensors=62 bound=62)
```

This proves artifact binding/validation on the actual Spark host. It does not
prove generation-time accept/reject correctness yet.

A second read-only live-fleet check at `2026-06-27 15:55 CDT` still found no
safe Spark slot for runtime smoke or a separate frontdoor alias:

- `spark-123a`: live DS4 frontdoor on `0.0.0.0:8000`
- `spark-ee82`: active Dynamo/vLLM Qwen3 Next workload
- `spark-cbee`: active vLLM Gemma DFlash workload on `0.0.0.0:8034`
- `spark-2f73`: active llama-server workloads on `8032`, `8042`, and `8043`

A fresh read-only live-fleet check at `2026-06-27 16:11 CDT` returned the same
blocking conclusion:

- `spark-123a`: live DS4 frontdoor process on `0.0.0.0:8000`
- `spark-ee82`: active Dynamo/vLLM Qwen3 Next workload
- `spark-cbee`: active vLLM Gemma DFlash workload on `0.0.0.0:8034`
- `spark-2f73`: active llama-server workloads on `8032`, `8042`, and `8043`
- safe DFlash test-slot hosts: none

A fresh read-only live-fleet check at `2026-06-27 16:52 CDT` again returned the
same deployment gate:

- `spark-123a`: live DS4 frontdoor process on `0.0.0.0:8000`
- `spark-ee82`: active Dynamo/vLLM Qwen3 Next workload
- `spark-cbee`: active vLLM Gemma DFlash workload on `0.0.0.0:8034`
- `spark-2f73`: active llama-server workloads on `8032`, `8042`, and `8043`,
  with GPU utilization observed at 95 percent during the audit
- safe DFlash test-slot hosts: none

A fresh read-only live-fleet check at `2026-06-27 17:30 CDT` again returned the
same deployment gate:

- `spark-123a`: live DS4 frontdoor process on `0.0.0.0:8000`
- `spark-ee82`: active Dynamo/vLLM Qwen3 Next workload
- `spark-cbee`: active vLLM Gemma DFlash workload on `0.0.0.0:8034`
- `spark-2f73`: active llama-server workloads on `8032`, `8042`, and `8043`
- safe DFlash test-slot hosts: none

The previous executable tree was archived to `spark-123a` at:

`/home/finite/ds4-dflash/ds4-dflash-clean-aba136d`

The archive is stamped with full commit
`aba136d74f0cd9a5b293926c1d468c108d47682b`. In that isolated tree,
`make cuda-spark`, `make dflash-config-test`, and `make dflash-summary-test`
passed. No DFlash server was started and no live route was mutated.

The Spark helper `remote-ready` check passed at `2026-06-27 17:31 CDT`, verifying
the staged archive commit stamp, target GGUF, DFlash artifact path, and remote
`ds4_dflash_config_test` plus `dflash_runtime_summary_test` without starting a
server or changing routes.

A fresh non-mutating refresh at `2026-06-27 17:35 CDT` kept the same conclusion:

- Spark-side `tools/ds4_dflash_test_slot.py validate` passed for the pending
  DFlash test aliases and blocked admission records.
- Spark-side `remote-ready` passed against `spark-123a`, verifying the
  `aba136d` archive stamp, target GGUF, DFlash artifact, remote config test, and
  remote summary parser test.
- The live Spark audit still found no safe DFlash test-slot host:
  `spark-123a` had the production DS4 frontdoor on `0.0.0.0:8000`,
  `spark-2f73` had active llama-server workloads on `8032`, `8042`, and `8043`,
  `spark-cbee` had the vLLM Gemma DFlash workload on `8034`, and `spark-ee82`
  had the active Dynamo/vLLM Qwen3 Next workload plus an Ornith runner process.
- No DFlash runtime smoke ran, no `8050` test server started, and no Dynamo
  Front Door route changed.

The current executable tree was then archived to `spark-123a` at:

`/home/finite/ds4-dflash/ds4-dflash-clean-c83c2eb`

The archive is stamped with full commit
`c83c2ebc7f018f31491743eaf659351f660d7fd4`. In that isolated tree,
`make cuda-spark`, `make dflash-config-test`, and `make dflash-summary-test`
passed. No DFlash server was started and no live route was mutated.

The Spark helper `remote-ready` check then passed at `2026-06-27 17:43 CDT`,
verifying the `c83c2eb` archive stamp, target GGUF, DFlash artifact path, and
remote `ds4_dflash_config_test` plus `dflash_runtime_summary_test` without
starting a server or changing routes.

A read-only live audit at `2026-06-27 17:44 CDT` still found no safe
DFlash test-slot host, so the runtime smoke and `8050` launch gate remain
closed.

The current executable tree was then archived to `spark-123a` at:

`/home/finite/ds4-dflash/ds4-dflash-clean-7f12901`

The archive is stamped with full commit
`7f12901c611124818e1e5a4ad267d31d134fb940`. In that isolated tree,
`make cuda-spark`, `make dflash-config-test`, and `make dflash-summary-test`
passed. No DFlash server was started and no live route was mutated.

The Spark helper `remote-ready` check then passed at `2026-06-27 17:54 CDT`,
verifying the `7f12901` archive stamp, target GGUF, DFlash artifact path, and
remote `ds4_dflash_config_test` plus `dflash_runtime_summary_test` without
starting a server or changing routes.

Spark `validate-smoke-evidence` was also checked against synthetic remote
evidence on `spark-123a`: a valid evidence directory passed with
`min_verified=2`, and an impossible accepted-anchor directory failed with
`accepted-anchor count exceeds drafted tokens plus verifier attempts`.

The latest read-only live audit at `2026-06-27 17:53 CDT` still found no safe
DFlash test-slot host, so the runtime smoke and `8050` launch gate remain
closed.

## Cutover recommendation

Do not keep investing in the older DS4 fork optimization stack as the active
line. Preserve its evidence and rollback anchors, but stop carrying that work
forward unless it is directly needed by DFlash.

Keep the deployed/frontdoor DS4 baseline as the operational baseline. Make this
DFlash branch the active fork development line. The old work should be treated
as archived evidence, not active roadmap.

Important: "throw away" should mean retire from the active line after anchors
are preserved, not destroy evidence. The old artifacts are still useful for
knowing what did not move the needle.

## Next executor steps

1. Run the gated DFlash runtime smoke as soon as a safe slot exists:
   - use the real DeepSeek V4 Flash target GGUF and staged DFlash artifact
   - run `tests/dflash_runtime_smoke.sh MODEL.gguf DFLASH_DIR`
   - preserve the emitted evidence directory for baseline equality, verifier
     execution, timing, and verified draft-token counts
   - do not run this against the live `spark-123a` DS4 process while it is
     carrying the production/frontdoor route

2. Tighten performance path after correctness:
   - replace sequential verifier with a batched target verifier only after the
     exact sequential path is proven
   - keep CPU DFlash as the reference path until a GPU DFlash executor exists

3. Only after local verifier correctness passes, create a separate Spark test
   slot/alias for DFlash-through-DS4-through-frontdoor.

## Deployment posture

No deployment yet.

The correct deployment posture is:

1. Prove local executor and verifier correctness with the real artifact/model
   pair.
2. Keep fail-closed `--dflash` behavior until that exactness evidence exists.
3. Deploy to a separate Spark test slot/alias, not the existing production DS4
   route.
4. Promote only after speed and quality evidence is better than the current DS4
   baseline.

## Bottom line

Yes: this is the cleaner move. The DFlash path has enough concrete progress and
enough upside to become the priority for the DS4 fork. The key discipline now is
to keep the next work executor-first and verifier-first, and not expose it
through the Spark frontdoor until the accept/reject loop is real.
