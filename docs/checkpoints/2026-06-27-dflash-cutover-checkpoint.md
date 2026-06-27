# DFlash cutover checkpoint - 2026-06-27

## Judgment

Cut the DS4 fork over to DFlash focus.

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
- Current head before this executor slice: `f93a3c6 Add DFlash hidden history checkpoint`
- Current branch state before this executor slice: clean, ahead of `origin/main`
  by 8 commits.

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
- These primitives are covered by focused C tests with a tiny safetensors
  fixture that exercises actual mapped BF16 bytes rather than synthetic arrays
  only.

## What is not done

- `--dflash` still fails closed for generation. It opens and validates the
  artifact, then prints that DFlash graph execution is not implemented unless
  run in inspect-only mode.
- The local proposal helper is not yet called by the speculative generation
  state machine.
- There is no verifier accept/reject wiring yet.
- There is no GPU DFlash executor yet. The CPU path is a correctness/reference
  path, not the production performance target.
- There is no Spark deployment slot or alias for DFlash-through-DS4 yet.

## Verification at this checkpoint

Commands run successfully in `/Users/plebdev/Desktop/Projects/finite/ds4-dflash-clean`:

- `make dflash-config-test`
- `make cpu`
- `make`
- `git diff --check`
- `./ds4_test --server`

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

1. Wire verifier accept/reject around `ds4_session_dflash_propose_argmax`:
   - verify tokens on target
   - accept matching prefix
   - reject and fall back to target token
   - preserve/restore target KV state correctly

2. Add local correctness instrumentation:
   - log draft count, accepted count, and first-miss position under a DFlash
     debug flag
   - keep `--dflash` fail-closed until accept/reject has exactness evidence

3. Only after local verifier correctness passes, create a separate Spark test
   slot/alias for DFlash-through-DS4-through-frontdoor.

## Deployment posture

No deployment yet.

The correct deployment posture is:

1. Finish local executor and verifier correctness.
2. Keep fail-closed `--dflash` behavior until verifier accept/reject works.
3. Deploy to a separate Spark test slot/alias, not the existing production DS4
   route.
4. Promote only after speed and quality evidence is better than the current DS4
   baseline.

## Bottom line

Yes: this is the cleaner move. The DFlash path has enough concrete progress and
enough upside to become the priority for the DS4 fork. The key discipline now is
to keep the next work executor-first and verifier-first, and not expose it
through the Spark frontdoor until the accept/reject loop is real.
