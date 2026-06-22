# Feature Dev Goal Ledger

## Run

- Run ID: ds4-cuda-f16-compressed-kv-2026-06-22
- Loop: plebdev Feature Dev
- Target repo: /Users/plebdev/Desktop/Projects/finite/ds4
- GitHub fork: finitecomputer/ds4
- Upstream repo: antirez/ds4
- Base branch: staging
- Feature branch: feature/cuda-f16-compressed-kv
- Human owner: plebdev
- Started: 2026-06-22
- Current status: issues #2 through #5 implemented locally; issue #6 Spark validation is next and currently HITL-blocked
- Skill setup status: complete for finite fork; upstream has AGENT.md and fork-local AGENTS.md plus docs/agents/* were added for the feature-dev loop

## Goal

Investigate and build a minimal, highly effective CUDA F16 compressed attention KV cache path for DS4 on DGX Spark, compare it against the current DS4 frontdoor benchmark checkpoints, and shape the result so it can plausibly become an upstreamable open-source PR.

## Durable Artifacts

- CONTEXT updates: root CONTEXT.md initialized as fork-local feature-dev glossary
- ADRs: none yet
- PRD issue: https://github.com/finitecomputer/ds4/issues/1
- Slice issues: https://github.com/finitecomputer/ds4/issues/2 through https://github.com/finitecomputer/ds4/issues/7
- Issue sessions: issues #2 through #5 completed locally; CUDA host validation remains parked in #6
- Agent briefs: none yet
- Review packets: none yet
- Local CodeRabbit report: not run yet
- PR URL: none yet

## Repo Skill Setup Decisions

- Issue tracker: GitHub Issues on `finitecomputer/ds4`; issues were enabled on 2026-06-22.
- Triage labels:
  - `needs-triage`: maintainer needs to evaluate
  - `needs-info`: waiting on reporter or human clarification
  - `ready-for-agent`: fully specified and ready for AFK agent work
  - `ready-for-human`: requires human implementation or judgment
  - `wontfix`: will not be actioned
- Domain docs: single-context repo; use a root `CONTEXT.md` plus root `docs/adr/` when ADRs are warranted.
- Upstream PR hygiene: `AGENTS.md`, `docs/agents/*`, `CONTEXT.md`, and `.scratch/feature-dev/*` are fork-local orchestration scaffolding. Strip them from any future `antirez/ds4` upstream PR unless upstream explicitly wants workflow metadata.

## Source Evidence

- Proposal handoff: /Users/plebdev/Downloads/Telegram Desktop/ds4_cuda_f16_compressed_kv_handoff_report.txt
- Current DS4 checkpoint repo: /Users/plebdev/spark-cluster
- DS4 speed probe: /Users/plebdev/spark-cluster/runs/2026-06-22-antirez-ds4-124k-frontdoor-smoke/speed-probe.json
- DS4 frontdoor smoke: /Users/plebdev/spark-cluster/runs/2026-06-22-antirez-ds4-124k-frontdoor-smoke/frontdoor-smoke.json
- DS4 tool/Hermes bench checkpoint: /Users/plebdev/spark-cluster/runs/2026-06-22-ds4-toolcall15-hermesagent20-frontdoor-fast/README.md

## Commands

- Install: no package install command identified yet
- Typecheck: not applicable yet
- Test: make test; make cuda-regression on CUDA host
- Build: make cuda-spark; make cuda CUDA_ARCH=sm_121
- Visual verification: not applicable
- Benchmark: ds4-bench and frontdoor smoke/tool/Hermes checkpoints from spark-cluster

## Slice Ledger

| Issue | Type | Status | Review thread | Fixes needed | Verified |
| --- | --- | --- | --- | --- | --- |
| #2 https://github.com/finitecomputer/ds4/issues/2 | AFK | implemented locally in 9e750c3 | self-review pass | no | local checks pass; CUDA host compile pending #6 |
| #3 https://github.com/finitecomputer/ds4/issues/3 | AFK | implemented locally in c244882 | self-review pass | no | local checks pass; CUDA host compile pending #6 |
| #4 https://github.com/finitecomputer/ds4/issues/4 | AFK | implemented locally in ce33a8e | self-review pass | no | local checks pass; CUDA host compile pending #6 |
| #5 https://github.com/finitecomputer/ds4/issues/5 | AFK | implemented locally in c94803c | self-review pass | no | local checks pass; CUDA host compile pending #6 |
| #6 https://github.com/finitecomputer/ds4/issues/6 | HITL | ready; blocked on Spark access | none | no | no |
| #7 https://github.com/finitecomputer/ds4/issues/7 | AFK | blocked by #6 | none | no | no |

## Parked HITL Slices

| Issue | Why parked | Blocks | Required human action | Final PR decision |
| --- | --- | --- | --- | --- |
| #6 https://github.com/finitecomputer/ds4/issues/6 | Requires DGX Spark host/runtime access and long-running model benchmarks | final performance claim and docs handoff | approve/run Spark validation when implementation reaches CUDA buildable state | required before upstream performance claims |

## Issue Session Ledger

| Issue | Fixed point | Worker session | Commit | Review result | Checks |
| --- | --- | --- | --- | --- | --- |
| #2 https://github.com/finitecomputer/ds4/issues/2 | 2b089a3 | main Codex session | 9e750c3 | self-review standards/spec pass | `make -j8`; `git diff --check`; `make test` partial; `./ds4_test --server`; `./ds4_test --metal-kernels`; CUDA make dry-runs |
| #3 https://github.com/finitecomputer/ds4/issues/3 | 02f15fa | main Codex session | c244882 | self-review standards/spec pass | `git diff --check`; `make -j8`; CUDA make dry-run |
| #4 https://github.com/finitecomputer/ds4/issues/4 | a488f06 | main Codex session | ce33a8e | self-review standards/spec pass | `git diff --check`; CUDA make dry-run |
| #5 https://github.com/finitecomputer/ds4/issues/5 | 9c3891b | main Codex session | c94803c | self-review standards/spec pass | `git diff --check`; `make -j8`; `./ds4_test --server`; `./ds4_test --metal-kernels`; default and experimental CUDA make dry-runs |

## Open Questions

- None.

## Resolved Alignment Decisions

- Initial CUDA F16 compressed attention KV support is opt-in only. The first implementation should make the storage mode explicitly selectable for CUDA experiments while preserving the existing CUDA default behavior until correctness and benchmark evidence justify changing it.
- Minimum correctness gate before speed comparison: the experimental CUDA F16 compressed attention KV build must pass `make cuda-regression`, a short CLI smoke, and at least one long-context prompt on DGX Spark. Official vector checks are desirable and should be run where practical, but they are best-effort for the first prototype if fixture/backend assumptions make them unsuitable as a hard blocker.
- Benchmark ladder before publishing or pitching the result as meaningful:
  1. Direct DS4 before/after with `ds4-bench` on the same Spark, same model, and same commit pair at 32K, 128K, and one stress context such as 512K or 1M when stable. Record prefill, decode, memory, and managed-KV fallback logs.
  2. Frontdoor smoke through `spark-cluster` using a distinct experimental F16 alias, rerunning the existing speed probe plus `/v1/models`, non-streaming, streaming, and tool-call smoke.
  3. Toolcall-15 because it is cheap and catches structured tool-use regressions.
  4. HermesAgent-20 only after the above pass, because it is slow and is mainly an agent-behavior regression check.
- Phase 1 scope is compressed attention KV only. Do not change raw sliding-window KV cache or indexer compressed cache in the first implementation, so correctness and performance deltas can be attributed to `layer_attn_comp_cache` storage.
- The opt-in surface for the first implementation is compile-time only. Use an overrideable macro/build define for CUDA experiments and do not add CLI, server, or runtime flags until the path is proven and the release-path decision changes.

## Validation Notes

- `make -j8` passed on local Darwin/Metal.
- `git diff --check` passed.
- `make -n cuda-spark UNAME_S=Linux` showed default CUDA build without `-DDS4_CUDA_ATTN_COMP_CACHE_F16=1`.
- `make -n cuda-spark UNAME_S=Linux DS4_CUDA_ATTN_COMP_CACHE_F16=1` showed the experimental define applied to `ds4.c`.
- `make test` built local tests and passed `tests/test_q4k_dot`, `./ds4-eval --self-test-extractors`, and `./ds4_agent_test`; full target stopped at `./ds4_test` because `ds4flash.gguf` is not present in this checkout.
- `./ds4_test --server` and `./ds4_test --metal-kernels` passed.

## Escalations

- DGX Spark SSH probe to `toor@192.168.0.180` timed out from this machine on 2026-06-22; CUDA compile, `make cuda-regression`, and benchmark validation remain in #6.
