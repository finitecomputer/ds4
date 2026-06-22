# Agent Instructions

Read `AGENT.md` first. It is the upstream project guidance for DS4/DwarfStar
development style, safety, layout, and testing.

This file adds fork-local workflow metadata for the finitecomputer feature-dev
loop. Keep this scaffolding out of an upstream `antirez/ds4` pull request unless
the upstream maintainer explicitly wants agent workflow files.

## Agent skills

### Issue tracker

Issues and PRDs for this fork are tracked in GitHub Issues on
`finitecomputer/ds4`. See `docs/agents/issue-tracker.md`.

### Triage labels

This fork uses the canonical five-label triage vocabulary:
`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and
`wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo. Use a root `CONTEXT.md` when glossary terms exist
and root `docs/adr/` only for warranted architectural decisions. See
`docs/agents/domain.md`.
