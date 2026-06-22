# Domain Docs

This fork is a single-context repo.

## Before Exploring

Read these when they exist and are relevant:

- `AGENT.md` for upstream DS4 development rules.
- `CONTEXT.md` at the repo root for glossary terms.
- `docs/adr/` for architectural decisions touching the work area.

If `CONTEXT.md` or `docs/adr/` do not exist yet, proceed silently. The
`grill-with-docs` skill creates them lazily only after terms or decisions are
actually resolved.

## Glossary Rules

Use the glossary vocabulary from `CONTEXT.md` when naming domain concepts in
issues, PRDs, review notes, or implementation plans.

Do not use `CONTEXT.md` as a spec or implementation scratchpad. It is glossary
only.

## ADR Rules

Create ADRs sparingly. An ADR is warranted only when the decision is hard to
reverse, surprising without context, and the result of a real trade-off.

## Upstream Boundary

These agent docs are fork-local workflow scaffolding. Keep them out of an
upstream `antirez/ds4` pull request unless the upstream maintainer asks for them.
