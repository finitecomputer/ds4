# Issue Tracker: GitHub

Issues and PRDs for this fork live as GitHub issues on `finitecomputer/ds4`.
Use the `gh` CLI for issue operations.

## Conventions

- Create an issue: `gh issue create --repo finitecomputer/ds4 --title "..." --body-file <file>`.
- Read an issue: `gh issue view <number> --repo finitecomputer/ds4 --comments`.
- List issues: `gh issue list --repo finitecomputer/ds4 --state open --json number,title,body,labels`.
- Comment on an issue: `gh issue comment <number> --repo finitecomputer/ds4 --body-file <file>`.
- Apply or remove labels: `gh issue edit <number> --repo finitecomputer/ds4 --add-label "..."` or `--remove-label "..."`.
- Close an issue: `gh issue close <number> --repo finitecomputer/ds4 --comment "..."`.

## Publishing

When a skill says to publish a PRD, generated issue, or agent brief to the issue
tracker, create a GitHub issue in `finitecomputer/ds4`.

## Upstream Boundary

The finitecomputer issue tracker is orchestration state for this fork. If this
work becomes an upstream `antirez/ds4` pull request, link only the useful public
evidence and avoid leaking fork-local process noise into the upstream PR.
