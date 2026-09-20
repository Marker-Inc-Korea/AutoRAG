<!--
AutoRAG PR template. Keep the headings; the comment blocks are guidance you can delete.
Fill in what you actually did — a command listed here is a claim that you ran it.
-->

## Summary

<!-- What changes and why, in one or two sentences. -->

## Linked issue

<!-- `Closes #123` for a fix/feature PR; `Refs #123` when it is partial. -->

Closes #

## Scope

<!-- What this PR deliberately does not change, when that is not obvious from the diff. -->

## Test plan

<!-- Exact commands, with their real outcome. Never list a command you did not run. -->

- [ ] `bun run lint`
- [ ] `bun run typecheck`
- [ ] `bun run test` (or the focused files: `bun run test test/path/to/file.test.ts`)

<!--
`bun run test` runs the complete suite. Add the platforms this change touches:
`make test-linux`, `make test-windows`, `make e2e-live-cold` for retrieval or
datasource behavior.
-->

- [ ] `bun run build`

Manual QA (real surface, exact command or UI path, and what you observed):

```text

```

## AI use

<!-- AutoRAG is built with AI agents; disclosure is expected, not discouraged. -->

- [ ] AI tools used for this PR are named in this section (or state "none")
- [ ] I understand every change in this PR, including anything AI-generated
- [ ] No claim in this description is unverified, and no test was weakened, skipped, or deleted to go green

The AI contribution policy (`AI_POLICY.md`) defines the full rule set.

## Checklist

- [ ] The diff stays inside the linked issue's scope
- [ ] Errors and diagnostics surface the underlying CLI exit status, stderr, and real paths verbatim
- [ ] Results keep source-native identities (datasource ids stay opaque, not fake OS paths)
- [ ] No secrets, tokens, cookies, or private corpus content added (config keeps env/keychain references only)
- [ ] New dependency or workflow action reviewed for license and supply chain
- [ ] Docs updated when behavior, CLI surface, configuration, or the output contract changed
- [ ] Behavior is covered by tests asserting machine-consumed values (not pinned prose)
