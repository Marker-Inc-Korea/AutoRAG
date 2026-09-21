# AI contribution policy

AutoRAG is itself an AI coding-agent product and already receives agent-authored pull requests. AI tools are allowed. Unverified dumps are not.

[`AGENTS.md`](AGENTS.md) is the in-repo map for people already working in this checkout. This file is the **external inbound rule** for anyone opening an issue or pull request, including first-time and drive-by contributors.

## Allowed

You may use AI coding assistants (Claude Code, Cursor, Codex, Senpi, Amp, Copilot, and similar) to draft, search, or test a change. Maintainers do the same. Grammar and translation tools are fine.

## You are liable

The human submitter owns the result: correctness, license, IP, secrets, and review follow-up. Naming a tool is not a disclaimer. The [DCO](CONTRIBUTING.md#licensing-and-the-dco) still applies. You must confirm that no unclear-license code was pasted in.

## What a pull request must say

The pull request description (not the commit subject) must state that you understood and verified the change. The template already has checkboxes for this. In that section:

1. Name the AI tools used, or write `none`.
2. Say what you personally read, ran, and changed after any model drafted text or code.
3. Point at the exact commands or tests you used to verify behavior.

If a reviewer can tell the patch is unreviewed model output, it is not ready.

## Closed without review

Maintainers may close the following without further review. This is governance, not hostility:

- Large auto-generated pull requests with no prior issue
- Pull requests that are clearly unedited model dumps
- Contributors who cannot explain the change without re-asking a model
- Unapproved bots opening issues, pull requests, or comments
- Entirely AI-generated issue or pull-request text pasted without human editing

A Ghostty-style extra we will use: inbound AI pull requests that are not linked to an existing issue may be closed unread. Open the issue first for anything larger than a typo or a failing test.

## Security reports

The bar is stricter than a normal bug. Follow [SECURITY.md](SECURITY.md):

- Use private vulnerability reporting. Do not file a public issue.
- Include a reproduction procedure and an impact scope, or the report is not triaged as a vulnerability.
- If AI helped, disclose that in the first sentence, reproduce it yourself, and write the report in your own words.
- Fabricated or copy-pasted AI security reports are not accepted and may lead to a ban.

## What we want instead

Small, understood patches. Tests that assert machine-consumed values, not pinned prose. Iteration in your own voice. Link the issue.
