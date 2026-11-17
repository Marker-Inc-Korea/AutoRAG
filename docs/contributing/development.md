---
title: "Development workflow"
description: "Set up the AutoRAG 2.0 checkout, run the Make targets, follow the commit and review rules, and edit the docs site."
---

# Development workflow

This page is the contributor procedure for AutoRAG 2.0: how to lay out a checkout, which commands to run, what the guardrails allow and forbid, how a commit and a pull request are shaped, and how the documentation site is written.

The authoritative contributor file is [`CONTRIBUTING.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/CONTRIBUTING.md); the guardrails for people and agents working inside a clone are in [`AGENTS.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/AGENTS.md).

## Repository layout

AutoRAG is a monorepo with two independent trees:

| Tree | What lives there | Toolchain |
|---|---|---|
| repository root | AutoRAG 2.0 — the TypeScript/Bun librarian agent: `src/`, `test/`, `scripts/`, `skills/`, `internal-skills/`, `plugins/`, the `autorag` CLI, and the `@autorag/librarian` npm package | Node.js 24+, Bun, `make` |
| `legacy/` | the legacy Python AutoRAG (the AutoML pipeline optimizer), published as the `autorag` pip package | Python 3.10+, uv or pip, pytest |

One pull request touches one tree. Root CI runs for root paths; the legacy suite (`.github/workflows/test.yml`) runs only for `legacy/**` diffs, so a job skipping on your pull request is expected, not a failure. Legacy setup lives in [`legacy/README.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/legacy/README.md).

It is worth knowing the top-level split before you edit: `src/` is the engine (see [Architecture](architecture.md#source-layout)), `test/` holds the per-process vitest files, `scripts/` holds CI, live-E2E, manual-QA, and supply-chain tooling, `skills/` and `internal-skills/` ship the agent skills, and `docs/` is the documentation site.

## Setup

Requirements:

- Node.js 24 or newer (`package.json` declares `engines.node >= 24.0.0`).
- [Bun](https://bun.sh) — package manager, test runner, and build tool.
- `git`, `make`, and a POSIX shell. On Windows, run the Make targets from Git Bash/MSYS2 (`make test-windows` verifies that path) or call the underlying `bun run …` scripts directly.
- Optional: the Rust toolchain, only to build MinSync/Jikji from source or run the live end-to-end lanes.

```bash
git clone https://github.com/Marker-Inc-Korea/AutoRAG.git
cd AutoRAG
bun install --frozen-lockfile
```

Do not commit `bun.lock` changes that do not correspond to a dependency change in `package.json`, and do not commit `dist/`, `.autorag/`, or local live-E2E state.

## Everyday commands

The `Makefile` is the entry point; each target wraps a `bun run …` script.

| Command | What it runs |
|---|---|
| `make install` | `bun install --frozen-lockfile` |
| `make lint` | `bun run lint` (Biome format + lint check, no writes) |
| `make format` | `bun run check` (Biome formatting and safe fixes) |
| `make typecheck` | `bun run typecheck` (`tsc --noEmit`) |
| `make test` / `make test-all` | `bun run test` — the complete suite, one vitest file per process |
| `make build` | `bun run build` — library, CLI, and declarations into `dist/` |
| `make ci` | `lint` → `typecheck` → `test` → `build`, the sequence CI runs |
| `make supply-chain` | the license, `NOTICE`, and local CycloneDX gate |
| `make test-macos` | the complete suite, requiring a macOS host |
| `make test-windows` | the complete suite, requiring Git Bash/MSYS2 on Windows |
| `make test-linux` | lint, typecheck, the suite, and build inside a `linux/amd64` container |
| `make tui` | run the current sources in the TUI without a build (extra flags via `TUI_ARGS`) |

Run one test file while iterating:

```bash
bunx vitest run test/agent/agent.test.ts
```

`make ci` is the gate we ask for in every pull request. If a check fails for a reason unrelated to your change, report it with the verbatim output instead of working around it.

## Guardrails

These bind every agent and human working in a checkout. The `main` rulesets, CI, and CODEOWNERS enforce them; they are the instruction.

### Allowed

Edits inside `src/`, `test/`, `scripts/`, `skills/`, `internal-skills/`, and `docs/` that stay within the task.

### Forbidden

- Changing the `LICENSE` license identifier.
- Adding or committing secrets (`.env`, `*.key`, tokens, cookies, private corpus dumps).
- Rewriting git history.
- Attaching `node_modules` to releases.
- Sending corpus text to a remote provider without an explicit trusted-config opt-in. Remote embedders and rerankers are supported; the local MinSync embedder stays the zero-config default.
- Hard-coding provider credentials in config files or argv; store only environment-variable, keychain, or profile references.
- Creating git worktrees: this clone is the isolation boundary.

### Required

`bun run check && bun run lint && bun run typecheck`, plus the tests that cover the change. Dependency or license-allowlist edits also need the supply-chain gate (`bun run supply-chain`).

### Ask first

A new runtime dependency, a license-allowlist change, `SECURITY.md` / `GOVERNANCE.md` / the AI policy, an embedding backend change or its license, and any change to P2P defaults.

## Branching, commits, and review

- Branch off the latest `main`, in your fork. Never commit to `main` directly, and keep unrelated cleanup out of the branch.
- One logical change per pull request. The work branch must derive from an up-to-date `main`; never start from a stale or leftover branch, or detached HEAD.
- Commit subjects follow the history: `feat(agent): …`, `fix(minsync): …`, `docs(datasource): …`, `test(…)`, `chore(…)`.
- Write the pull request body for a reader who did not watch you work: what changed, why, and the exact commands you ran with their result. Link the issue (`Fixes #123`). Report failures verbatim. If an agent wrote part of the change, say so; the inbound rule is [`AI_POLICY.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/AI_POLICY.md).
- A maintainer reviews every pull request; the required status checks plus one approving review are the merge gate, and merges to `main` are squash-only. Never merge a pull request on your own initiative.
- Security problems never go in a public issue; follow [`SECURITY.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/SECURITY.md).

### DCO sign-off

Every non-merge commit in a pull request must carry a `Signed-off-by:` trailer whose email matches the commit's author or committer email; the `dco` job (`.github/workflows/dco.yml` → `scripts/ci/check-dco.mjs`) fails the pull request otherwise.

```bash
# Enable the hook once per clone; it adds the trailer to every commit.
git config core.hooksPath .githooks

# Or sign explicitly:
git commit -s
git commit --amend -s
git rebase --signoff <base>

# Check before pushing:
DCO_BASE_SHA=origin/main DCO_HEAD_SHA=HEAD node scripts/ci/check-dco.mjs
```

The trailer certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/). Force-pushing your own feature branch to fix sign-off is allowed.

## Writing the docs

The Markdown under `docs/` is the single source of truth and is what coding agents read. It is published as a VitePress site at [https://autorag-docs.nomadamas.org](https://autorag-docs.nomadamas.org), but the GitHub Markdown — not the rendered HTML — is what agents consume. Where a page and the code disagree, the code wins.

- The table of contents is `docs/.vitepress/sidebar.mts`. A page cannot ship without a sidebar entry, and a sidebar entry cannot ship without a page. The VitePress config, the generated `llms.txt`, and the contract tests all read that file.
- Preview and build locally:

```bash
cd docs
bun install
bun run dev      # local preview
bun run build    # static build (dead links fail the build)
bun run preview  # serve the build
```

### Page format

Each page must satisfy the following; the contract tests enforce it.

- It begins with YAML frontmatter containing exactly `title:` and `description:` (one plain sentence, 50–160 characters, no angle brackets).
- One `# H1` follows whose text equals the `title`, then a short lead sentence.
- Use `##` / `###` only below the H1; never skip a level, and keep heading text unique within the page so anchors are stable.
- Relative links to other docs pages carry the `.md` extension (for example `[Architecture](architecture.md)`). Links to files outside `docs/` use absolute GitHub blob URLs; a link must never leave `docs/`.
- Every fenced code block is labeled with a language (`bash`, `json`, `text`, `ts`, `yaml`). Shell examples use real commands and flags.
- Callouts use VitePress/GitHub-safe containers (`::: tip`, `::: warning`, `::: danger`).

### Contract tests

The docs are tested like code, under `test/docs/`:

- `test/docs/site-contract.test.ts` checks the sidebar-to-page mapping, the page format above, every relative link and anchor, every `autorag <command>` and flag against the real CLI, every `autorag.<tool>` name against the MCP server, and that no removed or pre-2026 guidance leaks in.
- `test/docs/*.test.ts` also pins specific pages against the code their subject owns (for example the MinSync state paths and the remote-agents server surface).
- `test/helpers/docs-site.ts` exposes the shared helpers (frontmatter, links, anchors, fences) that those tests build on.

Run them with:

```bash
bunx vitest run test/docs/site-contract.test.ts
```

## Related pages

- [Architecture](architecture.md) for the engine and its state layout.
- [Manual QA and live E2E](manual-qa.md) for the isolated QA and live-run procedure.
- [Supply chain and licenses](supply-chain.md) for the dependency, SBOM, and license gates.
- [Adding a datasource](new-datasource.md) for the datasource-skill path.
