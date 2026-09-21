# Contributing to AutoRAG

AutoRAG is an over-powered librarian agent for document collections. Contributions
of every size are welcome: bug reports, documentation fixes, tests, retrieval
methods, datasource skills, and platform support.

This file is the contributor-facing procedure. You do **not** need
[`AGENTS.md`](AGENTS.md) to contribute — that file documents how coding agents and
maintainers work inside a clone, and it is only a reference if you contribute with
an agent.

**The short version:** fork → branch off the latest `main` → make one logical change
→ run `make ci` → open a pull request whose commits carry a `Signed-off-by` trailer
(`git commit -s`) and whose description links the issue it addresses.

## Which tree are you in?

AutoRAG is a monorepo with two independent trees:

| Tree | What lives there | Toolchain |
|---|---|---|
| repository root | AutoRAG 2.0 — the TypeScript/Bun librarian agent: `src/`, `test/`, `skills/`, CLI `autorag`, npm package `@autorag/librarian` | Node.js 24+, Bun, `make` |
| `legacy/` | legacy Python AutoRAG (the AutoML pipeline optimizer), published as the `autorag` pip package | Python 3.10+, uv or pip, pytest |

Rules for the split:

- One pull request touches one tree. Do not mix a `legacy/` fix with a 2.0 change.
- Root CI (`build`, `compatibility (…)`) runs for root paths. The legacy suite
  (`.github/workflows/test.yml`) runs only for `legacy/**` diffs. Some jobs skipping
  on your pull request is expected, not a failure.
- Legacy setup and commands: `legacy/README.md` and `.github/copilot-instructions.md`.

## Before you start

- Search existing issues and pull requests first; the fix may already be open.
- Small fixes (typos, documentation, a failing test) can go straight to a pull request.
- For a new feature, a new datasource skill, a retrieval change, or anything touching
  several areas, open an issue (or comment on the existing one) and agree on the
  approach before writing code. It saves a rewrite.
- Issues labelled `good first issue` are scoped to be doable without deep project
  knowledge; `help wanted` means we would like someone to pick it up. Asking questions
  in the issue is normal, not a nuisance.
- Security problems never go in a public issue: follow [SECURITY.md](SECURITY.md).

## Development setup (AutoRAG 2.0)

Requirements:

- Node.js 24 or newer
- [Bun](https://bun.sh) — package manager, test runner, and build tool
- `git`, `make`, and a POSIX shell. On Windows, run the Make targets from Git
  Bash/MSYS2 (`make test-windows` verifies that path) or call the underlying
  `bun run …` scripts directly.
- Optional: the Rust toolchain — only needed when you build MinSync/Jikji from source
  or run the live end-to-end lanes.

```bash
git clone https://github.com/Marker-Inc-Korea/AutoRAG.git
cd AutoRAG
bun install --frozen-lockfile
```

Do not commit `bun.lock` changes that do not correspond to a dependency change in
`package.json`, and do not commit `dist/`, `.autorag/`, or local live-E2E state.

## Everyday commands

The `Makefile` is the entry point; each target wraps a `bun run …` script.

| Command | What it does |
|---|---|
| `make install` | install dependencies from `bun.lock` |
| `make lint` | Biome format + lint check over the repository |
| `make format` | apply Biome formatting and safe fixes |
| `make typecheck` | TypeScript type checking (`tsc --noEmit`) |
| `make test` | the complete AutoRAG 2.0 suite (vitest, one file per process) |
| `make test-all` | alias for `make test` |
| `make build` | build the library, CLI, and type declarations into `dist/` |
| `make ci` | `lint` → `typecheck` → `test` → `build`, the sequence CI runs |
| `make supply-chain` | license, NOTICE, and local CycloneDX gates |

Run one test file while iterating:

```bash
bunx vitest run test/agent/agent.test.ts
```

`make ci` is the gate we ask for in every pull request. If a check fails for a reason
unrelated to your change, say so in the pull request with the verbatim output instead
of working around it.

## What CI runs on your pull request

| Check | Scope |
|---|---|
| `build` (required) | Biome lint, typecheck, full test suite, build. Its steps are skipped when the diff contains no AutoRAG 2.0 paths, and the job still reports success so the required check never blocks an unrelated change |
| `compatibility (macos-latest)`, `compatibility (windows-latest)` | typecheck, tests, build on both platforms |
| `Legacy Unit Test` | pytest suite under `legacy/`, only for `legacy/**` diffs |
| `Supply chain` | dependency review, OSV scan, license + NOTICE gate |
| `dco` | every commit in the pull request carries a `Signed-off-by` trailer (see below) |

## Commits and pull requests

- Branch off the latest `main`, in your fork. Never commit to `main` directly, and do
  not put unrelated cleanup in the same branch.
- One logical change per pull request. If you fixed a bug and also tidied a module,
  split it into two pull requests.
- Commit subjects follow the history: `feat(agent): …`, `fix(minsync): …`,
  `docs(datasource): …`, `test(…)`, `chore(…)`.
- Every commit needs a sign-off trailer — see
  [Licensing and the DCO](#licensing-and-the-dco).
- Write the pull request body for a reader who did not watch you work: what changed,
  why, and the evidence (the exact commands you ran and their result). Link the issue
  (`Fixes #123`, `Refs #123`).
- Report failures verbatim: real error text, exit status, and paths stay in the
  description. Do not summarise a failure into a sentence that hides it.
- If an agent wrote part of the change, say so in the pull request body.
- Force-pushing your own branch during review is fine, and commit history inside the
  pull request does not have to be perfect: merges to `main` are squash-only. The
  squashed commit message should still describe the change.

## Review

- A maintainer reviews every pull request; the required status checks plus one approval
  are the merge gate, and merges to `main` are squash-only. See [MAINTAINERS](MAINTAINERS)
  for who covers which area.
- Expect questions about scope, tests, and failure handling — that is the review
  working, not a rejection.
- If your pull request goes quiet for about a week, ping it with a comment (mentioning
  [@vkehfdl1](https://github.com/vkehfdl1) is fine).
- Behaviour decisions that a reviewer cannot settle go to the project lead; ownership
  decisions (license, repository, assets) are reserved to Marker Inc. under
  [GOVERNANCE.md](GOVERNANCE.md).

## Code conventions (AutoRAG 2.0)

- TypeScript ESM, formatted by Biome. Run `make format` before pushing; `make lint`
  is the check.
- Errors belong to the user: retrieval, refresh, and datasource failures surface the
  underlying text verbatim — exit status, stderr, and real filesystem paths included.
  Do not replace a failure with a generic message, do not drop it because it contains
  a path, and do not swallow it. Bounding runaway output by length is fine; removing
  content is not.
- Secrets stay external: environment variables, keychain, or CLI profiles only. Never
  persist tokens, cookies, passwords, or refresh credentials into config files, argv,
  or captured snapshots.
- Datasource results carry opaque source-native identities such as
  `/discord/<instance>/chunks/<chunk>`. They are not filesystem paths — never pass them
  to `cat`, a shell, or any path API, and never synthesise a fake OS-absolute path.
- Tests must not depend on timing luck. No fixed sleeps: subscribe to the exact event
  or state change before triggering the action, then await it with a bounded timeout.
  Never pin prose, prompt wording, or documentation text with a test — test the values
  a machine consumes (parsed fields, sentinel tokens, shipped-copy equality). A pure
  prose change ships with no new test.
- When your change alters user-visible behaviour, update the docs in the same pull
  request (`README.md`, `docs/`), and note it in the pull request body.

## Datasource skill contributions

Datasource skills federate CLI-owned stores in place. If you add one:

- Start from the issue template `.github/ISSUE_TEMPLATE/datasource-skill.md`.
- Spawn the external CLI with its own default store. Never force a
  `--workspace`/`--config` (or an env override) into an empty AutoRAG-managed
  directory, and never copy the native store's data.
- Keep failure isolation per CLI: one failing CLI degrades to diagnostics and an
  `unsearched` entry, and never crashes the search loop. A retrieval method that
  cannot answer throws the CLI's own error (failure kind, exit status, stderr) rather
  than returning an empty result set.
- Ship a datasource skill that documents native commands plus `<binary> --help`
  guidance, so the agent knows which CLI backs the datasource.
- Add focused tests, then run the manual QA harness for the source
  (`scripts/manual-qa/`, indexed in `docs/manual-qa-datasources.md`) before
  registering the datasource. A native store that exists on your machine must be
  exercised, not skipped.

## Live end-to-end (only when your change needs it)

Run the live lanes when your change touches retrieval, MinSync or embedding
configuration, datasource lanes, the agent tool surface, or the output contract. For
documentation, formatting, and unit-test-only changes, `make ci` is enough.

The live environment is clone-local: each clone owns its own `.autorag-e2e` state, so
two clones must not share one `E2E_ROOT`. Bootstrap is explicit — the targets never
create the corpus root for you.

```bash
export AUTORAG_LIVE_E2E_ROOT="$PWD/scripts/live-e2e"
node scripts/live-e2e/runner.mjs bootstrap --root "$AUTORAG_LIVE_E2E_ROOT"

make e2e-live-cold E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"   # fresh state + core checks
make e2e-live      E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"   # warm run, reusing state
```

- `E2E_DATASOURCES` only narrows the default lane set (e.g.
  `E2E_DATASOURCES=local`). Narrowing away an installed native store leaves a QA gap,
  not a green run.
- A lane whose CLI or native store is missing reports `SKIP` with a reason; an
  installed and configured lane that fails reports `FAIL`. `SKIP` is never reported as
  `PASS`.
- Keep `OPENAI_API_KEY` and `AUTORAG_OPENAI_API_KEY` unset. Embeddings are local: the
  `autorag-gateway` `qwen3-embedding-0.6b` profile, loopback only. Do not configure a
  remote embedding endpoint and do not send corpus text off the machine.
- MinSync semantic QA is only complete when you have observed all of: `refresh
  --method minsync` exiting successfully with `.minsync/cursor.json` present, a
  semantic query returning a hit for the fixture document, that hit mapped to an
  OS-absolute original `source`, that path existing and reading back, and no request
  leaving the machine.
- Cleanup is limited to runner-owned state: `rm -rf .autorag-e2e` from your own clone.
  Leave discrawl, crawler, qmd, rclone, mailcrawl, and Spotlight native stores
  untouched, and do not stage `.debug-journal.md`.
- If a behavioural change invalidates previously recorded cold/warm evidence,
  regenerate it in the same pull request. Stale green evidence is not proof.

## Licensing and the DCO

AutoRAG is released under the [MIT License](LICENSE), and contributions are accepted
inbound under the same terms (inbound = outbound).

**Every commit needs a sign-off**, which is how we enforce the inbound license:

```bash
git commit -s                     # sign the commit you are about to make
git commit --amend -s             # sign a commit you already made
git rebase --signoff <base>       # sign a whole branch
```

That adds a trailer to the commit message:

```text
Signed-off-by: Your Name <you@example.com>
```

The email must match the commit's author or committer email. The text you are
certifying is the [Developer Certificate of Origin 1.1](https://developercertificate.org/)
— in short, that you wrote the contribution or otherwise have the right to submit it
under this project's license.

CI verifies this in the `dco` job (`.github/workflows/dco.yml` →
`scripts/ci/check-dco.mjs`). Merge commits and bot authors are exempt. When it fails,
the job prints the offending commits verbatim, and the usual fix is
`git rebase --signoff <base> && git push --force-with-lease`.

### Decision: DCO, not a CLA — and why

We chose the **Developer Certificate of Origin** over a Contributor License Agreement:

1. **No relicensing or dual-licensing plan exists.** AutoRAG ships under MIT only, and
   there is no commercial edition whose rights a CLA would need to protect. The CLA's
   core benefit — a standing right to relicense contributions — is not in use.
2. **A CLA is a legal instrument we would have to maintain.** It needs drafted terms,
   a signature flow, and a record of who signed. The DCO is one trailer on commits
   contributors already write, verifiable by CI in seconds.
3. **The reserved relicensing right in [GOVERNANCE.md](GOVERNANCE.md) is a decision
   right, not a plan.** Under DCO, each contribution is already licensed to everyone
   under MIT, so relicensing a version that contains those contributions to another
   license would require the contributors' consent. We accept that constraint
   deliberately: lower friction for contributors now, no unilateral relicensing of
   community code later.
4. **It is reversible upward, not downward.** If a dual-license or commercial edition
   ever enters the roadmap, the switch to a CLA is itself a documented decision (a pull
   request changing this section and `GOVERNANCE.md`), and it can only apply
   prospectively to new contributions.

If you disagree with this trade-off, open an issue — it is a project-owner decision and
is meant to be argued in public.

## Code of Conduct

All participants are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Report unacceptable behaviour to the addresses
listed in that file; reports are handled confidentially by the project team.
