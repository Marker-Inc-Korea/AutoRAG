# AutoRAG — Pi-Powered Librarian Agent

## Guardrails

These bind every agent and human working in this checkout. Force is CI, CODEOWNERS, and the `main` rulesets; this file is the instruction. Do not flatten the rest of this map into these four boxes.

### Allowed

Edits inside `src/`, `test/`, `scripts/`, `skills/`, and `docs/` that stay within the task.

### Forbidden

- Changing the `LICENSE` license identifier
- Adding or committing secrets (`.env`, `*.key`, tokens, cookies, private corpus dumps)
- Rewriting git history
- Attaching `node_modules` to releases
- Sending corpus text to a remote provider without an explicit trusted-config opt-in (remote embedders and rerankers are supported; the local gateway remains the zero-config default)
- Hard-coding provider credentials in config files or argv; store only environment-variable, keychain, or profile references
- Creating git worktrees (this clone is the isolation boundary)

### Required

`bun run check && bun run lint && bun run typecheck`, plus the tests that cover the change. Dependency or license-allowlist edits also need the supply-chain gate (`bun run supply-chain`).

### Ask first

A new runtime dependency, a license-allowlist change, SECURITY / GOVERNANCE / AI policy, an embedding profile or its license, and P2P default changes.

## Git Workflow (binding)

This checkout is one of several clones of the same repository. Those clones are the isolation boundary. **Never create a git worktree.** Do not run `git worktree add`, do not create a linked checkout, and do not isolate a PR, review, or feature in a new worktree.

This section is the base workflow for this clone. If another agent skill, default, or habit (including worktree-based isolation) conflicts with it, follow this section.

### Before any PR review, PR work, or new feature

1. `git fetch origin`.
2. Update local `main` to match `origin/main` (`git checkout main` then `git pull --ff-only origin main`).
3. Check out the branch that belongs to the work, in this clone:
   - Existing PR or existing feature branch: `git checkout <branch>` (or `gh pr checkout <n>`), then sync that branch with its remote upstream (`git pull --ff-only`).
   - New feature: create the branch from the just-updated `main` (`git checkout -b <branch>`).
4. Only then edit, review, or test.

The work branch must be derived from latest `main`. Never start from a stale local branch, a leftover feature checkout, or detached HEAD.

### After the work is finished

Finished means the PR has been opened, or the review has been completed and no further edits remain in this checkout.

1. Commit and push **tracked** changes only (`git add -u`, then commit, then push). This is a standing request to commit at finish — do not wait for a per-session "please commit."
2. Do not add untracked files. `git add .` and `git add -A` are forbidden at this step. Untracked files stay untracked. If the finished work introduced new files that must ship, add those paths explicitly by name — never a recursive add of the working tree.
3. If there is nothing tracked to commit, skip commit/push and still return to `main`.
4. `git checkout main`.
5. Sync local `main` with `origin/main` (`git pull --ff-only origin main`).

Leave this clone on up-to-date `main` so the next session does not inherit a leftover feature branch.

### DCO sign-off (binding)

Every non-merge commit in a pull request must carry a `Signed-off-by:` trailer whose email matches the commit's author or committer email; the DCO check (`.github/workflows/dco.yml`, `scripts/ci/check-dco.mjs`) fails the PR otherwise. Sign-off is the default here:

1. Enable the hook once per clone: `git config core.hooksPath .githooks`. The committed `.githooks/prepare-commit-msg` then adds the trailer to every commit automatically; this clone is already configured.
2. If the hook is unavailable, sign explicitly: `git commit -s`, `git commit --amend -s`, or `git rebase --signoff <base>`.
3. Never finish a PR with an unsigned commit. To check before pushing: `DCO_BASE_SHA=origin/main DCO_HEAD_SHA=HEAD node scripts/ci/check-dco.mjs`.

Squash merges to `main` are still required, and force-pushing your own feature branch to fix sign-off is allowed (see CONTRIBUTING.md).

## Manual QA on a shared machine (binding)

The maintainer's machine is shared. Several AutoRAG clones on different branches and versions, and several checkouts of the AutoRAG Electron Finder app, are developed **at the same time** by different agent sessions. All of them read one global AutoRAG home: `~/.autorag` (`config.json`, `memory.json`, the embedding-model cache, TUI sessions, P2P policy). The Electron app searches with whatever `~/.autorag/config.json` says, and so does the maintainer's real daily AutoRAG usage.

That home belongs to the maintainer, not to your task. Every write your QA makes there silently changes every other clone, every app checkout, and the real setup. This already happened more than once: an ad-hoc QA step ran `autorag init --workspace "$TMP" --search-paths "$TMP/docs" --force` with no `--config` and no `AUTORAG_HOME`. `--workspace` does not move the config, so `--force` replaced the real `~/.autorag/config.json`. It dropped every configured datasource and pointed search at a temp directory that was deleted minutes later. From then on, every search on the machine, the desktop app included, failed with `AutoRAG search root does not exist: /var/folders/.../tmp.XXXX/docs` (#1743).

Rules for every manual QA, live check, and ad-hoc CLI or `AutoRAGAgent` run:

1. **Never touch the real `~/.autorag`.** Do not write, move, delete, or `--force` anything under it, and do not "fix" or "clean up" it. Reading is allowed only when the task needs it. If you think it is wrong, report it; never repair it.
2. **Isolate before the first command.** Create one temp root and point *both* AutoRAG variables at it. Export them in the same shell that runs every `autorag` / `node dist/cli/index.js` / `bun run src/cli/index.ts` command and every script that constructs `AutoRAGAgent`:

   ```bash
   QA_ROOT="$(mktemp -d)"
   export AUTORAG_HOME="$QA_ROOT/.autorag-home"
   export AUTORAG_CONFIG="$AUTORAG_HOME/config.json"
   mkdir -p "$QA_ROOT/docs" "$AUTORAG_HOME"
   ```

   A tool that starts a fresh shell per call (an agent `bash` tool, `tool.bash`, `Bun.$`, `spawn`) does not inherit a previous call's `export`. Repeat the exports in each call, or pass them through `env`.
3. **`--workspace` is not isolation.** It moves the workspace only. The config path comes from `--config`, then `AUTORAG_CONFIG`, then `$AUTORAG_HOME/config.json`, then `~/.autorag/config.json`. Never run `autorag init --force` (or anything else that writes config) unless `AUTORAG_CONFIG` or `--config` points inside your temp root.
4. **Check before you write.** Right before the first command that writes config, confirm the target is yours: `echo "$AUTORAG_CONFIG"` must print a path under `$QA_ROOT`. If it is empty or under `$HOME/.autorag`, stop.
5. **Leave other state alone.** Do not stop, restart, or reconfigure processes you did not start. That includes other clones' gateways and dev servers, a running Finder app, and `autorag watch` / refresh daemons. Do not edit crontabs or launch agents, and do not touch native datasource stores (see the live-E2E section). Use free ports rather than fixed ones.
6. **Prove it and clean up.** Hash the real config before and after the run (`shasum ~/.autorag/config.json`). The two hashes must match; record both in the task evidence. Then remove only your own `$QA_ROOT` and the processes you started.

The fixed live-E2E runner below is clone-local, but host execution still exposes the
host process and filesystem to ad-hoc mistakes. Use the Docker entry points for the
supported manual-QA boundary:

```bash
# Interactive manual-QA shell. Only the repository is mounted; host HOME is not.
make qa-shell

# Bootstrap the fixture root once, then run the live workflow in Docker.
export AUTORAG_LIVE_E2E_ROOT="$PWD/scripts/live-e2e"
node scripts/live-e2e/runner.mjs bootstrap --root "$AUTORAG_LIVE_E2E_ROOT"
make e2e-live-docker E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT" E2E_DATASOURCES=local
```

The container sets an ephemeral `HOME`, `AUTORAG_HOME`, and `AUTORAG_CONFIG`,
bind-mounts this clone at `/workspace`, and does not mount the host `~/.autorag`
or native datasource stores. Only the model credential allowlist in `QA_MODEL_ENV`
is forwarded; set it empty (`QA_MODEL_ENV=`) to forward no credentials at all.
Live E2E defaults to the local native MinSync embedder; set `E2E_EMBEDDER=gateway`
only when the Docker image has a compatible gateway runtime. Use `QA_PLATFORM`
and `QA_MODEL_ENV` to select a supported architecture or credential allowlist.
The live target keeps runner state inside the clone's `.autorag-e2e` directory;
evidence remains in the mounted clone under `.omo/evidence`.

`autorag init --force` also refuses to replace an existing implicit home config.
Use `--config <path>` or `AUTORAG_CONFIG=<path>` when replacement is intentional.
First-time implicit initialization remains allowed.

The QA image is based on Ubuntu 24.04 so current Linux MinSync release assets
run against the image's glibc; `make test-linux` continues to use the existing
Java 17 CI image. Native-store lanes are expected to skip inside the isolated
container unless their stores are deliberately provisioned inside it; never
mount a real host home to make a lane pass.
## Releases

Publishing the GitHub Release is not the announcement. People watching Discussions do not see the release feed. Every release also gets one Discussion in the **Announcements** category, with the same user-facing notes.

This applies to both tags:

- `v*` — AutoRAG 2.0, published by `.github/workflows/release.yml`
- `legacy-v*` — legacy Python, published by `.github/workflows/publish.yml`

After the release exists:

1. Search first. If an Announcements discussion for that tag already exists, comment on it. Do not open a second one.
   `gh discussion list --repo Marker-Inc-Korea/AutoRAG --category Announcements --search "vX.Y.Z"`
2. Post the user-facing notes (what changed, who it affects, upgrade steps) and link the tag:
   `gh discussion create --repo Marker-Inc-Korea/AutoRAG --category Announcements --title "AutoRAG vX.Y.Z" --body-file notes.md`
   Use `AutoRAG Legacy legacy-vX.Y.Z` as the title for a legacy tag.
3. Generated notes from commits are the source, not the post. Trim commit noise before posting.
4. Keep corpus text, secrets, tokens, cookies, and machine-local paths out of both the release notes and the discussion.

This is a maintainer step when cutting a release. It is not a requirement on contributor pull requests.

## Developer Commands

The repository root includes a `Makefile` for AutoRAG 2.0 validation:

- `make test` / `make test-all` — run the complete test suite.
- `make test-macos` — run the complete suite and require a macOS host.
- `make test-windows` — run the complete suite on a Windows host.
- `make test-linux` — run lint, typecheck, the complete suite, and build in a Docker container.
- `make lint`, `make typecheck`, `make build` — run individual checks.
- `make ci` — run the normal local lint, typecheck, complete test, and build sequence.

## Fixed live-E2E environment (the supported procedure)

Use the latest checkout's explicit, clone-local environment. Each clone owns its
`.autorag-e2e` state, so five independent clones may run concurrently without
sharing mutable state. Do not point two clones at the same `E2E_ROOT`.

The live-E2E environment is coupled to AutoRAG's behavior. When a change
modifies AutoRAG's major behavior or features — the agent tool surface,
retrieval methods, datasource skills, MinSync/embedding configuration, result
source identity rules, or the output contract — review whether the live-E2E
environment must change too (`scripts/live-e2e/`, `test/live-e2e/`, the corpus
manifest, preflight gates, and this procedure). A behavioral change that
invalidates the existing cold/warm QA evidence requires regenerating that
evidence; do not treat stale green evidence as proof for the new behavior.

Prerequisites:

- Node.js 24+, Bun, and the repository dependencies (`bun install --frozen-lockfile`).
- A bootstrapped corpus root. Bootstrap is explicit; live targets never create
  one implicitly:
  `export AUTORAG_LIVE_E2E_ROOT="$PWD/scripts/live-e2e"`
  `node scripts/live-e2e/runner.mjs bootstrap --root "$AUTORAG_LIVE_E2E_ROOT"`
- The gateway profile pinned in the clone-local `.autorag-e2e` home. The
  workflow self-ensures this via `autorag models prefetch --profile qwen3-embedding-0.6b`
  before starting the gateway; no manual Ollama serve, TEI adapter, or model
  pull is required.
- `OPENAI_API_KEY` and `AUTORAG_OPENAI_API_KEY` unset. Embeddings are local
  only; do not configure a remote embedding endpoint or send corpus text off
  the machine.

Run the cold path (fresh runner state and core local-file/MinSync verification):

```bash
make e2e-live-cold E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"
```

Run the warm path (reuse the same clone-local state after a successful cold run):

```bash
make e2e-live E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"
```

Datasource lanes run by default: with no `E2E_DATASOURCES` override the runner
executes the `local` lane plus every native CLI lane (lazykatok, discrawl, wacrawl,
telecrawl, slacrawl, lark, notcrawl, qmd, rclone, mailcrawl, and macOS Spotlight).
`E2E_DATASOURCES` only narrows this default (e.g. `E2E_DATASOURCES=local`
skips native lanes entirely). The summary separates the core MinSync result
(`commandsSummary.core`) from `datasourceLanes`. Native lanes whose CLI or
native store is missing are `SKIP` with a reason. An installed/configured
lane whose native check fails, including a successful harness without a valid
source-native identity, is `FAIL`; `SKIP` is never reported as PASS. On a
host where a native store genuinely exists, its lane must run and PASS —
leaving it `SKIP` by narrowing `E2E_DATASOURCES` is a QA gap, not a green
run. Native lanes are expected to remain `SKIP` only when no native store is
configured.
Native stores, profiles, and keychains remain owned by their CLIs: the runner
does not copy datasource data or force an AutoRAG workspace. Native datasource
references and setup details are in `docs/manual-qa-datasources.md` and the
individual scripts under `scripts/manual-qa/`, including the real lazykatok harness
at `scripts/manual-qa/run-qa-lazykatok-live.ts`.

Evidence is written to `.omo/evidence/task-6-fixed-live-e2e-environment.json`
for this task and to the runner result directory (by default
`.omo/evidence/live-core-cold/result.json` or `live-core-warm/result.json`).
Diagnostics and error text are reported verbatim, including absolute paths and
CLI stderr: an operator searching their own machine must be able to debug a
failure from the output alone. Assert local retrieval sources are absolute and readable; assert
native datasource results retain source-native identities such as
`/kakao/<instance>/chunks/<chunk>` (opaque slash-hierarchical, not an OS path),
not a retired `kakao:<chat>/<sender>/<chunk>` scheme and not a fake OS-absolute
filesystem path.

Cleanup is limited to runner-owned state. The cold command removes and rebuilds
`.autorag-e2e`; for manual cleanup, run `rm -rf .autorag-e2e` from this clone
only. Leave lazykatok, discrawl, crawler, qmd, rclone, mailcrawl, and Spotlight
native stores untouched.
Record the cleanup receipt in the task evidence; never stage `.debug-journal.md`.

## Required MinSync Live QA

When validating local-file retrieval changes, run a real semantic query
through the product default gateway path. The local gateway is the default;
a remote embedding service is used only when the operator explicitly
configures one.

The default product path uses the AutoRAG-owned `autorag-gateway` with the
`qwen3-embedding-0.6b` profile (1024 dimensions, no query/passage prefixes).
The gateway is started on demand by the semantic MinSync path and stays
loopback-only.

Run the isolated experiment. Both AutoRAG variables point inside the temp workspace, so `init --force` cannot reach the shared `~/.autorag` (see the manual-QA rules above):

```bash
WORKSPACE="$(mktemp -d)"
export AUTORAG_HOME="$WORKSPACE/.autorag-home"
export AUTORAG_CONFIG="$AUTORAG_HOME/config.json"
mkdir -p "$WORKSPACE/docs" "$AUTORAG_HOME"
printf '%s\n' \
  'Refund exceptions require director approval before payout.' \
  'Finance acknowledged the policy in the July review.' \
  > "$WORKSPACE/docs/refund-policy.txt"

cd "$WORKSPACE"
# Pin the model to the local cache (no Ollama, no adapter)
autorag models prefetch --profile qwen3-embedding-0.6b

# Init with the default gateway profile
autorag init \
  --search-paths "$WORKSPACE/docs" \
  --force

autorag refresh --method parsed,minsync --json
autorag search --json "semantic question about refund approval"
```

The QA gate is not complete until all of the following are observed:

1. `autorag refresh --method minsync` exits successfully and
   `.minsync/cursor.json` exists under the workspace.
2. The semantic query returns a hit for the fixture document.
3. AutoRAG maps that hit to an OS-absolute original `source` path.
4. `fs.existsSync(source)` and reading `source` succeed.
5. The run used the local gateway (no remote embedding endpoint configured).

If the model prefetch fails, the gateway is unavailable, or MinSync reports a
semantic failure, report the exact blocking diagnostic and do not claim live
MinSync verification.

### Legacy/manual variant (Ollama + TEI adapter)

For an existing workspace pinned to Ollama's EmbeddingGemma (768 dimensions),
keep the adapter available as a manually-started sidecar:

```bash
ollama pull embeddinggemma:latest
ollama serve
OLLAMA_EMBEDDINGS_URL=http://127.0.0.1:11434/api/embeddings \
  python3 scripts/manual-qa/ollama-tei-adapter.py
```

Then initialize the workspace explicitly with the TEI endpoint, in the same shell that exported `AUTORAG_HOME` and `AUTORAG_CONFIG` above:

```bash
cd "$WORKSPACE"
autorag init \
  --search-paths "$WORKSPACE/docs" \
  --embedder-id tei:embeddinggemma:latest \
  --embedder-base-url http://127.0.0.1:18080 \
  --embedder-dimension 768 \
  --minsync-max-chunk-size 1000 \
  --force
autorag refresh --method parsed,minsync --json
autorag search --json "semantic question about refund approval"
```

The adapter translates MinSync's `POST /embed` request to Ollama's
`POST /api/embeddings` request and returns the TEI response shape (a bare JSON
array of embedding arrays). Do not use this variant as a fresh-install
requirement or as an implicit fallback from the gateway.

Docker can reproduce the Linux job on macOS, Linux, or Windows hosts. The
`test-linux` target uses an isolated container volume for `node_modules`, so it
does not replace host-native dependencies, and pins `linux/amd64` to match
GitHub's Ubuntu runner. Windows
containers require a Windows kernel, so Windows compatibility is run natively
from Git Bash/MSYS2 with `make test-windows` (or directly with
`bun run test:windows`) and verified by the `windows-latest` GitHub-hosted
runner.

## Purpose

AutoRAG is an **over-powered librarian agent** for **document collections** — PDFs, wikis, notes, research papers, knowledge bases, and any unstructured text corpus. It is a customized [Pi](https://github.com/earendil-works/pi-mono) agent: the Pi agent loop configured into a librarian, used through one library/programmatic API (and a thin CLI).

Searches run in one agent loop: the librarian chooses retrieval methods, reads source files directly, judges the evidence, and curates structured results. The model and provider come from the user's authenticated runtime; the distributed package does not assume a private provider.

AutoRAG itself is the specialized librarian agent. It uses one configured
model for the whole search loop, so model selection should favor reliable tool
calling and structured output. High TPS and low first-token latency improve
interactive search speed because retrieval commonly spans several model turns;
they do not make the underlying BM25, MinSync, Jikji, filesystem, or indexing
operations faster.

**Primary target**: non-code document retrieval (manuals, legal docs, internal wikis, meeting notes, research literature).
Code repositories work too. AutoRAG's value is in the exploration + retrieval methods + curation layer that sit *on top* of raw search.

## Product Positioning

Three core values drive every design decision. Features and PRs that conflict
with any of them should be rejected or reshaped:

1. **Never migrate your data to search it.** AutoRAG federates CLI-owned
   stores (lazykatok, discrawl, qmd, msgvault, rclone, …) in place. No forced
   ingestion into a central index, no third-party server holding a copy of
   the corpus. Results carry source-native identities
   (`/kakao/<instance>/chunks/<chunk>`), and secrets
   stay with the tool that owns them.
2. **Just works — no RAG degree required.** A non-developer installs it and
   it works: minimal configuration, no pipeline tuning, no vector-DB
   operations, no OpenAI keys. The local embedder (EmbeddingGemma via Ollama)
   and MinSync auto-install handle the "RAG plumbing" invisibly.
3. **Fast by design.** One configured model owns the whole loop; retrieval
   runs locally over MinSync CDC chunks (BM25 / vector / hybrid); interactive
   search is optimized for low latency across several model turns.

Rationale for value 1 comes from the competitive landscape study
(see `docs/competitive-landscape-2026-09.md`): MCP-native, local-first, and
hybrid retrieval are commoditized, while harness-free federation with
source-native provenance is the durable differentiator.

## New CLI-backed datasource

External datasource CLIs (lazykatok, discrawl, slacrawl, qmd, rclone,
crawlers) are driven **directly** with their own native stores. There is no
AutoRAG-managed workspace/config forcing and no bash gate: the agent may run
these CLIs through `bash` as well as through the datasource tools.

Contributors and agents adding a CLI-backed datasource must:

- spawn the CLI with its own default store; never force
  `--workspace`/`--config`/env into an empty AutoRAG-managed directory unless
  the operator explicitly configured a workspace path;
- keep result sources as opaque slash-hierarchical datasource identities
  (e.g. `/kakao/<instance>/chunks/<chunk>`), never OS-absolute fake filesystem
  paths the agent could mistake for local files; they are not OS paths and
  must not be passed to `bash`/`cat`;
- provide a datasource skill with native command examples and `<binary>
  --help` guidance so the agent understands which CLI backs the datasource;
- keep failure isolation per CLI (one failing CLI degrades to diagnostics and
  an `unsearched` entry, never crashes the search loop);
- report failures verbatim: a retrieval method that cannot answer throws the
  CLI's own error (failure kind, exit status, stderr) instead of returning an
  empty result set, so the caller sees why the source was not searched;
- retain small, focused guards where they matter (e.g. discrawl's user-token
  rejection);
- add focused tests and live manual QA where a local store exists before
  registering the datasource.

Secrets must remain external: store only environment-variable, keychain, or
profile references, and never persist tokens, cookies, passwords, or refresh
credentials into config files or argv snapshots. This is about where
credentials live, not about muting errors — diagnostics and CLI stderr are
never scrubbed or suppressed on the way to the operator.

## Error Transparency

Errors belong to the user, not to the agent. Retrieval, refresh, and datasource
failures surface the underlying text verbatim — exit codes, stderr, and real
filesystem paths included — in diagnostics, `unsearched` reasons, and CLI
output. Do not classify a failure into a fixed enum in place of its message, do
not replace it with a generic sentence, and do not drop it because it contains a
path. Bounding runaway output by length is fine; suppressing content is not.

## Why AutoRAG Exists

Raw search tools return file paths and matching lines. A human still has to open each file, read the context, decide what's relevant, and synthesize an answer. AutoRAG eliminates that entire workflow:

1. **Search** across multiple retrieval methods (BM25, vector/MinSync, datasource skills — pluggable)
2. **Read** promising source files directly with the built-in bash tool
3. **Judge and curate** — extract key insights, resolve conflicts, and assess freshness
4. **Deliver** numbered knowledge units grounded in the sources
5. **Learn** — remember which methods worked and adapt strategy over time

The loop exists to serve the three core values above: it searches data where
it already lives (value 1), hides the retrieval plumbing behind curated
answers (value 2), and keeps the default retrieval path local and
latency-sensitive (value 3).

## Agent Tools

The librarian agent owns the full workflow:

| Tool | What it does | When to use |
|------|-------------|-------------|
| `bash` | Filesystem discovery and document reading with real paths (`ls`, `find`, `grep`, `cat`, etc.) | Direct source verification |
| `jikji_find` | Runs `jikji find ROOT "query"` and returns a policy-aware answer pack | Optional local discovery |
| `everything_search` | Windows only: instant file/folder name, extension, path, size, and date search over the configured search roots through the bundled voidtools Everything + ES | Locating files by name before reading them |
| `fsearch_search` | macOS/Linux only: instant file/folder name, extension, path, size, and date search over the configured search roots through the user's fsearch-cli (FSearch) database + watch daemon; degrades to a slow filesystem walk when fsearch-cli is not installed | Locating files by name before reading them |
| `search_all_documents` | Fan-out across configured retrieval methods and merge/rank candidates | Combined retrieval |
| `semantic_search_local_docs` | MinSync semantic/vector retrieval over parsed mirrors | Semantic retrieval |
| `search_datasource_<name>` | Search one datasource connection only; one tool is generated per configured connection (e.g. `search_datasource_discord`, `search_datasource_kakao_work`) and spawns no other datasource CLIs. This is the only datasource search surface — use it instead of any fan-out datasource tool | Targeted single-datasource retrieval |
| `check_memory` | Look up judged evidence from this conversation, similar past questions (hybrid BM25 + vector), and long-term insights | Advisory reference before searching |
| `load_datasource_skill` | Load instructions for a configured datasource skill | Datasource-specific searches |
| `scan_duplicate_documents` | Read-only dupey scan of configured local document roots | Duplicate-family review |
| `web_search` | Internet web search through the oh-my-pi-style provider chain; credential-free by default, keyed providers via env vars with quota-fallback | Current/public web information |
| `web_fetch` | Fetch a public http(s) URL and render it as markdown/text | Reading pages found via `web_search` or known URLs |
| `recommend_peer_targets` | Rank local SimpleX peer contacts (the profile a peer shared plus your local name and note) by keyword overlap | P2P routing; never contacts peers |
| `emit_fast_answer` | Internal non-terminating tool that delivers the fast-phase first answer | Two-phase progressive answers |
| `emit_autorag_results` | Terminating tool that returns curated results; `answer` is the complete answer, or only the delta (corrections + newly verified findings) when a fast answer already reached the caller | Final action |

There is no `lexical_search_local_docs` tool. BM25 runs inside MinSync (and some datasource methods) and is reached through `search_all_documents`. `recommend_peer_targets`, `web_search`, and `web_fetch` are omitted in remote P2P sessions.

`web_search`/`web_fetch` are ported from oh-my-pi's web module: a credential-free-only provider chain — model-native search reusing the agent's own model credentials (`gemini`/`anthropic`/`codex`/`xai`), the anonymous `perplexity` ask endpoint, Parallel's keyless MCP (`parallel`), then the scraped engines (`startpage`/`duckduckgo`/`ecosia`/`google`/`mojeek`, plus the `public` fan-out aggregate) with headless-browser escalation for bot challenges — where quota, auth, and bot-challenge failures automatically fall back to the next provider. No API key or signup is required; a self-hosted `SEARXNG_ENDPOINT` is the only env-gated, explicitly-advanced option. Web queries leave the machine: never include private corpus content or secrets in them.

## Architecture

```
Agent Tools                 AutoRAGAgent (customized Pi agent)
┌──────────────────┐       ┌──────────────────────────────────┐
│ bash / jikji_find │       │ Retrieval Memory (judged evidence)│
│ search_all_docs   │  ───▶ │ Curation Layer (LLM extraction)   │
│ semantic_search   │       │ check_memory (advisory reference) │
│ search_datasource │       │ Manifest System (indexed stores)  │
│ scan_duplicates   │       │ Retrieval Registry (pluggable)    │
│ peer_targets      │       │ Result Merger (cross-method)      │
└──────────────────┘       └──────────────────────────────────┘
```

## Retrieval Methods

AutoRAG is designed for **multi-method retrieval** — different methods for different document types:

| Method | Status | Best for |
|--------|--------|----------|
| BM25 (keyword) | Active | MinSync lexical ranking over shared CDC chunks from parsed mirrors |
| MinSync vector (semantic) | Active | Incrementally indexed semantic retrieval over the same MinSync chunks |
| Hybrid (vector+BM25) | Active | MinSync hybrid mode over the same canonical chunk IDs |
| Datasource skills | Active | External server-configured sources (e.g. KakaoTalk via `lazykatok`) |
| Vector (other backends) | Planned | Other dense-document backends, "find similar to X" |

The `RetrievalMethodRegistry` and `ResultMerger` are live: configured methods are registered and routed through `ParallelRetriever` + `ResultMerger`. New methods implement the `RetrievalMethod` interface and plug into the same pipeline. Plain-directory content search is handled directly through the agent's `bash` tool.

Jikji is intentionally not a retrieval method. It is an optional local-discovery layer: AutoRAG calls `jikji find ROOT "query" --json` through `jikji_find`, parses the upstream answer pack, and exposes `handoff_action`, `tool_call_policy`, and `agent_should_not_rerank` to the librarian. Direct `bash` reading remains available so Jikji never prevents source verification. `prepare`/`refresh` remain for indexing only and do not answer queries directly.

Datasource skills are retrieval-method factories plus indexing hooks for external, server-configured data sources. They remain inside the same pipeline — `RetrievalMethodRegistry` → `ParallelRetriever` → `filterDatasourceScope` → `ResultMerger`. Every connected datasource is searchable: a generated `search_datasource_<id>` tool exposes `{ query, topK?, scope? }`, where `scope` is ordinary per-query path filtering and `topK` bounds candidates. There is no datasource fan-out tool: every configured connection gets its own tool, and `search_all_documents` already spans every retrieval method including datasources. Results are not redacted — traceability is preferred over opacity, so pair AutoRAG with a local LLM when privacy matters.

CLI-backed datasources own their archive, lexical index, and vectors: KakaoTalk through the external `lazykatok` CLI, and **Discord** through the external [`discrawl`](https://github.com/openclaw/discrawl) CLI. AutoRAG only spawns them and maps results. AutoRAG never reads KakaoTalk databases directly; failures surface as diagnostics, and remote embedding egress settings are rejected before the CLI is spawned.

External crawler-backed skills cover **WhatsApp** (wacrawl), **Telegram** (telecrawl), **Slack** (slacrawl), and **Notion** (notcrawl); each crawler owns its archive, sync, credentials, and FTS search while AutoRAG provides bounded process execution, diagnostics, and retrieval mapping. The remaining connector-backed datasource skills use the shared framework (`src/datasource/connector.ts`, `chunk-store.ts`, `connector-skill.ts`): **GitHub**, **Google Drive**, **local mail export**, **Obsidian** (vault via external `qmd` CLI: incremental + BM25 + semantic), **RSS/news**, and **Spotlight**. Gmail, IMAP, and Maildir retrieval is provided by **mailcrawl**. Results remain traceable. Manual QA harnesses live in `scripts/manual-qa/` (see `docs/manual-qa-datasources.md`).

## Directory Access

The AutoRAG librarian navigates document collections directly with `bash`, using real paths for discovery and reading. Retrieval tools return bounded candidates; the librarian opens the source material, assesses sufficiency and freshness, resolves conflicts, and finalizes with `emit_autorag_results`.

Model authentication stays with the configured provider or authenticated local runtime; corpus indexes remain workspace-local under `<workspace>/.autorag`.

- **Tool surface** — the librarian owns `bash`, `check_memory`, `jikji_find`, `everything_search` (Windows, local sessions), `fsearch_search` (macOS/Linux, local sessions), `search_all_documents`, `semantic_search_local_docs`, one `search_datasource_<id>` tool per configured datasource connection, `load_datasource_skill`, `scan_duplicate_documents`, `recommend_peer_targets` (local sessions), `emit_fast_answer`, and `emit_autorag_results`.
- **Parsed mirrors** — `AutoRAGAgent.refresh()` parses supported files from configured source directories into `.autorag/parsed`; BM25 and MinSync index those parsed mirrors.
- **Document parsing** — `kordoc` is the default parser for `.hwp`, `.hwpx`, `.hml`, `.hwpml`, `.pdf`, `.docx`, `.xlsx` and `.xls`; it runs in-process (no Java, no subprocess) and keeps nested tables and per-sheet workbook structure. `.pptx`, `.eml` and plain text keep their own parsers. kordoc failures surface as `ParseError` with kordoc's own code and message verbatim, and kordoc warnings become `parser-warning` diagnostics.
- **Global language setting** — one `languages` list (config `languages`, `--languages`, or `AUTORAG_LANGUAGES`; default `["ko", "en"]`) describes the corpus. Accepted tags are curated in `src/language.ts` because every tag must map to an OCR engine configuration. Format parsing is language-agnostic; `languages` only selects OCR recognition languages (`ja` → `jpn`, `zh-hans` → `chi_sim`, …) for standalone images and scanned PDF pages. OCR stays opt-in, so a default refresh downloads no recognition model.
- **Jikji discovery** — `jikji_find` runs `jikji find ROOT "query" --json` and returns the answer pack to the librarian; direct file reading remains available. `prepare`/`refresh` remain for indexing only; AutoRAG-managed prepare runs with `--no-agent-rules` by default so it never rewrites the consumer repo's `AGENTS.md`/`CLAUDE.md`/`.cursorrules`. An explicit `writeAgentRules: true` opt-in re-enables upstream routing-block injection.
- **No indexing during inference** — a query (`searchDocuments`, TUI, MCP retrieve) only reads prebuilt indexes. It never runs `jikji prepare`, `minsync sync`/`init`, parsed-mirror sync, fsearch/Everything index builds or instance startup, or a binary auto-install. All of that runs only in `refresh`/`watch` (incremental: MinSync syncs changed mirrors, `jikji prepare` reuses unchanged docs, roots prepare in parallel). A never-refreshed source simply contributes no evidence.
- **External tool auto-install** — MinSync and Jikji binaries are cached under `<workspace>/.autorag/bin`. MinSync auto-installs from crates.io via `cargo install minsync` by default, falling back to verified GitHub release assets when cargo is unavailable (`minSync.autoInstall: false` opts out). Jikji auto-installs the `jikji-cli` crate from crates.io via cargo by default (`jikji.autoInstall: false` opts out; requires the Rust toolchain). New `autorag init` configs enable Jikji by default (`jikji: {}`). The KakaoTalk `lazykatok` and Discord `discrawl` CLIs remain manual, optional installs (`brew install openclaw/tap/discrawl`). All three degrade gracefully when missing.
- **Everything (Windows)** — the npm package bundles the unmodified voidtools portable Everything 1.4.1.1032 and ES 1.1.0.38 ZIPs (x64/ARM64) in `vendor/everything`, pinned by SHA-256 in `vendor/everything/manifest.json`, with their MIT (and PCRE BSD) texts in `licenses/` and NOTICE. On Windows only, AutoRAG extracts and verifies them into `<workspace>/.autorag/everything/<version>/<arch>/` and starts a named, user-level instance (`autorag-<hash>`) with its own `Everything.ini`/`Everything.db` that indexes only the configured search roots as folder indexes. It never requests elevation, installs the Everything service, indexes whole NTFS/ReFS volumes, or enables the HTTP/ETP servers, and it does not touch a user's own Everything. `refresh` (all methods, any parsed refresh, or `--method everything`) rewrites the config, restarts the instance, and waits for the index; failures surface as `everything-index-failed` with ES's exit code and stderr verbatim. `everything: false` disables it; on macOS/Linux it is absent. Remote P2P sessions never receive `everything_search`.
- **FSearch (macOS/Linux)** — the Everything-analog on non-Windows hosts, via the user's [`fsearch-cli`](https://github.com/NomaDamas/fsearch-mac) (GPL-2.0-or-later). fsearch-cli is NOT bundled: the user installs it separately and AutoRAG only spawns it as a separate process over its CLI and Unix-socket daemon protocol — mere aggregation per the FSF GPL FAQ, never a combined work, so AutoRAG keeps its own license. On macOS/Linux only, AutoRAG builds a per-workspace database at `<workspace>/.autorag/fsearch/fsearch.db` that indexes only the configured search roots (excluding `.autorag` state dirs) and keeps a `fsearch-cli watch` daemon (pid file `watch.pid`) live from FSEvents/inotify so `fsearch-cli search` answers over its socket — an explicit short socket at `/tmp/autorag-fsearch-<uid>-<sha12(db realpath)>.sock`, because fsearch-cli's default `<db>.sock` overflows the 104-byte unix sun_path limit for deep workspace paths and daemon/client rendezvous silently breaks; `stopFsearch()` terminates the daemon, and it never touches the user's own FSearch app database (`$XDG_DATA_HOME/fsearch/fsearch.db`). `refresh` (all methods, any parsed refresh, or `--method fsearch`) rebuilds the database and re-verifies the daemon; a missing binary surfaces as a `fsearch-binary-missing` warning (not an error), other failures as `fsearch-index-failed` with the CLI's exit code and stderr verbatim. When fsearch-cli is not installed, `fsearch_search` still answers through a bounded slow filesystem walk (substring/regex name matching) and labels its output as such. `fsearch: false` disables it; on Windows it is absent (Everything covers it). Remote P2P sessions never receive `fsearch_search`.
- **Datasource skills** — `AutoRAGAgent` can register `datasourceSkills`; their retrieval methods are merged with the normal retrieval pipeline, filtered before merging by per-query scope, and indexed during `refresh()`.

## Usage

```typescript
import { AutoRAGAgent } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
});
const response = await agent.searchDocuments("summarize the Q3 financial report");
console.log(response.answer);
```

`searchDocuments()` drives the Pi agent loop and returns a typed `SearchDocumentsResponse`; the caller consumes the structured payload directly, without parsing assistant text.

## Output Contract

**Caller sees curated, numbered knowledge units:**
```
[1] Revenue Summary — Q3 revenue grew 23% YoY to $4.2M, driven by enterprise contracts. (pages 3-5)
[2] Risk Factors — Three new risk factors added: supply chain, regulatory, talent retention. (pages 12-14)
```

Each result maps to an internal entry carrying its `source` (a real file path or datasource id), `method`, and cited evidence for retrieval memory. The curated `answer`/`results` are grounded in the sources; source paths may appear where relevant.

## Memory System

Retrieval memory is reference context, never instructions. After the final
answer (`emit_autorag_results`, or `emit_fast_answer` when Jev ends the run
early), every cited evidence is mapped back to the search query and method that
surfaced it and judged by Jev in one batched call: does the evidence really
support the sentence of the answer it backs? At P(supports) >= 0.7 the evidence
is stored as `judgedEvidence` in `~/.autorag/memory.json`; below 0.7 it is
discarded. With Jev disabled or unreachable nothing is judged or stored, and an
`evidence-judgment-fallback` diagnostic says why. Nothing is capped or evicted;
every 100 judged records are summarized once into long-term insights.

At search time the agent receives (1) everything judged earlier in the current
conversation, up to the 50 most recent records, (2) up to 25 similar past
questions found by hybrid BM25 + vector search with reciprocal-rank fusion, with
their judged evidence, and (3) matching long-term insights — injected as a
`<memory_context>` user message and offered through the `check_memory` tool.
Jev's datasource-selection step also receives up to 5 similar past questions as
hints. Memory never reorders search results. Evidence from datasources not
configured for the run is never shown, and remote P2P sessions never read or
write it. See [docs/retrieval-memory.md](docs/retrieval-memory.md).

## Files

| File | Role |
|------|------|
| `src/agent/agent.ts` | AutoRAGAgent class — the customized Pi agent and library API |
| `src/agent/bash-tool.ts` | Direct filesystem discovery and document-reading tool |
| `src/agent/fast-answer-tool.ts` | `emit_fast_answer` non-terminating tool for the fast-phase first answer |
| `src/agent/jev-extension.ts` | Shared Jev judge (`createJevJudge`) and the optional `jev` pi extension tool |
| `src/agent/query-routing.ts` | Jev query router: direct (intrinsic knowledge) / local branch, the decomposition check, the per-datasource search check, and the post-fast-answer follow-up check |
| `src/agent/query-decomposition.ts` | LLM question decomposition into at most five search queries |
| `src/agent/emit-results-tool.ts` | `emit_autorag_results` terminating tool that returns curated results as typed details |
| `src/agent/jikji-find-tool.ts` | `jikji_find` local-discovery tool |
| `src/agent/everything-search-tool.ts` | `everything_search` Windows file-name search tool |
| `src/everything/` | Bundled Everything extraction/verification (`bundle.ts`) and the per-workspace instance + ES client (`client.ts`) |
| `src/agent/fsearch-search-tool.ts` | `fsearch_search` macOS/Linux file-name search tool |
| `src/fsearch/` | fsearch-cli client: per-workspace DB + watch daemon (`client.ts`) and the slow-walk fallback (`walk.ts`) |
| `src/agent/search-all-tool.ts` | `search_all_documents` multi-method fan-out |
| `src/agent/search-minsync-tool.ts` | `semantic_search_local_docs` MinSync vector tool |
| `src/agent/web-search-tool.ts` | `web_search` internet search tool over the `src/web/search` provider chain |
| `src/agent/web-fetch-tool.ts` | `web_fetch` URL reader over the `src/web/fetch` render pipeline |
| `src/web/search/` | oh-my-pi-ported web search: provider chain, structured query parsing, keyed + credential-free providers |
| `src/web/fetch/` | oh-my-pi-ported URL render pipeline: page loader, HTML→markdown reader chain, feeds, content negotiation |
| `src/agent/dupey-tool.ts` | `scan_duplicate_documents` read-only dupey scan |
| `src/agent/peer-target-tool.ts` | `recommend_peer_targets` local SimpleX peer-contact ranking (shared profile + your name/note) |
| `src/agent/system-prompt.ts` | System prompt builder for the librarian agent |
| `src/memory/memory.ts` | Retrieval memory store: judged evidence, curated results, evidence chunks, insights, and v4→v5 file parsing |
| `src/memory/similar-queries.ts` | Similar past questions via hybrid BM25 + vector search with reciprocal-rank fusion and the SQLite question-vector cache |
| `src/memory/judged-evidence.ts` | `JudgedEvidenceRecord` shape and the 0.7 `EVIDENCE_SUPPORT_THRESHOLD` |
| `src/memory/context.ts` | Assembles current-conversation evidence, similar past questions, and insights into the advisory memory context |
| `src/memory/renderer.ts` | Renders retrieval memory as JSON-quoted advisory markdown |
| `src/memory/check-memory-tool.ts` | `check_memory` tool (pi-agent-core AgentTool) |
| `src/agent/evidence-judgment.ts` | Post-answer Jev step: one batched evidence-support call, the 0.7 threshold, and fallback |
| `src/agent/evidence-origins.ts` | Maps a cited evidence ref back to the search query and method that surfaced it |
| `src/manifest/loader.ts` | YAML/JSON manifest loader for indexed data stores |
| `src/retrieval/types.ts` | Core retrieval type definitions |
| `src/retrieval/registry.ts` | Method registry for multi-method orchestration |
| `src/retrieval/merger.ts` | Cross-method result merging and deduplication |
| `src/retrieval/rerank.ts` | Post-merge reranker seam + OpenRouter implementation (`OpenRouterReranker`, `createReranker`) |
| `src/minsync/method.ts` | MinSync retrieval method (vector / BM25 / hybrid over shared CDC chunks) |
| `src/language.ts` | Curated global language tags, normalization, and defaults |
| `src/parser/kordoc.ts` | Default document parser (kordoc) for HWP/HWPX/HWPML, PDF, DOCX, XLSX/XLS |
| `src/parser/ocr-engines.ts` | Language → tesseract traineddata mapping and the injected OCR provider |
| `src/datasource/` | Datasource skill contracts, scope filtering, polling metadata, diagnostics, and KakaoTalk/lazykatok skill implementation |
| `src/p2p/` | SimpleX P2P sharing: policy, injection/PII gates, approval store, wire protocol |
| `src/cli/commands/serve.ts` | `autorag serve` P2P peer query server |
| `src/cli/commands/p2p.ts` | `autorag p2p` peer trust and request approval |
| `src/cli/commands/p2p-policy.ts` | `autorag p2p policy` sharing-rule CLI |
| `src/datasource/connector.ts` | Connector contract + opaque-text/id sanitizers for connector-backed skills |
| `src/datasource/chunk-store.ts` | Persistent chunk store with BM25-style lexical search per skill instance |
| `src/datasource/connector-skill.ts` | Shared DatasourceSkill base composing a connector with the chunk store |
| `src/datasource/skills/` | Built-in skills: lazykatok, discrawl, wacrawl, telecrawl, slack, lark, clawgallery, notion, github, cloud-drive, mail-export, mailcrawl, obsidian, rss, spotlight (+ config factory) |
| `src/agent/search-single-datasource-tool.ts` | One generated `search_datasource_<id>` tool per configured datasource connection with model-safe `{ query, topK?, scope? }` parameters |
