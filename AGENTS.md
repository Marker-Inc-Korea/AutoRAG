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
- Sending corpus text to remote embedders (no `OPENAI_API_KEY` / remote embedding endpoint for corpus text)
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
telecrawl, slacrawl, notcrawl, qmd, rclone, mailcrawl, and macOS Spotlight).
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
through the product default gateway path. Do not use OpenAI credentials or
send corpus text to a remote embedding service.

The default product path uses the AutoRAG-owned `autorag-gateway` with the
`qwen3-embedding-0.6b` profile (1024 dimensions, no query/passage prefixes).
The gateway is started on demand by the semantic MinSync path and stays
loopback-only.

Run the isolated experiment:

```bash
WORKSPACE="$(mktemp -d)"
mkdir -p "$WORKSPACE/docs"
printf '%s\n' \
  'Refund exceptions require director approval before payout.' \
  'Finance acknowledged the policy in the July review.' \
  > "$WORKSPACE/docs/refund-policy.txt"

cd "$WORKSPACE"
# Pin the model to the local cache (no Ollama, no adapter)
AUTORAG_HOME="$WORKSPACE/.autorag-home" \
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
5. `OPENAI_API_KEY` is unset and no request leaves the local machine.

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

Then initialize the workspace explicitly with the TEI endpoint:

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
   (`/kakao/<instance>/chunks/<chunk>`) and scope-checked access, and secrets
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
answers (value 2), and keeps every step local and latency-sensitive
(value 3).

## Agent Tools

The librarian agent owns the full workflow:

| Tool | What it does | When to use |
|------|-------------|-------------|
| `bash` | Filesystem discovery and document reading with real paths (`ls`, `find`, `grep`, `cat`, etc.) | Direct source verification |
| `jikji_find` | Runs `jikji find ROOT "query"` and returns a policy-aware answer pack | Optional local discovery |
| `search_all_documents` | Fan-out across configured retrieval methods and merge/rank candidates | Combined retrieval |
| `semantic_search_local_docs` | MinSync semantic/vector retrieval over parsed mirrors | Semantic retrieval |
| `search_datasource_documents` | Search authorized external datasource skills | Server-bound datasource retrieval |
| `search_datasource_<name>` | Search one datasource connection only; one tool is generated per authorized connection (e.g. `search_datasource_discord`, `search_datasource_kakao_work`) and spawns no other datasource CLIs | Targeted single-datasource retrieval |
| `check_memory` | Query past search outcomes | Adaptive strategy |
| `load_datasource_skill` | Load instructions for an authorized datasource skill | Datasource-specific searches |
| `scan_duplicate_documents` | Read-only dupey scan of configured local document roots | Duplicate-family review |
| `web_search` | Internet web search through the oh-my-pi-style provider chain; credential-free by default, keyed providers via env vars with quota-fallback | Current/public web information |
| `web_fetch` | Fetch a public http(s) URL and render it as markdown/text | Reading pages found via `web_search` or known URLs |
| `recommend_peer_targets` | Rank local SimpleX peer personas by keyword overlap | P2P routing; never contacts peers |
| `emit_fast_answer` | Internal non-terminating tool that delivers the fast-phase first answer | Two-phase progressive answers |
| `emit_autorag_results` | Terminating tool that returns curated results | Final action |

There is no `lexical_search_local_docs` tool. BM25 runs inside MinSync (and some datasource methods) and is reached through `search_all_documents`. `recommend_peer_targets`, `web_search`, and `web_fetch` are omitted in remote P2P sessions.

`web_search`/`web_fetch` are ported from oh-my-pi's web module: a credential-free-only provider chain — model-native search reusing the agent's own model credentials (`gemini`/`anthropic`/`codex`/`xai`), the anonymous `perplexity` ask endpoint, Parallel's keyless MCP (`parallel`), then the scraped engines (`startpage`/`duckduckgo`/`ecosia`/`google`/`mojeek`, plus the `public` fan-out aggregate) with headless-browser escalation for bot challenges — where quota, auth, and bot-challenge failures automatically fall back to the next provider. No API key or signup is required; a self-hosted `SEARXNG_ENDPOINT` is the only env-gated, explicitly-advanced option. Web queries leave the machine: never include private corpus content or secrets in them.

## Architecture

```
Agent Tools                 AutoRAGAgent (customized Pi agent)
┌──────────────────┐       ┌──────────────────────────────────┐
│ bash / jikji_find │       │ Memory System (query history)     │
│ search_all_docs   │  ───▶ │ Curation Layer (LLM extraction)   │
│ semantic_search   │       │ check_memory (adaptive strategy)  │
│ search_datasource │       │ Manifest System (indexed stores)  │
│ scan_duplicates   │       │ Retrieval Registry (pluggable)    │
│ peer_targets      │       │ Result Merger (cross-method)      │
└──────────────────┘       │ Feedback Loop (learn from usage)  │
                           └──────────────────────────────────┘
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

Datasource skills are retrieval-method factories plus indexing hooks for external, server-configured data sources. They remain inside the same pipeline — `RetrievalMethodRegistry` → `ParallelRetriever` → `DatasourceResultFilter` → `ResultMerger`. Datasource access is default-deny and server-bound: LLM tool arguments cannot grant `allowedTags` or `allowedScopes`, and `search_datasource_documents` exposes only `{ query, topK?, scope? }` where `scope` can only narrow trusted access. Results are not redacted — traceability is preferred over opacity, so pair AutoRAG with a local LLM when privacy matters.

CLI-backed datasources own their archive, lexical index, and vectors: KakaoTalk through the external `lazykatok` CLI, and **Discord** through the external [`discrawl`](https://github.com/openclaw/discrawl) CLI. AutoRAG only spawns them and maps results. AutoRAG never reads KakaoTalk databases directly; failures surface as diagnostics, and remote embedding egress settings are rejected before the CLI is spawned.

External crawler-backed skills cover **WhatsApp** (wacrawl), **Telegram** (telecrawl), **Slack** (slacrawl), and **Notion** (notcrawl); each crawler owns its archive, sync, credentials, and FTS search while AutoRAG provides bounded process execution, diagnostics, and retrieval mapping. The remaining connector-backed datasource skills use the shared framework (`src/datasource/connector.ts`, `chunk-store.ts`, `connector-skill.ts`): **GitHub**, **Google Drive**, **local mail export**, **Obsidian** (vault via external `qmd` CLI: incremental + BM25 + semantic), **RSS/news**, and **Spotlight**. Gmail, IMAP, and Maildir retrieval is provided by **mailcrawl**. Results remain traceable and datasource access stays default-deny. Manual QA harnesses live in `scripts/manual-qa/` (see `docs/manual-qa-datasources.md`).

## Directory Access

The AutoRAG librarian navigates document collections directly with `bash`, using real paths for discovery and reading. Retrieval tools return bounded candidates; the librarian opens the source material, assesses sufficiency and freshness, resolves conflicts, and finalizes with `emit_autorag_results`.

Model authentication stays with the configured provider or authenticated local runtime; corpus indexes remain workspace-local under `<workspace>/.autorag`.

- **Tool surface** — the librarian owns `bash`, `check_memory`, `jikji_find`, `search_all_documents`, `semantic_search_local_docs`, `search_datasource_documents`, `load_datasource_skill`, `scan_duplicate_documents`, `recommend_peer_targets` (local sessions), `emit_fast_answer`, and `emit_autorag_results`.
- **Parsed mirrors** — `AutoRAGAgent.refresh()` parses supported files from configured source directories into `.autorag/parsed`; BM25 and MinSync index those parsed mirrors.
- **Jikji discovery** — `jikji_find` runs `jikji find ROOT "query" --json` and returns the answer pack to the librarian; direct file reading remains available. `prepare`/`refresh` remain for indexing only; AutoRAG-managed prepare runs with `--no-agent-rules` by default so it never rewrites the consumer repo's `AGENTS.md`/`CLAUDE.md`/`.cursorrules`. An explicit `writeAgentRules: true` opt-in re-enables upstream routing-block injection.
- **External tool auto-install** — MinSync and Jikji binaries are cached under `<workspace>/.autorag/bin`. MinSync auto-installs from crates.io via `cargo install minsync` by default, falling back to verified GitHub release assets when cargo is unavailable (`minSync.autoInstall: false` opts out). Jikji auto-installs the `jikji-cli` crate from crates.io via cargo by default (`jikji.autoInstall: false` opts out; requires the Rust toolchain). New `autorag init` configs enable Jikji by default (`jikji: {}`). The KakaoTalk `lazykatok` and Discord `discrawl` CLIs remain manual, optional installs (`brew install openclaw/tap/discrawl`). All three degrade gracefully when missing.
- **Datasource skills** — `AutoRAGAgent` can register `datasourceSkills`; their retrieval methods are merged with the normal retrieval pipeline, filtered before merging by trusted datasource access, and indexed during `refresh()`.

## Usage

```typescript
import { AutoRAGAgent } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
});
const response = await agent.searchDocuments("summarize the Q3 financial report");
console.log(response.answer);
agent.recordFeedbackByNumbers(response.sessionId, [1, 3], [2]);
```

`searchDocuments()` drives the Pi agent loop and returns a typed `SearchDocumentsResponse`; the caller consumes the structured payload directly, without parsing assistant text.

## Output Contract

**Caller sees curated, numbered knowledge units:**
```
[1] Revenue Summary — Q3 revenue grew 23% YoY to $4.2M, driven by enterprise contracts. (pages 3-5)
[2] Risk Factors — Three new risk factors added: supply chain, regulatory, talent retention. (pages 12-14)
```

Each result maps to an internal entry carrying its `source` (a real file path or datasource id), `method`, and evidence for feedback tracking. The curated `answer`/`results` are grounded in the sources; source paths may appear where relevant.

## Memory System (Self-Evolving)

AutoRAG remembers past search outcomes across sessions:
- Tracks which queries + methods succeeded or failed
- Prioritizes methods that historically work for similar queries
- `check_memory` tool lets the LLM query this history before searching
- Feedback loop: callers mark results as useful/not-useful → improves future searches

## Feedback Flow

1. Caller references results by session ID + number (e.g., session "abc", [1,3] useful)
2. Agent resolves numbers → session registry (populated from `emit_autorag_results` details) → sources
3. Sources → memory entries updated (useful/not_useful)
4. Memory informs future search strategy

## Files

| File | Role |
|------|------|
| `src/agent/agent.ts` | AutoRAGAgent class — the customized Pi agent and library API |
| `src/agent/bash-tool.ts` | Direct filesystem discovery and document-reading tool |
| `src/agent/fast-answer-tool.ts` | `emit_fast_answer` non-terminating tool for the fast-phase first answer |
| `src/agent/emit-results-tool.ts` | `emit_autorag_results` terminating tool that returns curated results as typed details |
| `src/agent/jikji-find-tool.ts` | `jikji_find` local-discovery tool |
| `src/agent/search-all-tool.ts` | `search_all_documents` multi-method fan-out |
| `src/agent/search-minsync-tool.ts` | `semantic_search_local_docs` MinSync vector tool |
| `src/agent/web-search-tool.ts` | `web_search` internet search tool over the `src/web/search` provider chain |
| `src/agent/web-fetch-tool.ts` | `web_fetch` URL reader over the `src/web/fetch` render pipeline |
| `src/web/search/` | oh-my-pi-ported web search: provider chain, structured query parsing, keyed + credential-free providers |
| `src/web/fetch/` | oh-my-pi-ported URL render pipeline: page loader, HTML→markdown reader chain, feeds, content negotiation |
| `src/agent/dupey-tool.ts` | `scan_duplicate_documents` read-only dupey scan |
| `src/agent/peer-target-tool.ts` | `recommend_peer_targets` local SimpleX persona ranking |
| `src/agent/system-prompt.ts` | System prompt builder for the librarian agent |
| `src/memory/memory.ts` | Feedback persistence and method priority scoring |
| `src/memory/renderer.ts` | Memory context renderer for system prompt |
| `src/memory/check-memory-tool.ts` | check_memory tool (pi-agent-core AgentTool) |
| `src/manifest/loader.ts` | YAML/JSON manifest loader for indexed data stores |
| `src/retrieval/types.ts` | Core retrieval type definitions |
| `src/retrieval/registry.ts` | Method registry for multi-method orchestration |
| `src/retrieval/merger.ts` | Cross-method result merging and deduplication |
| `src/minsync/method.ts` | MinSync retrieval method (vector / BM25 / hybrid over shared CDC chunks) |
| `src/datasource/` | Datasource skill contracts, trusted access context, result filtering, polling metadata, diagnostics, and KakaoTalk/lazykatok skill implementation |
| `src/p2p/` | SimpleX P2P sharing: policy, injection/PII gates, approval store, wire protocol |
| `src/cli/commands/serve.ts` | `autorag serve` P2P peer query server |
| `src/cli/commands/p2p.ts` | `autorag p2p` peer trust and request approval |
| `src/cli/commands/p2p-policy.ts` | `autorag p2p policy` sharing-rule CLI |
| `src/datasource/connector.ts` | Connector contract + opaque-text/id sanitizers for connector-backed skills |
| `src/datasource/chunk-store.ts` | Persistent chunk store with BM25-style lexical search per skill instance |
| `src/datasource/connector-skill.ts` | Shared DatasourceSkill base composing a connector with the chunk store |
| `src/datasource/skills/` | Built-in skills: lazykatok, discrawl, wacrawl, telecrawl, slack, clawgallery, notion, github, cloud-drive, mail-export, mailcrawl, obsidian, rss, spotlight (+ config factory) |
| `src/agent/search-datasource-tool.ts` | `search_datasource_documents` tool with model-safe `{ query, topK?, scope? }` parameters |
| `src/cli/commands/ui.ts` | `autorag ui` loopback dashboard for connecting and managing datasource skills |
| `src/ui/` | Local datasource UI catalog, config store, probes, HTML, and 127.0.0.1 HTTP server |
