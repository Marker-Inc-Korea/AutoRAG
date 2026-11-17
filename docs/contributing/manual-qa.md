---
title: "Manual QA and live E2E"
description: "Run AutoRAG's manual QA and live end-to-end checks in an isolated home, and verify local MinSync retrieval against the fixture corpus."
---

# Manual QA and live E2E

This page is the procedure for exercising AutoRAG against real stores, real embedders, and real datasource CLIs without touching the maintainer's machine state. It covers the isolation boundary, the clone-local live-E2E runner, the required MinSync live check, the per-datasource harnesses, and the KakaoTalk checklist.

Run the live lanes when a change touches retrieval, MinSync or embedding configuration, datasource lanes, the agent tool surface, or the output contract. For documentation, formatting, and unit-test-only changes, `make ci` is enough.

## Isolation boundary

Every manual QA, live check, and ad-hoc CLI or `AutoRAGAgent` run must keep the host's real state untouched. All clones read one global AutoRAG home, `~/.autorag` (`config.json`, `memory.json`, TUI sessions), and MinSync's model cache is shared at `~/.cache/minsync/models`.

::: danger **Never touch the real AutoRAG home.**
Do not write, move, delete, or `--force` anything under `~/.autorag`, and do not "fix" or clean it up. Reading is allowed only when the task needs it; if you think it is wrong, report it.
:::

For host runs, isolate before the first command. Create one temp root and point both AutoRAG variables at it, in the same shell that runs every AutoRAG command:

```bash
QA_ROOT="$(mktemp -d)"
export AUTORAG_HOME="$QA_ROOT/.autorag-home"
export AUTORAG_CONFIG="$AUTORAG_HOME/config.json"
mkdir -p "$QA_ROOT/docs" "$AUTORAG_HOME"
```

A tool that starts a fresh shell per call (an agent `bash` tool, `spawn`) does not inherit a previous call's `export`; repeat the exports in each call, or pass them through `env`. `--workspace` moves the workspace only, not the config path: the config comes from `--config`, then `AUTORAG_CONFIG`, then `$AUTORAG_HOME/config.json`, then `~/.autorag/config.json`. Right before the first command that writes config, confirm `echo "$AUTORAG_CONFIG"` prints a path under `$QA_ROOT`.

Prove it and clean up: hash the real config before and after (`shasum ~/.autorag/config.json`); the two hashes must match. Remove only your own `$QA_ROOT` and the processes you started. Do not stop or reconfigure processes you did not start, including other clones' `minsync serve` daemons and dev servers.

`autorag init --force` also refuses to replace an existing implicit home config; pass `--config <path>` or set `AUTORAG_CONFIG` when replacement is intentional. First-time implicit initialization remains allowed.

### The Docker QA shell

The supported manual-QA boundary is Docker. Only the repository is mounted; the host `HOME` and native datasource stores are not.

```bash
make qa-shell                  # interactive manual-QA shell
make e2e-live-docker E2E_ROOT="$PWD/scripts/live-e2e" E2E_DATASOURCES=local
```

The container sets an ephemeral `HOME`, `AUTORAG_HOME`, and `AUTORAG_CONFIG`, bind-mounts this clone at `/workspace`, and does not mount `~/.autorag`. Only the model credential names in `QA_MODEL_ENV` are forwarded; set it empty (`QA_MODEL_ENV=`) to forward none. `make qa-image` builds the image, which is based on Ubuntu 26.04 so current Linux MinSync release assets run against its glibc.

Native-store lanes are expected to skip inside the isolated container unless their stores are deliberately provisioned inside it; never mount a real host home to make a lane pass.

## The live-E2E runner

The live environment is clone-local: each clone owns its own `.autorag-e2e` state directory, so independent clones can run concurrently. Do not point two clones at the same `E2E_ROOT`.

Prerequisites:

- Node.js 24+, Bun, and repository dependencies (`bun install --frozen-lockfile`).
- A bootstrapped corpus root. Bootstrap is explicit; live targets never create one implicitly.
- MinSync 0.7.1 or newer on `PATH` (the floor is `MINSYNC_MIN_VERSION` in `src/minsync/version.ts`; the pinned install is 0.7.3). The runner starts a loopback `minsync serve` daemon for the run, checks its `/health` endpoint, and stops it afterward. The Docker QA image ships the pinned, sha256-verified `minsync`, so `make e2e-live*` needs nothing on the host. MinSync publishes a Linux x86_64 build only, so that image is `linux/amd64` (the default `QA_PLATFORM`); on an Apple Silicon host it runs under emulation, and building it for arm64 fails for that reason. Keep `MINSYNC_VERSION`/`MINSYNC_SHA256` in `scripts/ci/qa.Dockerfile` in step with `MINSYNC_VERSION` in `src/minsync/installer.ts`.
- `OPENAI_API_KEY` and `AUTORAG_OPENAI_API_KEY` unset. Embeddings stay local during live E2E; do not configure a remote `minSync.embedder` endpoint or send corpus text off the machine.

### Bootstrap

```bash
export AUTORAG_LIVE_E2E_ROOT="$PWD/scripts/live-e2e"
node scripts/live-e2e/runner.mjs bootstrap --root "$AUTORAG_LIVE_E2E_ROOT"
```

Bootstrap copies the digest-pinned fixture corpus and manifest into the root. `E2E_ROOT` defaults to the current repository path and honors `AUTORAG_LIVE_E2E_ROOT`; the Make targets never create or bootstrap the root for you.

### Cold and warm runs

```bash
make e2e-live-cold E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"   # delete .autorag-e2e, rebuild, run core checks
make e2e-live      E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"   # warm run, reusing clone-local state
```

A warm run compares the persisted fingerprint (runner schema, corpus version and digest, git commit and dirtiness, embedding model and dimension, embedding service, parser config, MinSync config) and refuses stale state with `live-e2e-fingerprint-mismatch`; rerun with cold mode to rebuild. The core run verifies local-file and MinSync behavior: an incremental refresh via the shared embedder, a `.minsync/cursor.json` cursor, a Korean stem that only MinSync's sparse encoder can reach, a semantic hit mapped to its OS-absolute source, and the `autorag lite retrieve --json` machine envelope (including time-window and degraded-surface assertions).

### Datasource lanes

With no override, the runner executes the `local` lane, the self-contained `obsidian` vault lane, and every native CLI lane (lazykatok, discrawl, wacrawl, telecrawl, slacrawl, lark, notcrawl, mailcrawl, wombat, hister). `E2E_DATASOURCES` only narrows this default:

```bash
E2E_DATASOURCES=local make e2e-live-docker E2E_ROOT="$AUTORAG_LIVE_E2E_ROOT"
```

The summary separates the core MinSync result (`commandsSummary.core`) from `datasourceLanes`. A native lane whose CLI or native store is missing is `SKIP` with a reason; an installed and configured lane whose native check fails is `FAIL`. `SKIP` is never reported as `PASS`. On a host where a native store genuinely exists, its lane must run and pass; narrowing it away with `E2E_DATASOURCES` is a QA gap, not a green run. The runner does not copy datasource data or force an AutoRAG workspace: native stores, profiles, and keychains remain owned by their CLIs.

## Required MinSync live QA

When validating local-file retrieval changes, run a real semantic query through the shared embedder. The local MinSync backend is the default; a remote embedding service is used only when the operator explicitly configures `minSync.embedder`.

The default path uses MinSync's Qwen3-Embedding-0.6B engine (1024 dimensions; queries carry Qwen3's `Instruct: …\nQuery:` header, passages are raw). Vectors come from a loopback `minsync serve` daemon when one answers, otherwise from MinSync's in-process engine. AutoRAG never starts or stops the daemon; `autorag setup` and `autorag status` only report the resolved `embedding.backend` (`daemon`, `in-process`, or `remote`) with its provider, model, and dimension, or why none is alive.

Run the isolated experiment. Both AutoRAG variables point inside the temp workspace, so `init --force` cannot reach the shared `~/.autorag`:

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
autorag init --search-paths "$WORKSPACE/docs" --force
autorag refresh --method parsed,minsync --json
autorag search --json "semantic question about refund approval"
```

The QA gate is not complete until all of the following are observed:

1. `autorag refresh --method minsync` exits successfully and the workspace's MinSync cursor (`<workspace>/.autorag/minsync/.minsync/cursor.json`) exists.
2. The semantic query returns a hit for the fixture document.
3. AutoRAG maps that hit to an OS-absolute original `source` path.
4. `fs.existsSync(source)` and reading `source` both succeed.
5. The run used a local backend (`daemon` or `in-process`), with no remote `minSync.embedder` endpoint configured.

If MinSync is missing, no daemon answers, the in-process engine is absent or fails (`embedder-unavailable`), or MinSync reports a semantic failure, report the exact blocking diagnostic and do not claim live MinSync verification.

### Legacy Ollama variant

For an existing workspace pinned to Ollama's EmbeddingGemma (768 dimensions), keep the adapter available as a manually started sidecar. Do not use this variant as a fresh-install requirement or as an implicit fallback from the MinSync embedder.

```bash
ollama pull embeddinggemma:latest
ollama serve
OLLAMA_EMBEDDINGS_URL=http://127.0.0.1:11434/api/embeddings \
  python3 scripts/manual-qa/ollama-tei-adapter.py
```

Then initialize explicitly with the TEI endpoint, in the same shell that exported `AUTORAG_HOME` and `AUTORAG_CONFIG`:

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

## Per-datasource harnesses

Each harness uses its CLI's native launch surface; AutoRAG supplies only explicitly configured paths and bounded process execution. It never materializes or forces an AutoRAG-managed archive for a datasource CLI.

| Harness | Coverage |
|---|---|
| `bun scripts/manual-qa/run-qa.ts` | Protocol-accurate local mocks of GitHub APIs, a real Obsidian-vault filesystem fixture, and a local RSS feed |
| `bun scripts/manual-qa/run-qa-live.ts` | The public GitHub REST API (this repo's issues) and a real RSS feed (hnrss.org), credential-free |
| `bun scripts/manual-qa/run-qa-obsidian-live.ts` | A real local vault through MinSync sparse + semantic + folder scope + incremental re-index |
| `bun scripts/manual-qa/run-qa-discrawl-live.ts` | A real Discord archive through the `discrawl` CLI (FTS + semantic + hybrid, incremental re-sync) |
| `bun scripts/manual-qa/run-qa-lazykatok-live.ts` | A real KakaoTalk archive through the `lazykatok` CLI (source-native `/kakao/<instance>/chunks/<chunk>` identity, keyword/BM25 retrieval) |
| `bun scripts/manual-qa/run-qa-mailcrawl.ts` | The deterministic mailcrawl process boundary, missing-binary diagnostics, and the agent datasource loop |
| `bun scripts/manual-qa/run-qa-mailcrawl-live.ts` | A real mailcrawl fixture and `--source files` (`.eml`/`.mbox`) sync plus BM25/semantic/hybrid retrieval |
| `bun scripts/manual-qa/run-qa-github-gist-live.ts` | A real GitHub REST API with the `gh` token: incremental gist index, no-op re-index, lexical and semantic search |
| `bun scripts/manual-qa/run-qa-lark-live.ts` | A real `lark-cli` on a logged-in tenant; prints `/lark/...` identities and refuses to pass with no hits |
| `bun scripts/manual-qa/run-qa-clawgallery-live.ts /path/to/images "query"` | The real ClawGallery CLI plus a local image folder (incremental bootstrap + hybrid search) |
| `bun scripts/manual-qa/run-qa-wombat-live.ts` | Real `wombat` + `minsync` in an isolated home: sparse, semantic, and hybrid search, session labels, incremental append, delete, and mirror isolation |
| `WOMBAT_QA_DATA_DIR=<dir> bun scripts/manual-qa/run-qa-wombat-real-sessions.ts [--jev]` | The operator's real coding-agent sessions, indexed into an isolated data dir and searched through the agent |
| `bun scripts/manual-qa/run-qa-hister-live.ts` | The Hister browser-history datasource against a real Hister install |
| `bun scripts/manual-qa/run-qa-datasource-aliases.ts` | Universal alias registration plus all-channel and channel-allowlisted chat retrieval |
| `bun scripts/manual-qa/run-qa-cloud-mount-live.ts`, `run-qa-network-mount-live.ts` | Opt-in cloud-mount and network-mount lanes (macOS mounts a throwaway folder; Windows uses a synthetic provider) |
| `bunx vitest run test/datasource/embedding-env.test.ts` | The real child-process boundary for mailcrawl and lazykatok on the shared embedder: provider/URL/model/dimension env, remote key hand-over and egress opt-in |
| `bunx vitest run test/datasource/skills/wacrawl.test.ts` (also `telecrawl`, `slacrawl`, `notcrawl`, `lark`, `obsidian`, `mailcrawl`, `github-gist`, `hister`, `wombat`) | Real child-process boundaries with deterministic fake executables: argv, JSON parsing, env isolation, missing binary, malformed output, indexing, retrieval |

WhatsApp, Telegram, Slack, and Notion are verified against deterministic fake crawler executables that exercise the actual spawn/stdout/stderr contract without private data. For a live check, install the crawler, grant its documented access, configure `datasources.<name>`, then run `autorag refresh --method datasources`. Set `datasources.slack.connector.configPath` / `datasources.notion.connector.configPath` for the slacrawl and notcrawl configs.

## Datasource checklist

Every harness above automates these checks. Use the list when adding a datasource by hand.

### Setup and wiring

- `buildDatasourceSkills` materializes every configured skill from the trusted `datasources` config section; unknown names are skipped at `buildAgentOptions` with an `unknown-datasource-skill` diagnostic so unrelated commands still run.
- Skills register their retrieval methods through the existing `RetrievalMethodRegistry` pipeline on agent construction.

### Indexing

- `agent.refresh(true, { methods: ["datasources"] })` indexes all configured skills; each returns an ok result with a chunk count.
- Connector-backed skills persist their own state under the configured connector or CLI-native store; AutoRAG-managed chunk stores are used only by connector skills that own that storage.
- Polling metadata (`mode`, `intervalMs`, `lastIndexedAt`, `lastPolledAt`, `lastError`) tracks success and failure; RSS applies a dedupe window.

### Progressive disclosure and search

- Configured skills appear as `datasource-<name>` in the system prompt; unconfigured skills are omitted entirely.
- `load_datasource_skill` returns full path-opaque instructions for configured names and a not-available result for unknown names.
- Each configured connection's dedicated `search_datasource_<id>` tool returns hits with opaque slash-hierarchical sources (`/<skill>/<instance>/chunks/<id>`): no `#` fragments, no real filesystem paths. There is no datasource fan-out tool.
- `scope` narrows results for scope-capable datasources (for example `/github/**` excludes Slack hits) and can never widen visibility.

### Security

- Every configured or already-connected datasource is searchable without access setup; there is no tag or scope authorization layer.
- Skill and descriptor `tags` are descriptive metadata only, and tool arguments cannot change which connections exist.
- User scope narrows scope-capable datasource results before merge (`filterDatasourceScope`); `#` fragments are always rejected.

### Diagnostics

- Wrong tokens map to `datasource-auth-error`; permission problems to `datasource-permission-denied`; HTTP 429 to `datasource-rate-limited`; unreachable services to `datasource-unavailable`.
- Failure payloads and warnings never contain tokens, URLs, email addresses, or absolute filesystem paths (`sanitizeOpaqueText`).
- Per-item failures (a denied Slack channel, a 404 repo, one bad feed, an unparseable mbox message) degrade to warnings without failing the whole index run.

## KakaoTalk and lazykatok manual checks

This checklist validates the KakaoTalk datasource skill and its connected-datasource behavior.

Preconditions:

- `lazykatok` is installed and on `PATH`, or `LazykatokClient({ binaryPath })` points to it.
- KakaoTalk access is already granted to `lazykatok` outside AutoRAG.
- AutoRAG is configured with a `LazykatokSkill`:

```ts
new AutoRAGAgent({
  searchPaths: ["/docs"],
  datasourceSkills: [new LazykatokSkill({ instanceId: "personal" })],
});
```

Required checks:

1. **Configured connection is searchable.** Run `agent.searchSingleDatasourceDocuments("kakao", "hello")`. The generated `search_datasource_kakao` tool returns lazykatok results, or a diagnostic when the archive is empty; no error is thrown.
2. **Refresh status.** Run `agent.refresh()`. `components.datasources` is `configured` or `degraded`.
3. **Datasource-native filtering.** lazykatok's own chat/channel filtering and chat identity metadata remain intact; AutoRAG does not apply a virtual source scope (lazykatok exposes no per-source scope filtering).
4. **Missing binary or permission failure.** Point `LazykatokClient` at a nonexistent binary, or run without required OS permissions. No throw; the failure surfaces as a warning or error diagnostic.
5. **Public response curation.** Run `searchDocuments()` and let the librarian curate a KakaoTalk-supported answer; the visible `answer` and `results` contain facts grounded in the datasource evidence.

CI and most development containers have no real KakaoTalk profile and no macOS app-container permissions. In those environments, perform checks 1 and 4 with the test/fake lazykatok client, and record the real-data checks as blocked by missing local KakaoTalk credentials rather than bypassing the safety requirements.

## Evidence and cleanup

Core live-E2E evidence is written to the runner result directory (by default `.omo/evidence/live-core-cold/result.json` or `live-core-warm/result.json`). Diagnostics and error text are reported verbatim, absolute paths included, so an operator can debug a failure from the output alone. Assert that local retrieval sources are absolute and readable, and that native datasource results keep source-native identities such as `/kakao/<instance>/chunks/<chunk>`.

Cleanup is limited to runner-owned state: the cold command removes and rebuilds `.autorag-e2e`; for manual cleanup run `rm -rf .autorag-e2e` from your own clone. Leave lazykatok, discrawl, crawler, and mailcrawl native stores untouched. If a behavioral change invalidates previously recorded cold/warm evidence, regenerate it in the same pull request; stale green evidence is not proof.

## Related pages

- [Development workflow](development.md) for the everyday commands and guardrails.
- [Architecture](architecture.md) for the state layout the harnesses inspect.
- [Supply chain and licenses](supply-chain.md) for the dependency gates.
