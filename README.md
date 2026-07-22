# AutoRAG

**A self-evolving librarian agent for document collections.**

> [!IMPORTANT]
> **Looking for the original AutoRAG (RAG AutoML / pipeline optimization tool)?**
> This repository now hosts **AutoRAG 2.0**, a complete reimagining of AutoRAG as a self-evolving librarian agent. The original Python-based AutoRAG — the RAG AutoML tool for automatically finding an optimal RAG pipeline for your data — now lives in the [`legacy/`](legacy/) directory of this repository.
>
> **The legacy AutoRAG is NOT abandoned.** It continues to be maintained (bug fixes, dependency updates, and PyPI releases via `pip install AutoRAG`) in maintenance mode. Existing users can keep using it exactly as before — see the [legacy README](legacy/README.md) for its documentation, and file issues in this repository as usual. New feature development is focused on AutoRAG 2.0.

AutoRAG searches your PDFs, wikis, notes, research papers, and knowledge bases — then curates the results into clean, numbered knowledge units. No raw grep dumps. Just answers.

AutoRAG is a customized [Pi](https://github.com/earendil-works/pi-mono) agent — the Pi agent loop configured into a librarian. Searches use a two-tier workflow: a parent orchestrator delegates exploration to explorer agents. The roles and providers are independently configured from the models available in the user's authenticated runtime; AutoRAG does not ship a private provider default.

## Why AutoRAG

### The problem with search tools

Every search tool gives you the same thing: a list of file paths and matching lines. Then *you* have to:
- Open each file
- Read the surrounding context
- Decide what's relevant
- Synthesize an answer
- Remember what worked for next time

That's the human doing all the hard work. The tool just points.

### AutoRAG does the hard work

AutoRAG is not a search tool. It's a **librarian** — it searches, reads, thinks, and reports back:

```
You ask:  "What were the key findings in the Q3 report?"

AutoRAG:
[1] Revenue grew 23% YoY to $4.2M, driven by enterprise contracts. (pages 3-5)
[2] Three new risk factors: supply chain, regulatory, talent retention. (pages 12-14)
[3] Headcount target missed by 12 — engineering hiring bottleneck. (page 8)
```

No file paths. No line numbers. Just curated knowledge you can act on.

### It gets smarter over time

AutoRAG has a **self-evolving memory system**. Every search teaches it something:

- Which retrieval methods work for which types of queries
- Which document areas are most productive
- What the caller found useful (via explicit feedback)

A fresh AutoRAG tries everything. A seasoned one knows exactly where to look. This is not a static configuration — it's learned behavior from real usage.

### Multiple retrieval methods, one interface

Different documents need different search strategies:

| Your documents | Best method | Why |
|---|---|---|
| Plain text, config files | grep (pattern matching) | Fast, precise, literal |
| Research papers, dense prose | Vector search (semantic) | Understands meaning, not just keywords |
| Legal documents, specifications | BM25 (keyword ranking) | Handles domain terminology well |
| Mixed collections | Hybrid (vector + BM25) | Combines precision and recall |

AutoRAG supports **pluggable retrieval methods**. It ships with lexical **BM25** and semantic **MinSync** methods wired through the `RetrievalMethodRegistry`, and the architecture is ready for additional vector and hybrid backends. The parent orchestrator owns process-bound retrieval tools and gives bounded seed packs to explorers; explorers use read-only `read`/`grep`/`find`/`ls` tools to inspect the underlying documents. The `ResultMerger` handles cross-method score normalization and deduplication — you get one unified result set regardless of how many methods contributed.

BM25 and MinSync are **enabled by default** — no explicit configuration is needed for standard lexical + semantic retrieval. Both can be disabled by setting `"bm25": false` or `"minSync": false` in the config file. MinSync uses a pre-installed binary (`autoInstall: false`); configure `minSync.embedder` via `autorag init --embedder-*` flags for remote embedding endpoints. AutoRAG never forces TEI or any external embedding service.

### Real directory access

AutoRAG reads configured source directories through delegated explorer tasks. Each explorer is assigned exactly one normalized configured search root as its `cwd`; the top-level `subagent` invocation sets `agentScope: "user"` and `artifacts: false` exactly once for single, `tasks`, `chain`, or `parallel` dispatch, and nested explorer task items omit both fields. Project-local `.pi-subagents` debug artifacts are disabled. Explorers use read-only `read`/`grep`/`find`/`ls`; the parent orchestrator owns retrieval seed tools and must delegate document reading before curating. Curated answers are returned as a structured `SearchDocumentsResponse`; results carry their real source (file path or datasource id) in the internal mapping for feedback and curation. BM25 and MinSync index parsed markdown mirrors under `.autorag`.

Each explorer `task` contains an Assignment V1 block — a sentinel-wrapped JSON body with `originalQuery` (the caller query verbatim), `method` (the selected retrieval path), and `queryVariants` (a nonempty array), followed by canonical role lines requiring `retrievedAt` and temporal metadata:

```text
<<<AUTORAG_ASSIGNMENT_V1>>>
{"originalQuery":"<caller query verbatim>","method":"<selected retrieval method>","queryVariants":["<variant 1>","<variant 2>"]}
<<<END_AUTORAG_ASSIGNMENT_V1>>>
Required handoff: include retrievedAt.
Required handoff: include temporal metadata.
```

A legacy labeled format (`Original query:`, `Selected retrieval method:`, `Query variants:`) is accepted for compatibility. Missing or null top-level `artifacts`, `agentScope`, and leaf `model` fields are safely autofilled before validation; explicit wrong values (`artifacts: true`, `agentScope: "project"`, a non-configured model) remain rejected. There is no single-agent fallback. See [docs/subagent-orchestration.md](docs/subagent-orchestration.md) for the full dispatch contract, templates, anti-examples, and stable coded dispatch errors.

### Optional Jikji discovery and indexing

AutoRAG can opt into [Jikji](https://github.com/NomaDamas/jikji) as a local CLI-backed **find-first discovery and indexing** layer. Jikji is optional: AutoRAG does not vendor it, install it, or register it as a retrieval backend when enabled.

When Jikji is configured, AutoRAG calls `jikji find ROOT "query" --json` via a policy-aware `jikji_find` tool as the first local-discovery action. The tool parses and validates the upstream answer-pack and honors its `handoff_action` (`direct_use` / `jikji_retry` / `raw_fallback_after_retry`), `tool_call_policy` (`stop_after_find`, `forbidden_tools`, `allowed_followups`), and `agent_should_not_rerank`. Explorer `read`/`grep`/`find`/`ls` discovery is the fallback only when the answer-pack permits raw fallback (`raw_fallback_after_retry`, after the required retry) or when Jikji is unavailable/unconfigured. `prepare`/`refresh` remain for indexing only and do not answer queries directly.

Programmatic use:

```typescript
const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
  jikji: { binaryPath: "jikji" },
});
await agent.prepareJikji();
```

The same `.autorag/jikji.json` shape configures Jikji when present:

```json
{
  "enabled": true,
  "binaryPath": "jikji",
  "timeoutMs": 10000,
  "maxBufferBytes": 1048576,
  "includeHidden": false,
  "includeSensitive": false,
  "maxFiles": 0,
  "writeAgentRules": false,
  "enableMediaIndex": false,
  "exclude": []
}
```

Call `agent.prepareJikji()` (or `agent.refresh()`) to prepare configured source roots. Hidden files, sensitive files, and media indexing are disabled by default; AutoRAG does not pass `--include-hidden`, `--include-sensitive`, or `--enable-media-index` unless the corresponding option is `true`. AutoRAG-managed prepare runs with `--no-agent-rules` by default, so it never rewrites the consumer repo's `AGENTS.md`/`CLAUDE.md`/`.cursorrules`; an explicit `writeAgentRules: true` opt-in re-enables upstream routing-block injection. AutoRAG passes `--enable-media-index` only when `enableMediaIndex: true`.

The upstream Rust `PrepareArgs` defines reference defaults that AutoRAG does not override unless explicitly configured: parse timeout `5.0`, max hash bytes `512 MiB`, doc text max chars `2,000,000`, doc text chunk chars `1,000,000`, and media index max MB `25.0`. AutoRAG emits `--parse-timeout`, `--max-hash-bytes`, `--doc-text-max-chars`, `--doc-text-chunk-chars`, and `--media-index-max-mb` only when the matching option is set, so the upstream defaults apply otherwise. AutoRAG answers queries through `jikji find` (find-first) plus the Pi agent loop and its registered retrieval methods; `prepare`/`refresh` are indexing-only.

### Datasource skills

Datasource skills let AutoRAG search external, server-configured data sources through the same retrieval pipeline as local documents. A skill describes what it indexes, how it should be refreshed, what source instances exist, and which permission tags/scopes bound access. Retrieval still flows through `RetrievalMethodRegistry` → `ParallelRetriever` → datasource result filtering → `ResultMerger`; datasource skills do not create a parallel search path.

Security defaults are intentionally strict:

- datasource access is default-deny unless trusted server/API configuration supplies `datasourceAccess.allowedTags` and `datasourceAccess.allowedScopes`;
- model/tool arguments never grant datasource tags or scopes;
- `search_datasource_documents` accepts only `{ query, topK?, scope? }`, and `scope` can only narrow trusted access.

#### Supported datasources

| Datasource | Skill | Connects via | Notes |
|---|---|---|---|
| KakaoTalk | `katok` | external [`katok`](https://github.com/NomaDamas/katok) CLI | first datasource skill; AutoRAG never reads KakaoTalk databases directly |
| Slack | `slack` | Slack Web API (bot token) | workspace/channel history; per-channel scope failures degrade to warnings |
| Discord | `discord` | Discord REST v10 (bot token) | guild/channel messages; Hangul channel names work as scopes |
| Notion | `notion` | Notion API (integration token) | pages/databases shared with the integration; block-tree text |
| GitHub Issues/PRs | `github` | GitHub REST (token optional) | issues + PR bodies per `owner/repo`; public repos work unauthenticated |
| Google Drive | `gdrive` | Drive REST v3, or **[`rclone`](https://rclone.org) CLI** (`backend: "rclone"`) | Docs/Sheets exported as text; the rclone backend also opens any of rclone's 70+ remotes |
| Gmail / IMAP | `gmail` | Gmail REST v1, or **[`himalaya`](https://pimalaya.org) CLI** (`backend: "himalaya"`) | the himalaya backend indexes any IMAP/Maildir account it has configured — no OAuth plumbing |
| Local mail exports | `mail-export` | filesystem (`.mbox` / `.eml`) | classic `From_` splitting, mailparser-based; count-only warnings |
| Obsidian vault | `obsidian` | filesystem (markdown) | frontmatter/inline tags, wiki links `[[...]]`, embeds `![[...]]` |
| RSS / news | `rss` | HTTP feed polling | RSS 2.0 + Atom, feed/category hierarchy, 24h dedupe window |

All of them share one framework: a trusted connector fetches documents at refresh time, chunks persist under `<workspace>/.autorag/datasources/<skill>/<instance>/`, and queries run against a local BM25 lexical index (Korean-aware prefix matching) through `search_datasource_documents`. Tokens are referenced by environment variable name only (e.g. `SLACK_BOT_TOKEN`), never stored in config. Auth/permission/rate-limit failures surface as path/PII-opaque diagnostics. See [docs/manual-qa-datasources.md](docs/manual-qa-datasources.md) for the QA harnesses.

Configure them in `config.json` (CLI) or pass `datasourceSkills` programmatically:

```jsonc
{
  "datasources": {
    "slack":    { "connector": { "tokenEnv": "SLACK_BOT_TOKEN" } },
    "github":   { "connector": { "repos": ["owner/repo"] } },
    "gmail":    { "connector": { "backend": "himalaya", "account": "gmail", "folder": "INBOX" } },
    "gdrive":   { "connector": { "backend": "rclone", "remote": "gdrive:" } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } },
    "rss":      { "connector": { "feeds": [{ "url": "https://example.com/feed.xml" }] } }
  },
  "datasourceAccess": {
    "allowedTags": ["slack", "github", "gmail", "gdrive", "obsidian", "rss"],
    "allowedScopes": ["/slack/**", "/github/**", "/gmail/**", "/gdrive/**", "/obsidian/**", "/rss/**"]
  }
}
```

#### KakaoTalk (katok)

KakaoTalk was the first datasource skill. It uses the external [`katok`](https://github.com/NomaDamas/katok) CLI only — AutoRAG never reads KakaoTalk databases directly. `katok` failures return diagnostics instead of throwing, and remote embedding egress configuration is rejected before the CLI is spawned.

```typescript
import { AutoRAGAgent, KatokSkill } from "@autorag/librarian";

const kakao = new KatokSkill({
  instanceId: "personal",
  tags: ["kakaotalk", "personal", "pii"],
  // Optional: client: new KatokClient({ binaryPath: "katok" })
});

const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
  datasourceSkills: [kakao],
  datasourceAccess: {
    allowedTags: ["kakaotalk"],
    allowedScopes: ["/kakao/personal/**"],
  },
});

await agent.refresh(); // refreshes parsed mirrors, BM25/MinSync, and datasource indexes
const results = await agent.searchDatasourceDocuments("meeting with Mina", { topK: 5 });
```

A datasource skill should provide polling/cron metadata for routine indexing, source descriptions for the agent prompt, slash-hierarchical opaque source paths such as `/kakao/personal/chunks/<chunk-id>`, and permission tags that match your server-side access policy.

### Primary target: document collections

AutoRAG is built for **non-code document retrieval**: manuals, legal docs, internal wikis, meeting notes, research literature, knowledge bases, PDFs.

Code repositories work too (the explorer's `grep` is useful), but AutoRAG's real value shows on unstructured text where simple pattern matching isn't enough.

## Configuration and state

The default home state is kept outside the workspace:

```text
~/.autorag/
├── config.json
├── memory.json
├── logs/
│   └── runs.jsonl
└── pi-agent/
    ├── auth.json
    ├── models.json
    ├── settings.json
    └── sessions/
```

`config.json` selects sources, the workspace, memory path, retrieval settings, and the two agent models. Configure the roles independently with `agents.orchestrator` and `agents.explorer`. Provider and model IDs must refer to models available in the user's authenticated runtime:

```json
{
  "searchPaths": ["/path/to/documents"],
  "workspacePath": "/path/to/workspace",
  "memoryPath": "/Users/you/.autorag/memory.json",
  "agents": {
    "orchestrator": { "provider": "provider-name", "id": "reasoning-model" },
    "explorer": { "provider": "provider-name", "id": "exploration-model" }
  }
}
```

`autorag init` leaves `agents` unset when no role-model flags are supplied. At search time AutoRAG resolves an authenticated local provider when possible; otherwise configure both roles explicitly.

Config path precedence is `--config` > `AUTORAG_CONFIG` > `~/.autorag/config.json`. When the home config is absent and `<cwd>/autorag.config.json` exists, AutoRAG copies the legacy file to `~/.autorag/config.json` without deleting or modifying the legacy file. The legacy cwd file is a migration source, not the default location.

`memory.json` stores retrieval memory, `logs/runs.jsonl` records run events, and durable Pi models, settings, and sessions stay under `~/.autorag/pi-agent`. Corpus indexes remain workspace-local: refresh keeps parsed mirrors and BM25/MinSync indexes under `<workspace>/.autorag`.

`autorag refresh` and `autorag index reset|rebuild` accept `--method <csv>` (e.g. `--method bm25,minsync,parsed`) to scope which indexing methods run or which index directories are removed. When omitted, all methods run. `autorag init` accepts `--embedder-*` flags to configure the MinSync embedder endpoint in the config file.

`autorag health` checks model/provider auth and explorer subagent setup before a search — it resolves both role models, verifies credential presence, and optionally probes a completion call per role. Use it to diagnose model, provider, auth, timeout, or subagent dispatch failures. `autorag status` remains the model-free index-health command (corpus freshness and BM25/MinSync readiness); it does not check models or subagent dispatch. When `autorag search` fails for a model/provider/subagent reason, the error output includes a hint pointing to `autorag health`.

## Installation

Published as `@autorag/librarian` (dist bundled with Bun, runtime Node ≥ 24 or Bun):

```bash
bun add @autorag/librarian          # library
bun install -g @autorag/librarian   # autorag CLI
# or run directly from the repo:
bun add github:NomaDamas/AutoRAG-2.0
```

Git-based installs build `dist/` via the `prepare` script and require Bun on the installing machine.
External tool binaries auto-install on first use into `<workspace>/.autorag/bin`: **MinSync** downloads a verified GitHub release asset (on by default; `minSync.autoInstall: false` to opt out), and **Jikji** compiles the [`jikji-cli`](https://crates.io/crates/jikji-cli) crate via cargo (requires the [Rust toolchain](https://rustup.rs); `jikji.autoInstall: false` to opt out). New `autorag init` configs enable Jikji find-first discovery by default. KakaoTalk (`katok`) stays a manual, optional install. All of them degrade gracefully when missing — core BM25 search works without any of them.

## Quick Start

```typescript
import { AutoRAGAgent } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
});

const response = await agent.searchDocuments("summarize the compliance requirements");
console.log(response.answer);
for (const result of response.results) {
  console.log(`[${result.number}] ${result.title} — ${result.summary}`);
}

// Mark which results were useful — AutoRAG remembers for next time
agent.recordFeedbackByNumbers(response.sessionId, [1, 3], [2]);
```

`searchDocuments()` runs the Pi agent loop — it searches, reads, consults memory, curates, and finalizes through the `emit_autorag_results` structured tool — then returns a typed `SearchDocumentsResponse`. The caller consumes the structured payload directly; no assistant text parsing.

## How It Works

```
    You ask a question
           │
           ▼
    ┌──────────────┐
    │ Plan + seed  │ ← parent check_memory, Jikji, and retrieval tools
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Delegate     │ ← pi-subagents → gpt-5.6-luna explorer
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Search + read│ ← explorer read/grep/find/ls
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Curate     │ ← Extract key insights, not raw lines
    └──────┬──────┘
           ▼
    [1] Finding A — summary (location)
    [2] Finding B — summary (location)

           │
    Caller says "1 was useful, 2 wasn't"
           │
           ▼
    ┌─────────────┐
    │   Memory     │ ← Remembers what worked, adapts next time
    └─────────────┘
```

## License

MIT
