# AutoRAG

**A self-evolving librarian agent for document collections.**

> [!IMPORTANT]
> **Looking for the original AutoRAG (RAG AutoML / pipeline optimization tool)?**
> This repository now hosts **AutoRAG 2.0**, a complete reimagining of AutoRAG as a self-evolving librarian agent. The original Python-based AutoRAG — the RAG AutoML tool for automatically finding an optimal RAG pipeline for your data — now lives in the [`legacy/`](legacy/) directory of this repository.
>
> **The legacy AutoRAG is NOT abandoned.** It continues to be maintained (bug fixes, dependency updates, and PyPI releases via `pip install AutoRAG`) in maintenance mode. Existing users can keep using it exactly as before — see the [legacy README](legacy/README.md) for its documentation, and file issues in this repository as usual. New feature development is focused on AutoRAG 2.0.

AutoRAG searches your PDFs, wikis, notes, research papers, and knowledge bases — then curates the results into clean, numbered knowledge units. No raw grep dumps. Just answers.

AutoRAG is a customized [Pi](https://github.com/earendil-works/pi-mono) agent — the Pi agent loop configured into a librarian. The AutoRAG librarian retrieves candidates, reads source files directly, judges the evidence, and curates the structured answer. Its model and provider come from the user's authenticated runtime; AutoRAG does not ship a private provider default.

AutoRAG itself is the specialized search agent, not a coordinator for other
model roles. You configure one model, and that model owns the complete
retrieval, reading, judgment, and curation loop.

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

AutoRAG supports **pluggable retrieval methods**. Local lexical BM25, semantic vector, and hybrid retrieval all go through **MinSync** over one shared CDC chunk lifecycle, wired through the `RetrievalMethodRegistry`. The librarian invokes retrieval tools, reads the underlying documents directly through `bash`, and curates one unified result set after `ResultMerger` score normalization and deduplication. External datasources keep their own archive/index lifecycle.

BM25, vector, and hybrid are **enabled by default** whenever MinSync is enabled. Disable local indexing with `"minSync": false`, or disable only lexical search with `"bm25": false`. MinSync auto-installs a verified release into `<workspace>/.autorag/bin` on first use (`autoInstall: true`); set `"autoInstall": false` only when managing the binary yourself. Configure `minSync.embedder` via `autorag init --embedder-*` flags for remote embedding endpoints, and set `minSync.maxChunkSize` (or `--minsync-max-chunk-size`) when a local embedder has a smaller context window. AutoRAG never forces TEI or any external embedding service.

See [docs/minsync-setup.md](docs/minsync-setup.md) for automatic installation,
managed binary paths, and the local EmbeddingGemma QA flow.

### Real directory access

AutoRAG reads configured source directories directly through its built-in `bash` tool. Retrieval tools can supply candidate paths, but the same agent opens the source material before curating. Answers are returned as a structured `SearchDocumentsResponse`; results carry their real source (file path or datasource id) in the internal mapping for feedback and curation. MinSync indexes parsed markdown mirrors under `.autorag` for BM25, vector, and hybrid retrieval.

### Thin PDF extraction retry

The default PDF parser performs a cheap quality check for multi-page PDFs.
When local markdown is unusually sparse (fewer than 800 characters or fewer
than 40 characters per detected page, for at least three pages), it retries
through OpenDataLoader's `docling-fast` hybrid backend with `hybridMode:
"auto"` and a 30-second timeout. Dense PDFs are not retried, and hybrid is
never used for single-page PDFs or as the first path for images.

The gate is parser-owned and can be tuned through trusted programmatic
`parserOptions`:

```typescript
new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
  parserOptions: {
    thinExtract: {
      minPages: 3,
      minChars: 800,
      minCharsPerPage: 40,
      timeoutMs: 30_000,
      hybrid: "docling-fast",
      hybridMode: "auto",
    },
  },
});
```

If the hybrid sidecar is missing, times out, or fails, AutoRAG keeps the local
markdown and emits `pdf-extract-thin` plus `pdf-hybrid-unavailable`
diagnostics; refresh remains successful.

### Default Jikji discovery and indexing

AutoRAG uses [Jikji](https://github.com/NomaDamas/jikji) by default as a local CLI-backed **find-first discovery and indexing** layer. Jikji is not registered as a retrieval backend: it supplies bounded discovery answer packs while the librarian still reads original files directly.

The default agent calls `jikji find ROOT "query" --json` via the `jikji_find` tool. The tool parses and validates the upstream answer pack and exposes its `handoff_action`, `tool_call_policy`, and `agent_should_not_rerank` to the librarian. Direct file reading remains available for source verification. `prepare`/`refresh` remain for indexing only and do not answer queries directly. If the binary or Rust toolchain is unavailable, Jikji reports a diagnostic and the normal filesystem/retrieval paths continue.

New `autorag init` configs include `"jikji": {}`. To opt out, set `"jikji": false` in `config.json`; programmatic callers can pass `jikji: false`.

Programmatic use:

```typescript
const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
});
await agent.prepareJikji();
```

### Duplicate document management with dupey

AutoRAG can use the external [`dupey`](https://github.com/NomaDamas/dupey) CLI
to detect exact, near, and containment document families.

```bash
autorag duplicates /path/to/documents
autorag duplicates --json
```

The command is read-only: it reports exact duplicate groups and review
guidance, but never moves or deletes source files. The `scan_duplicate_documents`
Agent tool exposes the same read-only scan to the orchestrator.

Exact duplicate exclusion is enabled by default during parsed-mirror refresh.
For each exact canonical-text hash, the newest filesystem copy is indexed and
older copies are omitted. Disable it in `config.json` when both copies must be
searchable:

```json
{
  "dupey": { "enabled": true },
  "excludeExactDuplicates": false
}
```

If `dupey` is not installed or fails, refresh continues without exclusion and
reports no destructive action; install it with `cargo install dupey`.

The same configuration shape customizes Jikji when needed:

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

Every datasource can be registered multiple times through a connection alias:
use the config key as the unique name and set `type` to the reusable backend
(`gmail`, `github`, `slack`, `discord`, `kakao`, `cloud-drive`, and so on).
Each alias becomes an independently loadable agent skill with its own source
scope and workspace namespace. Chat aliases search all channels by default;
trusted `channels.ids` / `channels.names` allowlists can expose a particular
channel or group chat as its own datasource.

Security defaults are intentionally strict:

- datasource access is default-deny unless trusted server/API configuration supplies `datasourceAccess.allowedTags`; `allowedScopes` applies only to datasource methods that advertise the `scoped` capability;
- model/tool arguments never grant datasource tags or scopes;
- `search_datasource_documents` accepts only `{ query, topK?, scope? }`, and `scope` can only narrow trusted access.

#### Supported datasources

| Datasource | Skill | Connects via | Notes |
|---|---|---|---|
| KakaoTalk | `katok` | external [`katok`](https://github.com/NomaDamas/katok) CLI | first datasource skill; AutoRAG never reads KakaoTalk databases directly |
| WhatsApp | `whatsapp` | external [`wacrawl`](https://github.com/openclaw/wacrawl) CLI | local-first incremental archive + FTS5 search; live desktop ingestion is macOS-only |
| Telegram | `telegram` | external [`telecrawl`](https://github.com/openclaw/telecrawl) CLI | local-first archive + FTS5 search; live desktop ingestion is macOS-only |
| Slack | `slack` | external [`slacrawl`](https://github.com/openclaw/slacrawl) CLI | local-first workspace/channel/thread archive + FTS5 search |
| Discord | `discord` | external [`discrawl`](https://github.com/openclaw/discrawl) CLI | guild/channel/thread/DM archive; FTS5 + semantic + hybrid retrieval, incremental sync |
| Notion | `notion` | external [`notcrawl`](https://github.com/openclaw/notcrawl) CLI | local-first page/database/block archive + FTS5 search |
| GitHub Issues/PRs | `github` | GitHub REST (token optional) | issues + PR bodies per `owner/repo`; public repos work unauthenticated |
| Cloud drives | `cloud-drive` | **[`rclone`](https://rclone.org) CLI** | Incremental Google Drive Tier-1; OneDrive/network remotes; iCloud experimental |
| Gmail | `gmail` | Gmail REST v1 | OAuth access token is referenced by environment variable name |
| Local mail exports | `mail-export` | filesystem (`.mbox` / `.eml`) | classic `From_` splitting, mailparser-based; count-only warnings |
| Mail archives | `mailcrawl` | external [`mailcrawl`](https://github.com/NomaDamas/mailcrawl) CLI | local email sync plus BM25, semantic, and hybrid retrieval |
| Obsidian vault | `obsidian` | external [`qmd`](https://github.com/tobi/qmd) CLI | incremental `qmd update`, BM25 `qmd search`, semantic `qmd vsearch`; vault path via `connector.vaultPath` |
| RSS / news | `rss` | HTTP feed polling | RSS 2.0 + Atom, feed/category hierarchy, 24h dedupe window |

Connector-backed skills fetch documents into AutoRAG's local chunk store. External-crawler skills such as KakaoTalk, WhatsApp, Telegram, Slack, and Notion leave incremental archive and FTS ownership with their CLI and map query results into the same retrieval pipeline. Obsidian uses the external `qmd` CLI (incremental update + BM25 + semantic). Tokens are referenced by environment variable name only, never stored in config. Process/API failures surface as path/PII-opaque diagnostics. See [docs/manual-qa-datasources.md](docs/manual-qa-datasources.md) for the QA harnesses.

Configure them in `config.json` (CLI) or pass `datasourceSkills` programmatically:

```jsonc
{
  "datasources": {
    "whatsapp": { "instanceId": "personal", "connector": { "binaryPath": "wacrawl" } },
    "telegram": { "instanceId": "personal", "connector": { "binaryPath": "telecrawl" } },
    "slack":    { "connector": { "binaryPath": "slacrawl", "configPath": "/path/to/slacrawl.yaml", "syncSource": "primary" } },
    "notion":   { "connector": { "binaryPath": "notcrawl", "configPath": "/path/to/notcrawl.yaml" } },
    "github":   { "connector": { "repos": ["owner/repo"] } },
    "gmail":    { "connector": { "tokenEnv": "GMAIL_ACCESS_TOKEN", "labelIds": ["INBOX"] } },
    "mailcrawl": { "instanceId": "personal", "connector": { "account": "personal", "mailbox": "INBOX", "binaryPath": "mailcrawl" } },
    "personal-google-drive": { "type": "cloud-drive", "instanceId": "personal", "connector": { "provider": "google-drive", "remote": "personal-gdrive:", "include": ["**/*.md"] } },
    "company-onedrive": { "type": "cloud-drive", "instanceId": "work", "connector": { "provider": "onedrive", "remote": "company-onedrive:Documents" } },
    "obsidian": { "connector": { "vaultPath": "/path/to/vault" } },
    "rss":      { "connector": { "feeds": [{ "url": "https://example.com/feed.xml" }] } }
  },
  "datasourceAccess": {
    "allowedTags": ["whatsapp", "telegram", "slack", "notion", "github", "gmail", "mailcrawl", "cloud-drive", "obsidian", "rss"],
    "allowedScopes": ["/whatsapp/**", "/telegram/**", "/slack/**", "/notion/**", "/github/**", "/gmail/**", "/mailcrawl/**", "/personal-google-drive/**", "/company-onedrive/**", "/obsidian/**", "/rss/**"]
  }
}
```

Install wacrawl with `brew install openclaw/tap/wacrawl`. AutoRAG invokes `wacrawl sync` during datasource refresh and `wacrawl --json --sync never search` during retrieval. Optional trusted connector fields are `binaryPath`, `databasePath`, and `sourcePath`. The child process receives only a restricted environment; unrelated model/provider secrets are not forwarded. Live WhatsApp Desktop discovery requires macOS and the permissions documented by wacrawl, while an existing portable archive can be queried on other supported platforms.

Install telecrawl with `brew install openclaw/tap/telecrawl`. AutoRAG invokes `telecrawl import` during datasource refresh and `telecrawl --json search` during retrieval. It uses the same optional trusted connector fields and restricted child environment as wacrawl. Live Telegram Desktop discovery requires macOS and the permissions documented by telecrawl, while an existing portable archive can be queried on other supported platforms.

Install slacrawl with `brew install openclaw/tap/slacrawl`. AutoRAG invokes `slacrawl sync` during datasource refresh and `slacrawl --json search` during retrieval. Optional trusted connector fields are `binaryPath`, `configPath`, and `syncSource`. Slack credentials and source definitions remain in slacrawl's own configuration rather than AutoRAG.

Install notcrawl with `brew install openclaw/tap/notcrawl`. AutoRAG invokes `notcrawl sync` during datasource refresh and `notcrawl search --json` during retrieval. Optional trusted connector fields are `binaryPath` and `configPath`. Notion credentials and workspace definitions remain in notcrawl's own configuration rather than AutoRAG.

Install and configure [`mailcrawl`](https://github.com/NomaDamas/mailcrawl)
`@nomadamas/mailcrawl@0.1.4` or newer separately. AutoRAG invokes
`mailcrawl sync` followed by `mailcrawl index` during datasource refresh, then
calls `mailcrawl search` in BM25, semantic, or hybrid mode. The archive remains
in mailcrawl's native store unless the operator explicitly sets
`connector.dataDir`; Himalaya credentials and provider configuration remain
owned by mailcrawl. 0.1.3 and earlier fail a repeated
`index` after a no-op sync.
Use `mailcrawl` for all Himalaya-backed IMAP/Maildir retrieval. The retired
`gmail` connector option `backend: "himalaya"` is no longer registered; migrate
that configuration to an explicit `mailcrawl` datasource.

Install and authenticate rclone separately (`brew install rclone && rclone
config` on macOS), then configure the provider-neutral `cloud-drive` skill.
`cloud-drive` is a reusable `type`: each datasource config key is a connection
alias and becomes a separate agent skill and scope. Multiple Google accounts,
or Google Drive plus OneDrive/iCloud, can therefore be loaded and searched
independently.
AutoRAG inventories with `rclone lsjson`, keeps a workspace-local manifest and
managed mirror, and downloads only added or changed indexable files. Google
Drive is Tier-1. OneDrive, Dropbox, SMB/SFTP/WebDAV, and mounted drives share
the same manifest contract. iCloud Drive is experimental because rclone marks
that backend Tier 4 and Apple ID/2FA sessions periodically require
reauthentication. See [docs/datasource-skills.md](docs/datasource-skills.md)
for filtering, size, concurrency, bandwidth, dry-run, and agent tool-calling
details.

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
  },
});

await agent.refresh(); // refreshes parsed mirrors, BM25/MinSync, and datasource indexes
const results = await agent.searchDatasourceDocuments("meeting with Mina", { topK: 5 });
```

#### Discord (discrawl)

Discord uses the external [`discrawl`](https://github.com/openclaw/discrawl) CLI, which owns the SQLite archive, the FTS5 index, and the message vectors. AutoRAG never calls the Discord API itself.

```bash
brew install openclaw/tap/discrawl
```

Two archive sources are supported. `wiretap` (the default) reads the local Discord Desktop cache and needs **no token at all**; `discord` uses a bot token, which is the ToS-sanctioned automation path. AutoRAG refuses to spawn the CLI when a Discord *user* token is present in the environment — automating a user account violates Discord's Community Guidelines and can get the account terminated.

```typescript
import { AutoRAGAgent, DiscrawlClient, DiscrawlSkill } from "@autorag/librarian";

const discord = new DiscrawlSkill({
  client: new DiscrawlClient({ source: "wiretap", root: process.cwd() }),
  instanceId: "community",
});

const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
  datasourceSkills: [discord],
  datasourceAccess: {
    allowedTags: ["discord"],
  },
});
```

Or through the trusted config factory:

```json
{
  "datasources": {
    "discord": {
      "instanceId": "community",
      "connector": {
        "source": "wiretap",
        "embeddingProvider": "ollama",
        "embeddingModel": "embeddinggemma",
        "defaultMode": "hybrid"
      }
    }
  }
}
```

Two defaults are deliberate and worth keeping:

- **`defaultMode: "hybrid"`** — discrawl's FTS index strips newlines without substituting a space, welding words across line breaks into a single unsearchable token (measured at ~47% of post-newline words on a real archive). Semantic recall covers that gap. See [#1413](https://github.com/Marker-Inc-Korea/AutoRAG/issues/1413).
- **`embeddingProvider: "ollama"` + `embeddingModel: "embeddinggemma"`** — semantic search requires an embedding provider (`ollama serve && ollama pull embeddinggemma`). Configure these values in discrawl's own config; AutoRAG uses discrawl's native store unless an explicit `connector.configPath` is supplied. EmbeddingGemma (Gemma 3 300M, 768-dim, 100+ languages) is the same model family katok uses for KakaoTalk, so all CLI-backed datasources can share one local embedder. Do **not** use `nomic-embed-text`: it is English-only and collapses non-English text into one narrow similarity band, silently degrading semantic search to noise. AutoRAG emits a diagnostic when an English-only model is configured. See [#1414](https://github.com/Marker-Inc-Korea/AutoRAG/issues/1414).

A datasource skill should provide polling/cron metadata for routine indexing, source descriptions for the agent prompt, slash-hierarchical opaque source paths such as `/kakao/personal/chunks/<chunk-id>`, and permission tags that match your server-side access policy.

### Primary target: document collections

AutoRAG is built for **non-code document retrieval**: manuals, legal docs, internal wikis, meeting notes, research literature, knowledge bases, PDFs.

Code repositories work too (direct `grep` is useful), but AutoRAG's real value shows on unstructured text where simple pattern matching isn't enough.

## Configuration and state

The default home state is kept outside the workspace:

```text
~/.autorag/
├── config.json
├── memory.json
└── logs/
    └── runs.jsonl
```

`config.json` selects sources, the workspace, memory path, retrieval settings, and the agent model. Provider and model IDs must refer to a model available in the user's authenticated runtime:

```json
{
  "searchPaths": ["/path/to/documents"],
  "workspacePath": "/path/to/workspace",
  "memoryPath": "/Users/you/.autorag/memory.json",
  "model": { "provider": "provider-name", "id": "reasoning-model" }
}
```

`autorag init` leaves `model` unset when no model flags are supplied. At search time AutoRAG resolves an authenticated local provider when possible; otherwise configure the model explicitly.

For fast interactive search, prefer a model with reliable tool calling, high
output TPS, and low first-token latency. A query can require several short
model turns while AutoRAG alternates between retrieval tools and direct source
reading, so model throughput has a visible effect on end-to-end response time.
It does not accelerate BM25, MinSync, Jikji, filesystem access, or indexing
itself. Larger reasoning models remain useful for difficult synthesis,
conflicting evidence, and specialized domain judgment, but they are not a
requirement for ordinary retrieval.

Config path precedence is `--config` > `AUTORAG_CONFIG` > `~/.autorag/config.json`. When the home config is absent and `<cwd>/autorag.config.json` exists, AutoRAG copies the legacy file to `~/.autorag/config.json` without deleting or modifying the legacy file. The legacy cwd file is a migration source, not the default location.

`memory.json` stores retrieval memory and `logs/runs.jsonl` records run events. Model authentication remains with the user's configured provider or authenticated local runtime. Corpus indexes remain workspace-local: refresh keeps parsed mirrors and BM25/MinSync indexes under `<workspace>/.autorag`.

`autorag refresh` and `autorag index reset|rebuild` accept `--method <csv>` (e.g. `--method bm25,minsync,parsed`) to scope which indexing methods run or which index directories are removed. When omitted, all methods run. `autorag init` accepts `--embedder-*` flags to configure the MinSync embedder endpoint in the config file.

`autorag health` checks model/provider auth before a search — it resolves the model, verifies credential presence, and optionally probes one completion call. Use it to diagnose model, provider, auth, or timeout failures. `autorag status` remains the model-free index-health command (corpus freshness and BM25/MinSync readiness). When `autorag search` fails for a model/provider reason, the error output includes a hint pointing to `autorag health`.

`autorag ui` opens a loopback-only page (`127.0.0.1`) to connect local folders and datasource skills without editing JSON. It writes the same trusted `datasources` / `datasourceAccess` fields as a hand-edited config, stores env-var *names* rather than secrets, and refuses non-loopback binds. Use `--no-open` to print the URL without launching a browser.

For a deliberately deployed UI, opt in explicitly in `config.json`. Keep the
session token in the environment, set the public URL used by the browser, and
allow only the frontend origins that should make credentialed API requests:

```json
{
  "ui": {
    "host": "0.0.0.0",
    "port": 8787,
    "allowRemote": true,
    "publicOrigin": "https://autorag.example.com",
    "corsOrigins": ["https://autorag.example.com"],
    "tokenEnv": "AUTORAG_UI_TOKEN"
  }
}
```

Start it with `AUTORAG_UI_TOKEN` set to a random value of at least 16
characters. Local use remains the safe default: omit `ui` (or leave
`allowRemote` false) and AutoRAG binds to loopback, including a working
`localhost` URL on systems that resolve it to IPv6.

## Installation

Published as `@autorag/librarian` (dist bundled with Bun, runtime Node ≥ 24 or Bun):

```bash
bun add @autorag/librarian          # library
bun install -g @autorag/librarian   # autorag CLI
# or run directly from the repo:
bun add github:NomaDamas/AutoRAG-2.0
```

PDF parsing requires **Java 11 or newer** because the bundled
`@opendataloader/pdf` runtime is compiled for Java 11. This applies to
Windows, macOS, and Linux. Verify the runtime that will be selected from
`PATH` before indexing PDFs:

```bash
java -version
```

On Windows, ensure a Java 11+ installation appears before older Java
installations in `PATH` (or set `JAVA_HOME` to the Java 11+ installation and
start a new shell). Java 8 is not supported by the PDF parser.

Git-based installs build `dist/` via the `prepare` script and require Bun on the installing machine.
External tool binaries auto-install on first use into `<workspace>/.autorag/bin`: **MinSync** downloads a verified GitHub release asset (on by default; `minSync.autoInstall: false` to opt out), and **Jikji** compiles the [`jikji-cli`](https://crates.io/crates/jikji-cli) crate via cargo (requires the [Rust toolchain](https://rustup.rs); `jikji.autoInstall: false` to opt out). New `autorag init` configs enable Jikji find-first discovery by default. KakaoTalk (`katok`) and Discord (`discrawl`) stay manual, optional installs. All of them degrade gracefully when missing — core BM25 search works without any of them.

### Experimental TUI (beta)

The latest TUI work is currently available on the `feat/experimental-tui`
branch. To use this beta version from source:

```bash
git clone --branch feat/experimental-tui https://github.com/Marker-Inc-Korea/AutoRAG.git
cd AutoRAG
bun install
bun run build

# Configure your document roots and model once:
node dist/cli/index.js init --search-paths /path/to/documents

# Start the beta TUI:
node dist/cli/index.js tui
```

Inside the TUI, use `/resume` to browse saved sessions. Selecting a session
replaces the current view with that session instead of mixing the two
conversation histories. This branch is experimental and may change before the
feature is included in a published release.

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
    │ Plan + search│ ← check_memory, Jikji, and retrieval tools
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Direct read  │ ← bash find/grep/cat
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
