---
title: "Terminal UI and search"
description: "Run the model-backed librarian from the CLI or TUI, with its tools, two-phase search, and numbered citations."
---

# Terminal UI and search

`autorag search` answers one question end to end and `autorag tui` hosts the same librarian
interactively. This page covers both surfaces: the flags, the agent's tools, the two-phase search,
the numbered-citation output contract, and how to steer a running search.

## Searching from the CLI

`autorag search "<query>"` runs the librarian agent loop: it routes the query, retrieves candidate
evidence, reads and verifies the sources, and prints a curated answer with numbered results. It
requires a configured chat model and a reachable decision model (Jev, the `jev` config section);
the command checks the decision model first and stops before searching when it cannot answer.

```bash
autorag search "What are our primary Q3 deliverables?"
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--top-k <n>` | Maximum curated results. |
| `--scope <scope>` | Narrow retrieval to an opaque virtual-path scope. |
| `--fast-thinking <lvl>` | Thinking level for the fast first answer. |
| `--final-thinking <lvl>` | Thinking level for the verified final answer. |
| `--permission-mode <m>` | Disk permission mode: `readonly` (default; OS-level read-only) or `full`. See [Permission modes](permissions.md). |

Thinking levels are `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. The two phases
default to `minimal` (fast) and `high` (verification). Global flags such as `--json`, `--debug`,
`--config`, `--search-paths`, `--workspace`, and `--model-provider` are accepted by every command;
see [Global flags](../reference/cli.md#global-flags).

`autorag search` defaults to `readonly` so unattended runs cannot modify files on disk. An explicit
`--permission-mode`, `AUTORAG_PERMISSION_MODE`, or config value still wins.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | A curated answer was produced. |
| `1` | The search failed at runtime (retrieval, datasource, or model error). |
| `2` | Usage or model-configuration error (empty query, no model, invalid flag). |
| `3` | Decision model unavailable (`decision-model-unavailable`). |

### JSON output

With `--json`, progress lines and the answer are machine-readable:

- Progress: one JSON object per line, `{"type":"progress","text":"…"}`.
- Fast answer: `{"type":"preliminary","response":{…}}`.
- Final answer: `{"answer":"…","results":[{"number":1,"title":"…","summary":"…","source":"…","locations":[…]}]}`.

`--debug` adds `sessionId`, `query`, `searched`, and `diagnostics` to the envelope, and per-result
`confidence` and `evidence` (each excerpt with its line, page, or section). Without `--debug` the
JSON carries only the answer and the numbered results.

## The interactive terminal UI

`autorag tui` runs Pi's interactive mode hosted on the librarian. Pi owns the terminal UI, provider
credentials and OAuth, model selection, session persistence, resume/fork/compaction, settings, and
extension loading; AutoRAG registers its domain tools and runs the two-phase search in the same
session. The model's progress notes, fast answer, and final answer are ordinary assistant messages.

```bash
autorag tui
```

### Pi host commands

These commands are available in-session:

| Command | Purpose |
| --- | --- |
| `/login`, `/logout` | Provider credentials and OAuth. |
| `/model` | Pick the model. Required before the first search when no model is configured. |
| `/resume`, `/new`, `/tree` | Resume, start, or branch sessions. |
| `/compact`, `/settings` | Compact context, edit settings. |

### Typing while a search runs

Input typed during a search behaves like Pi's own turns:

| Key | Effect |
| --- | --- |
| `Enter` | Steer the running search: a correction applied before its next model request, then verified. |
| `Alt+Enter` | Queue a new question for after the current search answers. |
| `Esc` | Stop the search. |

### Embedding controls

The footer shows the embedding backend state and re-probes it periodically. `Ctrl+Shift+E` starts or
stops the local `inl serve` daemon, and `/embedding [on|off|toggle|status]` does the same
without a bound key. When `minSync.embedder` is a remote endpoint there is no local daemon to
toggle. See [Running the embedding daemon](../indexing/embedding-backends.md#running-the-embedding-daemon)
and [TUI footer and the /embedding command](../indexing/embedding-backends.md#tui-footer-and-the-embedding-command).

### Messenger bridges

| Flag | Effect |
| --- | --- |
| `--telegram` | Load the Telegram bridge so a paired Telegram chat can ask, steer, and read answers. |
| `--whatsapp` | Control the session from WhatsApp; pair once with `/whatsapp`. |
| `--whatsapp-group <jid>` | With `--whatsapp`, answer only this WhatsApp group. Requires `--whatsapp`. |
| `--discord` | Control the session from Discord; creates a session channel. |

A bridged TUI has nobody to answer a terminal approval dialog, so it defaults to `full` permission
unless overridden. A bridged session does not register the `ask_user` tool. See
[Messenger connections](messengers.md).

## Agent tools

Every tool takes a short `i` argument: one line describing what the call checks, shown to the user
as live progress. The host registers Pi's built-in `read`, `bash`, `edit`, `write`, `grep`, `find`,
and `ls`, plus the AutoRAG tools below. Tools whose backend is disabled or unavailable for the host
are not registered.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `bash` | `command` | Filesystem discovery and document reading with real paths (`ls`, `find`, `grep`, `cat`); in local sessions it also performs user-requested file organization. |
| `write` | `path`, `content` | Create or save a file in a local session. |
| `edit` | `path`, old/new text | Update or replace an existing file in a local session. |
| `search_all_documents` | `query`, `topK?`, `scope?`, `since?`, `until?` | Fan out across every configured retrieval method and merge/rank candidates. |
| `semantic_search_local_docs` | `query`, `topK?`, `scope?` | MinSync semantic/vector retrieval over parsed mirrors. |
| `search_datasource_<id>` | `query`, `topK?`, `scope?`, `since?`, `until?` | One generated tool per configured datasource connection; searches only that connection. |
| `jikji_find` | `query`, `topK?`, `first?` | Local discovery through Jikji answer packs. |
| `everything_search` | `query`, `alternatives?`, `regex?`, `matchCase?`, `matchPath?`, `wholeWord?`, `kind?`, `path?`, `sort?`, `offset?`, `maxResults?` | Windows-only file/folder name search over the configured roots. |
| `fsearch_search` | `query`, `alternatives?`, `regex?`, `matchCase?`, `matchPath?`, `kind?`, `path?`, `sort?`, `offset?`, `maxResults?` | macOS/Linux file/folder name search; falls back to a bounded filesystem walk. |
| `load_datasource_skill` | `name` | Load instructions for a configured datasource skill. |
| `check_memory` | `query` | Inspect evidence earlier searches found, similar past questions, and long-term insights. |
| `scan_duplicate_documents` | none | Read-only dupey scan of configured local document roots. |
| `convert_files` | `inputs`, `to`, `outputDir?`, `recursive?`, `quality?`, `overwrite?`, `skipExisting?`, `keepDates?` | Offline file conversion (Filee). |
| `list_file_conversions` | `from?` | List known formats, or what one source format can become. |
| `calculator` | `operation`, `expression?`, `timeZone?`, `toTimeZone?`, `date?`, `datetime?`, `from?`, `to?`, plus per-operation count fields | Exact arithmetic, date, and time-zone math. |
| `manage_automations` | `action`, `id?`, `automation?` | Create, list, change, delete, and test file-triggered automations. |
| `web_search` | `query`, `recency?`, `limit?`, `num_search_results?` | Public-internet search through the credential-free provider chain. |
| `web_fetch` | `url`, `raw?` | Read a public http(s) URL as markdown/text. |
| `jev` | `state`, `questions`, `confidence_threshold?` | Ask the Jev judgment model for calibrated probabilities. |
| `ask_user` | `question?`, `context?`, `options?`, `allowMultiple?`, `allowFreeform?`, `questions?` | Ask the user a question at the terminal. Terminal sessions only; see [Asking you questions](ask-user.md). |

There is no separate lexical-search tool: BM25-style ranking runs inside MinSync and is reached
through `search_all_documents`. There is no datasource fan-out tool — every configured connection
gets its own `search_datasource_<id>` tool.

External MCP-server tools (`mcp__*`) are registered through Pi alongside these; they are not covered
by the file-permission gate.

## Two-phase search and query routing

A search is progressive. Before any model turn, Jev routes the question to one of four branches:
`local` (the answer needs the user's files, chats, mail, or public web), `organize` (the user wants
files moved, renamed, converted, or tidied), `extra` (answerable from the model's own knowledge or a
non-organization tool task), or `config` (inspect or change AutoRAG's own settings). A non-local
branch needs Jev confidence of at least `0.75`, otherwise the run falls back to a full local search.
See [Jev routing and decisions](../retrieval/jev.md).

A local search then runs in two phases:

1. **Fast answer.** Baseline MinSync and Jikji retrieval is gathered up front; the model writes a
   complete, self-contained first answer from that evidence with minimal thinking, without calling
   local search, retrieval, datasource, or file-reading tools. The `calculator`, `web_search` /
   `web_fetch`, and `ask_user` tools are the named exceptions.
2. **Verification.** Jev checks whether the fast answer needs follow-up; if so the run continues
   into a verification phase that re-checks, fills gaps, and finalizes with the configured final
   thinking level. A steer from the user always forces verification.

The other branches (organize, extra, config) skip the fast answer and verification and run one
tool-capable turn.

## Directory access boundaries

Local retrieval sources are absolute filesystem paths and may be read with `bash` after verifying
the returned path. Datasource retrieval sources use slash-prefixed virtual identifiers such as
`/kakao/...`, `/mailcrawl/...`, `/slack/...`, `/discord/...`, and `/github/...`; they are not OS
paths and must never be passed to `cd`, `cat`, or other filesystem tools — search them through the
connection's `search_datasource_<id>` tool.

The librarian may organize, create, and modify local files with `bash`/`write`/`edit` when the user
asks, but never performs permanent, unrecoverable deletion: files are moved to the OS trash instead.
In `ask` and `auto` permission modes, changes to existing files go through the approval gate. See
[Permission modes](permissions.md).

## Progress and steering

While a search runs, the host shows a progress widget built from the run's routing, decomposition,
per-source hit/failure counts, and model phases, and each tool call displays its `i` line. In the
TUI, `Enter` steers the running search, `Alt+Enter` queues a follow-up, and `Esc` stops it. A steer
is framed to the model as a correction of the running search, cuts in-flight retrieval short, and is
restated to the verification phase. A steer that lands after verification's last request still gets
one more verification turn.
