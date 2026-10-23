---
name: autorag
description: Search, summarize, compare, and answer questions from an already configured AutoRAG librarian over local documents and configured datasources. Use when the user asks AutoRAG to search PDFs, wikis, notes, or a knowledge base. Use autorag-setup for install, model, roots, indexing, or datasource changes.
license: Apache-2.0
---

# AutoRAG Librarian Skill

Use this skill when AutoRAG is already configured and the user asks to search,
summarize, compare, or answer questions from local PDFs, wikis, notes, research
papers, knowledge bases, or configured datasources.

AutoRAG is the specialized librarian agent. One configured model plans the
search, calls MinSync, Jikji, datasource, and filesystem tools, reads
sources, judges evidence, and curates the final answer. There is no subagent or
separate model role.

Indexing and refresh read source documents and write only under the configured
workspace `.autorag/` directory and Jikji's central store (`<OS data dir>/jikji/`,
plus a `.jikji_agent_map.md` guide in each indexed root). The librarian may
organize, create, or modify the user's own files in local sessions when the
user asks; it never deletes permanently (OS trash only) and lists every change
as old path → new path. Do not move, rename, edit, or delete source files on your own
initiative while searching or diagnosing.

## Preflight

Confirm a config exists at `--config`, `AUTORAG_CONFIG`,
`$AUTORAG_HOME/config.json`, or `~/.autorag/config.json`. Inspect only
non-secret `searchPaths` and `model` metadata, then run:

```bash
autorag duplicates --json
autorag status --json
autorag health --json
```

### Duplicate-file review

Use `autorag duplicates` when the user asks to find duplicate files, choose
likely latest copies, or reduce corpus/index space. The command and the
`scan_duplicate_documents` Agent tool are read-only and never delete or move
source files. Exact means dupey's canonical extracted-text hash matches;
near/contains families require human review. Exact duplicate exclusion during
refresh is enabled by default and can be disabled with
`"excludeExactDuplicates": false` in `config.json`.

### Converting files

The `convert_files` and `list_file_conversions` Agent tools run the
[Filee](https://github.com/KnifeLemon/Filee) command line to convert local
files offline: documents, spreadsheets, slides, HWP/HWPX, images, e-books,
archives, audio, video, data, and subtitles. They work on any path the user
names, not only the configured roots. This is the one exception to the
"never write source files" rule above: a conversion writes a new file and
never touches its source. An existing output name gets a number added unless
the user asks to replace it, in which case pass `overwrite: true`. Never
delete the source unless the user asked.

Each converted file is checked with `dupey compare` against its source, using
dupey's own thresholds, so the report says whether a later scan will group
them: `exact` (same canonical text, for example DOCX to HWPX), `near` (same
document in another format, for example Markdown to TXT), `different` (text
differs, typically layout formats such as PDF), `unsupported` (dupey cannot
read the format, for example HTML or images), or `unavailable` (dupey not
installed). The command line is `filee.binaryPath` if set, else a `filee`/`filee-cli`
on `PATH`, else the pinned [filee-cli](https://github.com/NomaDamas/filee-cli)
release that `autorag setup` (or the first conversion) downloads into
`<AUTORAG_HOME>/tools/filee-cli/`; `"filee": { "autoInstall": false }` forbids that
download. The system prompt carries the conversion map (which format family
converts to which, and the two-step chain when no direct route exists), so plan
from it instead of probing. `"filee": { "enabled": false }` in `config.json` removes both
tools; `filee.timeoutMs` (default 30 minutes) and `filee.dataDir` (Filee's
own settings and engines, `FILEE_DATA_DIR`) tune them.

### Excluding local files from the index

`"excludePaths"` in `config.json` lists files or folders (absolute, or relative
to `workspacePath`) that refresh keeps out of the parsed mirror and MinSync.
A folder entry excludes everything under it. Removing an entry and running
`autorag refresh --method minsync` indexes the file again. Excluded sources are
recorded as `user-excluded` skips, so they are not reported as stale. Jikji
indexes the source folders directly and is not refreshed here, so AutoRAG also
drops excluded paths from the `jikji_find` answer pack and the baseline prefetch
at retrieval time; the on-disk `.jikji_agent_map.md` stays complete, and direct
file reads remain available.

`status` is model-free and path-opaque. `health` resolves the single model,
checks credential presence, and normally probes one live completion. If the
model, authentication, configuration, or indexes are unhealthy, use
`autorag-setup` rather than guessing private provider details.

MinSync and Jikji should normally be healthy. Answering a question never
builds or installs them: `autorag refresh` auto-installs both by default and
builds their indexes incrementally. If they are missing or stale, run a full
`autorag refresh` or return to setup rather than silently degrading to
lexical-only search.

## Search

Prefer `--json --debug` when another agent will consume the result.
`--json` alone omits `sessionId`. `--debug` adds session/diagnostics fields and
does not print filesystem paths.

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5 --json --debug
```

`--json --debug` includes `answer`, numbered `results` (`number`, `title`,
`summary`, optional `source`), and `sessionId`. `--json` without `--debug` is
only `answer` plus `results`. Every
bracketed `[n]` citation in `answer` resolves to a `results[].number` of the
same response; an unmatched citation is removed and reported as a
`citation-without-result` diagnostic. Within one conversation (an interactive
session, including a resumed one), a source keeps the `number` it was first
shown with and a new source gets the next unused number, so `results[].number`
can skip numbers and need not start at 1 on a later turn.

To inspect the exact persisted evidence behind numbered results, use:

```bash
autorag evidence <sessionId> --result 1 --json
```

The response includes the original source, retrieval method, stable evidence
ID, raw excerpt/content, and any available `chunkIndex`, `lineNumber`,
`retrievalResultId`, and metadata. Omit `--result` to inspect every result in
the session. Prefer this command whenever the caller wants detailed chunk text
rather than only the curated summary.

- `--scope` narrows datasource retrieval to a requested sub-path (ordinary
  per-query filtering).
- `--json` is required for programmatic consumption.
- `--debug` is required for `sessionId` and diagnostics in search output.
- `autorag evidence` is the detailed source/chunk inspection path.

Do not bypass the librarian with ad hoc raw search when the user requested
AutoRAG. The search loop can use Jikji, MinSync lexical/vector/hybrid retrieval, datasource retrieval, and
direct source reading as appropriate. If search fails because of model,
provider, auth, or timeout problems, diagnose with `autorag health --json`.

Every search is two-phase: a fast answer, then verification. With Jev on (the
default, OpenRouter), Jev first routes the question:

- General knowledge or small talk is answered directly.
- A request to view or change AutoRAG's own settings (model, providers,
  datasources) is handled by the agent itself: it loads the setup skill, edits
  the active config, verifies it, and reports what changed. The running agent
  keeps its startup model; changes apply to the next `autorag` launch.
- A request to act on files or folders (move, bulk rename, rename by content,
  convert, sort) is carried out in one tool turn with no retrieval and no
  verification, and reports every change as old path → new path.
- Anything else that needs no retrieval (commit, push, open a PR, run a build)
  is carried out in one tool turn the same way.
- Private-data questions use local search; the agent decides on web search
  after the fast answer.
- A multi-part question is split into up to five parallel search queries.
- On local search, Jev also picks which registered datasources (Slack,
  Discord, KakaoTalk, email, ...) to search before the fast answer, from each
  datasource's description and where similar past questions were answered;
  their chunks are reranked together with local file evidence.

After the fast answer, Jev ends the run if the answer is complete and
evidence-backed. So `results` may come straight from the fast answer, with no
verification phase. `--debug` diagnostics show the decision: `query-routed`
(branch and queries), `datasources-selected` (datasources searched or skipped,
with probabilities), `follow-up-skipped` (fast answer final), or
`query-route-fallback` (Jev unavailable, single local search).

## Maintenance

```bash
autorag setup --format json
autorag status --json
autorag health --json
autorag refresh --json
autorag refresh --method minsync,jikji --json
autorag watch --once --json
autorag watch
autorag refresh --force --json
autorag index rebuild --yes --json
autorag index reset --method parsed --yes --json
autorag memory inspect --json
autorag tui
autorag automate status --json
autorag automate install --json
autorag automate uninstall --json
autorag automate run invoice --event create:/path/to/file.pdf --json
autorag automate daemon
```

File-triggered automations live in `~/.autorag/automations.json` (`automations`
array of `{ id, paths, events, include, exclude, run | agent, appendFiles, debounceMs, timeoutMs }`).
Edit `automations.json` directly to add or change rules; matched `[{ type, path }]`
events are piped on stdin to `run`. Use `autorag automate install` to register
the background daemon with launchd, `systemd --user`, or Windows Task Scheduler.

A rule runs exactly one of `run` (a command) or `agent` (an AI agent started in
the daemon on each batch of changes):

```json
{ "id": "inbox-triage", "paths": ["~/Downloads"], "include": ["**/*.pdf"],
  "agent": { "profile": "file-triage", "task": "Summarize each new PDF.",
             "retrieval": { "searchAll": true, "datasources": ["kakao"] } } }
```

`agent.profile` names a profile defined in code: `file-triage` (reads, lists and
searches files; changes nothing) or `file-organizer` (also `bash`, `edit`,
`write`). A rule never carries a prompt or a tool list, so the rules file cannot
grant an agent more than its profile has. `retrieval` optionally adds
`search_all_documents` and one `search_datasource_<id>` per listed datasource.
Retrieval memory is never connected to an automation agent.

Every agent run gets a loop guard appended to its system prompt, naming the
rule's folders, file types and events, and forbidding any change that would
fire the rule again, editing the rules file, and running `autorag automate`. At
most one agent run is in flight across all rules, and an agent rule is
disabled after 6 runs within 60 seconds (command rules: 20). The run's final
reply is the `stdout` of its log entry.

Prefer a full refresh so parsed mirrors, MinSync, Jikji, configured
datasources, and (on Windows) the bundled Everything file-name index stay
aligned. `--method` accepts
`parsed,minsync,datasources,jikji,everything,all`. BM25 is a MinSync retrieval mode, not
a `--method` name. Use `--method` only for deliberate narrowing. Scheduled
maintenance should use non-daemon `autorag watch --once`, typically every
1 hour, with the same config used by search and no overlapping runs.
`autorag tui` is the shipped beta terminal UI.

Reset and rebuild commands remove only selected workspace `.autorag` indexes.
Never target source documents. `memory inspect` is read-only and path-opaque.

## Rules

- Use only configured and approved search paths.
- Never expose provider credentials or authentication payloads.
- Never invent provider identities or model ids.
- A Pi-usable subscription is valid; a subscription Pi cannot invoke is not.
- Preserve real source mapping.
- Prefer `--json --debug` when another agent consumes search output.
- Do not invent CLI commands. `autorag --help` is the command list of record,
  including the shipped `setup`, `models`, `tui`, and `lite`
  commands. `autorag <command> --help` prints that command's own flags, and
  `autorag --version` prints the installed version. See
  `docs/minsync-setup.md#embedding-backends` for MinSync setup and embedding-backend selection.
