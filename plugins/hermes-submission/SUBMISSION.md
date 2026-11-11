# Hermes Plugin Hackathon: submission kit

Nothing here has been posted, opened, or submitted. This directory is the prep.
The plugin itself lives in [`../hermes`](../hermes) (the only tree the catalog installs).

## What is verified (run on Hermes git `6cff782b`, 2026-11-10)

| Check | Result |
| --- | --- |
| `hermes plugins validate plugins/hermes` | passes (portable manifest, security scan `safe`) |
| `hermes plugins doctor plugins/hermes --ci` | passes |
| `plugin-catalog/autorag.yaml` parsed by upstream `entry_from_mapping` | accepted (with a syntactically valid sha) |
| `hermes plugins install file://...` then `enable` | installs, MCP server `autorag` registers 10 tools |
| Real Hermes gateway turn (`-t autorag -z`) with `glm-5p3-flash` | calls `mcp__autorag__*` tools and answers from files, Slack, Discord, Telegram |
| `autorag.duplicates` through Hermes | finds the exact duplicate pair |
| Real `~/.autorag/config.json` sha1 before and after | identical |
| Model-driven setup on a first-time sandbox user (no `~/.autorag`, synthetic Documents + Slack/Telegram/Discord archives), prompt "Set up AutoRAG for my Documents folder." | finished in 12 min (722 s) with 57 tool calls; wrote the config, merged the Slack, Telegram and Discord entries with a backup, built the index, started the embedding daemon, and reported accurately. Its `mcp` tools only appeared after `hermes pm install --extra mcp` and a new session. |

### On the Hermes the desktop app actually uses (v0.23.0 updated to v0.23.1, stable channel)

| Check | Result |
| --- | --- |
| `hermes update` on `~/.hermes/hermes-agent` | v0.23.0 -> v0.23.1 (`3b4215785882`); `update --check` then reports up to date |
| `hermes plugins install Marker-Inc-Korea/AutoRAG/plugins/hermes --ref <sha> --no-enable`, then `enable` | installs to `~/.hermes/plugins/autorag`; `plugins validate` passes; custom-source warning shown (not a catalog entry yet) |
| MCP registration on the running gateway | first connect after `enable` was cancelled after about 28 s (`CancelledError`, cold `npx`); a later reconcile registered 16 tools: 9 core (status, search, search_files, datasources_list/get, duplicates, evidence, refresh, report) plus 7 `search_datasource_*` (mailcrawl, kakao, telegram, whatsapp, slack, nomadamas, browser) |
| Question to the live agent: list the AutoRAG tool names | returned exactly those 16 names in 10 s |
| In-process `registry.dispatch` of `autorag_datasources_list` and `autorag_status` in the install's own PM environment | both returned; `status` took 35 s and reports `state: failed` from an earlier refresh in the real index, which I did not touch |
| Real `~/.autorag/config.json` sha1 before and after | identical |

What did not work: a live-agent request to call `autorag_status` and `autorag_datasources_list` issued both
calls and then produced no result for 170 s, so the run was cut off. The same two calls complete when
dispatched in-process (above) and over a raw stdio session, so I could not reproduce the stall outside the
agent loop. Cause not found; do not claim live-agent tool calls work on this install until that is explained.

Side effects of the update on this machine: `hermes update` stopped the running gateway without respawning
it. I restarted it with `hermes gateway ensure`. Any open desktop session was interrupted. A backup of
`config.yaml` and `auth.json` was taken first under `~/.hermes/backup/pre-autorag-plugin-*`.

Not verified: Windows and Linux, a second Hermes version, `hermes plugins validate --install-deps`
(portable package, no Python deps), the catalog CI job itself (it only runs on the upstream PR), and a
second setup run (the 12 min figure is one sample on a loaded machine; cargo was present so `minsync`
compiled, which dominated it). In that run the model also hit two problems the skill did not cover and
patched the skill itself (now folded into its Troubleshooting): a Hermes installer PATH precheck, and
Discord staying keyword-only because discrawl's own config has embeddings off.

## Findings that change the plan

1. **MCP SDK.** Hermes loads the plugin's MCP server only when its Python `mcp` SDK is importable.
   `mcp` is part of Hermes's `[all]` extra, so a normal install should have it; I did not verify a
   stock install. My throwaway `uv sync` environment lacked it and the plugin then registered zero
   tools while the model quietly fell back to `terminal`. Mention `hermes pm install --extra mcp`
   in the PR description as the fix.
2. **Cold start.** With npm audit, fund, and update-notifier left on, the first `npx` start took
   about 20 s to 145 s depending on machine load (Hermes's connect budget is 60 s). A second start
   after the package is cached took about 1.4 s. Hermes's code retries a failed first connect on a
   cooldown (not tested here), so expect the first question after install to possibly find no tools. The `timeout: 1800` in `plugin.json` only widens tool calls, not the connect.
3. **`Hermes -z` and `hermes chat` go through a gateway daemon.** With no gateway running, the first
   call spawns one; it can lose a race with a one-shot client. For a recording, start the gateway
   first and wait for `registered 10 tool(s)` in `logs/agent.log`.
4. **Tool names:** `mcp__autorag__autorag_search`, not `mcp__autorag__autorag.search`; dots become
   underscores. The skills describe this.
5. **Config must exist.** Without a config the server starts but searches `.` (Hermes's data dir)
   and `autorag.search` returns `index-not-ready`. The `setup` skill handles first-time init.

## Before opening the catalog PR

1. Done: `plugins/hermes` is on `Marker-Inc-Korea/AutoRAG` `main` (#2123), pinned to `@autorag/librarian@2.13.11` (#2136, #2137).
2. `catalog/autorag.yaml` has `sha: 8b0ea0ab9bf5860676ce097cf0854ea915afe7ff`, the main commit with that pin. Checked with
   Hermes's own `entry_from_mapping`, and `hermes plugins validate` / `doctor --ci` on the tree at that commit. If the
   plugin changes again before the upstream PR is merged, set `sha:` to the new 40-char main commit (never a branch).
3. In `NousResearch/hermes-agent`, add that one file as `plugin-catalog/autorag.yaml`.
4. PR description must carry the rule-13 disclosure. Draft below.
5. Check `requires_hermes` against the Hermes release current on submission day; it is `>=0.22.3`
   here because the `servers.<name>.timeout` extension is only honoured from that release (the
   official `meshy` entry declares the same floor for the same key).

### PR description draft

> **AutoRAG** adds local search over your own files and chat/mail archives to Hermes.
> Portable Agent Plugins v1 package (`plugin.json` + `mcp.json` + four skills + a Desktop search pane). Source:
> `Marker-Inc-Korea/AutoRAG` `plugins/hermes` (submitted by a contributor to that repository; confirm owner or major-contributor status before opening, rule 5).
>
> - Surfaces used: stdio MCP server (`@autorag/librarian@2.13.11`, pinned), skills, and a Hermes Desktop pane with a small
>   backend router (`dashboard/plugin_api.py`). No hooks, no middleware, no core overrides. Three skills bundle a
>   standard-library Python helper that the agent runs only when the user asks (macOS/Linux): `setup` (a read-only
>   folder inventory: lists and stats files, never opens or writes them), `automate` (writes an AutoRAG automation rule and
>   two small scripts so a file event runs a Hermes cron job), and `schedule` (creates no-agent Hermes cron jobs that
>   refresh the index or report new duplicate files). Nothing is created at install time.
> - The Desktop backend runs `npm install` of the same pinned `@autorag/librarian` once into
>   `$HERMES_HOME/plugin-data/autorag/runtime` (no lifecycle scripts) and then runs `autorag lite find` with `node`.
> - Network: retrieval is local, but chunks a search returns are read by the user's Hermes model. First start downloads the pinned package from npm via `npx`. The first refresh downloads
>   `minsync` (cargo or sha256-verified GitHub release asset), `jikji` (cargo), and two local embedding
>   models from Hugging Face (about 1.5 GB). Search itself needs no network for local files and local crawler archives; Notion, GitHub, RSS and Hister connectors do.
> - Reads outside plugin data: the folders in the user's AutoRAG config and local crawler archives the
>   user configures. Writes only AutoRAG's own index and report state. Never edits user documents.
> - Writes beyond that, only on request: `~/.autorag/automations.json` (backed up first) and
>   `$HERMES_HOME/plugin-data/autorag/automations/` plus one gate script in `$HERMES_HOME/scripts/`. `remove` deletes them.
> - Processes: long-lived stdio child; refresh spawns local helper binaries.
> - No telemetry, no self-update (version pinned in `mcp.json`), no credentials read.
> - Tools are annotated `readOnlyHint` except `autorag.refresh` and `autorag.report`.
> - Tested with `hermes plugins validate` and `hermes plugins doctor --ci`.
