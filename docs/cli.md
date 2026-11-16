# CLI command reference

All `autorag` commands.

## CLI Command Reference

| Command | Description |
|---|---|
| `autorag init` | Initialize `~/.autorag/config.json` with search roots, document languages, and model settings |
| `autorag refresh` | Refresh parsed mirrors, MinSync CDC chunks, datasources, Jikji, and the platform file-name index (Everything on Windows, FSearch on macOS/Linux) |
| `autorag search "<query>"` | Run the librarian agent to curate structured answers |
| `autorag status` | Inspect corpus freshness, indexing status, and vector readiness |
| `autorag health` | Check model provider authentication, token validity, and API reachability |
| `autorag models list` | List chat models the pi runtime can resolve (built-ins, `models.json`, custom/extension providers) with provider auth status; `--decision` lists decision models (`jev.backend`/`jev.model` candidates) instead; never prints credential values |
| `autorag update-check` | Compare the running `autorag` against the published npm version (also runs on `autorag tui` launch) |
| `autorag tui` | Open Pi's interactive librarian TUI (`/login`, `/model`, `/resume`, …); the agent can stop and ask you a multiple-choice or open question through the [`ask_user` tool](ask-user.md); `--telegram` / `--whatsapp` / `--discord` add the Telegram / WhatsApp / Discord remote-control bridges; `--permission-mode ask\|auto\|full` (Shift+Tab, `/permissions`) sets the [approval gate for file changes](permissions.md) |
| `autorag duplicates [DIR]` | Read-only scan for exact and near-duplicate document families with `dupey` |
| `autorag lite ...` | CLI bootstrap, indexing repair, terminal maintenance, and fallback interface; agents use MCP for normal Lite operation |
| `autorag memory inspect` | Render a read-only, path-opaque snapshot of retrieval memory (judged evidence, curated results, evidence chunks, insights); model-free |
| `autorag evidence <session>` | Inspect persisted evidence behind numbered results; MCP clients normally use `autorag.evidence`, and the CLI stays available for terminal maintenance |
