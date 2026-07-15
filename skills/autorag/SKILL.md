---
name: autorag
description: Use an already configured AutoRAG librarian agent to search, summarize, compare, and answer questions from local document collections. Use autorag-setup instead for provider/model discovery, first-time configuration, or selecting folders to index.
---

# AutoRAG Librarian Skill

Use this skill when AutoRAG is already configured and the user asks to search,
summarize, compare, or answer questions from local PDFs, wikis, notes, research
papers, or knowledge bases. For first-time configuration, missing model/provider
settings, subscription or API-provider detection, or changing indexed folders,
use the `autorag-setup` skill instead.

AutoRAG is invoked through the `autorag` CLI. It is non-destructive: it reads
source files and writes indexes under the configured workspace's `.autorag/`
directory and, when enabled, Jikji's per-source `.jikji/` caches. Never move,
rename, or delete source files.

## Preflight

Confirm that `~/.autorag/config.json` exists (or an explicit `--config` /
`AUTORAG_CONFIG` path). Inspect that non-secret config for usable
`searchPaths`/`agents` without printing credentials. Then run:

```bash
autorag status
autorag health
```

`status` is model-free and path-opaque: it reports corpus freshness and index
health, not absolute filesystem paths or role-model auth. `health` checks
model/provider auth and explorer subagent setup: it resolves both role models,
verifies credential presence, and (unless `--skip-probes`) probes a completion
call per role. Use `health` to diagnose model, provider, auth, timeout, or
subagent dispatch failures before searching. If configuration, authentication,
role models, or indexes are missing or unhealthy, stop this workflow and use
`autorag-setup`; do not guess private providers or model IDs.

## Searching

```bash
autorag search "what were the key findings in the Q3 report" --top-k 5
```

AutoRAG returns curated, numbered knowledge units grounded in sources plus a
`sessionId` needed for feedback. Use:

- `--scope` to narrow to a configured virtual sub-path
- `--tags tag1,tag2` only when trusted datasource access is already configured
  server-side; tags narrow already-allowed results and never grant new access
- `--json` for structured output (`sessionId`, numbered `results`, `answer`)
- `--debug` only when diagnostics are needed

Do not bypass AutoRAG with ad hoc raw search when the user explicitly requested
the librarian agent. Search requires a resolvable orchestrator/explorer model
pair from config, flags, env, or the authenticated local runtime. When `autorag
search` fails for a model, provider, auth, timeout, or subagent reason, the
error output includes a hint pointing to `autorag health` for diagnosis.

Record feedback so retrieval memory learns which results were useful:

```bash
autorag feedback <sessionId> --useful 1,3 --not-useful 2
```

Supply at least one of `--useful` or `--not-useful`. Numbers refer to the
numbered knowledge units from that session's search output.

## Maintenance

```bash
autorag status
autorag health
autorag refresh
autorag refresh --method bm25,minsync
autorag watch --once
autorag watch
autorag refresh --force
autorag index rebuild --yes
autorag index reset --method bm25 --yes
autorag memory inspect
```

Use `refresh` after source documents change (parses sources and resyncs BM25 /
MinSync / datasources / optional Jikji prepare). Prefer bounded refresh over
reset. `watch --once` is the preferred single tick for scheduled jobs; long-running
`watch` keeps an fs event loop open for interactive sessions.
`--method <csv>` (e.g. `--method bm25,minsync,parsed`) restricts which methods
refresh/index run; when omitted all methods run. BM25 and MinSync are enabled
by default — no explicit configuration is needed for standard lexical + semantic
retrieval. MinSync uses a pre-installed binary (`autoInstall: false`); configure
`minSync.embedder` via `autorag init --embedder-*` flags for remote embedding.

### Keep indexes fresh on a schedule (agent responsibility)

There is **no always-on network service** in the CLI. Agents and operators must
install an OS-appropriate periodic job that runs a model-free index tick:

```bash
# Preferred scheduled command (single non-daemon refresh)
NODE_OPTIONS=--max-old-space-size=16384 autorag watch --once
# Equivalent one-shot refresh
NODE_OPTIONS=--max-old-space-size=16384 autorag refresh
```

Schedule guidance by OS (typical interval: every 15–30 minutes; never more often
than the corpus can finish refreshing):

| OS | Preferred scheduler | Pattern |
|---|---|---|
| macOS | `cron` or `launchd` (`~/Library/LaunchAgents`) | `*/30 * * * * ... autorag watch --once` or a KeepAlive=false StartInterval plist |
| Linux | `cron` / `systemd --user` timer | crontab `*/30 * * * *` or a oneshot service + timer |
| Windows | Task Scheduler | repeating task every 30 minutes running `autorag watch --once` under the user profile |

Rules for scheduled watch:

1. Prefer `autorag watch --once` (or `autorag refresh`) over a permanent
   long-running `autorag watch` daemon in user agents — daemons die on reboot
   and are harder to supervise from skill workflows.
2. Use the same config the search path uses (`~/.autorag/config.json` or an
   explicit `--config` / `AUTORAG_CONFIG`).
3. Raise Node heap for large home trees: `NODE_OPTIONS=--max-old-space-size=16384`.
4. Redirect logs somewhere under the home/user temp tree, never into source
   document folders.
5. After install/setup, **create or verify** the scheduled job before claiming
   ongoing indexing is covered. Re-check with `crontab -l`, `systemctl --user list-timers`,
   or Task Scheduler inspection when the user asks about freshness.
6. Do not schedule concurrent overlapping ticks; if a prior refresh is still
   running, skip or wait (lock/log rather than stampeding MiniSync/BM25 writers).

Destructive index commands:

- `autorag index reset --yes` removes parsed, BM25, and MinSync directories under
  workspace `.autorag` only. Add `--method bm25|minsync|parsed` to scope which
  indexes are removed (e.g. `--method bm25` removes only the BM25 index).
- `autorag index rebuild --yes` resets those indexes then forced-refreshes.
  `--method` scopes both the reset and the rebuild refresh.

Never run reset/rebuild against source documents. `memory inspect` is
read-only and path-opaque.

## Rules

- Use only configured and approved search paths.
- Never expose provider credentials or authentication payloads.
- Never invent or reveal private provider names or model IDs.
- Never treat a consumer subscription as API access without runtime evidence.
- Preserve real source mapping and numbered feedback identifiers.
- Prefer `--json` when another agent must consume the result programmatically.
