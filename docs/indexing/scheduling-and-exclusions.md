---
title: "Scheduling and exclusions"
description: "Limit background indexing by power and time, exclude files and folders, drop exact duplicates, and run the watch daemon."
---

# Scheduling and exclusions

This page shows how to keep indexing under your control: gate background refreshes by power source and local time, exclude files and folders, drop exact duplicates, and run a watch daemon.

## Indexing only when you want it

Background indexing — `autorag watch`, the auto-refresh tick, and the filesystem watcher — can be limited by power source and local time. An explicit `autorag refresh` is never gated.

```json
{ "refresh": { "schedule": { "power": "ac", "windows": ["22:00-07:00"], "days": ["mon", "tue", "wed", "thu", "fri"] } } }
```

- `power`: `any` (default) ignores power; `ac` pauses while on battery and runs only while plugged in. An unknown power source (a desktop, or a failed probe) counts as plugged in, so a failed probe never silently stops indexing.
- `windows`: local-time `HH:MM-HH:MM`, start inclusive and end exclusive. `22:00-07:00` wraps midnight and belongs to the day it starts on. Omitted means all day.
- `days`: `sun` through `sat`; omitted means every day.

`autorag watch` accepts the same policy as flags, which override the config: `--power`, `--window` (repeatable), and `--days` (CSV). A `--once` tick that falls outside the schedule prints `skipped` (and `skipped: true` with `--json`) and exits 0. A long-running watcher defers changes seen while the schedule is closed and indexes once it opens.

A malformed `refresh.schedule` fails open — a bad edit does not stop indexing — but the config loader rejects it loudly where a user can see it.

## Excluding files and folders

Leave generated workspaces or archives out when initializing with repeated `--exclude`:

```bash
autorag init --search-paths "$HOME/Documents" \
  --exclude "$HOME/Documents/Build output" \
  --exclude "$HOME/Documents/Archive, old"
```

Each `--exclude` argument is one complete path; spaces and commas are preserved. Relative CLI paths resolve from the current directory, even when `--workspace` points elsewhere. The generated config stores absolute paths.

For an existing configuration, edit the `excludePaths` array and run `autorag refresh` to remove excluded documents from the parsed mirror and update the indexes:

```json
{
  "searchPaths": ["/path/to/docs"],
  "excludePaths": ["/path/to/docs/Build output", "/path/to/docs/**/*.log"]
}
```

An entry without `*`, `?`, or `[` is a literal file or directory path: it excludes that exact path and all descendants of an excluded directory. An entry with a glob character is a glob over the whole absolute path (`**/node_modules/**`, `**/Downloads/**`, `**/*.log`) and is never resolved against a directory. Relative paths in JSON resolve against the configured `workspacePath`, or the config file's directory when `workspacePath` is absent.

Exclusions apply before the walk descends, so excluded contents do not consume the walk budget. Native fsearch and Everything results are filtered before pagination, including exact file exclusions and stale index entries; a native filename index may retain metadata for an individually excluded file, but the filter keeps it out of search results. Excluded files never enter the parsed mirror, so MinSync never chunks them.

## Exact-duplicate exclusion

`excludeExactDuplicates` (default `true`) invokes the external `dupey` CLI before parsed-mirror indexing. AutoRAG groups files by their exact canonical-text hash and keeps the newest copy, excluding older byte-identical copies from the mirror with a `duplicate-excluded` informational diagnostic. Set `"excludeExactDuplicates": false` to index every copy. Missing dupey never fails a refresh: duplicate exclusion is skipped and the rest of the run proceeds.

Cloud and network files join the same exclusion through stored fingerprints rather than being read again (see [Cloud and network folders](cloud-folders.md)).

## Watch daemon

`autorag watch` keeps the parsed mirror and configured indexes current from filesystem events. It is debounced, backpressure-limited (one refresh in flight plus at most one coalesced rerun), stoppable, and safe under rapid change bursts. It excludes `.autorag`, `.git`, and `node_modules` segments and does not follow symlinks.

```bash
autorag watch                       # long-running recursive watch
autorag watch --once                # one tick, for cron / launchd / Task Scheduler
```

| Flag | Meaning |
| --- | --- |
| `--once` | Run one refresh tick and exit. |
| `--immediate` | Refresh once before reading fs events (default true). |
| `--debounce-ms <n>` | Debounce window for fs events (default 1500). |
| `--force` | Re-index even when the mirror looks unchanged. |
| `--power <p>` | `any` or `ac`; with `ac`, index only while plugged in. |
| `--window <HH:MM-HH:MM>` | Index only inside this local-time window; repeatable. |
| `--days <csv>` | Index only on these weekdays. |

Each configured search root gets a watcher, up to a watcher cap; directories beyond it fall back to manual or polling refresh and report a `watch-limited` warning. A watcher that cannot be established reports `watch-failed`. A long-running watch handles `SIGINT`/`SIGTERM` itself and exits 0; the one-shot `--once` tick exits 130 or 143 when a signal stops it.

## Related pages

- [Reference: configuration](../reference/configuration.md) — the full `refresh` and `minSync` config schema.
- [Refresh operations](refresh-operations.md) — diagnostics and failure behavior.
