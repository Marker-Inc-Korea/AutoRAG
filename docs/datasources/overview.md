---
title: "Overview"
description: "What a datasource skill is, every builtin datasource with its backend CLI, and how cloud folders and local documents are indexed instead."
---

# Overview

This page explains what a datasource skill is, lists every builtin datasource with its backend and authentication owner, and points to the page for each one. Use it to decide which datasource to connect and how it differs from plain local documents.

## What a datasource skill is

A datasource skill connects AutoRAG to an external, server-bound source of retrieval evidence: a chat archive, a mailbox, a browser history server, a SaaS tenant. A skill is two things at once:

1. an indexing hook (`index()` plus `polling()` metadata); and
2. a retrieval-method factory (`retrievalMethods()`), whose methods are registered in the ordinary AutoRAG pipeline.

Skills are the trusted bridge to one external system. They are constructed only from the trusted `datasources` section of the config ([`src/datasource/skills/factory.ts`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/datasource/skills/factory.ts)); model and tool arguments never decide which connections exist. Data stays in each tool's native store — AutoRAG does not copy foreign databases into a centralized store. See [Connecting a datasource](./connecting.md) for the config shape and [Search semantics](./search-semantics.md) for how results are named and filtered.

Each skill also returns `describeSources()` (what data exists) and a progressive-disclosure `skillManifest()` whose body the model loads on demand. Failures never throw: `index()` returns a fail variant with a diagnostic.

## Builtin datasources

Every name below is a builtin template accepted under `datasources`. `kakao` is built only on macOS; every other name is available wherever its backing program runs.

| Name | Backend | Platform | Authentication owner | Page |
|---|---|---|---|---|
| `telegram` | `telecrawl` CLI | all | `telecrawl` (its own archive) | [Telegram](./telegram.md) |
| `whatsapp` | `wacrawl` CLI | all | `wacrawl` | [WhatsApp](./whatsapp.md) |
| `slack` | `slacrawl` CLI | all | `slacrawl` (desktop cache or token) | [Slack](./slack.md) |
| `discord` | `discrawl` CLI | all | `discrawl` (token-free wiretap) | [Discord](./discord.md) |
| `clawgallery` | `clawgallery` CLI | all | `clawgallery` | [ClawGallery](./clawgallery.md) |
| `notion` | `notcrawl` CLI | all | `notcrawl` | [Notion](./notion.md) |
| `lark` | `lark-cli` (official) | all | `lark-cli` keychain | [Lark / Feishu](./lark.md) |
| `hister` | Hister server (HTTP) | all | Hister (`tokenEnv`) | [Browser history (Hister)](./hister.md) |
| `kakao` | `lazykatok` CLI | macOS only | `lazykatok` | [KakaoTalk](./kakao.md) |
| `github` | GitHub REST API | all | AutoRAG (`GITHUB_TOKEN`) | [GitHub](./github.md) |
| `github-gist` | GitHub REST API | all | AutoRAG (`tokenEnv`, `gh` fallback) | [GitHub Gists](./github-gist.md) |
| `mailcrawl` | `mailcrawl` CLI | all | `mailcrawl` (Himalaya or files) | [Mail (mailcrawl)](./mailcrawl.md) |
| `obsidian` | MinSync (no CLI) | all | AutoRAG-owned index | [Obsidian](./obsidian.md) |
| `rss` | Native HTTP poller | all | none | [RSS](./rss.md) |
| `wombat` | `wombat` CLI + MinSync | all | `wombat` | [Coding-agent sessions (wombat)](./wombat.md) |

CLI-backed skills own their archive, index, and credentials, so `autorag setup` never requires an environment token for them. AutoRAG-owned stores (`obsidian`, `github-gist`, chat sidecars) embed through the one shared embedder; see [Connecting a datasource](./connecting.md).

## Cloud drives and local documents

Cloud storage and network shares are not datasource skills. They are folders: add the provider's desktop-sync folder to `searchPaths` and the parsed mirror indexes it like any other folder. The sync app owns the mount, its authentication, and its placeholder lifecycle. See [Cloud and network folders](../indexing/cloud-folders.md).

Local documents (the filesystem trees under `searchPaths`) are served by the same file index, not by a datasource skill. Google Docs, Sheets, and Slides that appear in a sync folder as pointer files are a known gap.

## Migrating from removed templates

Two former templates were removed. A config entry that still names one is skipped, never migrated or shimmed, and config resolution does not fail:

| Removed template | Reported as | What to do |
|---|---|---|
| `spotlight` | `retired-datasource-skill` warning | Delete the entry; file content stays searchable through `searchPaths`. |
| `cloud-drive` | `unknown-datasource-skill` warning | Delete the entry; add the provider's sync folder to `searchPaths` and refresh. |

The factory keeps a `RETIRED_TEMPLATES` set containing only `spotlight`, which is why it gets the dedicated `retired-datasource-skill` diagnostic while every other unrecognized name (including `cloud-drive`) is collected as `unknown` and reported with `unknown-datasource-skill`. Both warnings are emitted at agent construction and do not prevent unrelated commands from running.

## Where to go next

- [Connecting a datasource](./connecting.md) — config keys, aliases, embedding, timeouts, retries.
- [Search semantics](./search-semantics.md) — time windows, scope, result identity, diagnostics.
- [Hybrid search for chat archives](./chat-hybrid-search.md) — the sidecar behind Slack, Telegram, WhatsApp, KakaoTalk, and Discord ranking.
