---
title: "Overview"
description: "What AutoRAG Agent is: a personal librarian that searches local files and connected datasources in place and cites every answer."
---

# Overview

AutoRAG Agent is a personal librarian agent that finds, reads, and cites what is on your computer. This page explains what the engine does, which surfaces it runs on, and what stays on your machine.

## What the librarian does

AutoRAG is a customized [Pi](https://github.com/earendil-works/pi-mono) agent configured into a librarian. One configured model runs a single loop over a search request: it chooses retrieval methods, reads source files directly, judges the evidence, and curates a final answer whose claims are linked to the passages behind them. It is a library and a thin CLI, not a service you stand up.

The primary target is non-code document retrieval — manuals, legal documents, internal wikis, meeting notes, research literature — but code repositories work too. Its value is the exploration, retrieval-method, and curation layer on top of raw search, not search itself.

**Why it exists.** Raw search returns file paths and matching lines. AutoRAG runs the whole workflow instead: search across multiple retrieval methods, read the promising sources directly, judge and curate, deliver numbered knowledge units grounded in the source, and remember the evidence that judged well so later questions start warm.

**In-place federation.** AutoRAG does not migrate your data to search it. Local documents are parsed into a workspace mirror under `<workspace>/.autorag` and indexed by MinSync; external sources such as KakaoTalk, Discord, Slack, WhatsApp, Telegram, Notion, and mail are read through the CLI or connector that already owns their store. Results keep source-native identities such as `/kakao/<instance>/chunks/<chunk>`, and secrets stay with the tool that owns them. See [Datasources](../datasources/overview.md).

## Surfaces

One engine, several ways in. The desktop app is the polished entry point; everything else drives the same librarian.

| Surface | Use it for | Start here |
|---|---|---|
| Desktop app (Dazzi) | Browsing and asking in one window | [dazziapp.com](https://dazziapp.com) · [NomaDamas/AutoRAG-app](https://github.com/NomaDamas/AutoRAG-app) |
| Terminal UI | Interactive, steerable search sessions | [CLI and TUI](../reference/cli.md) |
| Messenger | Asking the librarian from a chat app | [Messengers](../agents/messengers.md) |
| CLI | Scripts, indexing, health checks, repair | [CLI reference](../reference/cli.md) |
| MCP server (Lite) | Feeding model-free retrieval to Claude Code, Codex, and other agents | [MCP server](../agents/mcp-server.md) |
| TypeScript API | Embedding the librarian in your own app | [TypeScript API](../reference/typescript-api.md) |
| Agent skills | Letting a coding agent install and drive AutoRAG | [Agent skills](../agents/skills.md) |

```text
Agent Tools                 AutoRAGAgent (customized Pi agent)
┌──────────────────┐        ┌──────────────────────────────────┐
│ bash / jikji_find │        │ Retrieval Memory (judged evidence)│
│ search_all_docs   │  ────▶ │ Curation Layer (model extraction) │
│ semantic_search   │        │ check_memory (advisory reference) │
│ search_datasource │        │ Datasource Skills (external stores)│
│ scan_duplicates   │        │ Retrieval Registry (pluggable)    │
│                   │        │ Result Merger (cross-method)      │
└──────────────────┘        └──────────────────────────────────┘
```

## Privacy model

- **Embedded on your machine.** Semantic indexing runs on MinSync's local `Qwen/Qwen3-Embedding-0.6B` engine (1024 dimensions), which MinSync provisions and runs in its own cache (`~/.cache/minsync/models`). AutoRAG owns no embedding model, and file contents are never sent to a remote embedder unless you explicitly configure a remote `minSync.embedder` endpoint.
- **Keys stay out of config.** Sign-in tokens and provider API keys live in the platform's encrypted store or Pi's `auth.json`; a config file records only environment-variable names such as `apiKeyEnv`. See [Models and providers](models.md).
- **Bring your own model.** You can use the hosted [Dazzi Models](models.md) plan, or connect your own provider credential and pay that provider directly.
- **What leaves the machine.** The librarian's chat model, the required Jev decision model, an optional rerank model, and `web_search`/`web_fetch` calls all reach the network you configure for them. A search question is sent to the decision model. Retrieval, indexing, and local document reading stay local.

## Legacy AutoRAG

This repository also contains the original Python AutoRAG — the RAG AutoML tool that automated finding an optimal RAG pipeline. It lives in [`legacy/`](https://github.com/Marker-Inc-Korea/AutoRAG/tree/main/legacy), still installs as `pip install AutoRAG`, and is maintained in maintenance mode; new feature work is AutoRAG Agent. See the [legacy README](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/legacy/README.md).
