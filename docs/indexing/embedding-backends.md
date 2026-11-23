---
title: "Embedding backends"
description: "im-not-llm owns the embedding model and daemon; choose a local daemon, the in-process engine, a remote endpoint, or a sparse-only index."
---

# Embedding backends

This page explains which embedding model AutoRAG uses, how it resolves a backend, how to run the embedding daemon, and where the embedding state lives.

## im-not-llm owns the embedding model

AutoRAG owns no embedding model, no llama.cpp, and no GGUF assets, and it does not download, cache, import, or verify model weights. The im-not-llm daemon (`inl`) provisions `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions in its own cache at `~/.cache/im-not-llm/models` and runs the engine. MinSync links the same `inl-server` and serves the same models through the same cache, so the two share one runtime. The default local path needs no Ollama, no TEI adapter, and no API key.

MinSync is required. AutoRAG auto-installs its binary when no usable executable is found on `PATH` or in the workspace cache. `minSync: false` and `minSync.enabled: false` are rejected as config errors; `minSync.autoInstall: false` only stops the auto-install. A search or query never starts indexing infrastructure, and never answers from the other retrieval paths when MinSync is missing — it fails with a `MinSyncRequiredError` and a non-zero exit.

One shared embedder serves everything that embeds inside AutoRAG: MinSync indexing and query, retrieval-memory similar-question search, the chat-crawler sidecars (Slack, Telegram, WhatsApp, Kakao), the github-gist vector index, and the endpoint external CLIs such as discrawl are pointed at.

## Backend resolution order

The shared embedder resolves a backend in this order. `autorag setup` and `autorag status` report which one is alive (`daemon`, `in-process`, or `remote`) with provider, model, and dimension, or why none is alive. Neither starts anything.

1. **Remote endpoint.** If `minSync.embedder` names a remote endpoint, it is used for everything; no daemon is started and no local model is loaded. `id` names the protocol and model (`openai:<model>`, `tei:<model>`, or any id together with `baseUrl`). OpenAI uses `POST {baseUrl}/v1/embeddings`; TEI uses `POST {baseUrl}/embed`; `openai:` without a `baseUrl` means `https://api.openai.com`. Optional fields: `apiKeyEnv` (the **name** of an environment variable holding the bearer token, never the token), `dimension`, `batchSize`, and `timeoutMs`.
2. **`inl serve` daemon.** Otherwise a running im-not-llm daemon is preferred. It is loopback only. It serves TEI `POST /embed`, OpenAI `POST /v1/embeddings`, and `GET /health`. Running instances are discovered through `~/.cache/im-not-llm/serve/<instance>/` (`host`, `port`, `pid`, `lock`); a matching daemon is found by the `kind` and `model_id` its `/health` reports, never by a directory name. AutoRAG never starts or stops it for you; it is looked up again on every call, so one you start or stop while AutoRAG runs is noticed. `minsync serve` is a compatible launcher for the same state root.
3. **In-process engine.** When no daemon answers, AutoRAG's own consumers can load MinSync's embedding engine inside the AutoRAG process through the optional Node-API addon `@nomadamas/minsync-native`. It runs the same engine and weights as `minsync`, so vectors are bit-for-bit identical to the daemon's. It is not bundled. This is separate from the `minsync` CLI, which embeds in-process by itself for the MinSync datasource.

Configure a remote endpoint through config or `autorag init` flags:

```json
{
  "minSync": {
    "embedder": {
      "id": "openai:<model-id>",
      "baseUrl": "https://<host>/v1",
      "apiKeyEnv": "<API_KEY_ENV_VAR>",
      "dimension": 1024,
      "batchSize": 64
    }
  }
}
```

```bash
autorag init \
  --search-paths /path/to/docs \
  --embedder-id "openai:<model-id>" \
  --embedder-base-url "https://<host>/v1" \
  --embedder-api-key-env "<API_KEY_ENV_VAR>" \
  --embedder-dimension 1024 \
  --embedder-batch-size 64
```

`--embedder-query-prefix`, `--embedder-passage-prefix`, and `--embedder-timeout-ms` set the remaining allowlisted fields. Only environment-variable names are ever stored, never key values.

## Query instruction and passage prefix

Qwen3-Embedding is instruction-aware: a search query is embedded as `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:<query>` and a document is embedded raw. AutoRAG applies this on every semantic path:

- MinSync's own index and search get it through `[embedder].query_prefix` in MinSync's config file, which MinSync prepends to the query only.
- AutoRAG-owned stores (chat sidecars, GitHub Gist) embed through the shared embedder: the embedder prepends the prefix itself, so neither the daemon nor a remote server is asked to.
- mailcrawl receives the prefix through `MAILCRAWL_QUERY_PREFIX`.
- Hister keeps its own vector store and is pointed at the shared daemon through its own `config.yml` `semantic_search` block; `autorag setup --sync-hister` writes those keys from the resolved shared endpoint, byte for byte.
- Similar-past-question lookup compares a question with questions, so both sides are embedded the same way, without the instruction.

The default applies only to MinSync's local Qwen3 (no `id`, or `native:Qwen/Qwen3-Embedding-0.6B`, and no `baseUrl`). An operator-set `minSync.embedder.queryPrefix` or `passagePrefix`, including `""`, always wins, and a remote endpoint or another model gets no default prefix.

## Running the embedding daemon

Semantic search over everything except the MinSync datasource needs an embedding backend that is alive: a remote endpoint, a running `inl serve`, or the optional in-process addon. Without one, memory, the chat-sidecar semantic search, gist, discrawl, mailcrawl, lazykatok, and Hister fall back to keyword search or report `embedding-unavailable`; the MinSync datasource keeps working because `minsync` embeds in-process by itself.

Without a remote endpoint, keep the daemon running. From inl-server 0.3.4 `inl serve` serves until stopped by default, and a restarted instance comes back on the port it used before. Older versions (and `minsync serve` before the MinSync release that takes 0.3.4) exit after 300 s without a request, which frees the ~2.8 GB model but makes the next semantic call fail until the daemon is started again, and choose a new port on every start; pass `--idle-timeout 0` there. Do not pass `--state-dir`: only the default state directory is discovered.

**macOS (launchd).** Save as `~/Library/LaunchAgents/dev.im-not-llm.inl-serve.plist` (replace `YOU` and the path from `which inl`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>dev.im-not-llm.inl-serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/.cargo/bin/inl</string>
    <string>serve</string>
    <string>--idle-timeout</string>
    <string>0</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
```

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.im-not-llm.inl-serve.plist
launchctl kickstart -k gui/$(id -u)/dev.im-not-llm.inl-serve   # restart
launchctl bootout gui/$(id -u)/dev.im-not-llm.inl-serve        # stop and unload
```

**Linux (systemd user unit).** Create `~/.config/systemd/user/inl-serve.service` with `ExecStart=%h/.cargo/bin/inl serve --idle-timeout 0` and `Restart=always`, then run `systemctl --user enable --now inl-serve`.

**Without a service manager**, run `inl serve --idle-timeout 0` in a terminal, or press **Ctrl+Shift+E** / type `/embedding on` in `autorag tui`. If you do not have `inl`, `minsync serve --idle-timeout 0` serves the same state root.

## TUI footer and the /embedding command

`autorag tui` shows the embedding state in the footer and re-checks it every 15 s and after each toggle:

| Footer | Meaning |
| --- | --- |
| `embedding: daemon ● up` | An `inl serve` daemon answers `GET /health`. |
| `embedding: in-process ●` | No daemon, but the in-process addon is loadable. |
| `embedding: remote ● up` / `remote ○ down (…)` | The configured remote's health. |
| `embedding: off ○ (/embedding on, or ctrl+shift+e, starts inl serve)` | Nothing can embed. |

**Ctrl+Shift+E** and **`/embedding`** do the same thing. The key toggles; the command takes `on`, `off`, `toggle` (the default with no argument), or `status` (a one-line report without changing anything). `on` starts the daemon (`inl serve --idle-timeout 0`, so it stays on until you stop it, preferring an `inl` binary on `PATH` and falling back to `minsync serve`) and `off` stops it with `SIGTERM`. With a remote endpoint configured they only explain that there is no local daemon to manage. A daemon started by a launchd service is stopped too, and launchd then restarts it (`KeepAlive`); use `launchctl bootout` to stop it for good.

## MinSync workspace state

By default the MinSync workspace is `<workspace>/.autorag/minsync`. MinSync keeps its own state in a `.minsync` subdirectory of that workspace, alongside the staged `files/` tree:

```text
<workspace>/.autorag/minsync/.minsync/autorag-embedding-identity.json
<workspace>/.autorag/minsync/.minsync/cursor.json
<workspace>/.autorag/minsync/.minsync/config.toml
```

The identity file records provider, model, immutable model revision, dimension, query prefix, and passage prefix. The cursor records sync progress. MinSync's config file holds the allowlisted embedder fields AutoRAG rewrites so your configuration wins over `minsync init` defaults. An explicit `minSync.workspacePath` setting replaces the default `<workspace>/.autorag/minsync` location, and MinSync then keeps its `.minsync` state under that configured path instead.

## Sparse-only index

Set `"minSync": { "embedder": { "id": "none" } }` to build a SPLADE-only index: MinSync encodes sparse vectors and no dense ones, so no dense model is downloaded or run. Every MinSync search is then a sparse search, an Obsidian vault offers only `obsidian-sparse`, the shared embedder reports no backend, and neither `autorag status` nor `autorag setup` warns about a missing embedder. See [Sparse index and chunking](sparse-and-chunking.md).

## Model cache

im-not-llm provisions `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions in `~/.cache/im-not-llm/models`. The daemon and the in-process addon read that cache; AutoRAG makes no model download of its own. There is no `AUTORAG_HOME` model or runtime cache and no AutoRAG prefetch, import, or verify command for embedding weights. `autorag models list` lists chat models only.

A 768-dimensional workspace cannot be kept by choosing a profile: reindex for the default MinSync identity, or configure an explicit remote endpoint (`minSync.embedder.id` and `baseUrl`). AutoRAG writes only the supported MinSync embedder fields and never writes secrets.

## Rebuilding after the embedding gateway

Daemon and in-process vectors are identical, so stores keyed on that identity do not re-embed when the daemon appears or disappears. The vectors from the removed llama-server runtime are not compatible.

- Memory vector stores (`memory.json.vectors.db`), chat-sidecar semantic stores, and the github-gist `vectors.json` re-embed automatically on the next refresh because their recorded identity changed.
- A MinSync workspace created through the old gateway profile (a `tei:Qwen3-Embedding-0.6B-Q8_0.gguf` id, or an old `profile` field) must be rebuilt explicitly:

  ```bash
  autorag index rebuild --method minsync
  ```

  `autorag index reset --method minsync --yes` followed by `autorag refresh --method minsync` has the same effect. Legacy `profile` and `.gguf` ids in `config.json` are ignored on read, without an error.

## Legacy Ollama/TEI adapter

The repository's Python adapter is no longer the product path; keep it only for a legacy workspace or manual compatibility QA. It translates Ollama's `/api/embeddings` response to MinSync's TEI `/embed` response and must remain bound to loopback:

```bash
ollama pull embeddinggemma:latest
ollama serve
OLLAMA_EMBEDDINGS_URL=http://127.0.0.1:11434/api/embeddings \
  python3 scripts/manual-qa/ollama-tei-adapter.py
```

A legacy/manual QA workspace may be initialized with `--embedder-id tei:<model> --embedder-base-url http://127.0.0.1:18080 --embedder-dimension 768`. Do not use this path as a fresh-install requirement or as an implicit fallback: AutoRAG never silently switches an existing vector store between Ollama/TEI and MinSync's local embedder.

## Related pages

- [Overview](overview.md) — the pipeline and workspace layout.
- [Sparse index and chunking](sparse-and-chunking.md) — SPLADE, sparse-only workspaces, and chunk size.
- [Refresh operations](refresh-operations.md) — embedding diagnostics and failure behavior.
