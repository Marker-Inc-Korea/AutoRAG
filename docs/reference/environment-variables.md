---
title: "Environment variables"
description: "Every environment variable the AutoRAG code reads, where it is read, what it changes, and its default."
---

# Environment variables

AutoRAG keeps all secrets in the environment and reads them at startup; nothing credential-shaped is written to the config file. This page lists every environment variable the shipped code reads, where it is read, what it changes, and its default. For the file half of the same settings see [Configuration file](configuration.md); for flags see [CLI commands](cli.md).

## Core paths and search settings

Read in `src/cli/config.ts` (via `resolveConfig`) and `src/config/home.ts`.

| Variable | Effect | Default |
|---|---|---|
| `AUTORAG_HOME` | Operator home directory; owns `config.json`, `memory.json`, `attachments/`, `automations.json`, the MCP owner token, and downloaded tools. Absolute paths resolve as given; relative values resolve against the current directory | `~/.autorag` |
| `AUTORAG_CONFIG` | Explicit config file path (same effect as `--config`; marks the path explicit for `init --force`) | unset |
| `AUTORAG_SEARCH_PATHS` | Comma-separated folders to index and search (same as `--search-paths`) | `["."]` |
| `AUTORAG_WORKSPACE` | Directory that owns `.autorag` state (same as `--workspace`) | current directory |
| `AUTORAG_MEMORY_PATH` | Retrieval-memory file (same as `--memory-path`) | `$AUTORAG_HOME/memory.json` |
| `AUTORAG_LANGUAGES` | Comma-separated OCR/parser languages (same as `--languages`); also read by `autorag init` | `ko,en` |
| `AUTORAG_PERMISSION_MODE` | Approval gate for file changes: `auto`, `ask`, or `full` (same as `--permission-mode`) | `auto` |
| `AUTORAG_MODEL_PROVIDER` | Model provider override (used with `AUTORAG_MODEL_ID`) | from config |
| `AUTORAG_MODEL_ID` | Model id override (used with `AUTORAG_MODEL_PROVIDER`) | from config |

Precedence is flag → environment variable → config file → default. `AUTORAG_MODEL_PROVIDER` and `AUTORAG_MODEL_ID` must be set together.

## Authentication and cloud

| Variable | Read where | Effect | Default |
|---|---|---|---|
| `AUTORAG_AUTH_MODE` | `src/cloud/credential.ts`, `src/cli/config.ts`, `src/agent/local-model.ts`, `src/agent/decision-backends.ts` | When set to `environment`, stored credentials (pi `auth.json`, OAuth, the Dazzi `/login` token) are ignored, and external model-config discovery is disabled; only environment variables authenticate | unset |
| `AUTORAG_BASE_URL` | `src/cloud/config.ts` | Server root for the hosted Dazzi Models plan; the API is served under `<root>/v1` | `https://api.dazziapp.com` |

## Update check

| Variable | Read where | Effect | Default |
|---|---|---|---|
| `AUTORAG_NO_UPDATE_CHECK` | `src/cli/update-check.ts` | Any non-empty value other than `0` skips the npm version lookup | unset (check runs) |
| `AUTORAG_UPDATE_CHECK_URL` | `src/cli/update-check.ts` | Registry endpoint override for the `latest` version | `https://registry.npmjs.org/@autorag/librarian/latest` |

## Model and decision-model credentials

Chat and decision credentials are read through pi-ai's provider environment conventions (`findEnvKeys`/`getEnvApiKey`) and by the Jev backend resolver; a configured endpoint with a `baseUrl` instead reads the variable named by `model.apiKeyEnv` (which defaults to `${PROVIDER}_API_KEY`).

| Variable | Effect |
|---|---|
| `OPENROUTER_API_KEY` | Credential for the `openrouter` provider (chat and Jev backend) |
| `OPENAI_API_KEY` | Credential for the `openai` provider |
| `ANTHROPIC_API_KEY` | Credential for the `anthropic` provider |
| `GEMINI_API_KEY`, `GOOGLE_API_KEY` | Credential for the Google/Gemini provider |
| `XAI_API_KEY` | Credential for the `xai` provider |
| `TYPESAFE_API_KEY` | Credential for the `typesafe` Jev backend |
| `AI_GATEWAY_API_KEY` | Credential for the `vercel-ai-gateway` Jev backend |
| `CLOUDFLARE_API_KEY` | Credential for the `cloudflare-workers-ai` Jev backend |
| `OPENCODE_API_KEY` | Credential for the `opencode` Jev backend |
| `<PROVIDER>_API_KEY` | Generic convention for a configured endpoint's provider |
| the var named by `minSync.embedder.apiKeyEnv` | Bearer key for a remote embedder; passed to the MinSync child as `OPENAI_API_KEY` |

A provider is "available" exactly when its documented variable holds a non-empty value; `autorag health` and `autorag models list --available` report this without printing the value.

## Web search and page fetch

| Variable | Read where | Effect | Default |
|---|---|---|---|
| `SEARXNG_ENDPOINT` | `src/web/search/providers/searxng.ts` | Base URL of a self-hosted SearXNG instance — the only env-gated search provider | unset (provider unavailable) |
| `SEARXNG_TOKEN` | same | Optional bearer token | unset |
| `SEARXNG_BASIC_USERNAME`, `SEARXNG_BASIC_PASSWORD` | same | Optional Basic auth; both are required together | unset |
| `SEARXNG_ENGINES` | same | Restrict SearXNG engines | unset |
| `SEARXNG_CATEGORIES` | same | Restrict SearXNG categories | unset |
| `SEARXNG_SAFESARCH` | same | Safe-search level: `0` off, `1` moderate, `2` strict | unset |
| `ANTHROPIC_SEARCH_MODEL` | `src/web/search/providers/anthropic.ts` | Override the model used for Anthropic's built-in web search | provider default |
| `GEMINI_SEARCH_MODEL` | `src/web/search/providers/gemini.ts` | Override the model used for Gemini grounding | provider default |
| `FIRECRAWL_API_KEY` | `src/web/fetch/html-renderer.ts` | Key for the Firecrawl HTML renderer | unset (renderer skipped) |
| `JINA_API_KEY` | `src/web/fetch/html-renderer.ts` | Optional key for the Jina reader (`https://r.jina.ai/`, keyless without it) | unset |
| `AUTORAG_WEB_BROWSER_PATH` | `src/web/search/browser-loader.ts` | Path to a browser binary for bot-challenge escalation | unset |
| `CHROME_PATH`, `PUPPETEER_EXECUTABLE_PATH` | same | Fallback browser paths, checked after `AUTORAG_WEB_BROWSER_PATH` | unset |

## MCP server (`autorag-mcp`)

Read in `src/mcp/index.ts`.

| Variable | Effect | Default |
|---|---|---|
| `AUTORAG_MCP_TOKEN` | Owner token every HTTP request must present as `Authorization: Bearer <token>` (at least 32 characters). When unset, a random token is generated and stored owner-only at `$AUTORAG_HOME/mcp-http-token` | generated token file |
| `AUTORAG_MCP_TOOLS` | Comma-separated allowlist of tools to expose | all tools |
| `AUTORAG_MCP_READ_ONLY` | stdio transport only: `1`/`true` restricts the server to read-only tools; unset exposes read-write tools. The HTTP transport ignores it and uses `--allow-write` (which defaults to read-only) | unset (read-write) |
| `AUTORAG_MCP_PUBLIC_URL` | `https` origin a cloud agent reaches the HTTP server at (tunnel/proxy); enables the OAuth endpoints connector-only clients need | unset |

## Datasource CLIs

AutoRAG's connector-backed datasources run an external CLI that owns its own credentials and cache. These variables are read by that CLI (AutoRAG forwards them) or by AutoRAG when locating/pointing it.

| Variable | Read where | Effect |
|---|---|---|
| `HISTER_CONFIG` | `src/datasource/skills/hister/setup.ts` | Path to Hister's `config.yml` when it is not at a standard location |
| `MAILCRAWL_DATA_DIR` | `src/datasource/skills/mailcrawl/client.ts` | mailcrawl archive/data directory |
| `WOMBAT_DATA_DIR` | `src/datasource/skills/wombat/agents.ts`, `client.ts` | wombat data directory (`~/.wombat` by default) |
| `WOMBAT_EMBEDDER` | `src/datasource/skills/wombat/client.ts` | Embedder selector passed to wombat |
| `MINSYNC_SERVER` | `src/datasource/skills/wombat/client.ts` | MinSync server mode forwarded to the wombat child |
| `INL_CACHE_DIR` | `src/datasource/crawler-client.ts`, `src/datasource/skills/*/client.ts` | im-not-llm model-cache root, forwarded to every CLI that loads or discovers models so a relocated inl root reaches it (default `~/.cache/im-not-llm/models`) |
| `CLAWGALLERY_CONFIG_DIR` | `src/datasource/skills/clawgallery/client.ts` | ClawGallery config directory |
| `KATOK_EMBED_API_KEY` | `src/datasource/embedding-env.ts` | Embedder key for the lazykatok child when it embeds for itself |

Each crawler CLI also reads its own `SLACRAWL_*`, `MAILCRAWL_*`, `WOMBAT_*`, `CLAWGALLERY_*`, `HISTER_*`, and `KATOK_*` variables for authentication and transport; see the datasource pages for the connector-specific names.

::: tip **Embedder key for external CLIs**
When mailcrawl or lazykatok embeds for itself, AutoRAG points it at the same backend as the rest of AutoRAG by setting that CLI's documented `*_EMBED_*` / `KATOK_EMBED_*` variables in the child environment. The remote key travels as the value of the variable named by `minSync.embedder.apiKeyEnv`; you only need to set that one variable yourself.
:::

## Standard variables the code reads

AutoRAG also reads conventional variables for platform and home discovery — they are not AutoRAG-specific and need no setup:

| Variable | Read where | Effect |
|---|---|---|
| `HOME`, `USERPROFILE` | `src/config/home.ts`, `src/embedding/daemon.ts` | Base for `~/.autorag` and im-not-llm's daemon state root `~/.cache/im-not-llm/serve`; an absolute `HOME`/`USERPROFILE` wins, else `os.homedir()` |
| `PATH` | binary resolution in `src/minsync`, `src/jikji`, `src/cli/setup.ts`, `src/automation/service.ts` | Locates `minsync`, `jikji`, crawler CLIs, and the automation service |
| `CARGO_HOME` | `src/datasource/skills/wombat/binary.ts`, jikji installer | cargo install location when auto-installing a Rust tool |
| `XDG_DATA_HOME`, `XDG_CONFIG_HOME` | `src/cli/setup.ts`, `src/automation/service.ts`, `src/datasource/skills/hister/setup.ts` | XDG paths for tools and the automation service on Linux |
| `LOCALAPPDATA`, `APPDATA` | `src/embedding/daemon.ts`, `src/datasource/skills/hister/setup.ts` | Windows per-user data locations |
| `NODE_ENV` | installers in `src/convert`, `src/minsync` | Installer behavior in non-production runs |

Variables that exist only for the repository's QA and live-E2E scripts (for example `AUTORAG_EMBED_ENDPOINT`, `AUTORAG_CALC_QA_*`, `E2E_*`, `MINSYNC_BINARY`, `AUTORAG_OPENAI_API_KEY`) are not read by the shipped CLI and are intentionally omitted here.
