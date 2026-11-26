---
title: "Models and providers"
description: "Configure the chat model, the Jev decision model, and credentials, including the hosted Dazzi Models plan and bring-your-own keys."
---

# Models and providers

AutoRAG uses one configured chat model for the whole search loop — routing, retrieval tool calls, reading, and curation — plus a required decision model (Jev) and an optional rerank model. This page covers discovering models, writing the config, credentials, the hosted Dazzi Models plan, and reranking.

Prefer a chat model with reliable tool calling and structured output, enough context for source excerpts, high output TPS, and low first-token latency: retrieval commonly spans several model turns, so latency compounds. A larger reasoning model pays off only when difficult synthesis matters more than speed.

## Discover models

```bash
autorag models list                        # chat models the pi runtime can resolve
autorag models list --available            # only providers with complete auth configured
autorag models list --provider openrouter  # one provider
autorag models list --decision --available # decision-model candidates (jev.backend / jev.model)
autorag models list --format json
```

`autorag models list` discovers models through the pi runtime (built-in catalog plus `~/.pi/agent/models.json`, custom, and extension providers) and reports whether each provider has complete auth. It never prints credential values.

## Configure the model

Record the model in the trusted config. `provider`/`id` alone resolves against the pi runtime catalog; add `api`, `baseUrl`, and `apiKeyEnv` for a custom or non-catalog endpoint.

```json
{
  "model": {
    "provider": "openrouter",
    "id": "<model-id>",
    "api": "openai-completions",
    "baseUrl": "https://openrouter.ai/api/v1",
    "apiKeyEnv": "OPENROUTER_API_KEY"
  }
}
```

| Field | Meaning |
|---|---|
| `provider` | pi provider id (`openai`, `anthropic`, `openrouter`, `dazzi`, …), or a custom provider. |
| `id` | Catalog model id, or any id when `api`/`baseUrl` are spelled out. |
| `api` | Wire protocol: `openai-completions`, `openai-responses`, `anthropic-messages`, `openai-codex-responses`, or `azure-openai-responses`. |
| `baseUrl` | Endpoint override (wins over the provider's registered endpoint). |
| `apiKeyEnv` | **Name** of the environment variable holding the key. Never the key itself. |

When `provider`/`id` names a catalog model, the catalog entry stays the base and your `baseUrl`, `api`, and any declared `reasoning`, `input`, `contextWindow`, or `maxTokens` override only those fields. An id outside the catalog gets a generic text model with a 128k context window unless those fields are declared.

You can also set the model at init time or per command with the global flags `--model-provider` and `--model-id` (they go together):

```bash
autorag init --search-paths "$HOME/docs" --model-provider openrouter --model-id "<model-id>"
autorag search "..." --model-provider openrouter --model-id "<model-id>"
```

**Warning:** `autorag init --force` replaces the whole config, including datasource and parser settings. To change only the model, edit the `model` block or the TUI, not `init --force`.

## Credentials

Credentials come from, in order, the pi runtime's stored auth (`~/.pi/agent/auth.json`, an API key or OAuth established with `pi /login` or the TUI's `/login`) and then the provider's environment variables. Store only references — an `apiKeyEnv` variable name, a keychain entry, or a profile — in config; never a literal secret.

```bash
export OPENROUTER_API_KEY="..."   # value in the environment, name in the config
autorag health --json             # confirms the credential resolves and works
```

## Health check

```bash
autorag health                 # resolve model + auth + one live completion probe
autorag health --json
autorag health --skip-probes   # skip the network probe; proves config shape only
autorag health --timeout-ms 20000
```

`autorag health` checks model resolution, provider auth, and a live completion probe; it does not check index health (use `autorag status`). Exit codes: `0` ok, `2` config or model-resolution failure, `1` auth missing, provider unreachable, completion failure, or timeout. `--skip-probes` never proves live provider access.

## Jev (decision model)

Jev is required: `autorag search` checks first that it can answer and stops with `decision-model-unavailable` (exit 3) otherwise. `jev: false` and `jev.enabled` are config errors — Jev cannot be turned off. On a local search Jev routes the question, decides whether to decompose it, decides per datasource whether to search it before the fast answer, and after the fast answer decides whether verification is needed.

```json
{
  "jev": { "backend": "openrouter", "model": "typesafe/jev-1.13" }
}
```

The default backend is `dazzi` when signed in to Dazzi Models, otherwise `openrouter`. The decision model asked per provider when the config names none:

| Provider | Default decision model |
|---|---|
| `openrouter` | `typesafe/jev-1.13` |
| `openai` | `gpt-6-luna` |
| `typesafe` | `jev-latest` |
| `vercel-ai-gateway` | `typesafe-ai/jev-1.13.0` |
| `cloudflare-workers-ai` | `typesafe/jev` |
| `opencode` | `jev-1.13` |
| `bivia` | `dec-1` |

List the candidates with `autorag models list --decision --available`; never invent a model id. A failing Jev stops `autorag search` with `[decision-model-unavailable]` and a typed `reason`: `missing-credential`, `unauthorized`, `model-not-found`, `payment-required`, `unreachable`, or `rate-limited`.

`jev.backend: "sllm"` decides with the small chat model alone and needs no Jev credential. When Jev is unavailable, the session can run on a fallback small chat model (`jev.sllm.model`) and warns `decision-model-fallback`; set `"fallback": false` to fail instead.

## Question decomposition

On by default. `queryDecomposition` splits a multi-part question into at most five parallel search queries. The default model is `dazzi/openai/gpt-6-luna` when signed in to Dazzi Models, otherwise `openrouter/qwen/qwen3.7-flash`. Any catalog `provider`/`id` works with the same fields as the top-level `model`; `"queryDecomposition": false` decomposes with the search model itself.

## Hosted Dazzi Models provider

AutoRAG registers the hosted **Dazzi Models** plan as a first-class provider named `dazzi`, available to `autorag tui`, `autorag search`, `autorag health`, the `AutoRAGAgent` library path, and the Pi interactive session. It speaks the server's OpenAI Responses API surface under `<root>/v1`, so streaming, tools, and usage ride Pi's built-in `openai-responses` implementation.

### Sign in

```text
autorag tui
/login
```

Choose **Dazzi Models**. A browser opens `<root>/desktop/authorize`; on success it is redirected to `http://127.0.0.1:<ephemeral port>/callback` and shows "Signed in to Dazzi Models. You can close this tab." The loopback listener closes after the first callback, after a 5-minute timeout, or on cancellation.

If the browser cannot reach this machine (remote shell, container, another host), paste the final redirect URL — or just the `code` — into the prompt the login shows. Both paths are raced; `state` is always verified, and a mismatch aborts the login.

The credential `/login` stores in Pi's `auth.json` is the **only** credential. The model provider, Jev, and rerank all read that same entry. The token is long-lived and revocable: there is no refresh token and no expiry. Revocation surfaces as HTTP 401 on the next request; run `/login` again. There is no API-key environment variable — `AUTORAG_API_KEY` is not read, and `rerank.apiKeyEnv`/`model.apiKeyEnv` are rejected for `dazzi`. A headless login flow (no browser on any reachable machine) is not available yet ([#1913](https://github.com/Marker-Inc-Korea/AutoRAG/issues/1913)).

### Base URL

| Variable | Default | Meaning |
|---|---|---|
| `AUTORAG_BASE_URL` | `https://api.dazziapp.com` | Server **root** (not the `/v1` path). The API base is `<root>/v1`. |

### Choose a model

The **chat** catalog is discovered live from `GET <root>/v1/models` — no chat model id is hardcoded. The Jev id (`jev-1.13`) and the rerank models are the ids the server lists under `GET <root>/v1/models?operation=jev|rerank`. After login, the models your plan returns appear in the model selector (`/model`) and in `autorag models list`; select one as `dazzi/<model-id>`.

On a fresh agent home there is no catalog snapshot yet, so pin the model with an explicit endpoint:

```json
{
  "model": {
    "provider": "dazzi",
    "id": "anthropic/claude-haiku-5.5",
    "baseUrl": "https://api.dazziapp.com/v1",
    "api": "openai-responses"
  }
}
```

The credential is still the stored login; no key is spelled out. Once a `/login` or model refresh has persisted the snapshot, the `provider`/`id` shorthand resolves as well:

```json
{ "model": { "provider": "dazzi", "id": "anthropic/claude-haiku-5.5" } }
```

A brand-new home with no snapshot resolved offline fails the shorthand with `Unknown configured model: dazzi/<id>` because offline resolution never reaches the network. Catalog fields map to Pi model fields as follows:

| Server field | Pi model field |
|---|---|
| `id`, `name` | `id`, `name` (`name` falls back to `id`) |
| `context_window` | `contextWindow` (default 128000) |
| `max_output_tokens` | `maxTokens` (default 16384; a session asks for at most 32,000) |
| `input_modalities` | `input` (`text`/`image`; unknown modalities dropped) |
| `reasoning` | `reasoning` |
| `pricing.{input,output,cache_read,cache_write}` | `cost` (USD per million tokens) |

A session never asks a model for more than 32,000 output tokens, whatever the catalog lists: pi sizes each request's `max_tokens` from its estimate of the window left, and that estimate excludes the system prompt and tool declarations, so a catalog `maxTokens` near the context window overshoots and the provider answers 400. The context guard budgets the input against this same 32,000-token reserve.

### Request size and context overflow

The hosted API rejects a request body above about 4.49 MB with HTTP 413. Every `dazzi` model declares `inputLimits` (`DAZZI_INPUT_LIMITS`): a 3.5 MB transcript cap and a 1 MB cap per image. Pi resends every image a tool returned on each later turn, so before each model request AutoRAG drops the oldest images until the transcript fits and leaves a note telling the model to read the file again if needed. Text is never dropped; the session record keeps every image.

When a request exceeds the model's context window, the server answers `400 context_length_exceeded` (or HTTP 413 for an oversized body) with its own message, and pi compacts the conversation and retries once. Because compaction keeps recent turns whole, the `dazzi` provider notes each overflow and the context guard then holds that model's requests to half the estimated size of the one that failed, ratcheting down on further overflows. The guard may cut retrieval chunks only, lowest-ranked first, keeping each list's best chunk before any list's second; the query, earlier turns, and system prompt are never edited. Sizes are counted with the `o200k_base` BPE, never below pi's chars/4 guess.

### Defaults when signed in

With a stored login, AutoRAG prefers the hosted plan wherever it picks a default; OpenRouter stays the fallback when signed out.

| What | Signed in to Dazzi Models | Signed out (fallback) |
|---|---|---|
| Chat model (no `model` in config) | `dazzi/z-ai/glm-5.3-flash`, else the catalog's first model | CLI: Pi default, then the local Codex runtime; `autorag tui`: Pi's own picker, no local fallback |
| Jev (`jev.backend` unset) | `dazzi` | `openrouter` |
| Question decomposition | `dazzi/openai/gpt-6-luna` | `openrouter/qwen/qwen3.7-flash` |
| Rerank block written by `autorag init` | none (reranking is opt-in) | none (reranking is opt-in) |

Explicit config always wins. The chat default needs a catalog snapshot; without one it is skipped. **Embeddings are not part of this**: they stay on MinSync's local `Qwen/Qwen3-Embedding-0.6B` engine at 1024 dimensions.

### Jev and rerank on the hosted plan

The same sign-in serves Jev and the rerank stage:

```json
{
  "jev": { "backend": "dazzi", "model": "jev-1.13" },
  "rerank": { "provider": "dazzi", "model": "qwen/qwen3-reranker-8b" }
}
```

| Capability | Endpoint | Default model | Credential |
|---|---|---|---|
| Jev (`jev.backend: "dazzi"`) | `POST <root>/v1/jev` | `jev-1.13` | stored `/login` |
| Rerank (`rerank.provider: "dazzi"`) | `POST <root>/v1/rerank` | none; runs only when `rerank.model` names one | stored `/login` |

Both read `AUTORAG_BASE_URL` for the server root; an explicit `rerank.baseUrl` or `jev.model` overrides the defaults. A `rerank.baseUrl` for `dazzi` is the API base **including** `/v1` (e.g. `https://api.dazziapp.com/v1`), because the rerank client appends `/rerank`; `AUTORAG_BASE_URL` stays the root without `/v1`. Without a stored login, both report a named "needs a sign-in" error instead of silently degrading.

### Environment-only mode

`AUTORAG_AUTH_MODE=environment` disables stored model credentials, Pi model/settings discovery, and the Codex config fallback for CLI model resolution and decision backends. It does not change normal standalone authentication when unset. The model-free Lite MCP retrieval tools still work.

## Reranking

Reranking is opt-in and runs only when `rerank.model` names a model. Without it (the default), each source contributes its own top 10 to the fast first answer and the diagnostic `rerank-disabled` is reported once. A rerank model widens the prefetch pool (Jikji 30, MinSync 100, each datasource 20) and the reranker picks the best. See [Reranking](../retrieval/reranking.md).
