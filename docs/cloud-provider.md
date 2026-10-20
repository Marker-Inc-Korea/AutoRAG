# Hosted Dazzi Models provider (`dazzi`)

AutoRAG registers the hosted **Dazzi Models** plan as a first-class model provider
named `dazzi`. The provider is available to `autorag tui`, `autorag search`,
`autorag serve`, `autorag health`, the `AutoRAGAgent` library path, and the Pi
interactive session.

The server exposes an OpenAI Responses API surface under `<root>/v1`:

| Capability | How it is provided |
|---|---|
| Streaming, tools, usage | Pi's built-in `openai-responses` API (`api: "openai-responses"`, `baseUrl: "<root>/v1"`). No custom stream code. |
| Login | OAuth2 authorization code + PKCE (S256) against `<root>/desktop/authorize?client=agent`, with a loopback redirect on `127.0.0.1`. Token exchange at `POST <root>/api/desktop/token`. This is the **only** way to authenticate. |
| Model catalog | The **chat** catalog is discovered live from `GET <root>/v1/models` — no chat model id is hardcoded. The Jev id (`jev-1.13`) and the default reranker are the ids the server lists under `GET <root>/v1/models?operation=jev\|rerank`; Jev and rerankers are not chat LLMs (the 2026+ model rule covers chat LLMs only). |

## Sign in

```text
autorag tui
/login
```

Choose **Dazzi Models**. A browser opens `<root>/desktop/authorize`; on success it is
redirected to `http://127.0.0.1:<ephemeral port>/callback` and shows
"Signed in to Dazzi Models. You can close this tab." The loopback listener closes
after the first callback, after a 5-minute timeout, or on cancellation.

If the browser cannot reach this machine (remote shell, container, another host),
paste the final redirect URL — or just the `code` — into the prompt the login
shows. Both paths are raced; whichever finishes first wins. `state` is always
verified, and a mismatch aborts the login.

The credential `/login` stores in Pi's `auth.json` is the **only** credential. The
model provider, Jev, and rerank all read that same entry. The token is long-lived
and revocable: there is no refresh token and no expiry. Revocation shows up as
HTTP 401 on the next request; run `/login` again.

There is no API-key environment variable. `AUTORAG_API_KEY` is not read, and
`rerank.apiKeyEnv` / `model.apiKeyEnv` are not used for `dazzi`. A headless login
flow (no browser on any reachable machine) is not available yet; it is tracked in
[#1913](https://github.com/Marker-Inc-Korea/AutoRAG/issues/1913).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AUTORAG_BASE_URL` | `https://api.dazziapp.com` | Server **root** (not the `/v1` path). The API base is `<root>/v1`. |

## Choose a model

After login, the models the server returns for your plan appear in the model
selector (`/model`) and in `autorag models list`. Select one as `dazzi/<model-id>`.

On a fresh agent home there is no catalog snapshot yet, so pin the model with an
explicit endpoint:

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

The credential is still the stored login: no key is spelled out in the config. Once
a TUI `/login` or a model refresh has persisted the snapshot, the `provider`/`id`
shorthand resolves as well:

```json
{
  "model": { "provider": "dazzi", "id": "anthropic/claude-haiku-5.5" }
}
```

Catalog fields are mapped as follows:

| Server field | Pi model field |
|---|---|
| `id`, `name` | `id`, `name` (`name` falls back to `id`) |
| `context_window` | `contextWindow` (default 128000) |
| `max_output_tokens` | `maxTokens` (default 16384; a session request asks for at most 32,000 of it, see below) |
| `input_modalities` | `input` (`text`/`image`; unknown modalities are dropped) |
| `reasoning` | `reasoning` |
| `pricing.{input,output,cache_read,cache_write}` | `cost` — USD per million tokens, Pi's cost unit |

A session never asks a model for more than 32,000 output tokens, whatever the
catalog lists. pi sizes each request's `max_tokens` from its own estimate of the
window left, and that estimate leaves out the system prompt and the tool
declarations, so a catalog `maxTokens` near the context window (some OpenRouter
models list about 90% of it) overshoots the window and the provider answers 400.
The context guard budgets the input against this same 32,000-token reserve.

### Offline resolution

The catalog is persisted after any refresh (a `/login`, or a normal interactive
startup). Model resolution for the CLI and library paths runs with
`allowModelNetwork: false`, so it restores the persisted snapshot instead of
calling the server. That keeps `model: { provider: "dazzi", id }` resolvable
offline **once a snapshot exists**.

A brand-new agent home has no snapshot to restore, and offline resolution never
reaches the network, so the shorthand alone fails with
`Unknown configured model: dazzi/<id>`. Use the explicit-endpoint shape above
until a `/login` or refresh has populated the snapshot.

### When you are not signed in

The provider still registers: it appears under `/login`, contributes no models, and
makes no network call. `/model` refreshes without an error; the model list simply
has no `dazzi` entries. Jev and rerank report a named "needs a sign-in" error
instead of silently degrading.

### Explicit configured endpoint

The fresh-start shape above is an explicit OpenAI-compatible endpoint: `baseUrl`
and `api` are spelled out, so the model resolves without a catalog snapshot. The
explicit `baseUrl` wins over the registered provider's endpoint, and the bearer
token is the stored `/login` credential.

### Request size

The hosted API rejects a request body above about 4.49 MB with HTTP 413. Every
`dazzi` model, whether it comes from the live catalog, a snapshot, or an explicit
endpoint, declares `inputLimits` (`DAZZI_INPUT_LIMITS` in `src/cloud/config.ts`):
a 3.5 MB transcript cap and a 1 MB cap for each image.

pi keeps every image a tool returned and resends all of them on each later turn, so
a search that reads several photos would otherwise cross the limit and end without
an answer. Before each model request AutoRAG drops the oldest images until the
transcript fits, and puts a short note in their place telling the model to read the
file again if it still needs it. Text is never dropped. The session record keeps
every image; only the request is trimmed. The same guard enforces `inputLimits` for
any model that declares them.

## Defaults when signed in

With a stored Dazzi Models login, AutoRAG prefers the hosted plan wherever it
picks a default. OpenRouter stays the fallback when you are signed out.

| What | Signed in to Dazzi Models | Signed out (fallback) |
|---|---|---|
| Chat model (no `model` in config, none in Pi settings) | `dazzi/z-ai/glm-5.3-flash`, else the catalog's first model | CLI: Pi default, then the local Codex runtime. `autorag tui`: Pi's own picker, no local fallback |
| Jev (`jev.backend` unset) | `dazzi` | `openrouter` |
| Question decomposition (`queryDecomposition.model` unset) | `dazzi/openai/gpt-6-luna` | `openrouter/qwen/qwen3.7-flash` |
| Rerank block `autorag init` writes | `dazzi` / `qwen/qwen3-reranker-8b` | `openrouter` / `voyageai/rerank-3-lite` |

Explicit config always wins. An existing `config.json` keeps the blocks it already
has; `autorag init` writes the signed-in or signed-out set once, when it creates
the file. The chat default needs a catalog snapshot (a prior `/login` or refresh);
without one it is skipped.

**Embeddings are not part of this.** They stay on MinSync's local
Qwen3-Embedding-0.6B engine (1024 dimensions) by default, through a
`minsync serve` daemon or MinSync's in-process engine. Sending corpus text to a
remote embedder requires an explicit remote `minSync.embedder` endpoint, the
hosted embeddings endpoint is Pro-only, and making it the default would send
every indexed document off the machine.

## Jev and rerank on the hosted plan

The same sign-in serves Jev and the rerank stage, so a hosted setup needs no
other credential:

```json
{
  "jev": { "backend": "dazzi", "model": "jev-1.13" },
  "rerank": { "provider": "dazzi", "model": "qwen/qwen3-reranker-8b" }
}
```

| Capability | Endpoint | Default model | Credential |
|---|---|---|---|
| Jev (`jev.backend: "dazzi"`) | `POST <root>/v1/jev` (Jev's native dialect) | `jev-1.13` | stored `/login` |
| Rerank (`rerank.provider: "dazzi"`) | `POST <root>/v1/rerank` (OpenRouter-compatible rerank shape) | `qwen/qwen3-reranker-8b` | stored `/login` |

Both read `AUTORAG_BASE_URL` for the server root. An explicit `rerank.baseUrl` or
`rerank.model` (and `jev.model`) overrides the defaults. `rerank.apiKeyEnv` is
rejected for `dazzi` rather than ignored. Without a stored login the backend
reports a named, non-silent "needs a sign-in" error instead of silently degrading.

A `rerank.baseUrl` for `provider: "dazzi"` is the API base **including**
`/v1`, e.g. `https://api.dazziapp.com/v1`, because the rerank client appends
`/rerank`. `AUTORAG_BASE_URL` stays the server **root** without `/v1`
(`https://api.dazziapp.com`), and the derived rerank base is `<root>/v1` — a root
that already ends in `/v1` is not doubled.

## Implementation

The provider lives in `src/cloud/`. A single registration point,
`registerDazziProvider(runtime)`, is called where each Pi `ModelRuntime` is
created (`src/cli/config.ts`, `src/agent/pi-session.ts`, and
`src/cli/commands/models.ts` for `autorag models list`). `readDazziLoginToken`
(`src/cloud/credential.ts`) reads the stored login for Jev and rerank.
