# Hosted AutoRAG provider (`autorag`)

AutoRAG registers the hosted **AutoRAG** plan (Dazzi) as a first-class model provider
named `autorag`. The provider is available to `autorag tui`, `autorag search`,
`autorag serve`, `autorag health`, the `AutoRAGAgent` library path, and the Pi
interactive session.

The server exposes an OpenAI Responses API surface under `<root>/v1`:

| Capability | How it is provided |
|---|---|
| Streaming, tools, usage | Pi's built-in `openai-responses` API (`api: "openai-responses"`, `baseUrl: "<root>/v1"`). No custom stream code. |
| Login | OAuth2 authorization code + PKCE (S256) against `<root>/desktop/authorize?client=agent`, with a loopback redirect on `127.0.0.1`. Token exchange at `POST <root>/api/desktop/token`. |
| Headless auth | `AUTORAG_API_KEY` is used as the bearer key; no login required. |
| Model catalog | Discovered live from `GET <root>/v1/models`. **No model ids are hardcoded.** |

## Sign in

```text
autorag tui
/login
```

Choose **AutoRAG**. A browser opens `<root>/desktop/authorize`; on success it is
redirected to `http://127.0.0.1:<ephemeral port>/callback` and shows
"Signed in to AutoRAG. You can close this tab." The loopback listener closes after
the first callback, after a 5-minute timeout, or on cancellation.

If the browser cannot reach this machine (remote shell, container, another host),
paste the final redirect URL — or just the `code` — into the prompt the login
shows. Both paths are raced; whichever finishes first wins. `state` is always
verified, and a mismatch aborts the login.

The returned token is a long-lived, revocable API key: there is no refresh token
and no expiry. Revocation shows up as HTTP 401 on the next request; run `/login`
again.

### Headless (no login)

```bash
export AUTORAG_API_KEY=dz_...
```

Pi resolves `$AUTORAG_API_KEY` for this provider, so it is authorized without
`/login`. A stored credential from `/login` takes precedence over the environment
variable; log out from the Pi UI to fall back to it.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AUTORAG_BASE_URL` | `https://api.dazziapp.com` | Server **root** (not the `/v1` path). The API base is `<root>/v1`. |
| `AUTORAG_API_KEY` | – | Headless API key. |

## Choose a model

After login (or with `AUTORAG_API_KEY` set), the models the server returns for your
plan appear in the model selector (`/model`) and in `autorag models list`. Select one
as `autorag/<model-id>`, or pin it in `config.json`:

```json
{
  "model": { "provider": "autorag", "id": "anthropic/claude-haiku-5.5" }
}
```

Catalog fields are mapped as follows:

| Server field | Pi model field |
|---|---|
| `id`, `name` | `id`, `name` (`name` falls back to `id`) |
| `context_window` | `contextWindow` (default 128000) |
| `max_output_tokens` | `maxTokens` (default 16384) |
| `input_modalities` | `input` (`text`/`image`; unknown modalities are dropped) |
| `reasoning` | `reasoning` |
| `pricing.{input,output,cache_read,cache_write}` | `cost` — USD per million tokens, Pi's cost unit |

### Offline resolution

The catalog is persisted after any refresh (a `/login`, or a normal interactive
startup). Model resolution for the CLI and library paths runs with
`allowModelNetwork: false`, so it restores the persisted snapshot instead of
calling the server. That keeps `model: { provider: "autorag", id }` resolvable
offline.

### When you are not signed in

The provider still registers: it appears under `/login`, contributes no models, and
makes no network call. Nothing crashes; the model list simply has no `autorag`
entries.

### Explicit configured endpoint

An app may write the same provider as an explicit OpenAI-compatible endpoint:

```json
{
  "model": {
    "provider": "autorag",
    "id": "anthropic/claude-haiku-5.5",
    "baseUrl": "https://api.dazziapp.com/v1",
    "api": "openai-responses",
    "apiKeyEnv": "AUTORAG_API_KEY"
  }
}
```

This shape keeps resolving exactly as before: the explicit `baseUrl` wins over the
registered provider's endpoint, and the key is read from the named environment
variable.

## Implementation

The provider lives in `src/cloud/`. A single registration point,
`registerAutoRAGProvider(runtime)`, is called where each Pi `ModelRuntime` is
created (`src/cli/config.ts` and `src/agent/pi-session.ts`).
