# Jev Decisions

AutoRAG Agent can expose one optional extra tool, `jev`, backed by
[TypeSafe's Jev](https://docs.typesafe.ai/) judgment model. Jev is not a chat
model: it answers **typed questions about a state** — `noul` (yes/no
probability), `choice` (one option from a set), and `score` (position on an
ordered rubric) — with calibrated probabilities instead of generated prose.

That makes it a cheap, deterministic alternative to asking the main model to
classify or rank something. The number comes back to code, and **code owns the
threshold**: the model cannot grade its own work or drift between runs.

The tool is **disabled by default** because it calls a paid external API. It is
also always omitted for remote P2P sessions, because it sends its `state` to a
third party.

## Enable it

Add a `jev` section to `~/.autorag/config.json` (or the workspace config):

```json
{
  "jev": {
    "backend": "openrouter",
    "model": "jev-latest"
  }
}
```

`enabled: false` (or `"jev": false`) keeps the tool off. An empty `{}` section
enables it with defaults.

### Backends

| `backend`    | Credential environment variable                                  | Model id on the wire  |
| ------------ | ---------------------------------------------------------------- | --------------------- |
| `typesafe`   | `TYPESAFE_API_KEY`                                               | `jev-<version>`       |
| `openrouter` | `OPENROUTER_API_KEY`                                             | `~typesafe/jev-<v>`   |
| `vercel`     | `AI_GATEWAY_API_KEY` (or `VERCEL_API_KEY`)                       | `typesafe-ai/jev`     |
| `cloudflare` | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`                 | `typesafe/jev`        |

Omit `backend` to let the first authenticated backend win, in the order
`typesafe`, `openrouter`, `vercel`, `cloudflare`. The backend's own environment
variable is used unless `apiKeyEnv` names a different one.

### Configuration fields

| Field        | Meaning                                                              |
| ------------ | -------------------------------------------------------------------- |
| `backend`    | One of `typesafe`, `openrouter`, `vercel`, `cloudflare`.              |
| `model`      | Provider-neutral model id: `jev-latest` (default), `jev-1.13`, `jev-1.12`. |
| `apiKeyEnv`  | Environment variable holding the backend API key (never the key itself). |
| `timeoutMs`  | Per-request timeout, 1000–120000 (SDK default 30000).                 |
| `maxRetries` | Retries for transient failures, 0–5 (SDK default 2).                  |

## Using the tool

The agent calls `jev` once per state with any number of named questions. It
returns probabilities in the tool result:

```json
{
  "state": { "message": "Help! Payouts have been failing for three days." },
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": { "billing": "Payments, invoices, refunds", "technical": "Bugs, outages, integrations" }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

Guidance:

- Ask narrow, atomic questions; Jev answers exactly what is asked.
- Batch every question about one state into a single call.
- For `noul`, describe **both** the `true` and `false` outcomes or neither —
  OpenRouter rejects a partial description.
- For rankings, use one `score` question per item instead of a `choice` over orderings.
- Read `confidence` (choice/score) before acting on an uncertain answer; a
  low-confidence `choice` means no option clearly fits.

## Live verification

The offline suite never calls Jev. The live test is gated on both an explicit
opt-in and a real key:

```bash
AUTORAG_JEV_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/jev.test.ts
```

## Programmatic usage

```ts
import { AutoRAGLite } from "@autorag/librarian";

const agent = new AutoRAGLite({
  searchPaths: ["./docs"],
  jev: { backend: "openrouter", model: "jev-latest" },
});
```

The tool is registered through pi's extension surface: `createJevExtension`
builds a pi `ExtensionFactory` that calls `pi.registerTool`, and AutoRAG loads
it (and allow-lists the `jev` name) only when the config enables it. pi owns
tool activation and rendering; AutoRAG keeps the prompt line and the reserved
name. `createJevTool` and `createJevEvaluator` are also exported for callers
that compose their own agent; `createJevEvaluator` is the transport seam and
can be replaced with any `JevEvaluator` implementation.
