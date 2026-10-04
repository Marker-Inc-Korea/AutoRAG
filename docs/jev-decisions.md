# Jev Decisions

AutoRAG Agent can expose one optional extra tool, `jev`, backed by
[TypeSafe's Jev](https://docs.typesafe.ai/) judgment model through the
[`jev-use`](https://www.npmjs.com/package/jev-use) engine. Jev is not a chat
model: it answers **typed questions about a state** — `noul` (yes/no
probability), `choice` (one option from a set), and `score` (position on an
ordered rubric) — with calibrated probabilities instead of generated prose.

That makes it a cheap, deterministic alternative to asking the main model to
classify or rank something. The number comes back to the caller, and **the
caller owns the threshold**: the model cannot grade its own work or drift
between runs.

`jev-use` owns backend auto-selection, request screening, and response
validation; AutoRAG registers a thin pi extension tool (`createJevExtension`)
on top. The tool is **disabled by default** because it calls a paid external
API, and it is always omitted for remote P2P sessions because its `state`
leaves the machine.

## Enable it

Add a `jev` section to `~/.autorag/config.json` (or the workspace config):

```json
{
  "jev": { "backend": "openrouter", "model": "jev-latest" }
}
```

`enabled: false` (or `"jev": false`) keeps the tool off. An empty `{}` section
enables it with defaults.

### Backends

`jev-use` resolves the backend from the environment. Set `JEV_BACKEND` to force
one, or leave it unset to let the first credential present win, in the order
TypeSafe -> OpenRouter -> Vercel AI Gateway.

| `backend`    | Credential environment variable                     | Model id on the wire |
| ------------ | --------------------------------------------------- | -------------------- |
| `typesafe`   | `TYPESAFE_API_KEY`                                  | `jev-1.13.0`         |
| `openrouter` | `OPENROUTER_API_KEY`                                | `typesafe/jev-1.13`  |
| `vercel`     | `AI_GATEWAY_API_KEY`                                | `typesafe-ai/jev`    |

`JEV_MODEL` overrides the wire model id, and `JEV_BACKEND=mock` runs a keyless
dry run (used by the offline tests).

### Configuration fields

| Field                 | Meaning                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| `enabled`             | `false` disables the tool (same as `"jev": false`).                     |
| `backend`             | Force `typesafe`, `openrouter`, or `vercel`; omit to auto-select.       |
| `model`               | Model id sent with every call, e.g. `jev-latest`.                       |
| `confidenceThreshold` | Escalate verdicts below this confidence (0-1). Default: per-source.     |

## Using the tool

The agent calls `jev` once per state with any number of typed questions. Each
verdict carries the answer plus its confidence, and `escalate: true` when Jev
cannot decide (with a typed `reason`: `writing`, `open_ended`, `oversized`,
`unsure`, or `unreachable`):

```json
{
  "state": "Help! Payouts have been failing for three days.",
  "questions": [
    { "id": "is_urgent", "type": "noul", "question": "Does this convey urgency?" },
    {
      "id": "department",
      "type": "choice",
      "question": "Which team should handle this?",
      "options": { "billing": "Payments, invoices, refunds", "technical": "Bugs, outages, integrations" }
    },
    {
      "id": "frustration",
      "type": "score",
      "question": "How frustrated is the customer?",
      "levels": ["Calm", "Frustrated", "Very angry"]
    }
  ]
}
```

Guidance:

- Ask narrow, atomic questions; Jev answers exactly what is asked.
- Batch every question about one state into a single call.
- For `noul`, describe **both** the true and false outcomes with `criteria` or
  neither.
- Read `confidence` (and `confidenceFrom`) before acting on a close call; an
  `escalate` verdict means the LLM should take the step over.

## Live verification

The offline suite never calls Jev (it uses the `mock` backend). The live test is
gated on both an explicit opt-in and a real key:

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
name. For callers that want the raw engine, import `Jev`, `check`, `pick`, and
`rate` from `jev-use` directly.
