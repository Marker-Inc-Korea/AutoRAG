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
on top, and uses the same client for the query pipeline described below. Jev
is **on by default** with the OpenRouter backend, and it is always omitted for
remote P2P sessions because its `state` leaves the machine.

## Defaults and opt-out

With no `jev` section, AutoRAG behaves as if the config said:

```json
{
  "jev": { "backend": "openrouter" },
  "queryDecomposition": { "model": { "provider": "openrouter", "id": "qwen/qwen3.7-flash" } }
}
```

`autorag init` writes exactly these sections into new configs so they are
visible and editable. An empty `{}` keeps the defaults. `"jev": false` or
`enabled: false` turns Jev off completely: no routing, no follow-up check, no
`jev` tool. Both features send the question text to OpenRouter, so disable them
when questions must stay on the machine. Without `OPENROUTER_API_KEY`, routing
falls back to a single local search with a `query-route-fallback` diagnostic;
searches keep working.

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

The OpenRouter path is pinned to `typesafe/jev-1.13`: `jev-use` 0.8.0's own
OpenRouter default (`typesafe/jev-latest`) is not a live OpenRouter model id and
returns HTTP 400. Set `model` to override the pin.

### Configuration fields

| Field                 | Meaning                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| `enabled`             | `false` disables Jev (same as `"jev": false`).                          |
| `backend`             | `openrouter` (default), `typesafe`, or `vercel`.                        |
| `model`               | Wire model id sent with every call. Omit for the backend default.       |
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

## Query pipeline (routing, decomposition, follow-up check)

Enabling `jev` also turns on a Jev-driven pipeline that runs in the two-phase
search **before** `emit_fast_answer`. Jev answers two typed questions about the
user question in one batched call:

1. **Branch** (`choice`): `local`, `web`, `direct`, or `config`. The branch with the
   highest probability wins, even when Jev reports low confidence.
   - `local`: answering needs information only the user can reach (files on
     their computer, Discord/KakaoTalk/Slack chats, email, notes).
   - `web`: not answerable from general knowledge, but one public internet
     search would answer it.
   - `direct`: general knowledge, simple reasoning, or small talk.
   - `config`: the user wants to view, change, or test AutoRAG's own settings
     (model, providers, API-key environment variables, Jev, search roots,
     datasources). Offered only to local sessions, never to remote P2P peers.
2. **Decomposition** (`noul`): does the question need several search queries
   (multiple sub-questions, comparisons, several facts to confirm)? A
   probability of 0.5 or more means yes.

What happens next:

| Branch   | Pipeline                                                                                 |
| -------- | ---------------------------------------------------------------------------------------- |
| `direct` | Skips Jikji, MinSync, web search, and the verification phase; `emit_fast_answer` is final. |
| `config` | Skips retrieval, decomposition, `emit_fast_answer`, and verification. The turn prompt carries the full `autorag-setup` skill, the active config path, and the pi agent dir; the model edits the config with `bash`/`read`/`edit`/`write`, verifies with `autorag health`/`models list`, and reports old → new through `emit_autorag_results` (no results, and the run is not recorded in retrieval memory). If the skill cannot be loaded the run falls back to `local` with a `self-config-unavailable` diagnostic. |
| `local`  | Decompose (if needed) → Jikji + MinSync per query, in parallel → merged pool → rerank against the original question (when `rerank` is configured) → fast answer → follow-up check → verification (only if needed). |
| `web`    | Decompose (if needed) → `web_search` per query, in parallel → merged evidence → fast answer → follow-up check → verification (only if needed). |

After a `config` turn that changed the config file, the host re-resolves it the way the next launch will (JSON, schema, and that the model id exists in the pi catalog or a declared endpoint). If it no longer resolves, the pre-turn file is restored, a warning is appended to the report, and a `self-config-rolled-back` diagnostic is recorded, so a bad model id can never leave the agent unable to start. A missing credential is not a rollback reason; that is reported by `autorag health` as `auth_missing`. An unchanged file is never touched.

### Follow-up check after the fast answer

After `emit_fast_answer` on the `local` and `web` branches, Jev answers one more
`noul` about the question **and** the fast answer together: does the answer need
correction, clarification from the user, or further research? Below 0.5, the run
ends there: the fast answer becomes the final response (its numbered results and
sources are kept), the verification phase does not run, and a
`follow-up-skipped` diagnostic records the probability. At 0.5 or above, the
fast answer is published as the preliminary answer and verification continues
as usual. With Jev enabled, the preliminary is held until this decision, so an
answer that turns out to be final reaches the caller once, as the complete
response. If the check fails (missing credential, unreachable backend), the run
verifies (`follow-up-check-fallback`), because ending on an unchecked answer is
the costlier mistake.

Decomposition sends a short prompt to an LLM and keeps **at most five** search
queries. The default model is `openrouter/qwen/qwen3.7-flash`.
`queryDecomposition.model` takes the same fields and credential rules as the
top-level `model`; `"queryDecomposition": false` decomposes with the search
session's own model instead.

The default was picked on a live OpenRouter benchmark: the real decomposition
prompt, 6 questions (including 2 Korean), and 2 runs each, scored on valid
JSON, at most five queries, coverage of every sub-question, and language
preserved:

| Model | Score | p50 latency | Cost per 12 calls |
| ----- | ----- | ----------- | ----------------- |
| `qwen/qwen3.7-flash` (default) | 12/12 | 0.92s | $0.00011 |
| previous default (retired 2025 Gemini flash-lite) | 12/12 | 0.87s | $0.00039 |
| `qwen/qwen3.8-flash` | 12/12 | 1.19s | $0.00051 |
| `upstage/solar-mini4` | 12/12 | 0.86s | $0.00021 (not in the pi catalog) |
| `openai/gpt-6-luna` | 12/12 | 2.24s | $0.00038 (not in the pi catalog) |
| `xiaomi/mimo-v2.6-flash` | 12/12 | 5.19s | $0.00030 |
| `nvidia/nemotron-3.5-lightning` | 11/12 | 0.38s | $0.00021 |
| `inception/mercury-2.5` | 10/12 | 0.75s | $0.00010 (2 upstream timeouts) |
| `z-ai/glm-5.3-flash` | 0/12 | — | requires reasoning; rejected with reasoning off |

```json
{
  "queryDecomposition": {
    "model": { "provider": "openrouter", "id": "qwen/qwen3.7-flash" }
  }
}
```

Failures never block a search. A missing Jev credential, an unreachable Jev
backend, or an unusable verdict falls back to today's single local search for
the original question (diagnostic `query-route-fallback`). A `web` verdict with
web tools disabled also falls back to local search. A failed decomposition
searches the original question (`query-decomposition-failed`). Every routed run
records its branch and queries as a `query-routed` diagnostic (`--debug`
shows it). The pipeline never runs for remote P2P sessions. Every search is
two-phase (fast answer, then verification unless Jev ends the run); there is
no single-phase mode.

## Live verification

The offline suite never calls Jev (it uses the `mock` backend). The live test is
gated on both an explicit opt-in and a real key:

```bash
AUTORAG_JEV_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/jev.test.ts
```

## Programmatic usage

```ts
import { AutoRAGAgent } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["./docs"],
  jev: { backend: "openrouter" },
});
const response = await agent.searchDocuments("summarize the Q3 report");
```

The tool is registered through pi's extension surface: `createJevExtension`
builds a pi `ExtensionFactory` that calls `pi.registerTool`, and AutoRAG loads
it (and allow-lists the `jev` name) only when the config enables it. pi owns
tool activation and rendering; AutoRAG keeps the prompt line and the reserved
name. For callers that want the raw engine, import `Jev`, `check`, `pick`, and
`rate` from `jev-use` directly.
