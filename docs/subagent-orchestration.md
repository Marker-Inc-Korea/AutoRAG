# Subagent orchestration

AutoRAG uses a mandatory two-tier `pi-subagents` workflow for document
retrieval. A configured parent model is the orchestrator and configured child
models are explorers. There is no single-agent fallback.

## Contract

| Role | Model | Owns |
|------|-------|------|
| Orchestrator | user-configured reasoning/high-context model | judgment, sufficiency, conflicts, freshness, timing, follow-ups, and final curation |
| Explorer | user-configured fast, high-recall model | high-recall search/read work and candidate evidence handoff |

The roles are configured independently with `agents.orchestrator` and
`agents.explorer`; each value is a `{ "provider": "...", "id": "..." }`
object. AutoRAG does not ship a private provider default. Setup selects models
from the user's authenticated runtime or requires explicit role configuration.

The `pi-subagents` extension and its `subagent` capability are required. A
missing capability is fatal for the run; do not silently complete the request
with one agent.

## Preflight diagnostics

`autorag health` checks model/provider auth and explorer subagent setup without
touching indexes. It resolves both role models, verifies credential presence,
and (unless `--skip-probes` is set) runs a lightweight completion probe per
role. Use it before a search to confirm that the mandatory two-tier workflow
can dispatch:

```bash
autorag health
autorag health --skip-probes   # auth + model resolution only, no network probes
```

When `autorag search` fails for a model, provider, auth, timeout, or subagent
reason, the error output includes a hint pointing to `autorag health`:

```
error: Mandatory pi-subagents extension failed to load
Run autorag health to diagnose model/provider and explorer subagent setup.
```

`autorag status` remains the model-free index-health command (corpus freshness,
BM25/MinSync readiness). It does not check models or subagent dispatch — use
`autorag health` for that.

## Home State and Configuration

The default home state is separate from workspace indexes:

```text
~/.autorag/
├── config.json
├── memory.json
├── logs/
│   └── runs.jsonl
└── pi-agent/
    ├── auth.json
    ├── models.json
    ├── settings.json
    └── sessions/
```

Config path precedence is `--config` > `AUTORAG_CONFIG` >
`~/.autorag/config.json`. If the home config is absent and
`<cwd>/autorag.config.json` exists, AutoRAG copies the legacy file to the home
path without deleting or modifying it. Workspace parsed mirrors and indexes
remain under `<workspace>/.autorag`. Durable Pi models, settings, and sessions
stay under `~/.autorag/pi-agent`.

## Roles

### `gpt-5.6-sol` orchestrator

The orchestrator is the only agent that owns:

- relevance and evidence-quality judgment;
- deciding whether the collected evidence is sufficient;
- conflict resolution across documents, retrieval methods, and explorers;
- freshness judgment and selection of the relevant creation, publication,
  update, modification, or observation time;
- deciding when a document or artifact was created or modified for the answer;
- follow-up assignments and changes to the retrieval plan;
- final answer synthesis and curation.

Explorers may report signals and uncertainty, but their ranking or conclusion
is never the final decision.

### `gpt-5.6-luna` explorers

Explorers are high-recall retrieval and reading workers. They receive an
assignment containing:

1. the original query, unchanged;
2. one selected retrieval method;
3. multiple query variants, including exact, synonymous, identifier, and
   broader or narrower forms;
4. the allowed scope and inherited policy constraints.

Each explorer is assigned exactly one normalized configured search root as its
`cwd`. The top-level `subagent` invocation sets `artifacts: false` exactly once
for single, `tasks`, `chain`, or `parallel` dispatch. Nested explorer task
items omit `artifacts`; project-local `.pi-subagents` debug artifacts are
disabled.

They search and read a large set of candidate documents. They should return
weakly relevant candidates when those candidates may illuminate a conflict,
missing evidence, or an alternate time interpretation. They do not decide
whether the overall answer is sufficient, settle conflicts, make the final
freshness judgment, assign more work, or produce the caller-facing answer.

### Retrieval execution boundary

Read-only `read`/`grep`/`find`/`ls` exploration runs inside the luna child
process. BM25, MinSync, Jikji, and datasource methods are AutoRAG
process-bound tools because they close over the live indexes, policy gates, and
trusted datasource context. The sol orchestrator invokes those retrieval tools
only to build a bounded seed pack, then delegates the seed paths/results,
unchanged original query, and query variants to a luna explorer. The explorer
reads the underlying documents broadly and returns the evidence handoff. Seed
retrieval does not permit the orchestrator to skip delegation, read the
documents, or make a final answer without explorer evidence.

## Explorer handoff

Each explorer returns candidate findings with enough detail for the
orchestrator to make an independent decision:

```text
original_query: the unchanged caller query
retrieval_method: the selected method
query_variants: every variant actually tried
candidates:
  - source: real file path or authorized datasource id
    relevance: strong | moderate | weak
    evidence: excerpts or structured facts with location context
    retrievedAt: when this candidate was retrieved
    temporal_metadata:
      created_at: when available
      published_at: when available
      updated_at: when available
      modified_at: when available
      observed_at: when the value was observed
      asOf: source as-of time when available, otherwise explicit unknown
    temporal_basis: which timestamp supports the finding
    uncertainty: missing, ambiguous, or conflicting metadata
```

The fields are a handoff contract, not permission for an explorer to infer a
missing date. Preserve the source and method so the orchestrator can compare
results and record feedback accurately.

An explorer may report an unknown temporal value, but it must not manufacture
one from `retrievedAt`. The orchestrator decides which creation/modification
timing and freshness interpretation is relevant to the caller.

## Dispatch and decision loop

1. The orchestrator checks memory and chooses one or more retrieval methods.
2. For process-bound methods, it creates a bounded seed pack with the selected
   AutoRAG retrieval tool; for read/grep/find/ls it delegates discovery
   directly.
3. It dispatches `gpt-5.6-luna` explorers through `pi-subagents`, passing the
   original query, selected method, multiple query variants, and any seed pack.
4. Explorers search and read broadly, then return strong, moderate, and weak
   candidates with evidence and temporal metadata.
5. The orchestrator compares candidates, resolves conflicts, evaluates
   sufficiency and freshness, and assigns follow-ups when gaps remain.
6. Only the orchestrator performs final curation and calls
   `emit_autorag_results` exactly once as the final action.

If the extension cannot dispatch explorers, the run is blocked/degraded. Do
not silently replace the two-tier workflow with a single-agent search.

## Final curation and termination

Only the `gpt-5.6-sol` orchestrator may turn explorer handoffs into the
caller-facing answer. It must preserve source, method, evidence, and temporal
metadata in the curated mapping, then call `emit_autorag_results` exactly once
as its final action. Explorers never call the terminating tool, and no
assistant-prose answer follows it.

## Existing safety boundaries

The subagent workflow does not change retrieval policy:

- When Jikji is configured, `jikji_find` remains the first local-discovery
  action. Explorers must honor `answer_paths`,
  `agent_should_not_rerank`, `handoff_action`, and `tool_call_policy`. Raw
  `read`/`grep`/`find`/`ls` discovery is permitted only when the answer-pack
  allows the fallback after the required retry, or when Jikji is
  unavailable/unconfigured.
- Datasource access remains default-deny and server-bound. Explorers cannot
  grant themselves `allowedTags` or `allowedScopes`; `scope` may only narrow
  trusted access. Datasource results are filtered before merge.
- `emit_autorag_results` remains the structured terminating tool. Explorers
  return evidence to the orchestrator and never call it.

## Testing

Prompt tests should assert the role split and handoff fields as parsed contract
signals rather than snapshotting the full prompt. The RED-GREEN cases cover:

- mandatory `pi-subagents` and fatal missing-capability behavior;
- exclusive `gpt-5.6-sol` decisions and `gpt-5.6-luna` search/read work;
- original query, selected method, multiple variants, weak candidates,
  evidence, `retrievedAt`, and `asOf`/unknown metadata;
- unchanged Jikji, datasource trust, and exactly-once
  `emit_autorag_results` termination rules.
