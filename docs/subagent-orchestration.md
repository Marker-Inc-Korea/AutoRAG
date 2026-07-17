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

The assignment is a sentinel-wrapped JSON block (Assignment V1):

```text
<<<AUTORAG_ASSIGNMENT_V1>>>
{"originalQuery":"<caller query verbatim>","method":"<selected retrieval method>","queryVariants":["<variant 1>","<variant 2>"]}
<<<END_AUTORAG_ASSIGNMENT_V1>>>
Required handoff: include retrievedAt.
Required handoff: include temporal metadata.
```

The JSON body has exactly three keys: `originalQuery` (the caller query
verbatim), `method` (the selected retrieval or discovery path), and
`queryVariants` (a nonempty array of query variant strings). A legacy labeled
format (`Original query:`, `Selected retrieval method:`, `Query variants:`)
is accepted for compatibility; the V1 sentinel format is preferred and
required for multiline or special content.

Each explorer is assigned exactly one normalized configured search root as its
`cwd`. The top-level `subagent` invocation sets `agentScope: "user"` and
`artifacts: false` exactly once for single, `tasks`, `chain`, or `parallel`
dispatch. Nested explorer task items omit `agentScope` and `artifacts`;
project-local `.pi-subagents` debug artifacts are disabled.

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

## Dispatch templates and safe defaults

The canonical single-root and multi-root dispatch payloads set
`agentScope: "user"` and `artifacts: false` exactly once at the top level.
Each executable leaf carries `agent: "autorag-explorer"`, the configured
explorer model, an explicit `cwd` set to one allowed root, and a `task`
containing an Assignment V1 block.

**Single-root dispatch:**

```json
{
  "agentScope": "user",
  "artifacts": false,
  "agent": "autorag-explorer",
  "model": "<configured-model>",
  "cwd": "<allowed-root>",
  "task": "<Assignment V1 block>"
}
```

**Multi-root dispatch** (one task per configured root):

```json
{
  "agentScope": "user",
  "artifacts": false,
  "tasks": [
    {
      "agent": "autorag-explorer",
      "model": "<configured-model>",
      "cwd": "<allowed-root>",
      "task": "<Assignment V1 block>"
    }
  ]
}
```

### Safe autofill

Missing or null top-level `artifacts`, `agentScope`, and leaf `model` fields
are autofilled before validation: `artifacts` to `false`, `agentScope` to
`"user"`, and leaf `model` to the configured explorer model. This reduces
retry cascades from safe envelope omissions. Explicit wrong values
(`artifacts: true`, `agentScope: "project"`, a non-configured model) remain
rejected. Diagnostics (`list`, `get`, `models`, `status`, `doctor`) are
separate from launch dispatch and never receive these defaults. There is no
single-agent fallback.

### Anti-examples (rejected dispatches)

- **Nested `artifacts`/`agentScope` on task items** — these fields are set
  only once at the top level and are never injected into leaves; nested
  presence is rejected as malformed.
- **Wrong root (`cwd` outside configured roots)** — every executable leaf
  `cwd` must be one of the configured search roots; paths outside the
  allowed roots or with symlink escape are rejected.
- **Wrong agent (not `autorag-explorer`)** — every executable leaf `agent`
  must be exactly `"autorag-explorer"`.
- **Explicit unsafe values** — `artifacts: true`, `agentScope: "project"`,
  or a non-configured model are rejected even though their missing/null
  counterparts are safely autofilled.

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

## Stable coded dispatch errors

When a dispatch is rejected, AutoRAG emits a stable error code, the failing
field path, and a one-line `exactFix` string. The error format is:

```text
[<CODE>] field=<field> fix=<exactFix>
<selected skeleton>
```

`forceCorrectable` marks whether the caller can retry with the fix applied.
Non-correctable codes (`ADMIN_MUTATION_FORBIDDEN`, `CONTROL_FORBIDDEN`,
`SCHEDULE_FORBIDDEN`, `AGENT_IDENTITY`, `CWD_OUTSIDE_ROOTS`,
`NO_ACTIVE_QUERY`) require a structurally different request or an active
search context. Canonical role lines (retrievedAt, temporal metadata) are
normalized idempotently by `ensureRoleLines` rather than rejected — no
role-metadata rejection code exists in the catalog.

| Code | exactFix | forceCorrectable |
|------|----------|-----------------|
| `DISPATCH_MALFORMED` | remove fields not owned by this action | yes |
| `DISPATCH_ACTION_UNKNOWN` | use list\|get\|models\|status\|doctor or a supported launch shape | yes |
| `DISPATCH_ADMIN_MUTATION_FORBIDDEN` | do not mutate subagent definitions during AutoRAG search | no |
| `DISPATCH_CONTROL_FORBIDDEN` | launch a fresh autorag-explorer assignment instead of controlling an existing run | no |
| `DISPATCH_SCHEDULE_FORBIDDEN` | dispatch autorag-explorer work immediately; scheduling is disabled | no |
| `DISPATCH_ARTIFACTS_INVALID` | set args.artifacts = false | yes |
| `DISPATCH_AGENT_SCOPE_INVALID` | set args.agentScope = "user" | yes |
| `DISPATCH_AGENT_IDENTITY` | set every executable leaf agent = "autorag-explorer" | no |
| `DISPATCH_MODEL_MISMATCH` | set the referenced model field to the configured explorer model | yes |
| `DISPATCH_ASSIGNMENT_INVALID` | replace the task assignment with the canonical AUTORAG_ASSIGNMENT_V1 block | yes |
| `DISPATCH_QUERY_MISMATCH` | set originalQuery to the active user query verbatim | yes |
| `DISPATCH_CWD_MISSING` | set each executable leaf cwd to one configured search root | yes |
| `DISPATCH_CWD_OUTSIDE_ROOTS` | use a configured search root without symlink escape | no |
| `DISPATCH_NO_ACTIVE_QUERY` | dispatch only while an AutoRAG search query is active | no |

Diagnostics (`list`, `get`, `models`, `status`, `doctor`) bypass the
assignment, root, and model launch gates. They never receive artifacts,
agentScope, or model defaults. Control, mutation, and scheduling actions are
forbidden regardless of payload.

## Testing

Prompt tests should assert the role split and handoff fields as parsed contract
signals rather than snapshotting the full prompt. The RED-GREEN cases cover:

- mandatory `pi-subagents` and fatal missing-capability behavior;
- exclusive `gpt-5.6-sol` decisions and `gpt-5.6-luna` search/read work;
- original query, selected method, multiple variants, weak candidates,
  evidence, `retrievedAt`, and `asOf`/unknown metadata;
- unchanged Jikji, datasource trust, and exactly-once
  `emit_autorag_results` termination rules.
