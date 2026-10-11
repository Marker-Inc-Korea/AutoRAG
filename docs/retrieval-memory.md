# Retrieval Memory

Retrieval memory is AutoRAG's cross-session record of **evidence that survived
judgment**. It is reference context only, never instructions: it is injected as
a `<memory_context>` user message and offered through the `check_memory` tool,
and it never reorders search results and never overrides the evidence in front
of the agent. A past method or source is a hint, not a rule.

For the Jev query pipeline itself (routing, decomposition, the datasource
check, the follow-up check), see [jev-decisions.md](jev-decisions.md).

## What it stores

The only unit memory stores is a **judged evidence record**: one piece of
evidence a final answer cited, plus the question it was cited for, the search
query and retrieval method that surfaced it, and Jev's judgment of it. A record
carries:

`id` (`<sessionId>:<stableEvidenceId>`, unique per run and evidence),
`sessionId` (one `searchDocuments` call), `conversationId` (the agent
conversation; current-conversation memory is keyed on it), `question` (the
user's original question), `searchQuery` (the query that surfaced the evidence,
the question itself when no sub-query is known), `method`, `source` (the opaque
source identifier), `stableEvidenceId`, `resultNumber` and `title` (of the
curated result the evidence backs), `excerpt` (bounded evidence text),
`probability` (Jev's P(supports)), and `createdAt`.

Nothing is evicted: the store has no cap, and every judged record stays until
it is folded into a long-term insight (its own copy stays too). Alongside the
judged records the file keeps the curated results and evidence chunks of past
searches, the long-term insights, and warnings.

## How evidence is judged

Judgment runs **after the final answer**: after the final reply, or
after the fast reply when Jev ends the run early on the fast answer. At
that point every evidence the answer cites is mapped back to its origin — the
search query and retrieval method that surfaced it (`EvidenceOriginIndex`) —
and sent to Jev. A run Jev routes to the `config` branch (self-configuration)
reports on AutoRAG's own settings, not retrieved evidence, so it is never
judged or stored.

Before judgment, each inline citation in the answer (`[e3]`, `[file:<path>]`) is
resolved against the run's `EvidenceLedger`: retrieval-issued ids attach the
harness-recorded source, method, and chunk, not model-written text. The origin
index then recovers the query that retrieved that chunk. Local files read
outside retrieval may be cited by absolute path; remote sessions cannot. A
config report has no citations and does not enter retrieval memory.

- **Questions.** Each cited evidence becomes one `noul` question: *does this
  evidence directly support the sentence it backs, and is it needed to answer
  the user's question?* A sentence that never cites a result is judged against
  the claim that the answer does not cite it.
- **One batched call.** Every evidence of the run goes into a single Jev call,
  with the per-evidence questions running in parallel, so judging one piece or
  ten costs about the same.
- **What the state contains.** A shared instruction that each question is about
  one evidence piece judged against the sentence it backs, then
  `User question: "<question>"` (JSON-quoted) and the full answer marked as
  reference only. Each question repeats the search query and retrieval method
  (JSON-quoted) and the evidence text (JSON-quoted, bounded to 1500
  characters). Quoting keeps a newline or a forged `User question:` line inside
  user- or model-written text from breaking the state's structure.
- **Threshold.** `EVIDENCE_SUPPORT_THRESHOLD` is **0.7**. Jev's
  P(supports) >= 0.7 stores the record as `judgedEvidence`; below 0.7 it is
  discarded. A unit Jev leaves unanswered is simply absent.
- **Failure behavior.** With Jev disabled or unreachable, **nothing is judged
  and nothing is stored**, and an `evidence-judgment-fallback` diagnostic
  carries the verbatim error (or Jev's own hint when it answered with no usable
  number). Judging never blocks a search.

## What goes into the prompt

When a question is searched, memory contributes three things, all rendered as
advisory markdown; every section is omitted when empty. Question and evidence
text is JSON-quoted. Insight table cells escape backslashes before pipes and
newlines, preserving literal backslashes without letting them cancel an escape:

1. **Current conversation memory** — everything judged earlier in the current
   conversation (the agent instance), in run order, in full **up to the 50 most
   recent records**; older records are noted as omitted.
2. **Similar past questions** — up to **25** past questions from *other*
   conversations, found by **hybrid BM25 + vector search** fused by reciprocal
   rank (RRF constant 60), so a question found by both rankings outranks one
   found by either alone and a rephrasing sharing no word is still found
   semantically. Question vectors are cached in `memory.json.vectors.db`;
   when the embedder fails, the search degrades to BM25 only and a diagnostic
   records the verbatim reason ("Semantic matching of similar questions was
   unavailable; ranked by keywords only. …").
3. **Long-term insights** — matching insights (see below).

The same content is available to the agent as it works through the
`check_memory` tool, which takes a query and returns advisory context.

**Datasource-selection hints.** Jev's datasource-selection step also receives
up to **5** of the similar past questions (hybrid BM25 + vector search), each
with up to **4** result titles and where each result's evidence came from. The
question and titles are JSON-quoted, so a newline or a fake `User question:`
line inside them cannot forge the state's structure. This is a hint, not a
rule.

## Privacy and datasource scope

- **Evidence from an unconfigured datasource is never shown.** Memory is shared
  across workspaces and configs, so every record is filtered against the
  *current* run: a record whose source is a datasource this run does not
  configure is dropped. The datasource-selection hints apply the same rule more
  strictly — only evidence from a configured datasource, a configured search
  path, or the web contributes a past result. Evidence with no recognizable
  origin is dropped, and a search left with no result is not shown at all.
- **Remote P2P sessions never read or write judged memory.** A remote peer's
  run neither consults nor stores it; its `check_memory` answers as if memory
  were empty.
- **Evidence text leaves the machine.** The question wording is judged against
  evidence excerpts and titles, so those go to Jev's backend (OpenRouter by
  default). Warming the vector cache also sends the past question texts to the
  local embedding runtime. Disable Jev to keep everything on the machine (no
  judgment, no storage).

## Long-term insights

Every **100** judged records (`INSIGHT_BATCH_SIZE`) are summarized once into
long-term insights; that is the only summarization, and a batch is not retried
if extraction fails (an `insight-extraction-failed` warning is recorded and the
save continues). Records are grouped by question topic; a cluster becomes an
insight only when it has **at least 5 judged records from at least 2 search
runs**, so one lucky search cannot become a lesson. An insight names the
sources and methods that keep supplying evidence for the topic (up to 3 each)
and a confidence, and is advisory only — it never disables a method.

## File format

Memory lives at `~/.autorag/memory.json` (or the configured memory path),
version **5**:

| Field | Contents |
|-------|----------|
| `version` | `5` |
| `curatedResults` | The curated results of past searches, with their evidence ids |
| `evidenceChunks` | Deduplicated evidence chunks (source, method, excerpt hash, first/last seen) |
| `judgedEvidence` | Every piece of evidence Jev judged to support its question; never evicted |
| `insights` | Long-term lessons summarized from judged batches |
| `pendingInsightEntries` | Ids of judged evidence not yet summarized; summarized 100 at a time |
| `warnings` | Recent diagnostics (`memory-reset`, `insight-extraction-failed`), capped at 50 |

A question-vector cache lives beside it at `memory.json.vectors.db`, a SQLite
file (built-in `node:sqlite`) holding one Float32 BLOB per question, keyed by a
hash of the question and tied to the embedding identity (provider, model,
dimension); a stale identity empties the cache. Each vector is written as soon
as it is embedded, so concurrent processes never overwrite each other's
vectors and a question is embedded at most once per model. An unreadable cache
file is replaced and rebuilt. `memory.json` itself is written atomically (temp
file + rename) under a `memory.json.lock` file lock, and a save folds in what
other processes wrote since this instance loaded.

**v4 migration.** A version-4 file loads and is rewritten as version 5. Each
result's user verdict and remote-peer tag are dropped (feedback state no longer
exists), and a v4 insight's `supportingSignalCount` becomes its evidence count.
A file that is neither v4 nor v5 is reset, with a `memory-reset` warning.

## What was removed

The redesign deleted the old feedback and scoring machinery outright:

- **Explicit feedback**: the `autorag feedback` CLI command, the
  `autorag.feedback` MCP tool, `feedbackId`, numbered feedback, and
  `recordFeedback*` APIs.
- **Implicit signals**: follow-up/retry scoring and method/context hints.
- **Result re-ordering**: memory no longer weights or reorders search results
  in any way.
- **The 500-record cap**: nothing is capped or evicted anymore.

Judged evidence replaced all of it: memory is only ever reference context.
