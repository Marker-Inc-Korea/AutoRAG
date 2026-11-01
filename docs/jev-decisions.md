# Jev Decisions

AutoRAG Agent requires one decision model: Jev,
[TypeSafe's Jev](https://docs.typesafe.ai/) judgment model, called through the
[`jev-use`](https://www.npmjs.com/package/jev-use) engine and also exposed to the
model as the `jev` tool. Jev is not a chat
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
is **required**: routing, the follow-up check, citation and evidence judgment,
and the `auto` permission gate all ask it, so there is no mode that runs
without it. It uses the hosted Dazzi Models backend when you are signed in
(`/login`) and OpenRouter otherwise.

## Session check: `decision-model-unavailable`

The first question of every session puts one probe question to the decision
model before any model call: the first question of a TUI session (again after
`/new`), and the first `searchDocuments()` call of an `AutoRAGAgent` instance
outside a TUI session (every `autorag search` process). A later question in the
same session does not repeat it. When the probe gets a number back, the run
goes on and records a `decision-model-checked` info diagnostic naming the
backend and model that answered. When it does not, the run stops with
`DecisionModelUnavailableError` and nothing else happens:

- `code`: always `decision-model-unavailable`.
- `reason`: `missing-credential` (no API key or no `/login`), `unauthorized`
  (HTTP 401/403), `payment-required` (402), `model-not-found` (404, or the
  backend says the model does not exist), `rate-limited` (429), `unreachable`
  (network failure, timeout, 5xx), or `invalid-response` (no usable verdict).
- The message starts with `[decision-model-unavailable]`, says AutoRAG cannot be
  used, quotes the backend's own error, and says what to fix.

`autorag search` prints that error and exits with status **3**; with `--json`
the error envelope carries `code`, `reason`, and, when known, `backend` and
`model`. The TUI shows the same message in the conversation; the next question
checks again, so fixing the credential and asking again is enough. Library
callers can catch `DecisionModelUnavailableError` (exported with
`DECISION_MODEL_UNAVAILABLE_CODE`) and branch on `reason`.

## Defaults

With no `jev` section, AutoRAG behaves as if the config said, **when signed in to
Dazzi Models**:

```json
{
  "jev": { "backend": "dazzi" },
  "queryDecomposition": { "model": { "provider": "dazzi", "id": "openai/gpt-6-luna" } }
}
```

and, **when not signed in** (the fallback):

```json
{
  "jev": { "backend": "openrouter" },
  "queryDecomposition": { "model": { "provider": "openrouter", "id": "qwen/qwen3.7-flash" } }
}
```

The choice is made when AutoRAG reads the config, from whether a Dazzi Models
`/login` credential is stored. `autorag init` writes the matching block into new
configs so it is visible and editable; an existing config keeps what it says. An
empty `{}` keeps the defaults. There is no opt-out: `"jev": false` and
`jev.enabled` are rejected with a config error. Jev sends the question text (and,
for citation and evidence judgment, the cited excerpts) to the chosen backend.

### Backends

`jev-use` resolves the backend from the environment. Set `JEV_BACKEND` to force
one, or leave it unset to let the first credential present win, in the order
TypeSafe -> OpenRouter -> Vercel AI Gateway. `dazzi` is not resolved by
`jev-use`: `backend: "dazzi"` (or `JEV_BACKEND=dazzi`, with `options.backend`
unset) builds the hosted Dazzi Models backend in AutoRAG itself, using the
credential stored by `/login` and the server root at `AUTORAG_BASE_URL`.

| `backend`    | Credential                                          | Model id on the wire |
| ------------ | --------------------------------------------------- | -------------------- |
| `typesafe`   | `TYPESAFE_API_KEY`                                  | `jev-1.13.0`         |
| `openrouter` | `OPENROUTER_API_KEY`                                | `typesafe/jev-1.13`  |
| `vercel`     | `AI_GATEWAY_API_KEY`                                | `typesafe-ai/jev`    |
| `dazzi`      | stored `/login` credential                          | `jev-1.13`           |

`JEV_MODEL` overrides the wire model id, and `JEV_BACKEND=mock` runs a keyless
dry run (used by the offline tests).

The OpenRouter path is pinned to `typesafe/jev-1.13`: `jev-use` 0.8.0's own
OpenRouter default (`typesafe/jev-latest`) is not a live OpenRouter model id and
returns HTTP 400. Set `model` to override the pin.

### Configuration fields

| Field                 | Meaning                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| `backend`             | `dazzi` when signed in, else `openrouter` (default); or `typesafe`, `vercel`. |
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

## Query pipeline (routing, decomposition, datasource check, follow-up check)

> **Conversations.** The pipeline below runs only for the first query of an
> interactive conversation (and for every programmatic `searchDocuments()` call
> outside a bound TUI session). Every later query in the same session is a
> plain agent turn: the user's message reaches the model exactly as typed, with
> no Jev branch, decomposition, datasource check, baseline prefetch, fast
> answer, follow-up check, or per-turn instruction wrapper. The system prompt
> tells the model that later messages continue the conversation; it answers
> from evidence it already gathered when that suffices, and otherwise searches
> by tool call. Evidence ids are numbered across the whole conversation and
> saved in the session file, so an `[eN]` the model saw in any earlier turn
> resolves after any number of turns or searches, and after `/resume`.
> `/new` starts a fresh conversation with the full pipeline again.

Jev drives a pipeline that runs in the two-phase
search **before** the fast answer. Jev answers two typed questions about the
user question in one batched call:

1. **Branch** (`choice`): how must the assistant handle the question? Four
   options are always offered — `local`, `organize`, `extra` — plus `config`
   for local sessions. There is no web branch and no intrinsic-knowledge-only
   branch. The highest-probability branch wins once it clears the confidence
   floor (see below); a weak or probability-less `organize`, `extra`, or
   `config` verdict keeps `local`, whose full plan still runs (decomposition
   and the datasource check), with a `query-route-fallback` diagnostic naming
   the floor.
   - `local`: answering needs anything beyond the assistant's intrinsic
     knowledge — information only the user can reach (files on their computer,
     Discord/KakaoTalk/Slack chats, email, notes, calendar, history) or facts
     that may be recent, specific, or changing (public ones included: on this
     branch the web stays with the agent after the fast answer). Jev is told to
     prefer `local` whenever the question refers to the user's own life,
     situation, plans, or records — "my/I/our", a named friend, family member,
     or colleague, or "my case/hearing/appointment/routine" — and whenever it
     is unsure.
   - `config`: the user wants to view, change, or test AutoRAG's own settings
     (model, providers, API-key environment variables, Jev, search roots,
     datasources).
   - `organize`: the user wants files or folders **acted on**, not information
     found: move or copy into folders, tidy or sort a folder, bulk or pattern
     rename, rename by what a file's content says, convert formats, create
     folder structures, archive, move to the trash. A request that only asks to
     find or read a document is `local`.
   - `extra`: neither of the others fits and no search of the user's own
     records is needed first. Either the request is answerable without looking
     anything up — stable general knowledge, a definition, simple reasoning,
     small talk — or it is a task for the agent's tools that is not file
     organization and not AutoRAG configuration: committing code, pushing a
     branch, opening a pull request, running tests or a build, installing a
     package, running a script, writing or editing code (programming work is
     `extra` even when the code names the user's files, and so is a request to
     run something and report the outcome). One turn with every tool the
     session has; no retrieval, no fast answer, no verification. The prompt
     tells the model to answer from its own knowledge when that is reliable,
     to look facts up with the retrieval/datasource tools or `web_search` /
     `web_fetch` when they are not its own, and otherwise to carry the task
     out. There is no tool ban.
2. **Decomposition** (`noul`): does the question need several search queries
   (multiple sub-questions, comparisons, several facts to confirm)? A
   probability of 0.5 or more means yes.

**Confidence floor.** `NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD` (0.75) is the
minimum branch probability required to leave local search. A non-local branch at
or below it, or one Jev reported without a probability, becomes a `local` route
with the floor named in `query-route-fallback`; the rejected branch's
probability is still reported as `routeProbability`. The floor is not an outage:
the run keeps a full local plan (decomposition and the datasource check still
run) — only a missing credential, an unreachable backend, or an unusable verdict
keeps the bare single-query local search. Removed by #1995 and restored here
because a weakly-favored non-local verdict skips retrieval entirely, which is
how a question about the user's own records can be answered from general
knowledge.

### Branch calibration (live Jev)

Measured on live OpenRouter Jev (`typesafe/jev-1.13`) with 22 labeled questions
in Korean and English (5 general knowledge, 4 tool tasks, 7 personal records —
"my"/"our" phrasings included — 2 recent public facts, 2 file organization, 2
AutoRAG settings), through `routeQuery(..., { selfConfig: true })`, the way the
CLI runs it. Three consecutive runs scored 22/22: every non-local verdict
cleared the floor (lowest 0.77, highest 1.00), and every personal-records and
recent-fact question topped `local` at 0.65-1.00. The floor therefore pulled
nothing back in those runs.

It did in the wording iteration before the final one: "run the tests in this
repository and tell me which fail" won `local` at 0.63/0.65 against a weak
`extra` verdict. That is why the `extra` option now states outright that a
request to run something (tests, a build, a script, a command) and report the
outcome belongs to `extra` even when what it runs lives in the user's own
repository, and that programming work is `extra` even when the code names the
user's files or folders. The unit tests pin both the floor's decision and the
full local plan it keeps.

What happens next:

| Branch   | Pipeline                                                                                 |
| -------- | ---------------------------------------------------------------------------------------- |
| `organize` | Skips retrieval, decomposition, the fast answer, and verification. One tool-capable turn with every tool the session has: no route changes the active tool set, so the turn prompt names the tools the task uses (`bash`, the file tools, `convert_files`/`list_file_conversions`, the duplicate scan, and file-name discovery) and the retrieval, datasource, and web tools it must not call. The prompt also carries the packaged `organize` skill as a `<skill name="organize" location="autorag-skill://organize">` invocation, which holds the rules: plan first, no overwrites, no permanent deletion (trash only), a listing check afterwards, and a final report of every change as old path → new path. If the skill cannot be loaded the run falls back to `local` with an `organize-skill-unavailable` diagnostic. No results; the run is not recorded in retrieval memory. |
| `extra` | Skips retrieval, decomposition, the fast answer, and verification. One turn with **every** tool the session has and no tool ban: the prompt has the model answer from its own knowledge when that is reliable, look facts up with the retrieval/datasource tools or `web_search`/`web_fetch` when they are not, and otherwise do the work. The final message is either the answer (the same language and citation rules the other answer prompts use) or a short report of what was done. Cited retrieved evidence becomes results and feeds retrieval memory exactly like a local run (`recordStructuredResultsSession` without `isolateMemory`, plus the evidence judgment); an answer or report that cites nothing stays out of memory. |
| `config` | Skips retrieval, decomposition, the fast answer, and verification. The turn prompt carries the packaged `autorag-setup` skill as a `<skill name="autorag-setup" location="autorag-skill://autorag-setup">` invocation, the active config path, and the pi agent dir, names `bash`/`read`/`edit`/`write` as the tools to use, and keeps retrieval, datasource, and web search out of the turn. The model edits the config, verifies with `autorag health`/`models list`, and reports old → new as its final message (no results, and the run is not recorded in retrieval memory). If the skill cannot be loaded the run falls back to `local` with a `self-config-unavailable` diagnostic. |
| `local`  | Datasource check (configured datasources **and** the public web) starts together with the route check and runs alongside decomposition → Jikji + MinSync + every selected datasource + one web search per query (when the web was selected), per query, in parallel → merged pool → rerank against the original question (when `rerank` is configured) → fast answer → follow-up check → verification (only if needed). |

### One tool set for every route

No route changes the session's active tools: `organize` and `config` no longer
swap in a narrower or builtin-only list, and a run without a bound TUI session
keeps the tools it was built with. A stable tool list keeps the model's cached
prefix intact and never leaves it referring to a tool that just vanished, so
what a turn may use is stated in its prompt and in the skill it loads.

A packaged agent skill reaches the turn as a `<skill name="…" location="…">`
block whose location is opaque (`autorag-skill://<name>`) — never a filesystem
path. Skills live under two shipped directories: `skills/<name>/SKILL.md` for a
skill a parent agent may also copy (`autorag-setup`, which is both the
config-branch skill and a parent-agent skill, as the README documents), and
`internal-skills/<name>/SKILL.md` for a skill that is the librarian's own
(`organize`), so parent-agent skill folders never grow a skill that only makes
sense inside the librarian's own turn.

The public web is one more source the `local` branch can search, chosen by Jev
in the datasource check (see below) whenever the web tools are enabled. When it
is selected, one web search runs per search query in parallel with Jikji and
MinSync, and its hits join the same reranked pool under their URL as the source.
The fast phase may also call `web_search` itself (and `web_fetch`, to read a
promising page) when the question turns on a public fact that may be recent or
that changed since the model's training and the baseline does not settle it; no
other tool is allowed in that phase. Verification keeps its own web nudge, for
when local evidence turns out missing, thin, or possibly outdated.

After a `config` turn that changed the config file, the host re-resolves it the way the next launch will (JSON, schema, and that the model id exists in the pi catalog or a declared endpoint). If it no longer resolves, the pre-turn file is restored, a warning is appended to the report, and a `self-config-rolled-back` diagnostic is recorded, so a bad model id can never leave the agent unable to start. A missing credential is not a rollback reason; that is reported by `autorag health` as `auth_missing`. An unchanged file is never touched.

### Datasource check before the fast answer

On the `local` branch, Jev answers one more batched call: one `noul` per
registered datasource (configured, with retrieval methods), "Should the `<id>`
datasource be searched to answer the user question?". When the web tools are
enabled, the public web is one more candidate in that same batch, under the
reserved id `web`, asked in the same `noul` shape but worded for what the public
internet holds (current events, recent releases, prices, exchange rates, version
numbers, public facts beyond the model's knowledge or likely to have changed
since its training — never the user's private records). The gate runs even when
no datasource is configured, so a web-only run still asks it. The registered set
comes from the config: `autorag search` reads it on every call, while `autorag
tui` reads it at startup, so restart it after adding,
disabling, or re-describing a datasource. Every candidate at 0.5 or
above is searched with every search query (the original or the decomposed
ones), and its hits join Jikji and MinSync in the pool the reranker orders,
before the fast answer; the skipped ones are not searched before the fast answer
(the verification phase can still call `search_datasource_<id>`). A
`datasources-selected` diagnostic lists the selected and skipped candidates —
including `web (p=…)` — with their probabilities. If the check fails (missing
credential, unreachable
backend), neither a datasource nor the web is searched before the fast answer
(`datasource-selection-fallback`).

The check needs only the question, so it is sent **while the route check is
still in flight**; the two calls overlap and the run waits for the slower one
instead of their sum. It is speculative: when the route is not `local`
(`organize`, `extra`, `config`), or the routing verdict is unusable
(`query-route-fallback` without the confidence floor), its verdict is
discarded — it is not searched, its `datasource-select` progress step is
removed, and it records neither `datasources-selected` nor
`datasource-selection-fallback`, so a run that never used it never reports it.
On those routes the extra call is the only cost: a few hundred tokens of state.

Datasource ids come from the config's `datasources` keys, which are only
required to be non-empty, so an operator can name one `web`. When that happens
the web candidate is withheld and a `web-datasource-id-taken` diagnostic says
so: the question id `datasource:web` then names the configured datasource and
that alone, which is judged as usual, and the run searches no web before the
fast answer. (With web tools disabled there is no web candidate at all — no
question, and web-attributed past results are dropped from the state.)

A selected web candidate runs one web search per search query in parallel with
the other sources, on the same provider chain, ordering, and credentials as the
`web_search` tool. Its hits enter the merged pool with the URL as their source
and title plus snippet as their content, are reranked against the original
question with everything else, and are registered in the evidence ledger under
the `web_search` method, so the fast answer cites them as `[eN]` and they become
results whose source is the URL. A failed web search contributes no evidence and
a `web-search-failed` diagnostic naming the query and the reason; the run
continues without it.

The state Jev judges has three parts, followed by the question:

1. **Datasource catalog**: `id (type): description` for every datasource, plus
   the public web as `web (web): …` when its tools are enabled. Write each
   `description` in the config to say what the datasource holds
   (channels, rooms, people, topics, time range). The state tells Jev that
   every description is a **short, non-exhaustive summary**: a datasource can
   hold other topics, people, and conversations that its description does not
   mention, so a datasource is not ruled out just because its description is
   silent on the question's topic.
2. **Similar past questions** (only when retrieval memory has any): up to 5 of
   the similar past questions found by hybrid BM25 + vector search (see
   [retrieval-memory.md](retrieval-memory.md)), each with up to 4 result titles
   and where each result's evidence came from (`[kakao]`, `[local files]`,
   `[web]`, ...). The question and titles are JSON-quoted, so a newline or a
   fake `User question:` line inside them cannot forge the state's structure. A
   result titled as not found or negative tells Jev that datasource was
   searched and did not have the answer. Jev treats this as a hint, not a rule.

   These titles are model-written text about your private content and the
   state goes to the Jev backend (OpenRouter by default), so a past result is
   shown only when it is safe for the **current** run: a result is shown only
   when every place its evidence came from is a candidate this run searches —
   a datasource configured for this run, a configured search path (`local
   files`), or the web when the web candidate is offered. Evidence with no
   recognizable origin drops the whole result, a `[web]` hint is dropped when
   web tools are off, and a search left with no result is not shown at all.
3. **User question**.

The question wording was checked on live OpenRouter Jev (`typesafe/jev-1.13`)
with the maintainer's 10 real datasources (descriptions written from each
archive's actual content) and retrieval memory (113 past searches). The set has
10 labeled questions: 6 naming a datasource, 3 whose topic no description
mentions (a restaurant tip in KakaoTalk, a birthday in a Discord server, a
flight booking email), and 1 that only description content can resolve (the
shared-fund ledger lives in one of two Discord servers). Over 3 runs, 30/30
passed. Required datasources scored 0.71-0.96, and datasources the question
excluded scored at most 0.38
(`scripts/manual-qa/run-qa-jev-datasource-selection-live.ts`). With memory, a
similar past question sharpens the choice: for "dependabot PR 알림", Discord,
Spotlight, and NomaDamas dropped from 0.53-0.56 to at most 0.35 once memory
showed the earlier answer came from Slack, and the KakaoTalk choice for "구봉님
랄프톤 공지" rose from 0.91 to 0.98.

The web candidate was checked the same way on live Jev with the maintainer's 11
real datasources (web tools enabled). A recent public fact ("What is the latest
stable Node.js LTS version right now?") scored the web at 0.95 with every
datasource at 0.24 or below, and a question about the user's own records ("내
릴리즈 플랜 노트에 Q3 런치 날짜가 뭐라고 적혀 있어?") scored it 0.05 and selected
KakaoTalk (0.61), Slack (0.57), Spotlight (0.88), and NomaDamas (0.58) instead
(`bun src/cli/index.ts search "<q>" --json --debug`, plus
`test/live-e2e/jev-web-datasource.test.ts` for the prefetch end to end).

### Follow-up check after the fast answer

After the fast answer on the `local` branch, Jev answers one more
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
queries. The default model is `dazzi/openai/gpt-6-luna` when signed in to Dazzi
Models, and `openrouter/qwen/qwen3.7-flash` otherwise.
`queryDecomposition.model` takes the same fields and credential rules as the
top-level `model`; `"queryDecomposition": false` decomposes with the search
session's own model instead.

The OpenRouter fallback was picked on a live OpenRouter benchmark: the real decomposition
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
the original question (diagnostic `query-route-fallback`). A failed decomposition
searches the original question (`query-decomposition-failed`). A failed
datasource check searches no datasource before the fast answer
(`datasource-selection-fallback`). Every routed run records its branch and
queries as a `query-routed` diagnostic, and the datasource check's picks as
`datasources-selected` (`--debug` shows both). Every search is
two-phase (fast answer, then verification unless Jev ends the run); there is
no single-phase mode.

## Citation relevance check

The model sometimes cites a chunk that has nothing to do with the sentence it
backs or with what the user asked: a table of contents, a navigation bar, a
copyright footer, a generic introduction. Every `[eN]` the harness resolves
against the evidence ledger is real, but real is not the same as related, so
with Jev on, each cited chunk is checked before the answer is returned.

- **One `noul` question per cited chunk**, all in **one batched call**: *is the
  evidence relevant to the answer sentence that cites it, or to what the user
  asked?* The question carries only that chunk (JSON-quoted, bounded to 1500
  characters) and every distinct sentence that cites it.
- **What Jev reads as the user's side: everything the user typed in the
  conversation**, oldest first, each message JSON-quoted in the state: the
  questions, and the steers typed while a search ran. A follow-up such as
  "그럼 월세는?" is only meaningful next to the question before it, so the last
  message alone is not enough. It is the text as typed, never the harness prompt
  that wraps the first question with baseline evidence and formatting rules. It
  is saved in the session file next to the evidence ledger and result numbers, so
  `/resume` and tree moves restore exactly the queries on the branch in view; a
  run that owns no conversation starts empty. The answer itself is not sent:
  the citing sentences already carry what it claims.
- **Threshold.** `CITATION_RELEVANCE_FLOOR` is **0.3**: below it the chunk is
  removed from the citations; at or above it the chunk stays. This is
  deliberately looser than the memory gate below (0.7, "directly supports and is
  needed"): a chunk that is on topic, or backs only part of a claim, must stay.
  Only evidence with no information on the subject is removed.
- **What removal means.** Cited chunks are read straight from the answer's
  markers (`citedChunks`), judged, and then the answer is derived once, skipping
  the rejected chunks. Judging numbers nothing, so a rejected source never spends
  one of the conversation's never-reissued result numbers. A source whose every
  cited chunk was rejected disappears and the rest keep their numbers; a source
  with one rejected chunk among several keeps its marker and loses only that
  chunk. A `citation-irrelevant` diagnostic lists the removed sources. A removed
  chunk is not an invented id, so no `citation-without-result` warning is raised.
- **What is never judged.** Evidence that is only a path (a file the model
  located but did not read) has no text to judge and is kept.
- **Failure behavior.** If the call fails or Jev gives no usable
  verdict for any chunk, **every citation is kept** and a
  `citation-relevance-fallback` diagnostic carries the verbatim reason. A chunk
  Jev left unanswered is kept. The check never blocks a search.
- **Order.** It runs where the answer is derived (the fast answer, the final
  answer, and the `extra` turn's answer), before the preliminary is published and before
  the memory judgment below, so a rejected chunk is also never judged for or
  stored in memory.

Measured on live OpenRouter Jev (`typesafe/jev-1.13`) with 32 labeled cases
(14 on-topic, partial, or multi-turn chunks that must stay, 18 junk chunks that
must go: tables of contents, introductions, navigation, copyright and email
footers, chatter, bot notifications, unrelated sections), 5 runs each, through
the agent's own `citedChunks` and `judgeCitationRelevance`
(`scripts/manual-qa/run-qa-citation-relevance-live.ts`):

| Group | Correct |
| ----- | ------- |
| on-topic chunks retained | 35/35 |
| partially related chunks retained | 20/20 |
| tables of contents removed | 15/15 |
| introductions removed | 15/15 |
| boilerplate removed | 15/20 |
| off-topic chunks removed | 20/25 |
| multi-turn (elliptical last message), kept and removed | 30/30 |

Junk removed 80/90 and relevant chunks retained 70/70. The multi-turn cases are
the ones the last message alone cannot decide: "그럼 월세는?" keeps the rent
chunk and still drops a table of contents, and a topic change ("점심 메뉴
뭐였지?" after lease questions) is not rescued by the earlier lease turns.

Two cases fail every run, and both are on the removal side: a court case header
block cited for a division ruling scores about 0.40 and a chunk of party
background cited for the same ruling scores about 0.34, so both are kept
although they carry no ruling. They sit just above the floor, next to chunks that
must stay (the security-deposit chunk cited for the rent sentence scores about
0.45), so lowering the margin means more false removals of on-topic chunks.
The floor trades a few kept near-misses for never dropping a related chunk.

## Evidence judgment for memory

Retrieval memory stores only evidence that survived judgment, so after the
final answer (the final reply, or the fast reply when Jev ends the
run early) every cited evidence is put back to Jev. Each evidence is mapped to
the search query and retrieval method that surfaced it, then judged as one
`noul` question: *does the evidence directly support the sentence of the answer
it backs, and is it needed to answer the user's question?* The wording judges
one evidence piece against its backed sentence, and the full answer is supplied
only as reference; user-, model-, and evidence-written text is JSON-quoted so
none of it can forge the state's structure. Every question in a run is sent in
**one batched call**, so one or many pieces cost about the same.

At P(supports) >= **0.7** the evidence is stored in `~/.autorag/memory.json`;
below 0.7 it is discarded. With Jev unreachable nothing is judged or
stored, and an `evidence-judgment-fallback` diagnostic carries the verbatim
error. See [retrieval-memory.md](retrieval-memory.md) for the stored fields,
the prompt context, privacy rules, and the long-term insights.

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
