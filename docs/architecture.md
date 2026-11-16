# Architecture

How AutoRAG Agent is put together, moved out of the README.

## Core Values

<p align="center">
  <img src="assets/autorag-agent-principles.png" alt="AutoRAG Agent Core Principles" width="100%" />
</p>

Three principles drive every design decision in AutoRAG Agent:

1. **Never migrate your data to search it.** Traditional RAG systems force you to upload, ETL, and duplicate your files into a centralized vector database. AutoRAG Agent federates your data **in place**, querying CLI-native stores (`lazykatok`, `discrawl`, `slacrawl`, `mailcrawl`) where your data already lives. Results retain opaque, source-native identities (`/kakao/...`, `/slack/...`) that preserve local access control and privacy. *(See our [Competitive Landscape Study](competitive-landscape-2026-09.md) on why in-place federation is the durable differentiator).*

2. **Just works — no RAG degree required.** No pipeline tuning, no vector-DB maintenance, and no remote embedding keys required. AutoRAG Agent automatically manages [MinSync](minsync-setup.md) for incremental Change Data Capture (CDC) chunking and provides a local [embedding backend](minsync-setup.md#embedding-backends) out of the box with zero external telemetry. A [remote rerank model](rerank.md) is an opt-in extra: it runs only when you name one in `rerank.model`, and otherwise the first answer is built from each source's own top 10.

3. **Fast by design.** Rather than coordinating slow multi-agent hierarchies, a single configured model owns the entire retrieval, direct-read, and curation loop. Local CDC chunks (SPLADE sparse, vector, and hybrid modes) deliver rapid, low-latency turnaround across multi-turn research queries.

---

## Architecture & How It Works

AutoRAG Agent orchestrates five integrated subsystems:

1. **Retrieval Memory (`check_memory`):** After each answer, cited evidence Jev judges to genuinely support it is stored in `~/.autorag/memory.json`. Later searches receive this conversation's evidence, similar past questions, and long-term insights as **advisory reference** — it never reorders results or overrides current evidence. See [docs/retrieval-memory.md](retrieval-memory.md).
2. **Pluggable Multi-Method Retrieval:**
   - **SPLADE Sparse Search:** Learned-sparse keyword ranking via MinSync's KURE-Sparse-nano encoder over parsed markdown mirrors; it replaced BM25 in MinSync 0.6 and matches Korean inflected forms (`인증서를` → `인증서`) with no tokenizer library. Exact identifiers among many look-alike files rank less reliably than with BM25. See [docs/minsync-setup.md](minsync-setup.md#sparse-retrieval-splade).
   - **Semantic Vector Search:** Dense vector retrieval over CDC chunks via the shared MinSync embedder.
   - **Hybrid Search:** MinSync fuses sparse and vector ranking over the same chunks.
   - **Jikji Find-First Discovery:** Local CLI-backed fast discovery answer packs.
   - **Everything File-Name Search (Windows):** The bundled [voidtools Everything](https://www.voidtools.com/) indexes file and folder names under your search roots so the agent finds files by name, extension, path, size, or date instantly (`everything_search`).
   - **FSearch File-Name Search (macOS/Linux):** [fsearch-cli](https://github.com/NomaDamas/fsearch-mac) (FSearch) keeps a live, per-workspace name index of your search roots so the agent finds files by name, extension, path, size, or date instantly (`fsearch_search`); without fsearch-cli installed it degrades to a slow filesystem walk.
   - **Calculator:** Exact decimal arithmetic (40 significant digits, no floating-point drift) plus calendar and clock math — days between dates, date offsets with month-end clamping, weekdays, ISO weeks, business days with holidays, the current time, and DST-aware time zone conversion — through the `calculator` tool. The system prompt, including the fast-answer phase, orders the model to compute every number and date with it instead of in its head. It runs in-process, reads no files, and needs no configuration.
   - **Datasource Skills:** Federated retrieval across every configured external datasource.
3. **Result Merger & Scope Narrowing:** Cross-method deduplication, score normalization, and ordinary query-scope narrowing.
4. **Direct Evidence Reading (`bash`):** The agent directly opens and inspects promising files with `cat`, `grep`, or `find` to verify facts against ground truth.
5. **Curation:** The final answer is the model's plain reply; the harness resolves its inline `[eN]` citations into numbered, source-backed results; cited evidence that Jev judges to genuinely support the answer is remembered as reference context for later searches ([docs/retrieval-memory.md](retrieval-memory.md)).

### Pi host boundary

AutoRAG uses `@earendil-works/pi-coding-agent` as the runtime host for model sessions. Pi owns provider credentials and OAuth storage, model-runtime dispatch, session JSONL persistence/resume, extension loading, lifecycle events, and the built-in `read`/`bash`/`edit`/`write`/`grep`/`find`/`ls` tools. AutoRAG registers only its domain tools and keeps orchestration outside the host: datasource orchestration, MinSync/Jikji preparation, memory hints, fast-to-verification two-phase search, structured result emission, and remote-session filtering.

The CLI TUI (`autorag tui`) is Pi's interactive mode hosted on the AutoRAG librarian: there is no separate AutoRAG renderer. Pi owns the terminal UI and its native in-session commands — `/login`/`/logout`, `/model`, `/resume`, `/new`, `/tree`, `/compact`, and `/settings` — so provider sign-in and model selection work on first launch even when no AutoRAG model is configured. AutoRAG registers its retrieval tools and runs the two-phase search in the same Pi session, where the model's progress notes, first answer, and final answer are ordinary assistant messages, so resume and two-phase search keep working together.

A fresh session — at launch and after `/new` (or `/reload`) — opens on AutoRAG's own welcome screen: the AutoRAG logo, the version, and three rotating tips, centered and scaled down to a smaller logo or a plain wordmark on narrow terminals. Pi's own logo, key-hint list, and loaded skills/prompts/extensions listing are hidden by default (`quietStartup: "header"`). A `quietStartup` saved in Pi's global or project `settings.json` wins: `true` removes the banner, `false` brings Pi's resource listing back.

Typing while a search runs works like steering one of Pi's own turns. **Enter** steers the running search: AutoRAG treats the text as a correction of a search that is going wrong, delivers it before the search's next model request (also between phases, where no turn is streaming), cuts in-flight retrieval short so the correction lands promptly, and always runs the verification phase so the answer reflects the correction. **Alt+Enter** queues a new question that runs after the current search answers. **Esc** stops the search instead of letting it continue into its next phase. Pi's registered `/` commands still run during a search; any other line starting with `/` (a path, for example) is a steer like any other text. A query typed after a search answers is a new search in the same session, as before.

From the moment a question is submitted, a live progress widget sits above the editor: a spinner, the elapsed time, and one line per step the search is running (routing, splitting the question, each retrieval source with its hit count and time, ranking, the first answer, verification). A slow source shows its own running timer, a failing source shows its error text verbatim, and the widget disappears when the answer is final or **Esc** stops the search. While the model streams, Pi's own `Working` loader takes over and the widget keeps only the header and steps. Every tool call also carries a short model-written intent (the `i` argument): Pi's `Working` loader shows the newest one, and `autorag search` and the desktop app receive it as progress, so the model does not write separate progress notes before tool calls.

Update notices are AutoRAG's, not Pi's: `autorag tui` suppresses Pi's "run `pi update`" banner (Pi is a bundled host, not a separate install) and instead checks the published `@autorag/librarian` release, showing a one-line notice in the session when a newer version exists. Set `AUTORAG_NO_UPDATE_CHECK=1` to skip the lookup; `AUTORAG_UPDATE_CHECK_URL` overrides the registry endpoint.

---
