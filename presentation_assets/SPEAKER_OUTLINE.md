# Journey of AutoRAG — Speaker Outline (20 min)

Event: ICML Night · Instruct.KR × Exa — AI Search Technology Meetup, Seoul (English).
Deck: `presentation.html` (open in a browser, press `F` for fullscreen, `S` for speaker view + these notes, arrows to navigate).
Target: ~1 min/slide, 21 slides. Leave ~2 min buffer for Q&A / transitions.

All figures/images are the real assets from the repos + the NIPA proposal deck (translated to English).

---

## ACT 1 — AutoRAG: AutoML for RAG  (slides 1–9, ~8 min)

**01 · Title (0:00)**
- "The journey of AutoRAG — from AutoML for RAG to search infra for agents." Tie to the meetup theme: retrieval infra for agents.

**02 · whoami (0:45)**
- Creator of AutoRAG, work out of NomaDamas (AI hacker house, Seoul). Won the Minister of Science & ICT Award for RAG pipeline optimization.
- One obsession: which retrieval pipeline is actually best for THIS data / use case.

**03 · The Journey (1:30)**
- The whole talk on one slide: three eras, the GOAL kept changing — AutoML for RAG → research automation → search infra for agents.
- Point at the star history: AutoRAG to ~4.8K; the new agent-era repos (research, MinSync, agentdir, jikji) rising bottom-right.

**04 · Why AutoRAG began (2:30)**
- There were already tons of RAG frameworks and eval frameworks. The unanswered question was "which one for MY data?"
- The missing idea = AutoML (hyperparameter search) applied to retrieval → AutoRAG.

**05 · AutoRAG = AutoML for RAG (3:30)**
- 4.8K stars, Trendshift, arXiv paper, real adopters (Uber, SKT, AWS, Hanwha Life). Bring QA+corpus → get the best pipeline.

**06 · Data creation (4:15)**
- You can't optimize without data. AutoRAG ships parse → chunk → QA generation to turn raw docs into an eval set.

**07 · How it optimizes: nodes (5:00)**
- A RAG pipeline = an ordered chain of nodes. Each node has swappable modules; AutoRAG greedily finds the best module+params at each stage.

**08 · The search space (5:45)**
- Every node has many modules. One YAML defines the space; AutoRAG runs trials, evaluates (F1/Recall/nDCG/MRR/METEOR/ROUGE/SemScore), emits summary.csv.

**09 · Deploy & dashboard (6:30)**
- Optimize → inspect in the dashboard → deploy as code / API / web. Fully reproducible.

---

## ACT 2 — AutoRAG-Research: reproducible RAG  (slides 10–13, ~4 min)

**10 · Where AutoRAG hit its ceiling (7:30)**
- Agentic RAG arrived; AutoRAG's fixed "advanced RAG" node structure couldn't express new pipeline shapes.
- Plus: datasets are hard to build, and research doesn't reproduce (every paper claims SOTA).
- Punchline: a tool that optimizes ONE pipeline shape can't answer "which shape is even right?"

**11 · AutoRAG-Research (8:30)**
- Built to fix all three. Unifies datasets + SOTA pipelines + metrics. Benchmark your idea against the real SOTA with one command.

**12 · One PostgreSQL, everything pre-built (9:15)**
- Unified & pre-embedded datasets (BEIR, MTEB, RAGBench, MrTyDi, BRIGHT, ViDoRe v1–v3, VisRAG, Open-RAGBench — text AND image).
- SOTA pipelines from papers (DPR, BM25, HyDE, Query Rewrite, Hybrid RRF/CC, BasicRAG, IRCoT, ET2RAG, VisRAG, MAIN-RAG).
- 3 commands: ingest → restore pre-computed embeddings → run all pipelines into one leaderboard.

**13 · Plugins + agent skills (10:00)**
- Plugin system (no fork) + an agent skill that queries your results DB in natural language → experiment at agent speed.

---

## ACT 3 — AutoRAG 2.0: search infra for agents  (slides 14–21, ~8 min)

**14 · The agent era arrived (11:00)**
- Claude Code, Codex, autonomous agents everywhere. The only question: in THIS era, what should AutoRAG become?

**15 · The answer + partners (11:45)**
- Build the missing knowledge-base layer for agents. Supported by NIPA's open-source program.
- Consortium: Marker × NomaDamas (lead), BrainCrew / TeddyNote (LangChain Ambassador — agent validation, education, community), 2e Consulting (public/finance domains, pilots).

**16 · Why a knowledge-base layer (12:45)**
- Analogy from the proposal: org knowledge today = flyers on a fridge (dig, call, stale). AutoRAG 2.0 = a delivery app (search, categorized, always current).
- It's the agent-native middle layer that lets an agent use your documents at 200%.

**17 · 5-layer architecture (13:45)**
- The core figure (translated from the NIPA deck). Between local files and the upper agent:
  1 Document understanding → 2 Storage → 3 Search orchestration → 4 Agent interface → 5 Operations (self-tuning loop).
- Every layer is a swappable module — not locked to any vector DB, engine, or model.

**18 · Naive optimization → memory-based personalization (15:00)**
- AutoRAG 1.0 = one-shot grid search over a config space. AutoRAG 2.0 = a memory-driven loop that adapts retrieval to YOUR corpus, queries, and feedback. Optimization becomes personalization.

**19 · Building blocks: MinSync + agentdir (16:00)**
- MinSync (Rust): git-free incremental indexing — re-embed only changed chunks, keep the storage layer fresh cheaply.
- agentdir (Rust): read-only virtual directories via CoW/reflink — give agents a better working tree without moving files.
- Both being integrated into AutoRAG 2.0.

**20 · Jikji: file maps for agents (17:00)**
- A simple prepared local map yields huge gains for agentic file search.
- HippoCamp benchmark (551 cases), raw Hermes vs. + jikji find: Hit@1 0.67 → 0.79, LLM calls 6,420 → 551, tokens 21.3M → 0.25M (86×), time 31,232s → 1,164s.
- Prepare once, judge a ranked candidate slate once — instead of crawling.

**21 · Closing (18:30)**
- The destination: the all-purpose librarian search agent — manages many data sources, digs everywhere (local files included), keeps itself current via memory.
- Journey: AutoML for RAG → research automation → search infra for agents. Thank you.

---

## Notes / things to double-check before presenting
- Speaker name/title on slide 1–2: set to "Jeffrey Kim" — adjust if you present under a different name/handle.
- Star counts are current as of build (AutoRAG 4.8K, Research 144, MinSync 51, agentdir 51, jikji 35). Refresh `img/star-history-render.png` on the day if you want the very latest.
- Partner descriptions on slide 15 are summarized from the NIPA proposal; confirm 2e Consulting scope wording is OK to share publicly.
- The whole deck is offline-capable (reveal.js + all images are vendored locally). No network needed to present.
