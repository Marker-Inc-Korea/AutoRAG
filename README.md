# AutoRAG

**A self-evolving librarian agent for document collections.**

AutoRAG searches your PDFs, wikis, notes, research papers, and knowledge bases — then curates the results into clean, numbered knowledge units. No file paths. No raw grep dumps. Just answers.

Built on the [Pi framework](https://github.com/earendil-works/pi-mono).

## Why AutoRAG

### The problem with search tools

Every search tool gives you the same thing: a list of file paths and matching lines. Then *you* have to:
- Open each file
- Read the surrounding context
- Decide what's relevant
- Synthesize an answer
- Remember what worked for next time

That's the human doing all the hard work. The tool just points.

### AutoRAG does the hard work

AutoRAG is not a search tool. It's a **librarian** — it searches, reads, thinks, and reports back:

```
You ask:  "What were the key findings in the Q3 report?"

AutoRAG:
[1] Revenue grew 23% YoY to $4.2M, driven by enterprise contracts. (pages 3-5)
[2] Three new risk factors: supply chain, regulatory, talent retention. (pages 12-14)
[3] Headcount target missed by 12 — engineering hiring bottleneck. (page 8)
```

No file paths. No line numbers. Just curated knowledge you can act on.

### It gets smarter over time

AutoRAG has a **self-evolving memory system**. Every search teaches it something:

- Which retrieval methods work for which types of queries
- Which document areas are most productive
- What the caller found useful (via explicit feedback)

A fresh AutoRAG tries everything. A seasoned one knows exactly where to look. This is not a static configuration — it's learned behavior from real usage.

### Multiple retrieval methods, one interface

Different documents need different search strategies:

| Your documents | Best method | Why |
|---|---|---|
| Plain text, config files | grep (pattern matching) | Fast, precise, literal |
| Research papers, dense prose | Vector search (semantic) | Understands meaning, not just keywords |
| Legal documents, specifications | BM25 (keyword ranking) | Handles domain terminology well |
| Mixed collections | Hybrid (vector + BM25) | Combines precision and recall |

AutoRAG supports **pluggable retrieval methods**. It ships with a real-directory `posix` method wired through the `RetrievalMethodRegistry`, and the architecture is ready for vector, BM25, and hybrid backends. The `ResultMerger` handles cross-method score normalization and deduplication — you get one unified result set regardless of how many methods contributed.

### Real directory access

AutoRAG reads configured source directories directly. In extension mode it leaves Pi's built-in `grep`, `find`, `read`, and `ls` tools active instead of replacing them with a virtual filesystem layer. Programmatic retrieval still returns opaque root-relative source identifiers for feedback and curation, while MinSync continues to index parsed markdown mirrors under `.autorag`.

### Primary target: document collections

AutoRAG is built for **non-code document retrieval**: manuals, legal docs, internal wikis, meeting notes, research literature, knowledge bases, PDFs.

Code repositories work too (Pi's grep is ripgrep — already the best), but AutoRAG's real value shows on unstructured text where simple pattern matching isn't enough.

## Quick Start

### Interactive (Pi TUI)

```bash
pi --extension path/to/autorag/src/extension.ts
```

### Programmatic

```typescript
import { AutoRAGAgent } from "@autorag/librarian";
import { getModel } from "@earendil-works/pi-ai";

const agent = new AutoRAGAgent({
  model: getModel("anthropic", "claude-sonnet-4-20250514"),
  searchPaths: ["/path/to/documents"],
});

const session = await agent.prompt("summarize the compliance requirements");
// Mark which results were useful — AutoRAG remembers for next time
agent.recordFeedbackByNumbers(session.sessionId, [1, 3], [2]);
```

### Headless (single-shot)

```bash
pi --extension path/to/autorag/src/extension.ts \
  --print "What are the key deadlines in the project plan?"
```

## How It Works

```
    You ask a question
           │
           ▼
    ┌─────────────┐
    │  check_memory │ ← "Have I seen similar queries before?"
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Search     │ ← grep, find, vector, BM25 (pluggable)
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Read       │ ← Read promising files in full
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Curate     │ ← Extract key insights, not raw lines
    └──────┬──────┘
           ▼
    [1] Finding A — summary (location)
    [2] Finding B — summary (location)

           │
    Caller says "1 was useful, 2 wasn't"
           │
           ▼
    ┌─────────────┐
    │   Memory     │ ← Remembers what worked, adapts next time
    └─────────────┘
```

## License

MIT
