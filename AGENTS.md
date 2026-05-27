# AutoRAG — Pi-Powered Librarian Agent

## Identity

AutoRAG is an **over-powered librarian agent** built on the [Pi framework](https://github.com/earendil-works/pi-mono).
It searches, reads, curates, and reports information from codebases and document collections.

Unlike raw search tools, AutoRAG adds an **intelligent curation layer**: it doesn't just find files —
it reads them, extracts key insights, and delivers numbered knowledge units with no raw paths exposed.

## Architecture

Built on Pi's extension system, AutoRAG reuses Pi's battle-tested tools (grep, find, read, ls)
and adds its own intelligence layer on top:

```
Pi Built-in Tools          AutoRAG Extension Layer
┌──────────────────┐      ┌──────────────────────────────────┐
│ grep (ripgrep)   │      │ Memory System (query history)     │
│ find (fd)        │ ───▶ │ Curation Layer (LLM extraction)   │
│ read (file read) │      │ check_memory (adaptive strategy)  │
│ ls (directory)   │      │ Manifest System (indexed stores)  │
└──────────────────┘      │ Feedback Loop (learn from usage)  │
                          └──────────────────────────────────┘
```

## Usage

### As Pi Extension (Interactive)
```bash
pi --extension path/to/autorag/src/extension.ts
```

### As Library (Programmatic)
```typescript
import { AutoRAGAgent } from "@autorag/librarian";
import { getModel } from "@earendil-works/pi-ai";

const agent = new AutoRAGAgent({
  model: getModel("anthropic", "claude-sonnet-4-20250514"),
  searchPaths: ["/path/to/codebase"],
});
const session = await agent.prompt("find authentication middleware");
agent.recordFeedbackByNumbers(session.sessionId, [1, 3], [2]);
```

## Output Contract

**Caller sees:**
```
[1] authenticate() function — Middleware that extracts/verifies JWT. (lines 42-67)
[2] AuthConfig interface — JWT configuration type. (lines 5-12)
```

**Caller does NOT see:** file paths, retrieval method names, raw grep output.

## Files

| File | Role |
|------|------|
| `src/extension.ts` | Pi extension factory — registers tools, hooks events, injects system prompt |
| `src/agent/agent.ts` | AutoRAGAgent class for programmatic/library usage |
| `src/agent/system-prompt.ts` | System prompt builder (shared between extension and library modes) |
| `src/agent/parse-mapping.ts` | Internal mapping parser (number → source → method) |
| `src/memory/memory.ts` | Feedback persistence and method priority scoring |
| `src/memory/renderer.ts` | Memory context renderer for system prompt |
| `src/memory/check-memory-tool.ts` | check_memory tool (ToolDefinition + AgentTool) |
| `src/manifest/loader.ts` | YAML/JSON manifest loader for indexed data stores |
| `src/retrieval/types.ts` | Core retrieval type definitions |
| `src/retrieval/registry.ts` | Method registry for multi-method orchestration |
| `src/retrieval/merger.ts` | Cross-method result merging and deduplication |

## Memory System

AutoRAG remembers past search outcomes across sessions:
- Tracks which queries + methods succeeded or failed
- Prioritizes methods that historically work for similar queries
- `check_memory` tool lets the LLM query this history before searching
- Feedback loop: callers mark results as useful/not-useful → improves future searches

## Feedback Flow

1. Caller references results by session ID + number (e.g., session "abc", [1,3] useful)
2. Agent resolves numbers → internal mapping → source paths
3. Source paths → memory entries updated (useful/not_useful)
4. Memory informs future search strategy
