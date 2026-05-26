# AutoRAG — Librarian Agent

## Identity

AutoRAG is a **librarian agent** that searches, reads, curates, and reports
information from codebases and document collections.

## Architecture

```
Query → SEARCH (find candidates) → READ (examine files) → CURATE (extract insights) → OUTPUT (knowledge units)
```

The LLM is the curator. Search tools return raw file paths as working data.
The LLM reads files, extracts key information, and outputs numbered curated
knowledge units with no source paths exposed to the caller.

## Output Contract

**Caller sees:**
```
[1] authenticate() function — Middleware that extracts/verifies JWT. (lines 42-67)
[2] AuthConfig interface — JWT configuration type. (lines 5-12)
```

**Caller does NOT see:** file paths, retrieval method names, raw grep output.

**Internal mapping** (`<internal_mapping>` block) tracks `index → source path → method`
for feedback resolution only.

## Tools

| Tool | Purpose |
|------|---------|
| `search_posix` | File/content search via ripgrep |
| `read_file` | Read file contents with optional line range |
| `check_memory` | Query past search outcomes |

## Sessions

Each `prompt()` call creates a session with a unique ID. The caller uses the
session ID to reference results and submit feedback. Multiple concurrent
sessions are supported — each has its own isolated result registry.

```typescript
const session = await autorag.prompt("find auth middleware");
// session.sessionId = "abc-123"
// Caller sees curated info: [1] authenticate() function — ...

autorag.recordFeedbackByNumbers(session.sessionId, [1], [2]);
```

## Feedback Flow

1. Caller references results by session ID + number (e.g., session "abc", [1,3] useful)
2. Agent resolves numbers → internal mapping → source paths
3. Source paths → memory entries updated (useful/not_useful)
4. Memory informs future search strategy

## Files

| File | Role |
|------|------|
| `src/agent/agent.ts` | Agent class, system prompt, afterToolCall, mapping parser |
| `src/tool/read-file.ts` | read_file tool implementation |
| `src/tool/tool.ts` | Standalone search tool (raw output, no curation) |
| `src/retrieval/types.ts` | NumberedResult, CuratedResult, RetrievalMethod |
| `src/memory/memory.ts` | Feedback persistence and method priority |
| `src/cli/cli.ts` | CLI for raw search + feedback |
