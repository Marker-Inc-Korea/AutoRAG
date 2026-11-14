# MCP and AutoRAG Lite

AutoRAG as an MCP server (Lite, model-free) and as an MCP client.

## AutoRAG Lite MCP Server

Register the package's stdio server with the host; the host owns the process
lifecycle and sends MCP tool calls:

```bash
AUTORAG_CONFIG=/absolute/path/to/.autorag/config.json autorag-mcp
```

For Claude Code and Codex registration commands, use
[`skills/autorag-lite-setup/SKILL.md`](../skills/autorag-lite-setup/SKILL.md).
The MCP contract source of truth is [`src/mcp/server.ts`](../src/mcp/server.ts)
(tool registration, schemas, handlers) plus [`src/mcp/index.ts`](../src/mcp/index.ts)
(stdio entrypoint). Discover tools and schemas with MCP `tools/list`; do not
hard-code a tool count.

Core Lite tools include `autorag.status`, `autorag.search`,
`autorag.search.files`, `autorag.datasources.list`, `autorag.datasources.get`,
`autorag.refresh`, `autorag.report`, and `autorag.evidence`. Configured
integrated datasources expose additional scoped search tools. Searches take an
optional ISO-8601 `since` (inclusive) / `until` (exclusive) time window; see
[Time windows](datasource-skills.md#time-windows) for how each method dates its
results. Read-only MCP mode omits mutating tools such as refresh and report.

Always-on cloud agents (Grok Bot, OpenAI dots, Claude Duet, Meta Muse) connect by URL
instead: `autorag-mcp --http` serves the same tools over streamable HTTP with bearer
and OAuth auth, read-only by default. See [`remote-agents.md`](remote-agents.md).

## ⚡ AutoRAG Lite: Model-Free Retrieval Engine

Need blazing fast local search without configuring an LLM or paying for API tokens? Use **AutoRAG Lite**.

AutoRAG Lite provides the exact same high-performance indexing, BM25 ranking, and local MinSync vector/hybrid retrieval engine as the full librarian, but **runs 100% model-free**:

- **Zero LLM Token Usage:** Run purely local BM25 and vector search offline.
- **Agent Integration Ready:** Use the AutoRAG Lite MCP server to supply raw context chunks to an external model; the CLI remains a bootstrap and maintenance interface.
- **Fast Local CLI:** The CLI remains available for terminal-only indexing and repair.

```bash
# Initialize a model-free workspace
autorag lite init --search-paths ./documents

# Index local files and datasources
autorag lite refresh
```

Normal Lite retrieval runs through the MCP tools (`autorag.status`,
`autorag.search`); the CLI remains the bootstrap and maintenance interface for
indexing and repair.

---

## Using external MCP servers and agent skills

AutoRAG is also an MCP *client*. Servers listed in the pi agent directory's `mcp.json`
(`~/.pi/agent/mcp.json`, or `<piAgentDir>/mcp.json` when the agent is given one) are connected
when a TUI session or a one-shot `autorag search` starts, and their tools reach the model next to
AutoRAG's own tools. The file uses the format shared by other MCP clients:

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] },
    "docs": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer ${DOCS_TOKEN}" } }
  }
}
```

- **Exposure.** By default a server's tools are reached through the `codemode` tool (the model
  writes a short script that calls them); `"exposure": "direct"` declares them to the model
  like built-in tools, `"deferred"` loads them on demand through `tool_search`, and `"hidden"`
  turns a server off. `description` is shown to the model so it knows when to use the server.
- **Only the user-level file is read.** A project `.pi/mcp.json` is ignored on purpose: it would
  let any folder AutoRAG starts in choose which processes run.
- **Failures are reported verbatim.** An invalid entry or a server that fails to start is
  reported at session start with pi's own message; the other servers keep working.
- **Inside the TUI**, `/mcp` lists servers, reconnects them, signs in to OAuth servers, and
  changes exposure. Pi's `/skill:<name>` commands and `<available_skills>` list also work for
  any agent skill installed under the agent directory.
- MCP results are data from an external server. AutoRAG tells the model never to follow
  instructions found in them.

---
