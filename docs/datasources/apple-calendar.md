---
title: "Apple Calendar and Reminders"
description: "Read macOS Calendar events and Reminders live through the EventKit MCP server connected to AutoRAG or another MCP client."
---

# Apple Calendar and Reminders

AutoRAG has no Calendar datasource. On macOS you can still read Calendar events and Reminders today by connecting the external EventKit MCP server, [`mcp-server-apple-events`](https://github.com/FradSer/mcp-server-apple-events), either to the librarian's pi session or to any MCP client that also talks to AutoRAG. Nothing is indexed: queries hit Calendar.app and Reminders.app live through Apple's EventKit framework.

## Prerequisites

- macOS. EventKit is macOS-only; Linux, Windows, and headless servers are not served by this path.
- Node.js 20 or newer, to run the server with `npx`.
- Calendars and reminder lists already visible in Calendar.app and Reminders.app. Add iCloud, Google, Exchange/Microsoft 365, or any CalDAV account under System Settings → Internet Accounts first; the server needs no per-provider credentials of its own.
- macOS privacy grants for Calendars and Reminders, given to the application that launches the server (your MCP client or the AutoRAG terminal) under Privacy & Security → Calendars / Reminders. Each host needs its own approval, and a headless launch (SSH, launchd) cannot show the prompt: the server kills the call after 30 seconds with a readable error.

## Connect

AutoRAG's librarian runs on pi, which reads external MCP servers from `<piAgentDir>/mcp.json`. `piAgentDir` defaults to `~/.pi/agent` and is overridable with the `PI_CODING_AGENT_DIR` environment variable. Only this user-level file is read; pi would also load a project `.pi/mcp.json`, but AutoRAG deliberately ignores it so a directory AutoRAG starts in cannot choose which processes run. An entry with `"enabled": false` is skipped, and a rejected entry or an unreadable file is reported at session start without stopping the other servers.

Add the server next to any others already in the file:

```json
{
  "mcpServers": {
    "apple-events": {
      "command": "npx",
      "args": ["-y", "mcp-server-apple-events", "--require-write-confirmation"]
    }
  }
}
```

If your MCP client is a separate agent rather than AutoRAG's librarian, register the same server there, next to AutoRAG's own server (see [MCP server (Lite)](../agents/mcp-server.md)). Both tools then appear in one agent.

## What is available

EventKit reads every calendar and reminder list that Calendar.app and Reminders.app already show.

| Tool | Actions |
| --- | --- |
| `calendar_calendars` | Read; with a date range, only calendars that have events in it, plus counts. |
| `calendar_events` | Read, create, update, delete; date-range reads across calendars. |
| `reminders_lists` | Read, create, update, delete. |
| `reminders_tasks` | Read, create, update, delete; priority, tags, filters. |
| `reminders_subtasks` | Read, create, update, delete, toggle, reorder. |

Dates accept `YYYY-MM-DD`, `YYYY-MM-DD HH:mm:ss` (local), or ISO 8601.

## Read-only and trust

- Start the server with `--require-write-confirmation` as in the example. Without it, writes need no confirmation; with it, clients that cannot show MCP form elicitation cannot write at all.
- Event titles, notes, and attendees can come from invitations other people sent, so treat them as untrusted text. Results are tagged `provenance.trust = "untrusted"`; record content must never authorise another tool call.
- AutoRAG's own Lite MCP server is read-only unless it is started with `--allow-write` (see [MCP server (Lite)](../agents/mcp-server.md#read-only-mode)); the EventKit flag is separate and server-side.

## Limits

- macOS only. For Linux, Windows, or a headless server, there is no EventKit path.
- Live queries, not an index: no cross-source search with your indexed mail, chat, or documents, and no semantic search over events.
- Keep date ranges narrow: subscribed calendars (holidays, sports) and recurring series expand into many events.
- Only calendars Calendar.app knows about are visible.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Calendar and Reminders prompts never appear | A headless launch cannot show them; run from an app with a UI, or grant access under Privacy & Security → Calendars / Reminders. |
| A call fails after about 30 seconds | No privacy grant reached the launching app; grant it, then retry. |
| Writes are rejected | The server was started with `--require-write-confirmation` and your client cannot show MCP elicitation; use a client that can, or drop the flag after accepting the risk. |
| A calendar or list is missing | Add its account under System Settings → Internet Accounts so Calendar.app or Reminders.app shows it. |
