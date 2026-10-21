# Apple Calendar and Reminders on macOS (EventKit MCP)

AutoRAG has no Calendar datasource yet (tracked in #1313). On macOS you can
still read Calendar events and Reminders today by running a separate MCP server,
[`mcp-server-apple-events`](https://github.com/FradSer/mcp-server-apple-events),
next to `autorag-mcp` in the same MCP client. Nothing is indexed: the MCP client
queries Calendar.app / Reminders.app live through Apple's EventKit framework.

## What you get

EventKit reads every calendar and reminder list that Calendar.app and
Reminders.app already show. Add Google, iCloud, Exchange/Microsoft 365 or any
CalDAV account under **System Settings → Internet Accounts** first; the MCP
server needs no per-provider credentials of its own.

| Tool | Actions | Notes |
|---|---|---|
| `calendar_calendars` | `read` | Every calendar with its account `source`. With `startDate`/`endDate`, only calendars with events in the window plus counts. |
| `calendar_events` | `read`, `create`, `update`, `delete` | Date-range reads across calendars. |
| `reminders_lists` | `read`, `create`, `update`, `delete` | |
| `reminders_tasks` | `read`, `create`, `update`, `delete` | Priority, tags, filters. |
| `reminders_subtasks` | `read`, `create`, `update`, `delete`, `toggle`, `reorder` | |

Dates accept `YYYY-MM-DD`, `YYYY-MM-DD HH:mm:ss` (local) or ISO 8601.

## Setup

Add the server to your MCP client next to AutoRAG. Example `mcp.json`:

```json
{
  "mcpServers": {
    "autorag": { "command": "autorag-mcp" },
    "apple-events": {
      "command": "npx",
      "args": ["-y", "mcp-server-apple-events", "--require-write-confirmation"]
    }
  }
}
```

- Requires macOS and Node.js 20 or newer.
- On first use macOS asks for Calendar and Reminders access. Grant it to the
  application that launches the server (your MCP client), not just a terminal;
  each host needs its own approval under **Privacy & Security → Calendars /
  Reminders**. Headless launches (SSH, launchd) cannot show the prompt; the
  server kills the call after 30 s with a readable error.

## Read-only by default, please

Event titles, notes and attendees can come from invitations sent by other
people, so they are untrusted text. `mcp-server-apple-events` does **not**
require confirmation for writes unless started with
`--require-write-confirmation`, which is why the example above sets it. With the
flag, clients that cannot show MCP form elicitation cannot write at all. Results
are tagged `provenance.trust = "untrusted"`; do not let record content authorise
another tool call.

## Limits

- macOS only (EventKit). For Linux/Windows or a headless server, see #1313.
- Live queries, not an index: no cross-source search with mail/chat, and no
  semantic search over events. Keep date ranges narrow: subscribed calendars
  (holidays, sports) and recurring series expand into many events.
- Only calendars known to Calendar.app are visible.

## Verifying your setup

A live, read-only end-to-end test drives the server with AutoRAG's own MCP
client. It never writes, pins no calendar content, and tries one write to prove
the confirmation flag refuses it:

```bash
AUTORAG_EVENTKIT_LIVE=1 bun run vitest run test/live-e2e/eventkit-mcp.test.ts
```

It is skipped unless `AUTORAG_EVENTKIT_LIVE=1` on macOS, so CI and other
platforms never touch Calendar. Do not run it from `make qa-shell`/Docker: the
container has no Calendar access.
