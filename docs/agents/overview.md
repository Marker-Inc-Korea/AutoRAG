---
title: "Choosing an integration"
description: "Pick between the Lite MCP server, agent skills, the model-backed librarian, cloud agents, and the Hermes plugin."
---

# Choosing an integration

AutoRAG can be driven by a person or by another agent in several ways. This page
maps each integration to the case it fits, then gives the exact prompts that make
a coding agent install and configure AutoRAG for you.

## What each integration is

| Integration | What it is | Choose it when | Start with |
| --- | --- | --- | --- |
| Lite MCP server | `autorag-mcp`, a stdio MCP server with no search model; it returns ranked chunks | A coding agent (Claude Code, Codex, Cursor, Pi) should search your files with its own model | [MCP server (Lite)](mcp-server.md) |
| Agent skills | Markdown skills under `skills/` that teach an agent to install, configure, and repair AutoRAG | The agent must bootstrap or repair AutoRAG, not just query it | [Agent skills](skills.md) |
| Full librarian | The model-backed `autorag` CLI and TUI, with MinSync/Jikji retrieval and Jev routing | You want a curated answer (a synthesized, cited reply) rather than raw chunks | [Quickstart](../guide/quickstart.md) |
| Remote agents | `autorag-mcp --http` behind an HTTPS tunnel, for agents that run in a vendor cloud | An always-on cloud agent (Claude Duet, OpenAI dots, Grok Bot, Meta Muse) should reach your index by URL | [Always-on cloud agents](remote-agents.md) |
| Hermes plugin | A portable Agent Plugins v1 package that runs the Lite MCP server inside Hermes | You use Hermes and want retrieval plus a search pane | [Hermes plugin](hermes-plugin.md) |

The Lite MCP server and the full librarian share the same indexing and retrieval
engine. Lite adds no model and no agent loop: the calling agent plans the search
and writes the answer. The full librarian adds one configured model that plans,
reads, and curates the answer itself. See [Retrieval](../retrieval/overview.md)
for the engine both use.

Use the Lite MCP server for external-agent search. Use the skills only to
install, configure, or diagnose AutoRAG, or to drive the model-backed librarian.
A wording like `autorag-lite-search` names a removed skill: routine Lite
retrieval goes through MCP tools, not a shell command or a search skill.

## Prompts that make an agent install AutoRAG

Paste one of these into a coding agent that can read URLs. The agent fetches the
matching skill from `raw.githubusercontent.com` and follows it, asking before it
touches anything.

**Set up the AutoRAG librarian** ([`autorag-setup`](https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/skills/autorag-setup/SKILL.md)):

```text
Install and configure AutoRAG Agent on this computer by following
https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/skills/autorag-setup/SKILL.md
Ask me which folders to index and how to sign in to a model before you change anything.
When it is set up, run a health check and a test search, and show me the result.
Never move, rename, edit, or delete my documents while setting up.
```

**Give your coding agent model-free search over your files** ([`autorag-lite-setup`](https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/skills/autorag-lite-setup/SKILL.md)):

```text
Install and register the AutoRAG Lite MCP server for this agent by following
https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/skills/autorag-lite-setup/SKILL.md
Ask me which folders to index, build the index, then verify with a real search.
Never move, rename, edit, or delete my documents.
```

**Something broken?** ([`autorag-doctor`](https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/skills/autorag-doctor/SKILL.md)):

```text
Check AutoRAG by following https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/skills/autorag-doctor/SKILL.md
and finish with the per-datasource status table.
```

For search only, after setup, the agent calls the MCP tools directly: no
additional skill is needed. See
[Agent skills](skills.md#quick-agent-workflow) for the tool sequence.
