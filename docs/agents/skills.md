---
title: "Agent skills"
description: "The shipped Markdown skills that teach a coding agent to install, drive, and repair AutoRAG."
---

# Agent skills

AutoRAG ships Markdown skills that a coding agent can read to install, configure,
search, or diagnose AutoRAG. This page lists what each skill does, how to copy it
into an agent, and the sequence an agent follows to answer a question.

## Shipped skills

Each skill is a `SKILL.md` with YAML frontmatter (`name`, `description`,
`license`) and a body of rules and commands.

| Skill | Directory | Purpose |
| --- | --- | --- |
| `autorag` | [`skills/autorag/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/skills/autorag/SKILL.md) | Search, summarize, compare, and answer from an already configured model-backed librarian. |
| `autorag-setup` | [`skills/autorag-setup/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/skills/autorag-setup/SKILL.md) | Install AutoRAG and configure the model-backed librarian: the single search model, approved roots, indexes, datasources, health checks. |
| `autorag-lite-setup` | [`skills/autorag-lite-setup/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/skills/autorag-lite-setup/SKILL.md) | Install and register the model-free `autorag-mcp` server, write model-free config, build indexes, and verify MCP search. |
| `autorag-doctor` | [`skills/autorag-doctor/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/skills/autorag-doctor/SKILL.md) | Diagnose and repair a broken install: probe every datasource natively, repair locks, embedding mismatches, and stale indexes, and finish with a per-datasource status table. |

Pick the skill by intent. Use `autorag-lite-setup` for external-agent search over
MCP with no search model; use `autorag-setup` (plus `autorag`) only when the agent
must also drive the model-backed librarian. Use `autorag-doctor` only to repair an
existing install. See [Choosing an integration](overview.md) for the full decision
table.

## Installing skills

### Install the skills into your coding agent

Skills are not auto-discovered. From a clone of the repository, copy the skill
folders into the agent's skill directory. Claude Code reads `.claude/skills`;
other agents use their own directory (for example Cursor, Codex, and Pi each
have a per-agent or project skill location), so copy the same folder there and
reload the agent session.

```bash
mkdir -p .claude/skills
cp -R skills/autorag-lite-setup .claude/skills/
```

Copy `skills/autorag`, `skills/autorag-setup`, and `skills/autorag-doctor` too
when the agent should drive the model-backed librarian or repair an install:

```bash
mkdir -p .claude/skills
cp -R skills/autorag-lite-setup skills/autorag-setup skills/autorag-doctor .claude/skills/
```

The npm package ships the same `skills/` folder. A global install keeps it under
the package root:

```bash
AUTORAG_SKILLS="$(npm root -g)/@autorag/librarian/skills"
mkdir -p .claude/skills
cp -R "$AUTORAG_SKILLS/autorag-lite-setup" .claude/skills/
```

With a Bun global install the package root is
`~/.bun/install/global/node_modules/@autorag/librarian/skills`.

After copying, reload the agent so it picks up the skill, and run the setup skill
once to register the stdio MCP server with the host. Do not copy the removed
`autorag-lite-search` skill: routine Lite retrieval goes through MCP tools, not a
shell command.

### Skills in the npm package

`package.json` `files` publishes `dist`, `skills`, `internal-skills`, `README.md`,
`NOTICE`, `licenses`, and `vendor`.

- `skills/` holds the parent-agent skills above (`autorag`, `autorag-setup`,
  `autorag-lite-setup`, `autorag-doctor`).
- `internal-skills/` holds the librarian agent's own turn rules, for example
  [`internal-skills/organize/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/internal-skills/organize/SKILL.md).
  When Jev routes a request to `organize`, the run loads that file as a skill
  invocation. These are never parent-agent skills: do not copy `internal-skills/`
  into another agent.

### Plugin marketplaces

Three marketplace manifests declare one plugin, `autorag-remote`, sourced from
[`plugins/remote-agents/`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/plugins/remote-agents/README.md):
[`.claude-plugin/marketplace.json`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/.claude-plugin/marketplace.json),
[`.cursor-plugin/marketplace.json`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/.cursor-plugin/marketplace.json),
and [`.agents/plugins/marketplace.json`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/.agents/plugins/marketplace.json).
They ship the cloud-agent `search` skill, not an MCP server; you register your own
server URL. Install from the marketplace:

```bash
claude plugin marketplace add Marker-Inc-Korea/AutoRAG
claude plugin install autorag-remote@autorag
```

```bash
codex plugin marketplace add Marker-Inc-Korea/AutoRAG
codex plugin add autorag-remote@autorag
```

Then follow [Always-on cloud agents](remote-agents.md) to start the HTTP server
and add its URL.

## Using the skills

### Quick agent workflow

1. Confirm the CLI and config exist. `command -v autorag` should print a path; a
   config is resolved from `--config`, `AUTORAG_CONFIG`,
   `$AUTORAG_HOME/config.json`, or `~/.autorag/config.json`. If the CLI is
   missing, install it: `command -v autorag >/dev/null || bun install -g @autorag/librarian`.
2. Check readiness. `autorag status --json` reports corpus freshness and index
   health; `autorag duplicates --json` scans duplicate families read-only.
3. Build or refresh indexes if status is stale. `autorag refresh --json` runs all
   configured methods incrementally; `autorag refresh --method minsync --json`
   narrows the run. Indexing never happens while answering.
4. Search. For the model-backed librarian use `autorag search "your question" --json`
   (add `--debug` for `sessionId` and diagnostics); for Lite use the MCP tools
   `autorag.status` → `autorag.refresh` (when needed) → `autorag.search`.
5. Persist and re-open evidence when the caller needs exact chunks. See
   [Evidence and citations](../retrieval/evidence-and-citations.md).
6. On failure, read the diagnostics. `autorag health --json` checks the model and
   credentials; the `autorag-doctor` skill maps diagnostic codes to first moves.

Safety rules every skill states: setup, indexing, and refresh never move, rename,
edit, or delete source documents; never permanently delete (OS trash only); never
print or leak credential values (record environment-variable names instead); and
index artifacts go only under the workspace `.autorag/` directory and Jikji's
central store.
