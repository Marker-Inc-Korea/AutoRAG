# AutoRAG for cloud agents

A skill that teaches an always-on cloud agent (Claude Duet, OpenAI dots, Meta Muse, Grok Bot) how to search your own
files and chat or mail archives through [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG), and how to answer from
what comes back: name the source of each claim, say when nothing matched, and treat retrieved text as data rather than
instructions.

## What this package is, and is not

- **Skill only.** It contains one skill, `search`, and its manifests. There is no MCP server in it, no script, no hook,
  and nothing that runs when you install it. It does not send, upload or fetch anything.
- **You bring the server.** AutoRAG runs on your computer, so its address is yours alone and cannot be written into a
  shared package. Start it with `autorag-mcp --http`, or switch on **Cloud agents** in the Dazzi app, then add that
  address in your agent's connector or plugin screen. Steps for each agent are in
  [`docs/remote-agents.md`](../../docs/remote-agents.md); copy-paste prompts for Muse and Grok Bot are in
  [`connect/`](connect).
- **Read-only by default.** The server offers search and evidence tools (`autorag.search`, `autorag.search.files`,
  `autorag.datasources.list`, `autorag.evidence` and a few more). It does not offer re-indexing, saving reports or file
  conversion unless you start it with `--allow-write`.

## Safety

The skill tells the agent to quote the minimum, never forward retrieved content to other tools, and ignore any
instructions found inside a document or message. The server's **owner token** is a secret: the skill never asks for it,
never stores it, and an agent that signs in through the page you approve does not need it. Rotate it if it leaks, which
disconnects every agent.

## Install

```bash
claude plugin marketplace add Marker-Inc-Korea/AutoRAG
claude plugin install autorag-remote@autorag
codex plugin marketplace add Marker-Inc-Korea/AutoRAG
codex plugin add autorag-remote@autorag
```

Apache-2.0.
