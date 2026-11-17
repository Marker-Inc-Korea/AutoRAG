---
layout: home
title: "AutoRAG"
titleTemplate: Developer documentation

hero:
  name: AutoRAG
  text: Your personal librarian agent
  tagline: Finds, reads, and cites anything on your computer. Files, chats, mail, and code are searched in place, and every claim links to the passage behind it.
  image:
    src: /img/hero.webp
    alt: A folder on the left and an AI search conversation with cited answers on the right
  actions:
    - theme: brand
      text: Quickstart
      link: /guide/quickstart
    - theme: alt
      text: Use it from an agent
      link: /agents/overview
    - theme: alt
      text: GitHub
      link: https://github.com/Marker-Inc-Korea/AutoRAG

features:
  - title: Agents
    details: Install and drive AutoRAG from Claude Code, Codex, Cursor, Pi, or an always-on cloud agent through skills and the Lite MCP server.
    link: /agents/overview
  - title: Indexing
    details: How files are parsed, embedded locally with MinSync, scheduled, excluded, refreshed, and repaired.
    link: /indexing/overview
  - title: Datasources
    details: Slack, Discord, Telegram, WhatsApp, KakaoTalk, mail, GitHub, Notion, Obsidian, browser history, and coding-agent sessions.
    link: /datasources/overview
  - title: Retrieval
    details: Hybrid keyword and semantic search, reranking, Jev routing decisions, evidence ids, and citations.
    link: /retrieval/overview
  - title: Reference
    details: Every CLI command and flag, every config key, every environment variable, and the TypeScript API.
    link: /reference/cli
  - title: Contributing
    details: Architecture, the development workflow, adding a datasource, manual QA, and live end-to-end runs.
    link: /contributing/development
---
