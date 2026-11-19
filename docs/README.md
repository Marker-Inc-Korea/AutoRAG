# AutoRAG documentation

This folder is the single source of truth for how AutoRAG Agent is installed, indexed, connected, and operated. The same Markdown is published as a searchable site at **https://autorag-docs.nomadamas.org**.

## For coding agents

Read the Markdown here on GitHub, not the rendered site. Start from one of these:

- [`llms.txt`](https://autorag-docs.nomadamas.org/llms.txt): one line per page, each pointing at the raw Markdown on GitHub.
- [`.vitepress/sidebar.mts`](.vitepress/sidebar.mts): the table of contents; every page of the site is listed there exactly once.
- Task to page: install and verify = `guide/installation.md`, `guide/quickstart.md`; register AutoRAG with an agent = `agents/overview.md`, `agents/mcp-server.md`, `agents/skills.md`; index folders = `indexing/overview.md`; connect a chat, mail, or code source = `datasources/overview.md`; every command and flag = `reference/cli.md`; something broke = `guide/troubleshooting.md`.

Where a page and the code disagree, the code is correct: open an issue or a pull request that fixes the page.

## For contributors

- Pages are plain Markdown with `title` and `description` frontmatter. Add a page by creating the file and listing it in `.vitepress/sidebar.mts`.
- Link other pages with relative `.md` links (`../datasources/slack.md`); link files outside `docs/` with absolute `https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/...` URLs.
- Every `autorag ...` command, flag, and `autorag.*` MCP tool you write is checked against the real CLI and server by `test/docs/site-contract.test.ts`, as are links, anchors, and the sidebar.
- Preview locally: `cd docs && bun install --frozen-lockfile && bun run dev`. Production build: `bun run build`.
- The site is built by `.github/workflows/docs.yml` and deployed to GitHub Pages on every push to `main` that touches `docs/`.
