/**
 * The one table of contents for the AutoRAG docs.
 *
 * Every page under `docs/` must be listed here exactly once, and every link must point at a real file. The VitePress
 * config, the generated `llms.txt`, and `test/docs/site-contract.test.ts` all read this file, so a page cannot ship
 * without a sidebar entry and a sidebar entry cannot ship without a page.
 */
export interface SidebarItem {
	readonly text: string;
	readonly link: string;
}

export interface SidebarGroup {
	readonly text: string;
	readonly items: readonly SidebarItem[];
}

export interface Section {
	/** Top-nav label. */
	readonly text: string;
	/** URL prefix every page of this section lives under. */
	readonly base: string;
	readonly groups: readonly SidebarGroup[];
}

export const SECTIONS: readonly Section[] = [
	{
		text: "Guide",
		base: "/guide/",
		groups: [
			{
				text: "Get started",
				items: [
					{ text: "Overview", link: "/guide/overview" },
					{ text: "Installation", link: "/guide/installation" },
					{ text: "Quickstart", link: "/guide/quickstart" },
					{ text: "Models and providers", link: "/guide/models" },
					{ text: "Troubleshooting", link: "/guide/troubleshooting" },
				],
			},
		],
	},
	{
		text: "Agents",
		base: "/agents/",
		groups: [
			{
				text: "Use AutoRAG from an agent",
				items: [
					{ text: "Choosing an integration", link: "/agents/overview" },
					{ text: "Agent skills", link: "/agents/skills" },
					{ text: "MCP server (Lite)", link: "/agents/mcp-server" },
					{ text: "Always-on cloud agents", link: "/agents/remote-agents" },
					{ text: "Hermes plugin", link: "/agents/hermes-plugin" },
				],
			},
			{
				text: "The librarian agent",
				items: [
					{ text: "Terminal UI and search", link: "/agents/librarian" },
					{ text: "Permission modes", link: "/agents/permissions" },
					{ text: "Asking you questions", link: "/agents/ask-user" },
					{ text: "Host subagents", link: "/agents/subagents" },
				],
			},
			{
				text: "Messengers",
				items: [
					{ text: "Overview", link: "/agents/messengers" },
					{ text: "Telegram", link: "/agents/telegram" },
					{ text: "WhatsApp", link: "/agents/whatsapp" },
					{ text: "Discord", link: "/agents/discord" },
				],
			},
		],
	},
	{
		text: "Indexing",
		base: "/indexing/",
		groups: [
			{
				text: "How indexing works",
				items: [
					{ text: "Overview", link: "/indexing/overview" },
					{ text: "Parsing and languages", link: "/indexing/parsing-and-languages" },
					{ text: "Scheduling and exclusions", link: "/indexing/scheduling-and-exclusions" },
					{ text: "Refresh operations", link: "/indexing/refresh-operations" },
				],
			},
			{
				text: "MinSync",
				items: [
					{ text: "Embedding backends", link: "/indexing/embedding-backends" },
					{ text: "Sparse index and chunking", link: "/indexing/sparse-and-chunking" },
					{ text: "Cloud and network folders", link: "/indexing/cloud-folders" },
				],
			},
		],
	},
	{
		text: "Datasources",
		base: "/datasources/",
		groups: [
			{
				text: "Concepts",
				items: [
					{ text: "Overview", link: "/datasources/overview" },
					{ text: "Connecting a datasource", link: "/datasources/connecting" },
					{ text: "Search semantics", link: "/datasources/search-semantics" },
				],
			},
			{
				text: "Chat and mail",
				items: [
					{ text: "Slack", link: "/datasources/slack" },
					{ text: "Discord", link: "/datasources/discord" },
					{ text: "Telegram", link: "/datasources/telegram" },
					{ text: "WhatsApp", link: "/datasources/whatsapp" },
					{ text: "KakaoTalk", link: "/datasources/kakao" },
					{ text: "Lark / Feishu", link: "/datasources/lark" },
					{ text: "Mail (mailcrawl)", link: "/datasources/mailcrawl" },
					{ text: "Hybrid search for chat archives", link: "/datasources/chat-hybrid-search" },
				],
			},
			{
				text: "Knowledge and code",
				items: [
					{ text: "GitHub", link: "/datasources/github" },
					{ text: "GitHub Gists", link: "/datasources/github-gist" },
					{ text: "Notion", link: "/datasources/notion" },
					{ text: "Obsidian", link: "/datasources/obsidian" },
					{ text: "RSS", link: "/datasources/rss" },
					{ text: "Browser history (Nogon)", link: "/datasources/nogon" },
					{ text: "Coding-agent sessions (wombat)", link: "/datasources/wombat" },
					{ text: "ClawGallery", link: "/datasources/clawgallery" },
					{ text: "Apple Calendar and Reminders", link: "/datasources/apple-calendar" },
				],
			},
		],
	},
	{
		text: "Retrieval",
		base: "/retrieval/",
		groups: [
			{
				text: "Search and answers",
				items: [
					{ text: "Retrieval methods", link: "/retrieval/overview" },
					{ text: "Reranking", link: "/retrieval/reranking" },
					{ text: "Jev decisions", link: "/retrieval/jev" },
					{ text: "Evidence and citations", link: "/retrieval/evidence-and-citations" },
					{ text: "Retrieval memory", link: "/retrieval/memory" },
					{ text: "Learning to rank", link: "/retrieval/learning-to-rank" },
				],
			},
		],
	},
	{
		text: "Reference",
		base: "/reference/",
		groups: [
			{
				text: "Reference",
				items: [
					{ text: "CLI commands", link: "/reference/cli" },
					{ text: "Configuration file", link: "/reference/configuration" },
					{ text: "Environment variables", link: "/reference/environment-variables" },
					{ text: "TypeScript API", link: "/reference/typescript-api" },
				],
			},
		],
	},
	{
		text: "Contributing",
		base: "/contributing/",
		groups: [
			{
				text: "Work on AutoRAG",
				items: [
					{ text: "Architecture", link: "/contributing/architecture" },
					{ text: "Development workflow", link: "/contributing/development" },
					{ text: "Adding a datasource", link: "/contributing/new-datasource" },
					{ text: "Manual QA and live E2E", link: "/contributing/manual-qa" },
					{ text: "Supply chain and licenses", link: "/contributing/supply-chain" },
					{ text: "Design notes", link: "/contributing/design-notes" },
				],
			},
		],
	},
];

/** Every page the site publishes, in reading order. Links are site paths without an extension. */
export function allPages(): SidebarItem[] {
	return SECTIONS.flatMap((section) => section.groups.flatMap((group) => [...group.items]));
}

/** Source file (relative to `docs/`) for a sidebar link. */
export function sourceFileFor(link: string): string {
	return `${link.replace(/^\//, "")}.md`;
}
