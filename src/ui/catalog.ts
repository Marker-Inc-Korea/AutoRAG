/**
 * Operator-facing connection catalog for the local datasource UI.
 *
 * Each built-in factory key gets a small form: paths, env-var *names*,
 * feed/repo lists, and optional CLI binary paths. Secret values are never
 * catalog fields.
 */

import { BUILTIN_DATASOURCE_SKILL_NAMES } from "../datasource/skills/factory.ts";

export const LOCAL_FOLDERS_ID = "local-folders";

export type CatalogFieldKind = "text" | "path" | "path-list" | "env" | "textarea" | "select" | "checkbox";

export interface CatalogField {
	readonly key: string;
	readonly label: string;
	readonly kind: CatalogFieldKind;
	readonly help?: string;
	readonly required?: boolean;
	readonly placeholder?: string;
	readonly options?: readonly { readonly value: string; readonly label: string }[];
}

export interface DatasourceTypeCatalog {
	readonly type: string;
	readonly title: string;
	readonly summary: string;
	readonly defaultTags: readonly string[];
	readonly binaryName?: string;
	readonly installHint?: string;
	readonly fields: readonly CatalogField[];
}

const INSTANCE: CatalogField = {
	key: "instanceId",
	label: "Instance name",
	kind: "text",
	placeholder: "default",
	help: "Optional label for this connection. Leave blank to use “default”.",
};

const BINARY: CatalogField = {
	key: "connector.binaryPath",
	label: "CLI path",
	kind: "path",
	help: "Leave blank to use the program on your PATH.",
};

export const DATASOURCE_TYPE_CATALOG: readonly DatasourceTypeCatalog[] = [
	{
		type: "kakao",
		title: "KakaoTalk",
		summary: "Search a local KakaoTalk archive through the katok CLI.",
		defaultTags: ["kakaotalk", "personal", "pii"],
		binaryName: "katok",
		installHint: "Install the katok CLI, then connect. AutoRAG never reads KakaoTalk databases itself.",
		fields: [INSTANCE, BINARY],
	},
	{
		type: "whatsapp",
		title: "WhatsApp",
		summary: "Local WhatsApp archive through the wacrawl CLI.",
		defaultTags: ["whatsapp", "chat", "personal", "pii"],
		binaryName: "wacrawl",
		installHint: "brew install openclaw/tap/wacrawl",
		fields: [INSTANCE, BINARY],
	},
	{
		type: "telegram",
		title: "Telegram",
		summary: "Local Telegram archive through the telecrawl CLI.",
		defaultTags: ["telegram", "chat", "personal", "pii"],
		binaryName: "telecrawl",
		installHint: "brew install openclaw/tap/telecrawl",
		fields: [INSTANCE, BINARY],
	},
	{
		type: "slack",
		title: "Slack",
		summary: "Workspace archive through the slacrawl CLI. Credentials stay in slacrawl.",
		defaultTags: ["slack", "chat", "pii"],
		binaryName: "slacrawl",
		installHint: "brew install openclaw/tap/slacrawl",
		fields: [
			INSTANCE,
			BINARY,
			{
				key: "connector.configPath",
				label: "slacrawl config",
				kind: "path",
				help: "Optional path to slacrawl’s own config file.",
			},
			{ key: "connector.syncSource", label: "Sync source", kind: "text", placeholder: "primary" },
		],
	},
	{
		type: "discord",
		title: "Discord",
		summary: "Guild archive through the discrawl CLI.",
		defaultTags: ["discord", "chat", "pii"],
		binaryName: "discrawl",
		installHint: "brew install openclaw/tap/discrawl",
		fields: [INSTANCE, BINARY, { key: "connector.guildId", label: "Guild id", kind: "text" }],
	},
	{
		type: "notion",
		title: "Notion",
		summary: "Workspace archive through the notcrawl CLI. Credentials stay in notcrawl.",
		defaultTags: ["notion", "documents"],
		binaryName: "notcrawl",
		installHint: "brew install openclaw/tap/notcrawl",
		fields: [INSTANCE, BINARY, { key: "connector.configPath", label: "notcrawl config", kind: "path" }],
	},
	{
		type: "github",
		title: "GitHub issues",
		summary: "Index issues and pull requests for repositories you list.",
		defaultTags: ["github", "issues"],
		fields: [
			INSTANCE,
			{
				key: "connector.tokenEnv",
				label: "Token environment variable",
				kind: "env",
				placeholder: "GITHUB_TOKEN",
				help: "Name of the env var that holds the token. The token itself is never saved.",
			},
			{
				key: "connector.repos",
				label: "Repositories",
				kind: "textarea",
				required: true,
				placeholder: "owner/repo",
				help: "One owner/repo per line.",
			},
		],
	},
	{
		type: "gdrive",
		title: "Google Drive",
		summary: "Drive files via an access-token env var, or rclone if you already use it.",
		defaultTags: ["gdrive", "documents", "pii"],
		binaryName: "rclone",
		fields: [
			INSTANCE,
			{
				key: "connector.backend",
				label: "Backend",
				kind: "select",
				options: [
					{ value: "", label: "Google Drive API (token env)" },
					{ value: "rclone", label: "rclone" },
				],
			},
			{
				key: "connector.tokenEnv",
				label: "Access token environment variable",
				kind: "env",
				placeholder: "GDRIVE_ACCESS_TOKEN",
				help: "Used for the Drive API backend. Never paste the token here.",
			},
			{
				key: "connector.folderId",
				label: "Folder id",
				kind: "text",
				help: "Optional. Restrict to one Drive folder.",
			},
			{ key: "connector.includeSharedDrives", label: "Include shared drives", kind: "checkbox" },
			{
				key: "connector.remote",
				label: "rclone remote",
				kind: "text",
				placeholder: "gdrive:",
				help: "Used when backend is rclone.",
			},
		],
	},
	{
		type: "cloud-drive",
		title: "Cloud drive (rclone)",
		summary: "Google Drive, OneDrive, or another rclone remote.",
		defaultTags: ["cloud-drive", "rclone", "documents", "pii"],
		binaryName: "rclone",
		installHint: "Run `rclone config` once, then enter the remote name here.",
		fields: [
			INSTANCE,
			{
				key: "connector.provider",
				label: "Provider",
				kind: "select",
				options: [
					{ value: "google-drive", label: "Google Drive" },
					{ value: "onedrive", label: "OneDrive" },
					{ value: "icloud", label: "iCloud (experimental)" },
				],
			},
			{ key: "connector.remote", label: "rclone remote", kind: "text", required: true, placeholder: "gdrive:" },
			BINARY,
		],
	},
	{
		type: "gmail",
		title: "Gmail / IMAP",
		summary: "Gmail API via a token env var, or any IMAP account through himalaya.",
		defaultTags: ["gmail", "email", "pii"],
		binaryName: "himalaya",
		fields: [
			INSTANCE,
			{
				key: "connector.backend",
				label: "Backend",
				kind: "select",
				options: [
					{ value: "", label: "Gmail API (token env)" },
					{ value: "himalaya", label: "himalaya (IMAP)" },
				],
			},
			{
				key: "connector.tokenEnv",
				label: "Access token environment variable",
				kind: "env",
				placeholder: "GMAIL_ACCESS_TOKEN",
			},
			{
				key: "connector.labelIds",
				label: "Gmail labels",
				kind: "textarea",
				placeholder: "INBOX",
				help: "One label id per line.",
			},
			{ key: "connector.account", label: "himalaya account", kind: "text" },
			{ key: "connector.folder", label: "Mail folder", kind: "text", placeholder: "INBOX" },
			BINARY,
		],
	},
	{
		type: "mail-export",
		title: "Mail export",
		summary: "Index local .eml or .mbox files.",
		defaultTags: ["mail-export", "email", "pii"],
		fields: [
			INSTANCE,
			{
				key: "connector.paths",
				label: "Files or folders",
				kind: "path-list",
				required: true,
				help: "One path per line.",
			},
		],
	},
	{
		type: "obsidian",
		title: "Obsidian vault",
		summary: "Search a vault through the qmd CLI.",
		defaultTags: ["obsidian", "notes"],
		binaryName: "qmd",
		fields: [INSTANCE, { key: "connector.vaultPath", label: "Vault folder", kind: "path", required: true }, BINARY],
	},
	{
		type: "rss",
		title: "RSS / Atom feeds",
		summary: "Public feeds you list. No login.",
		defaultTags: ["rss", "news", "public"],
		fields: [
			INSTANCE,
			{
				key: "connector.feeds",
				label: "Feed URLs",
				kind: "textarea",
				required: true,
				placeholder: "https://example.com/feed.xml",
				help: "One URL per line. Optional category: URL | category",
			},
		],
	},
	{
		type: "spotlight",
		title: "Spotlight",
		summary: "macOS Spotlight queries, optionally limited to one folder.",
		defaultTags: ["spotlight", "files"],
		fields: [
			INSTANCE,
			{
				key: "connector.queries",
				label: "Queries",
				kind: "textarea",
				required: true,
				help: "One Spotlight query per line.",
			},
			{ key: "connector.onlyIn", label: "Only in folder", kind: "path" },
		],
	},
];

const BY_TYPE = new Map(DATASOURCE_TYPE_CATALOG.map((entry) => [entry.type, entry]));

export function getDatasourceType(type: string): DatasourceTypeCatalog | undefined {
	return BY_TYPE.get(type);
}

export function assertCatalogCoversBuiltins(): void {
	for (const name of BUILTIN_DATASOURCE_SKILL_NAMES) {
		if (!BY_TYPE.has(name)) throw new Error(`UI catalog missing built-in datasource type: ${name}`);
	}
}
