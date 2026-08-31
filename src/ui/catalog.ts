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
		type: "clawgallery",
		title: "ClawGallery",
		summary: "Search local screenshots and photos through the clawgallery CLI.",
		defaultTags: ["clawgallery", "screenshots", "images"],
		binaryName: "clawgallery",
		installHint: "cargo install clawgallery",
		fields: [
			INSTANCE,
			{
				key: "connector.path",
				label: "Image folder",
				kind: "path",
				required: true,
				help: "Folder containing screenshots or photos.",
			},
			BINARY,
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

export type SourcePickerKind = "folders" | "index" | "datasource";

export interface SourcePickerEntry {
	readonly type: string;
	readonly title: string;
	readonly summary: string;
	readonly kind: SourcePickerKind;
	readonly binaryName?: string;
	readonly installHint?: string;
	/** False for sources that cannot attach a second account (KakaoTalk). */
	readonly supportsMultiple: boolean;
	readonly extras: readonly PickerExtra[];
}

export type PickerExtraKind = "text" | "textarea" | "select" | "path";

export interface PickerExtra {
	readonly key: string;
	readonly label: string;
	readonly kind: PickerExtraKind;
	readonly placeholder?: string;
	readonly help?: string;
	readonly choices?: "rclone-remotes" | "mail-accounts";
	readonly allowOther?: boolean;
	/** Asked in the coding-agent prompt when this extra is left blank. */
	readonly question: string;
}

const EXTRAS_BY_TYPE: Readonly<Record<string, readonly PickerExtra[]>> = {
	kakao: [],
	whatsapp: [
		{
			key: "account",
			label: "Account or archive",
			kind: "text",
			placeholder: "personal",
			question: "Which WhatsApp account or archive should this connection use?",
		},
	],
	telegram: [
		{
			key: "account",
			label: "Account or archive",
			kind: "text",
			placeholder: "personal",
			question: "Which Telegram account or archive should this connection use?",
		},
	],
	slack: [
		{
			key: "workspace",
			label: "Workspace",
			kind: "text",
			placeholder: "company",
			question: "Which Slack workspace (and optional slacrawl sync source) should this connection use?",
		},
	],
	discord: [
		{
			key: "guildId",
			label: "Server / guild",
			kind: "text",
			placeholder: "guild id or name",
			question: "Which Discord server/guild should this connection index?",
		},
	],
	notion: [
		{
			key: "workspace",
			label: "Workspace",
			kind: "text",
			question: "Which Notion workspace should this connection use?",
		},
	],
	github: [
		{
			key: "repos",
			label: "Repositories",
			kind: "textarea",
			placeholder: "owner/repo",
			help: "One owner/repo per line.",
			question: "Which GitHub repositories (owner/repo) should this connection index?",
		},
	],
	clawgallery: [
		{
			key: "path",
			label: "Image folder",
			kind: "path",
			question: "Which local screenshot or photo folder should this connection use?",
		},
	],
	"cloud-drive": [
		{
			key: "remote",
			label: "rclone remote",
			kind: "select",
			choices: "rclone-remotes",
			allowOther: true,
			question: "Which rclone remote (Google Drive, OneDrive, …) should this connection use?",
		},
	],
	gmail: [
		{
			key: "account",
			label: "Mailbox",
			kind: "select",
			choices: "mail-accounts",
			allowOther: true,
			question: "Which email account should this connection use (Gmail, Outlook, iCloud, or another IMAP mailbox)?",
		},
	],
	"mail-export": [
		{
			key: "paths",
			label: "Export files or folders",
			kind: "textarea",
			placeholder: "/path/to/export.mbox",
			question: "Where are the .eml or .mbox files for this mailbox?",
		},
	],
	obsidian: [
		{
			key: "vaultPath",
			label: "Vault folder",
			kind: "path",
			question: "Which Obsidian vault folder should this connection use?",
		},
	],
	rss: [
		{
			key: "feeds",
			label: "Feed URLs",
			kind: "textarea",
			placeholder: "https://example.com/feed.xml",
			question: "Which RSS/Atom feed URLs should this connection index?",
		},
	],
	spotlight: [
		{
			key: "onlyIn",
			label: "Folder (optional)",
			kind: "path",
			question: "Which folder should Spotlight search be limited to, if any?",
		},
	],
};

export const SOURCE_PICKER: readonly SourcePickerEntry[] = DATASOURCE_TYPE_CATALOG.map((entry) => ({
	type: entry.type,
	title: entry.title,
	summary: entry.summary,
	kind: "datasource" as const,
	supportsMultiple: entry.type !== "kakao",
	extras: EXTRAS_BY_TYPE[entry.type] ?? [],
	...(entry.binaryName !== undefined ? { binaryName: entry.binaryName } : {}),
	...(entry.installHint !== undefined ? { installHint: entry.installHint } : {}),
}));

export function getPickerEntry(type: string): SourcePickerEntry | undefined {
	return SOURCE_PICKER.find((entry) => entry.type === type);
}

export function assertCatalogCoversBuiltins(): void {
	for (const name of BUILTIN_DATASOURCE_SKILL_NAMES) {
		if (!BY_TYPE.has(name)) throw new Error(`UI catalog missing built-in datasource type: ${name}`);
	}
}
