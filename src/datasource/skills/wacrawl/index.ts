import { CrawlerCliClient } from "../../crawler-client.ts";
import {
	CrawlerDatasourceSkill,
	type CrawlerDatasourceSkillOptions,
	type CrawlerSkillClient,
} from "../../crawler-skill.ts";
import type { CrawlerCliOptions, CrawlerHit, CrawlerProfile } from "../../crawler-types.ts";

export interface WacrawlOptions extends CrawlerCliOptions {}

const WACRAWL_PROFILE: CrawlerProfile = {
	binaryName: "wacrawl",
	allowedEnvPrefixes: ["WACRAWL_"],
	syncArgs: (options) => [...globalArgs(options), "sync"],
	searchArgs: (options, query, topK) => [
		...globalArgs(options),
		"--sync",
		"never",
		"search",
		"--limit",
		String(topK),
		query,
	],
	parseSyncCount: parseCount,
	parseHits,
};

const WHATSAPP_DEFINITION = {
	datasourceId: "whatsapp",
	skillType: "whatsapp-archive",
	description: "WhatsApp archive datasource via wacrawl",
	defaultTags: ["whatsapp", "chat", "personal", "pii"],
	contentType: "chat",
	manifestDescription:
		"Search archived WhatsApp messages, chats, senders, and media titles. Use for questions about WhatsApp conversations or who said what.",
	backendName: "wacrawl",
} as const;

export class WacrawlClient extends CrawlerCliClient {
	constructor(options: WacrawlOptions = {}) {
		super(WACRAWL_PROFILE, options);
	}
}

export interface WacrawlSkillOptions extends Omit<CrawlerDatasourceSkillOptions, "client"> {
	readonly client?: CrawlerSkillClient;
	readonly connectorOptions?: WacrawlOptions;
}

export class WacrawlSkill extends CrawlerDatasourceSkill {
	constructor(options: WacrawlSkillOptions = {}) {
		const { client, connectorOptions, ...skillOptions } = options;
		super(WHATSAPP_DEFINITION, {
			...skillOptions,
			client: client ?? new WacrawlClient(connectorOptions),
		});
	}
}

function globalArgs(options: CrawlerCliOptions): readonly string[] {
	return [
		"--json",
		...(options.databasePath !== undefined ? ["--db", options.databasePath] : []),
		...(options.sourcePath !== undefined ? ["--source", options.sourcePath] : []),
	];
}

function parseCount(stdout: string): number | undefined {
	const parsed = parseJson(stdout);
	if (parsed === undefined) return undefined;
	if (Array.isArray(parsed)) return parsed.length;
	if (!isRecord(parsed)) return undefined;
	for (const key of ["messages", "message_count", "messageCount", "imported", "count"]) {
		const value = parsed[key];
		if (typeof value === "number" && Number.isFinite(value)) return value;
	}
	return 0;
}

function parseHits(stdout: string): readonly CrawlerHit[] | undefined {
	const parsed = parseJson(stdout);
	const rows = Array.isArray(parsed)
		? parsed
		: isRecord(parsed) && Array.isArray(parsed.messages)
			? parsed.messages
			: undefined;
	if (rows === undefined) return undefined;
	const hits: CrawlerHit[] = [];
	for (const [index, row] of rows.entries()) {
		if (!isRecord(row)) return undefined;
		const id = stringValue(row, ["message_id", "event_id", "id"]);
		const content = stringValue(row, ["snippet", "text", "content"]);
		if (id === undefined || content === undefined) return undefined;
		const chatId = stringValue(row, ["chat_jid", "chat_id"]);
		const chatName = stringValue(row, ["chat_name"]);
		const senderName = stringValue(row, ["sender_name"]);
		const timestamp = stringValue(row, ["timestamp"]);
		hits.push({
			id,
			content,
			score: 1 / (index + 1),
			...(chatName !== undefined ? { title: chatName } : {}),
			...(chatName !== undefined || chatId !== undefined
				? { hierarchy: ["chats", chatName ?? chatId ?? "unknown"] }
				: {}),
			...(timestamp !== undefined && Number.isFinite(Date.parse(timestamp))
				? { publishedAt: Date.parse(timestamp) }
				: {}),
			metadata: {
				...(chatId !== undefined ? { chatId } : {}),
				...(chatName !== undefined ? { chatName } : {}),
				...(senderName !== undefined ? { senderName } : {}),
				...(timestamp !== undefined ? { timestamp } : {}),
			},
		});
	}
	return hits;
}

function parseJson(stdout: string): unknown {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) return [];
	try {
		return JSON.parse(trimmed);
	} catch {
		return undefined;
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(record: Record<string, unknown>, keys: readonly string[]): string | undefined {
	for (const key of keys) {
		const value = record[key];
		if (typeof value === "string" && value.length > 0) return value;
	}
	return undefined;
}
