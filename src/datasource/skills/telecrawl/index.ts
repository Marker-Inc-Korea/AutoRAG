import { CrawlerCliClient } from "../../crawler-client.ts";
import {
	CrawlerDatasourceSkill,
	type CrawlerDatasourceSkillOptions,
	type CrawlerSkillClient,
} from "../../crawler-skill.ts";
import type { CrawlerCliOptions, CrawlerHit, CrawlerProfile } from "../../crawler-types.ts";

export interface TelecrawlOptions extends CrawlerCliOptions {}

const TELECRAWL_PROFILE: CrawlerProfile = {
	binaryName: "telecrawl",
	allowedEnvPrefixes: ["TELECRAWL_"],
	syncArgs: (options) => [...globalArgs(options), "import"],
	searchArgs: (options, query, topK) => [...globalArgs(options), "search", "--limit", String(topK), query],
	parseSyncCount: parseCount,
	parseHits,
};

const TELEGRAM_DEFINITION = {
	datasourceId: "telegram",
	skillType: "telegram-archive",
	description: "Telegram archive datasource via telecrawl",
	defaultTags: ["telegram", "chat", "personal", "pii"],
	contentType: "chat",
	manifestDescription:
		"Search archived Telegram messages, chats, senders, topics, threads, and media titles. Use for questions about Telegram conversations or who said what.",
	backendName: "telecrawl",
} as const;

export class TelecrawlClient extends CrawlerCliClient {
	constructor(options: TelecrawlOptions = {}) {
		super(TELECRAWL_PROFILE, options);
	}
}

export interface TelecrawlSkillOptions extends Omit<CrawlerDatasourceSkillOptions, "client"> {
	readonly client?: CrawlerSkillClient;
	readonly connectorOptions?: TelecrawlOptions;
}

export class TelecrawlSkill extends CrawlerDatasourceSkill {
	constructor(options: TelecrawlSkillOptions = {}) {
		const { client, connectorOptions, ...skillOptions } = options;
		super(TELEGRAM_DEFINITION, {
			...skillOptions,
			client: client ?? new TelecrawlClient(connectorOptions),
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
		const topicName = stringValue(row, ["topic_name"]);
		const timestamp = stringValue(row, ["timestamp"]);
		hits.push({
			id,
			content,
			score: 1 / (index + 1),
			...(chatName !== undefined ? { title: chatName } : {}),
			...(chatName !== undefined || chatId !== undefined
				? { hierarchy: ["chats", chatName ?? chatId ?? "unknown", ...(topicName !== undefined ? [topicName] : [])] }
				: {}),
			...(timestamp !== undefined && Number.isFinite(Date.parse(timestamp))
				? { publishedAt: Date.parse(timestamp) }
				: {}),
			metadata: {
				...(chatId !== undefined ? { chatId } : {}),
				...(chatName !== undefined ? { chatName } : {}),
				...(senderName !== undefined ? { senderName } : {}),
				...(topicName !== undefined ? { topicName } : {}),
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
