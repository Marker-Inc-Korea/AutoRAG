import { CrawlerCliClient } from "../../crawler-client.ts";
import {
	CrawlerDatasourceSkill,
	type CrawlerDatasourceSkillOptions,
	type CrawlerSkillClient,
} from "../../crawler-skill.ts";
import type { CrawlerCliOptions, CrawlerHit, CrawlerProfile } from "../../crawler-types.ts";

export interface SlacrawlOptions extends CrawlerCliOptions {}

const SLACRAWL_PROFILE: CrawlerProfile = {
	binaryName: "slacrawl",
	allowedEnvPrefixes: ["SLACRAWL_"],
	syncArgs: (options) => [
		...globalArgs(options),
		"sync",
		...(options.syncSource !== undefined ? ["--source", options.syncSource] : []),
	],
	searchArgs: (options, query, topK) => [...globalArgs(options), "--json", "search", "--limit", String(topK), query],
	parseSyncCount: parseCount,
	parseHits,
};

const SLACK_DEFINITION = {
	datasourceId: "slack",
	skillType: "slack-archive",
	description: "Slack archive datasource via slacrawl",
	defaultTags: ["slack", "chat", "pii"],
	contentType: "chat",
	manifestDescription:
		"Search archived Slack messages across workspaces, channels, users, and threads. Use for questions about Slack conversations, decisions, or who said what.",
	backendName: "slacrawl",
} as const;

export class SlacrawlClient extends CrawlerCliClient {
	constructor(options: SlacrawlOptions = {}) {
		super(SLACRAWL_PROFILE, options);
	}
}

export interface SlackSkillOptions extends Omit<CrawlerDatasourceSkillOptions, "client"> {
	readonly client?: CrawlerSkillClient;
	readonly connectorOptions?: SlacrawlOptions;
}

export class SlackSkill extends CrawlerDatasourceSkill {
	constructor(options: SlackSkillOptions = {}) {
		const { client, connectorOptions, ...skillOptions } = options;
		super(SLACK_DEFINITION, {
			...skillOptions,
			client: client ?? new SlacrawlClient(connectorOptions),
		});
	}
}

function globalArgs(options: CrawlerCliOptions): readonly string[] {
	return options.configPath !== undefined ? ["--config", options.configPath] : [];
}

function parseCount(stdout: string): number | undefined {
	const parsed = parseJson(stdout);
	if (parsed === undefined) return undefined;
	if (Array.isArray(parsed)) return parsed.length;
	if (!isRecord(parsed)) return undefined;
	for (const key of ["messages", "message_count", "messageCount", "synced", "count"]) {
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
		const workspaceId = stringValue(row, ["workspace_id"]);
		const workspaceName = stringValue(row, ["workspace_name"]);
		const channelId = stringValue(row, ["channel_id"]);
		const channelName = stringValue(row, ["channel_name"]);
		const timestamp = stringValue(row, ["ts", "timestamp"]);
		const content = stringValue(row, ["normalized_text", "text", "content"]);
		const id = stringValue(row, ["message_id", "id"]) ?? derivedId(channelId, timestamp);
		if (id === undefined || content === undefined) return undefined;
		const userId = stringValue(row, ["user_id"]);
		const userName = stringValue(row, ["user_name"]);
		hits.push({
			id,
			content,
			score: 1 / (index + 1),
			...(channelName !== undefined ? { title: `#${channelName}` } : {}),
			...(channelName !== undefined || channelId !== undefined
				? {
						hierarchy: [
							"workspaces",
							workspaceName ?? workspaceId ?? "unknown",
							"channels",
							channelName ?? channelId ?? "unknown",
						],
					}
				: {}),
			...(timestamp !== undefined && Number.isFinite(Number.parseFloat(timestamp))
				? { publishedAt: Math.round(Number.parseFloat(timestamp) * 1000) }
				: {}),
			metadata: {
				...(workspaceId !== undefined ? { workspaceId } : {}),
				...(workspaceName !== undefined ? { workspaceName } : {}),
				...(channelId !== undefined ? { channelId } : {}),
				...(channelName !== undefined ? { channelName } : {}),
				...(userId !== undefined ? { userId } : {}),
				...(userName !== undefined ? { userName } : {}),
				...(timestamp !== undefined ? { timestamp } : {}),
			},
		});
	}
	return hits;
}

function derivedId(channelId: string | undefined, timestamp: string | undefined): string | undefined {
	if (channelId === undefined || timestamp === undefined) return undefined;
	return `${channelId}-${timestamp.replaceAll(".", "-")}`;
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
