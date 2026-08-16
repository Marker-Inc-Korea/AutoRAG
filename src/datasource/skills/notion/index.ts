import { CrawlerCliClient } from "../../crawler-client.ts";
import {
	CrawlerDatasourceSkill,
	type CrawlerDatasourceSkillOptions,
	type CrawlerSkillClient,
} from "../../crawler-skill.ts";
import type { CrawlerCliOptions, CrawlerHit, CrawlerProfile } from "../../crawler-types.ts";

export interface NotcrawlOptions extends CrawlerCliOptions {}

const NOTCRAWL_PROFILE: CrawlerProfile = {
	binaryName: "notcrawl",
	allowedEnvPrefixes: ["NOTCRAWL_"],
	syncArgs: (options) => [...globalArgs(options), "sync"],
	searchArgs: (options, query, topK) => [...globalArgs(options), "search", query, "--limit", String(topK), "--json"],
	parseSyncCount: parseCount,
	parseHits,
};

const NOTION_DEFINITION = {
	datasourceId: "notion",
	skillType: "notion-archive",
	description: "Notion archive datasource via notcrawl",
	defaultTags: ["notion", "documents"],
	contentType: "document",
	manifestDescription:
		"Search archived Notion pages, databases, titles, properties, and block text. Use for questions about Notion docs and wikis.",
	backendName: "notcrawl",
} as const;

export class NotcrawlClient extends CrawlerCliClient {
	constructor(options: NotcrawlOptions = {}) {
		super(NOTCRAWL_PROFILE, options);
	}
}

export interface NotionSkillOptions extends Omit<CrawlerDatasourceSkillOptions, "client"> {
	readonly client?: CrawlerSkillClient;
	readonly connectorOptions?: NotcrawlOptions;
}

export class NotionSkill extends CrawlerDatasourceSkill {
	constructor(options: NotionSkillOptions = {}) {
		const { client, connectorOptions, ...skillOptions } = options;
		super(NOTION_DEFINITION, {
			...skillOptions,
			client: client ?? new NotcrawlClient(connectorOptions),
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
	for (const key of ["pages", "page_count", "pageCount", "synced", "count"]) {
		const value = parsed[key];
		if (typeof value === "number" && Number.isFinite(value)) return value;
	}
	return 0;
}

function parseHits(stdout: string): readonly CrawlerHit[] | undefined {
	const parsed = parseJson(stdout);
	const rows = Array.isArray(parsed)
		? parsed
		: isRecord(parsed) && Array.isArray(parsed.results)
			? parsed.results
			: undefined;
	if (rows === undefined) return undefined;
	const hits: CrawlerHit[] = [];
	for (const [index, row] of rows.entries()) {
		if (!isRecord(row)) return undefined;
		const id = stringValue(row, ["page_id", "id"]);
		const content = stringValue(row, ["snippet", "text", "content"]);
		if (id === undefined || content === undefined) return undefined;
		const title = stringValue(row, ["title", "page_title"]);
		const parentTitle = stringValue(row, ["parent_title", "database_title"]);
		const url = stringValue(row, ["url"]);
		const lastEditedTime = stringValue(row, ["last_edited_time", "updated_at"]);
		hits.push({
			id,
			content,
			score: 1 / (index + 1),
			...(title !== undefined ? { title } : {}),
			hierarchy: ["pages", ...(parentTitle !== undefined ? [parentTitle] : []), title ?? id],
			...(lastEditedTime !== undefined && Number.isFinite(Date.parse(lastEditedTime))
				? { publishedAt: Date.parse(lastEditedTime) }
				: {}),
			metadata: {
				...(title !== undefined ? { title } : {}),
				...(parentTitle !== undefined ? { parentTitle } : {}),
				...(url !== undefined ? { url } : {}),
				...(lastEditedTime !== undefined ? { lastEditedTime } : {}),
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
