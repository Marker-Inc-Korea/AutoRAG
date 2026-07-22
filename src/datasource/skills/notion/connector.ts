/**
 * Notion workspace connector (issue #1302).
 *
 * Enumerates pages/databases via the Notion search API using a trusted,
 * server-configured integration token, then pulls one page of block children
 * per page for text content. Per-page block failures degrade to warnings.
 * Never throws; messages stay path/PII-opaque.
 */

import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, httpJson, parseEpochMs, resolveToken } from "../../http.ts";

export interface NotionConnectorOptions {
	/** Notion REST base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** Integration token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the integration token. Default `NOTION_TOKEN`. */
	readonly tokenEnv?: string;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxPages?: number;
	readonly maxDocuments?: number;
}

const DEFAULT_BASE_URL = "https://api.notion.com/v1";
const DEFAULT_TOKEN_ENV = "NOTION_TOKEN";
const NOTION_VERSION = "2022-06-28";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_DOCUMENTS = 500;
const TEXT_BLOCK_TYPES = [
	"paragraph",
	"heading_1",
	"heading_2",
	"heading_3",
	"bulleted_list_item",
	"numbered_list_item",
	"to_do",
	"quote",
	"callout",
	"code",
] as const;

export class NotionConnector implements DatasourceConnector {
	private readonly options: NotionConnectorOptions;

	constructor(options: NotionConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const token = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		if (token === undefined) return { ok: false, reason: "not-configured", message: "auth token not configured" };
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const request = {
			headers: {
				Authorization: `Bearer ${token}`,
				"Notion-Version": NOTION_VERSION,
				"Content-Type": "application/json",
			},
			timeoutMs: this.options.timeoutMs,
			fetchImpl: this.options.fetchImpl,
			signal,
		};
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;

		// 1. Enumerate pages and databases via cursor-paginated search.
		const entries: Record<string, unknown>[] = [];
		let cursor: string | undefined;
		for (let page = 0; page < maxPages; page += 1) {
			const body = JSON.stringify({ page_size: 100, ...(cursor !== undefined ? { start_cursor: cursor } : {}) });
			const result = await httpJson(`${baseUrl}/search`, { ...request, method: "POST", body });
			if (!result.ok) return { ok: false, reason: result.reason, message: `search failed: ${result.message}` };
			const envelope = asRecord(result.json);
			if (envelope === undefined) return { ok: false, reason: "invalid-data", message: "search response malformed" };
			for (const raw of asArray(envelope.results)) {
				const entry = asRecord(raw);
				if (entry !== undefined) entries.push(entry);
			}
			cursor = envelope.has_more === true ? asString(envelope.next_cursor) : undefined;
			if (cursor === undefined) break;
		}

		// 2. Build documents; pull one page of block children per page entry.
		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		for (const entry of entries) {
			if (documents.length >= maxDocuments) break;
			const id = asString(entry.id);
			if (id === undefined) continue;
			const objectType = asString(entry.object) ?? "page";
			const title = extractTitle(entry);
			let blockText = "";
			if (objectType === "page") {
				const blocksResult = await httpJson(`${baseUrl}/blocks/${id}/children?page_size=100`, request);
				if (blocksResult.ok) {
					blockText = extractBlockText(blocksResult.json);
				} else {
					warnings.push(`page content fetch failed: ${blocksResult.message}`);
				}
			}
			const content = [title, blockText].filter((part) => part.length > 0).join("\n\n");
			if (content.length === 0) continue;
			documents.push({
				docId: id,
				hierarchy: hierarchyFor(entry, objectType),
				...(title.length > 0 ? { title } : {}),
				content,
				publishedAt: parseEpochMs(entry.last_edited_time),
				metadata: {
					objectType,
					...(asString(entry.last_edited_time) !== undefined
						? { lastEditedTime: asString(entry.last_edited_time) }
						: {}),
				},
			});
		}

		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}

function hierarchyFor(entry: Record<string, unknown>, objectType: string): readonly string[] {
	if (objectType === "database") return ["databases", asString(entry.id) ?? "unknown"];
	const parent = asRecord(entry.parent);
	const parentType = asString(parent?.type);
	if (parentType === "database_id") {
		return ["databases", asString(parent?.database_id) ?? "unknown"];
	}
	if (parentType === "page_id") {
		return ["pages", asString(parent?.page_id) ?? "unknown"];
	}
	return ["pages"];
}

function extractTitle(entry: Record<string, unknown>): string {
	// Database: title[].plain_text at the top level.
	const directTitle = plainText(entry.title);
	if (directTitle.length > 0) return directTitle;
	// Page: find the property whose type is "title".
	const properties = asRecord(entry.properties);
	if (properties === undefined) return "";
	for (const value of Object.values(properties)) {
		const property = asRecord(value);
		if (property === undefined || property.type !== "title") continue;
		const text = plainText(property.title);
		if (text.length > 0) return text;
	}
	return "";
}

function extractBlockText(json: unknown): string {
	const envelope = asRecord(json);
	const lines: string[] = [];
	for (const raw of asArray(envelope?.results)) {
		const block = asRecord(raw);
		const type = asString(block?.type);
		if (block === undefined || type === undefined) continue;
		if (!(TEXT_BLOCK_TYPES as readonly string[]).includes(type)) continue;
		const payload = asRecord(block[type]);
		const text = plainText(payload?.rich_text);
		if (text.length > 0) lines.push(text);
	}
	return lines.join("\n");
}

function plainText(richText: unknown): string {
	return asArray(richText)
		.map((raw) => asString(asRecord(raw)?.plain_text) ?? "")
		.filter((text) => text.length > 0)
		.join("")
		.trim();
}
