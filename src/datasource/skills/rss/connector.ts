/**
 * RSS/news connector (issue #1316).
 *
 * Fetches RSS 2.0 / Atom feeds from a trusted, server-configured feed list
 * using the existing `fast-xml-parser` dependency. Feeds are referenced by
 * index in warnings — never by URL — so diagnostics stay opaque. Per-feed
 * failures degrade to warnings; only all feeds failing fails the fetch.
 */

import { XMLParser } from "fast-xml-parser";
import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, httpText, parseEpochMs } from "../../http.ts";

export interface RssFeedConfig {
	readonly url: string;
	/** Optional category segment inserted into the hierarchy. */
	readonly category?: string;
}

export interface RssConnectorOptions {
	/** Feeds to poll. Required trusted configuration. */
	readonly feeds?: readonly RssFeedConfig[];
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxDocuments?: number;
	readonly maxItemsPerFeed?: number;
}

const DEFAULT_MAX_DOCUMENTS = 500;
const DEFAULT_MAX_ITEMS_PER_FEED = 100;
/** Feeds are often slow static hosts; allow more headroom than API calls. */
const DEFAULT_FEED_TIMEOUT_MS = 30_000;
const MAX_CONTENT_CHARS = 20_000;
const ACCEPT_HEADER = "application/rss+xml, application/atom+xml, application/xml, text/xml";

export class RssConnector implements DatasourceConnector {
	private readonly options: RssConnectorOptions;
	private readonly parser = new XMLParser({ ignoreAttributes: false });

	constructor(options: RssConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const feeds = this.options.feeds ?? [];
		if (feeds.length === 0) return { ok: false, reason: "not-configured", message: "no feeds configured" };
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const maxItemsPerFeed = this.options.maxItemsPerFeed ?? DEFAULT_MAX_ITEMS_PER_FEED;

		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		let failedFeeds = 0;
		for (const [index, feed] of feeds.entries()) {
			if (documents.length >= maxDocuments) break;
			const result = await httpText(feed.url, {
				headers: { Accept: ACCEPT_HEADER },
				timeoutMs: this.options.timeoutMs ?? DEFAULT_FEED_TIMEOUT_MS,
				fetchImpl: this.options.fetchImpl,
				signal,
			});
			if (!result.ok) {
				failedFeeds += 1;
				warnings.push(`feed ${index + 1} failed: ${result.message}`);
				continue;
			}
			let parsed: unknown;
			try {
				parsed = this.parser.parse(result.text);
			} catch {
				failedFeeds += 1;
				warnings.push(`feed ${index + 1} failed: unparseable XML`);
				continue;
			}
			const normalized = normalizeFeed(parsed);
			if (normalized === undefined) {
				failedFeeds += 1;
				warnings.push(`feed ${index + 1} failed: unrecognized feed shape`);
				continue;
			}
			const feedTitle = normalized.title ?? `feed-${index + 1}`;
			const hierarchy = feed.category !== undefined ? ["feeds", feed.category, feedTitle] : ["feeds", feedTitle];
			for (const item of normalized.items.slice(0, maxItemsPerFeed)) {
				if (documents.length >= maxDocuments) break;
				const title = item.title ?? "(untitled)";
				const body = stripHtml(item.body ?? "");
				const content = body.length > 0 ? `${title}\n\n${body}` : title;
				documents.push({
					docId: item.id,
					hierarchy,
					title,
					content: content.slice(0, MAX_CONTENT_CHARS),
					...(item.publishedAt !== undefined ? { publishedAt: item.publishedAt } : {}),
					metadata: { feedIndex: index, feedTitle, categories: item.categories },
				});
			}
		}

		if (failedFeeds === feeds.length) {
			return { ok: false, reason: "unavailable", message: "all configured feeds failed" };
		}
		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}

interface NormalizedItem {
	readonly id: string;
	readonly title?: string;
	readonly body?: string;
	readonly publishedAt?: number;
	readonly categories: readonly string[];
}

interface NormalizedFeed {
	readonly title?: string;
	readonly items: readonly NormalizedItem[];
}

function normalizeFeed(parsed: unknown): NormalizedFeed | undefined {
	const root = asRecord(parsed);
	if (root === undefined) return undefined;
	const channel = asRecord(asRecord(root.rss)?.channel);
	if (channel !== undefined) {
		const items = asArray(channel.item ?? []).concat(asRecord(channel.item) !== undefined ? [channel.item] : []);
		return { title: textOf(channel.title), items: items.map(normalizeRssItem).filter(present) };
	}
	const atom = asRecord(root.feed);
	if (atom !== undefined) {
		const entries = asArray(atom.entry ?? []).concat(asRecord(atom.entry) !== undefined ? [atom.entry] : []);
		return { title: textOf(atom.title), items: entries.map(normalizeAtomEntry).filter(present) };
	}
	return undefined;
}

function normalizeRssItem(raw: unknown): NormalizedItem | undefined {
	const item = asRecord(raw);
	if (item === undefined) return undefined;
	const title = textOf(item.title);
	const guid = textOf(item.guid) ?? textOf(item.link) ?? title;
	if (guid === undefined) return undefined;
	return {
		id: guid,
		title,
		body: textOf(item.description),
		publishedAt: parseEpochMs(textOf(item.pubDate)),
		categories: categoriesOf(item.category),
	};
}

function normalizeAtomEntry(raw: unknown): NormalizedItem | undefined {
	const entry = asRecord(raw);
	if (entry === undefined) return undefined;
	const title = textOf(entry.title);
	const id = textOf(entry.id) ?? title;
	if (id === undefined) return undefined;
	return {
		id,
		title,
		body: textOf(entry.summary) ?? textOf(entry.content),
		publishedAt: parseEpochMs(textOf(entry.published) ?? textOf(entry.updated)),
		categories: categoriesOf(entry.category),
	};
}

/** Extract text from a string node or `{ "#text": … }` attribute node. */
function textOf(value: unknown): string | undefined {
	const direct = asString(value);
	if (direct !== undefined) return direct.trim();
	if (typeof value === "number") return String(value);
	const record = asRecord(value);
	if (record !== undefined) {
		const text = asString(record["#text"]) ?? asString(record["@_term"]);
		if (text !== undefined) return text.trim();
		if (typeof record["#text"] === "number") return String(record["#text"]);
	}
	return undefined;
}

function categoriesOf(value: unknown): readonly string[] {
	const values = Array.isArray(value) ? value : value !== undefined ? [value] : [];
	return values.map(textOf).filter((category): category is string => category !== undefined && category.length > 0);
}

function stripHtml(text: string): string {
	return text
		.replace(/<[^>]+>/gu, " ")
		.replace(/\s+/gu, " ")
		.trim();
}

function present(item: NormalizedItem | undefined): item is NormalizedItem {
	return item !== undefined;
}
