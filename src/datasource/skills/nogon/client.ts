/**
 * HTTP client for a running Nogon server (https://github.com/asciimoo/nogon, AGPL-3.0).
 *
 * Nogon is a separate program. AutoRAG never links, embeds, copies, or modifies its code: this file speaks
 * Nogon's public `GET /search` and `GET /api/config` JSON endpoints over HTTP, exactly like Nogon's own
 * `nogon search` CLI. Nogon owns its index, its browser-extension capture, its credentials, and its embeddings.
 */

import { createHash } from "node:crypto";
import { asArray, asNumber, asRecord, asString } from "../../http.ts";

/** Nogon `document.DocType`: 0 web page, 1 watched local file, 2 remote file snapshot. */
const DOC_TYPE_WEB = 0;

const DEFAULT_BASE_URL = "http://127.0.0.1:4433";
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_TOP_K = 20;
/** Nogon rejects a search with more terms than it can parse; keep a natural-language question bounded. */
const MAX_TERMS = 24;
const WEB_ONLY = "type:web";

export interface NogonClientOptions {
	/** Nogon server, default `http://127.0.0.1:4433`. */
	readonly baseUrl?: string;
	/** Name of the environment variable holding Nogon's `app.access_token`. The token itself is never configured. */
	readonly tokenEnv?: string;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export interface NogonHit {
	readonly url: string;
	readonly domain: string;
	readonly title: string;
	/** Highlighted-mark-free snippet (or the matched semantic chunk). */
	readonly snippet: string;
	readonly score: number;
	/** Epoch ms of the last visit/update. */
	readonly updatedAt: number | undefined;
	readonly addedAt: number | undefined;
	readonly language: string | undefined;
	readonly description: string | undefined;
}

export type NogonFailureKind = "unavailable" | "auth" | "http" | "invalid-output" | "semantic-disabled";

export interface NogonFailure {
	readonly ok: false;
	readonly kind: NogonFailureKind;
	/** Verbatim: HTTP status and body, or the connection error. */
	readonly message: string;
}

export type NogonResult<T> = { readonly ok: true; readonly value: T } | NogonFailure;

export interface NogonSearchOptions {
	readonly topK?: number;
	readonly signal?: AbortSignal;
	/** Restrict to one domain (exact match, Nogon `domain:`). */
	readonly domain?: string;
}

export class NogonClient {
	readonly baseUrl: string;
	private readonly tokenEnv: string | undefined;
	private readonly timeoutMs: number;
	private readonly fetchImpl: typeof fetch;
	private readonly env: Readonly<Record<string, string | undefined>>;

	constructor(options: NogonClientOptions = {}) {
		const raw = (options.baseUrl ?? DEFAULT_BASE_URL).trim();
		let parsed: URL;
		try {
			parsed = new URL(raw);
		} catch {
			throw new Error(`nogon baseUrl must be an http(s) URL, got ${JSON.stringify(raw)}`);
		}
		if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
			throw new Error(`nogon baseUrl must be an http(s) URL, got ${JSON.stringify(raw)}`);
		}
		this.tokenEnv = options.tokenEnv;
		this.env = options.env ?? process.env;
		if (this.tokenEnv !== undefined && parsed.protocol === "http:" && !isLoopback(parsed.hostname)) {
			throw new Error(
				`nogon baseUrl must use https when an access token is configured and the host is not loopback (${parsed.hostname})`,
			);
		}
		this.baseUrl = trimTrailingSlashes(raw);
		this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
	}

	/** Lexical search over web pages only. */
	async searchLexical(query: string, options: NogonSearchOptions = {}): Promise<NogonResult<readonly NogonHit[]>> {
		const text = buildWebQuery(query, options.domain);
		if (text === undefined) return { ok: true, value: [] };
		const response = await this.search(
			{ text, highlight: "HTML", limit: options.topK ?? DEFAULT_TOP_K, semantic_enabled: false },
			options.signal,
		);
		if (!response.ok) return response;
		return { ok: true, value: webDocuments(response.value.documents) };
	}

	/**
	 * Semantic search. Nogon embeds the query itself through the endpoint the operator configured on the server;
	 * AutoRAG sends only the question text. `semantic_hits` carry the matched chunk and similarity.
	 */
	async searchSemantic(query: string, options: NogonSearchOptions = {}): Promise<NogonResult<readonly NogonHit[]>> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return { ok: true, value: [] };
		const text = options.domain === undefined ? trimmed : `${trimmed} domain:${options.domain}`;
		const response = await this.search(
			{ text, highlight: "HTML", limit: options.topK ?? DEFAULT_TOP_K, semantic_enabled: true },
			options.signal,
		);
		if (!response.ok) return response;
		if (response.value.semanticEnabled !== true) {
			return {
				ok: false,
				kind: "semantic-disabled",
				message:
					"the Nogon server did not run a semantic search. Enable `semantic_search.enable` with an embedding endpoint in the Nogon server config (https://nogon.eu/docs/configuration), restart `nogon listen`, or set connector.semantic to false",
			};
		}
		const byUrl = new Map<string, Record<string, unknown>>();
		for (const document of response.value.documents) {
			const record = asRecord(document);
			const url = asString(record?.url);
			if (record !== undefined && url !== undefined) byUrl.set(url, record);
		}
		const hits: NogonHit[] = [];
		for (const raw of response.value.semanticHits) {
			const hit = asRecord(raw);
			if (hit === undefined) continue;
			const docId = asString(hit.doc_id);
			const similarity = asNumber(hit.similarity);
			if (docId === undefined || similarity === undefined) continue;
			// Nogon only returns the full document for semantic-only hits; a keyword overlap is resolved from
			// `documents`. A hit whose type cannot be established is dropped rather than assumed to be a web page.
			const document = asRecord(hit.document) ?? byUrl.get(docId);
			if (document === undefined || asNumber(document.type) !== DOC_TYPE_WEB) continue;
			const mapped = toHit(document, similarity, asString(hit.matched_chunk));
			if (mapped !== undefined) hits.push(mapped);
		}
		return { ok: true, value: hits };
	}

	/** Count documents matching a Nogon query. */
	async count(text: string, signal?: AbortSignal): Promise<NogonResult<number>> {
		const response = await this.search({ text, limit: 1, semantic_enabled: false }, signal);
		return response.ok ? { ok: true, value: response.value.total } : response;
	}

	/** Reachability and authentication probe. Returns Nogon's `/api/config` body. */
	async config(signal?: AbortSignal): Promise<NogonResult<Record<string, unknown>>> {
		const response = await this.request("/api/config", signal);
		if (!response.ok) return response;
		const record = asRecord(response.value);
		if (record === undefined) {
			return { ok: false, kind: "invalid-output", message: "nogon /api/config did not return a JSON object" };
		}
		return { ok: true, value: record };
	}

	private async search(
		query: Record<string, unknown>,
		signal: AbortSignal | undefined,
	): Promise<
		NogonResult<{
			readonly total: number;
			readonly documents: readonly unknown[];
			readonly semanticHits: readonly unknown[];
			readonly semanticEnabled: boolean | undefined;
		}>
	> {
		const response = await this.request(`/search?query=${encodeURIComponent(JSON.stringify(query))}`, signal);
		if (!response.ok) return response;
		const body = asRecord(response.value);
		const documents = body?.documents;
		if (body === undefined || (documents !== null && !Array.isArray(documents))) {
			return { ok: false, kind: "invalid-output", message: "nogon /search did not return a search result object" };
		}
		const error = asString(body.error);
		if (error !== undefined && error.length > 0)
			return { ok: false, kind: "http", message: `nogon rejected the query: ${error}` };
		return {
			ok: true,
			value: {
				total: asNumber(body.total) ?? 0,
				documents: asArray(documents),
				semanticHits: asArray(body.semantic_hits),
				semanticEnabled: typeof body.semantic_enabled === "boolean" ? body.semantic_enabled : undefined,
			},
		};
	}

	private async request(path: string, signal: AbortSignal | undefined): Promise<NogonResult<unknown>> {
		const headers: Record<string, string> = {
			// Nogon answers a request without this header with a bare 500.
			Accept: "application/json",
		};
		if (this.tokenEnv !== undefined) {
			const token = this.env[this.tokenEnv];
			if (token !== undefined && token.length > 0) headers["X-Access-Token"] = token;
		}
		const timeout = AbortSignal.timeout(this.timeoutMs);
		let response: Response;
		try {
			response = await this.fetchImpl(`${this.baseUrl}${path}`, {
				method: "GET",
				headers,
				signal: signal === undefined ? timeout : AbortSignal.any([timeout, signal]),
			});
		} catch (error) {
			const reason = error instanceof Error ? error.message : String(error);
			const cause = error instanceof Error && error.cause instanceof Error ? ` (${error.cause.message})` : "";
			return {
				ok: false,
				kind: "unavailable",
				message: `nogon server at ${this.baseUrl} is not reachable: ${reason}${cause}. Start it with \`nogon listen\` and check connector.baseUrl.`,
			};
		}
		let body: string;
		try {
			body = await response.text();
		} catch (error) {
			return { ok: false, kind: "unavailable", message: `nogon response body unreadable: ${String(error)}` };
		}
		if (!response.ok) {
			const kind: NogonFailureKind = response.status === 401 || response.status === 403 ? "auth" : "http";
			return {
				ok: false,
				kind,
				message: `nogon ${path.split("?")[0]} returned HTTP ${response.status}: ${body.trim()}`,
			};
		}
		try {
			return { ok: true, value: JSON.parse(body) as unknown };
		} catch {
			return {
				ok: false,
				kind: "invalid-output",
				message: `nogon ${path.split("?")[0]} did not return JSON (unexpected content: ${body.slice(0, 120).trim()})`,
			};
		}
	}
}

/**
 * Wrap a free-text question as a Nogon query confined to web pages.
 *
 * Nogon ANDs bare terms, so a natural-language question would match almost nothing. Plain words become one
 * alternation group `(a|b|c)`, while uppercase `AND`/`OR`/`NOT` tokens in natural-language questions are stripped by
 * {@link normalizeTerm} so Nogon 0.21+'s boolean operator parser never rejects casual English phrasing. Nogon's own
 * syntax (quoted phrases, `field:value`, wildcards, negation) passes through untouched, and a query using any of it is
 * sent with its plain words ANDed, exactly as written. A `sort:` directive is hoisted out of the group because Nogon
 * only honours it at the top level.
 */
export function buildWebQuery(question: string, domain?: string): string | undefined {
	const tokens = tokenize(question);
	let sort: string | undefined;
	const words: string[] = [];
	for (const token of tokens) {
		if (/^sort:-?(relevance|date|visits|domain)$/u.test(token)) sort = token;
		else words.push(token);
	}
	const terms = words.map(normalizeTerm).filter((term): term is NormalizedTerm => term !== undefined);
	const structured = terms.some((term) => term.operator);
	const flat = terms.flatMap((term) => term.parts).slice(0, MAX_TERMS);
	let group: string;
	if (structured) group = flat.join(" ");
	else group = flat.join("|");
	if (group.length === 0 && sort === undefined) return undefined;
	const parts: string[] = [];
	if (group.length > 0) parts.push(`(${group})`);
	if (domain !== undefined) parts.push(`domain:${domain}`);
	parts.push(WEB_ONLY);
	if (sort !== undefined) parts.push(sort);
	return parts.join(" ");
}

/**
 * Split on whitespace, keeping a double-quoted phrase (optionally field-prefixed and/or negated, like
 * `title:"a b"` or `-"a b"`) in one token. A linear scan, so no input can make it slow.
 */
function tokenize(input: string): string[] {
	const tokens: string[] = [];
	let i = 0;
	while (i < input.length) {
		if (/\s/u.test(input[i] ?? "")) {
			i += 1;
			continue;
		}
		const start = i;
		let quoted = false;
		while (i < input.length && !/\s/u.test(input[i] ?? "")) {
			if (input[i] === '"') {
				// A quote is only a phrase when it is the token's first character or follows an optional `-` and field prefix.
				if (!quoted && isPhraseStart(input, start, i)) {
					const close = input.indexOf('"', i + 1);
					if (close !== -1) {
						i = close + 1;
						quoted = true;
						break;
					}
				}
			}
			i += 1;
		}
		tokens.push(input.slice(start, i));
	}
	return tokens;
}

/**
 * True when the `"` at `quote` opens a phrase, i.e. the token so far is exactly one of `"`, `-"`, `field:"`,
 * `-field:"`, `field:-"` or `-field:-"`, where `field` is one or more of `A-Za-z_.`. Same language as the original
 * regular expression `(?:-?[A-Za-z_.]+:)?-?"[^"]*"`, without its backtracking.
 */
function isPhraseStart(input: string, start: number, quote: number): boolean {
	let j = start;
	if (input[j] === "-") j += 1;
	let k = j;
	while (k < quote && /[A-Za-z_.]/u.test(input[k] ?? "")) k += 1;
	if (k > j && input[k] === ":") {
		j = k + 1;
		if (input[j] === "-") j += 1;
		return j === quote;
	}
	return start + (input[start] === "-" ? 1 : 0) === quote;
}

function trimTrailingSlashes(text: string): string {
	let end = text.length;
	while (end > 0 && text[end - 1] === "/") end -= 1;
	return text.slice(0, end);
}

interface NormalizedTerm {
	/** True for Nogon syntax (phrase, `field:value`, negation) that must reach the server unchanged. */
	readonly operator: boolean;
	readonly parts: readonly string[];
}

function normalizeTerm(token: string): NormalizedTerm | undefined {
	if (token.startsWith('"') || /^-?[A-Za-z_.]+:/u.test(token) || token.startsWith("-")) {
		return { operator: true, parts: [token] };
	}
	const upper = token.toUpperCase();
	if (upper === "AND" || upper === "OR" || upper === "NOT") return undefined;
	// Strip punctuation that is meaningless to Nogon's tokenizer, and that would break its `(a|b)` grammar when it is
	// `|` or a parenthesis. A hyphenated word becomes its parts.
	const cleaned = token.replace(/[^\p{L}\p{N}*_]+/gu, " ").trim();
	if (cleaned.length === 0) return undefined;
	return { operator: false, parts: cleaned.split(/\s+/u) };
}

function webDocuments(documents: readonly unknown[]): NogonHit[] {
	const hits: NogonHit[] = [];
	for (const raw of documents) {
		const record = asRecord(raw);
		if (record === undefined) continue;
		// The query already says type:web; this is the second, independent barrier against local files.
		if (asNumber(record.type) !== DOC_TYPE_WEB) continue;
		const score = asNumber(record.score) ?? 0;
		const hit = toHit(record, score, undefined);
		if (hit !== undefined) hits.push(hit);
	}
	return hits;
}

function toHit(record: Record<string, unknown>, score: number, chunk: string | undefined): NogonHit | undefined {
	const url = asString(record.url);
	if (url === undefined || !/^https?:\/\//iu.test(url)) return undefined;
	const metadata = asRecord(record.metadata);
	const updated = asNumber(record.updated);
	const added = asNumber(record.added);
	const language = asString(record.language);
	return {
		url,
		domain: asString(record.domain) ?? hostOf(url),
		title: asString(record.title) ?? url,
		snippet: plainSnippet(chunk ?? asString(record.text) ?? ""),
		score,
		updatedAt: updated === undefined ? undefined : updated * 1000,
		addedAt: added === undefined ? undefined : added * 1000,
		language: language === undefined || language === "unknown" ? undefined : language,
		description: asString(metadata?.description),
	};
}

function hostOf(url: string): string {
	try {
		return new URL(url).hostname;
	} catch {
		return "unknown";
	}
}

/** Nogon snippets carry `<mark>` highlights and HTML entities; the model reads plain text. */
export function plainSnippet(snippet: string): string {
	return snippet
		.replace(/<\/?mark>/giu, "")
		.replace(/&#(\d+);/gu, (_m, code: string) => String.fromCodePoint(Number(code)))
		.replace(/&#x([0-9a-f]+);/giu, (_m, code: string) => String.fromCodePoint(Number.parseInt(code, 16)))
		.replace(/&quot;/gu, '"')
		.replace(/&apos;/gu, "'")
		.replace(/&lt;/gu, "<")
		.replace(/&gt;/gu, ">")
		.replace(/&amp;/gu, "&")
		.trim();
}

/** Stable, opaque id for one page: the same URL always maps to the same chunk id. */
export function pageId(url: string): string {
	return createHash("sha256").update(url).digest("hex").slice(0, 16);
}

export function isLoopback(hostname: string): boolean {
	const host = hostname.replace(/^\[|\]$/gu, "").toLowerCase();
	return host === "localhost" || host === "::1" || host === "127.0.0.1" || /^127\.\d+\.\d+\.\d+$/u.test(host);
}
