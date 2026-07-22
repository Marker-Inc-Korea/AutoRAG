/**
 * Gmail connector (issue #1304).
 *
 * Lists and fetches messages through the Gmail REST v1 API with a trusted,
 * server-configured OAuth access token. Per-message fetch failures degrade
 * to warnings. Never throws; messages stay path/PII-opaque (addresses appear
 * only inside indexed chunk content, never in diagnostics).
 */

import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, httpJson, resolveToken } from "../../http.ts";

export interface GmailConnectorOptions {
	/** Gmail REST base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** OAuth access token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the access token. Default `GMAIL_ACCESS_TOKEN`. */
	readonly tokenEnv?: string;
	/** Restrict to these label ids (e.g. `["INBOX"]`). */
	readonly labelIds?: readonly string[];
	/** Gmail search query (the `q` parameter). */
	readonly query?: string;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxPages?: number;
	readonly maxDocuments?: number;
}

const DEFAULT_BASE_URL = "https://gmail.googleapis.com/gmail/v1";
const DEFAULT_TOKEN_ENV = "GMAIL_ACCESS_TOKEN";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_DOCUMENTS = 500;
const MAX_BODY_CHARS = 50_000;

export class GmailConnector implements DatasourceConnector {
	private readonly options: GmailConnectorOptions;

	constructor(options: GmailConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const token = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		if (token === undefined) return { ok: false, reason: "not-configured", message: "auth token not configured" };
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const request = {
			headers: { Authorization: `Bearer ${token}` },
			timeoutMs: this.options.timeoutMs,
			fetchImpl: this.options.fetchImpl,
			signal,
		};
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;

		// 1. List message ids (token-paginated).
		const messageIds: string[] = [];
		let pageToken: string | undefined;
		for (let page = 0; page < maxPages && messageIds.length < maxDocuments; page += 1) {
			const url = new URL(`${baseUrl}/users/me/messages`);
			url.searchParams.set("maxResults", "100");
			for (const labelId of this.options.labelIds ?? []) url.searchParams.append("labelIds", labelId);
			if (this.options.query !== undefined) url.searchParams.set("q", this.options.query);
			if (pageToken !== undefined) url.searchParams.set("pageToken", pageToken);
			const result = await httpJson(url.toString(), request);
			if (!result.ok) return { ok: false, reason: result.reason, message: `message list failed: ${result.message}` };
			const envelope = asRecord(result.json);
			if (envelope === undefined) return { ok: false, reason: "invalid-data", message: "message list malformed" };
			for (const raw of asArray(envelope.messages)) {
				const id = asString(asRecord(raw)?.id);
				if (id !== undefined) messageIds.push(id);
			}
			pageToken = asString(envelope.nextPageToken);
			if (pageToken === undefined || pageToken.length === 0) break;
		}

		// 2. Fetch full messages; degrade per-message failures to warnings.
		const documents: ConnectorDocument[] = [];
		let failedFetches = 0;
		for (const id of messageIds.slice(0, maxDocuments)) {
			const result = await httpJson(`${baseUrl}/users/me/messages/${id}?format=full`, request);
			if (!result.ok) {
				if (result.reason === "auth") return { ok: false, reason: "auth", message: "gmail: unauthorized" };
				if (result.reason === "rate-limited") {
					return { ok: false, reason: "rate-limited", message: "gmail: rate limited" };
				}
				failedFetches += 1;
				continue;
			}
			const message = asRecord(result.json);
			if (message === undefined) {
				failedFetches += 1;
				continue;
			}
			const payload = asRecord(message.payload);
			const headers = headerMap(payload);
			const subject = headers.get("subject") ?? "(no subject)";
			const body = (extractBody(payload) ?? asString(message.snippet) ?? "").slice(0, MAX_BODY_CHARS);
			const labelIds = asArray(message.labelIds)
				.map((label) => asString(label))
				.filter((label): label is string => label !== undefined);
			const internalDate = asString(message.internalDate);
			const publishedAt =
				internalDate !== undefined && /^\d+$/u.test(internalDate) ? Number(internalDate) : undefined;
			const headerBlock = [
				`Subject: ${subject}`,
				...(headers.has("from") ? [`From: ${headers.get("from")}`] : []),
				...(headers.has("to") ? [`To: ${headers.get("to")}`] : []),
				...(headers.has("date") ? [`Date: ${headers.get("date")}`] : []),
			].join("\n");
			documents.push({
				docId: id,
				hierarchy: ["labels", this.options.labelIds?.[0] ?? "all"],
				title: subject,
				content: `${headerBlock}\n\n${body}`.trim(),
				...(publishedAt !== undefined ? { publishedAt } : {}),
				metadata: {
					...(asString(message.threadId) !== undefined ? { threadId: asString(message.threadId) } : {}),
					labelIds,
				},
			});
		}

		const warnings = failedFetches > 0 ? [`${failedFetches} message(s) failed to fetch`] : undefined;
		return { ok: true, documents, ...(warnings !== undefined ? { warnings } : {}) };
	}
}

function headerMap(payload: Record<string, unknown> | undefined): Map<string, string> {
	const map = new Map<string, string>();
	for (const raw of asArray(payload?.headers)) {
		const header = asRecord(raw);
		const name = asString(header?.name)?.toLowerCase();
		const value = asString(header?.value);
		if (name !== undefined && value !== undefined && !map.has(name)) map.set(name, value);
	}
	return map;
}

/** Depth-first search for text/plain part bodies (base64url-encoded). */
function extractBody(payload: Record<string, unknown> | undefined): string | undefined {
	if (payload === undefined) return undefined;
	const mimeType = asString(payload.mimeType);
	const data = asString(asRecord(payload.body)?.data);
	if (mimeType === "text/plain" && data !== undefined) return decodeBase64Url(data);
	const collected: string[] = [];
	for (const raw of asArray(payload.parts)) {
		const part = asRecord(raw);
		const text = extractBody(part);
		if (text !== undefined && text.length > 0) collected.push(text);
	}
	if (collected.length > 0) return collected.join("\n");
	// Fallback: top-level body regardless of mime type.
	return data !== undefined ? decodeBase64Url(data) : undefined;
}

function decodeBase64Url(data: string): string {
	try {
		return Buffer.from(data, "base64url").toString("utf8");
	} catch {
		return "";
	}
}
