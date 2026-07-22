/**
 * Slack workspace connector (issue #1300).
 *
 * Fetches channel history through the Slack Web API using a trusted,
 * server-configured bot token. Never throws; failures map onto the coarse
 * connector failure union with path/PII-opaque messages. Per-channel
 * failures degrade to warnings so one denied channel cannot fail the fetch.
 */

import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, httpJson, resolveToken } from "../../http.ts";

export interface SlackConnectorOptions {
	/** Slack Web API base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** Bot token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the bot token. Default `SLACK_BOT_TOKEN`. */
	readonly tokenEnv?: string;
	/** Restrict indexing to these channel names or ids. */
	readonly channels?: readonly string[];
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxPages?: number;
	readonly maxDocuments?: number;
}

const DEFAULT_BASE_URL = "https://slack.com/api";
const DEFAULT_TOKEN_ENV = "SLACK_BOT_TOKEN";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_DOCUMENTS = 500;
const AUTH_ERRORS = new Set(["invalid_auth", "not_authed", "token_revoked", "account_inactive"]);
const CHANNEL_SKIP_ERRORS = new Set(["missing_scope", "access_denied", "channel_not_found", "not_in_channel"]);

interface SlackChannel {
	readonly id: string;
	readonly name: string;
}

export class SlackConnector implements DatasourceConnector {
	private readonly options: SlackConnectorOptions;

	constructor(options: SlackConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const token = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		if (token === undefined) return { ok: false, reason: "not-configured", message: "auth token not configured" };
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const headers = { Authorization: `Bearer ${token}` };
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;
		const request = { headers, timeoutMs: this.options.timeoutMs, fetchImpl: this.options.fetchImpl, signal };

		// 1. Enumerate channels (cursor-paginated).
		const channels: SlackChannel[] = [];
		let cursor: string | undefined;
		for (let page = 0; page < maxPages; page += 1) {
			const url = new URL(`${baseUrl}/conversations.list`);
			url.searchParams.set("limit", "200");
			url.searchParams.set("types", "public_channel,private_channel");
			if (cursor !== undefined) url.searchParams.set("cursor", cursor);
			const result = await httpJson(url.toString(), request);
			if (!result.ok) return { ok: false, reason: result.reason, message: `channel list failed: ${result.message}` };
			const envelope = asRecord(result.json);
			if (envelope === undefined) return { ok: false, reason: "invalid-data", message: "channel list malformed" };
			if (envelope.ok !== true) {
				const error = asString(envelope.error) ?? "unknown-error";
				if (AUTH_ERRORS.has(error)) return { ok: false, reason: "auth", message: `slack: ${error}` };
				if (error === "ratelimited") return { ok: false, reason: "rate-limited", message: "slack: ratelimited" };
				return { ok: false, reason: "api-error", message: `slack: ${error}` };
			}
			for (const raw of asArray(envelope.channels)) {
				const channel = asRecord(raw);
				const id = asString(channel?.id);
				const name = asString(channel?.name);
				if (id !== undefined && name !== undefined) channels.push({ id, name });
			}
			cursor = asString(asRecord(envelope.response_metadata)?.next_cursor);
			if (cursor === undefined || cursor.length === 0) break;
		}

		const filter = this.options.channels;
		const selected =
			filter === undefined || filter.length === 0
				? channels
				: channels.filter((channel) => filter.includes(channel.id) || filter.includes(channel.name));

		// 2. Fetch message history per channel; degrade per-channel failures.
		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		for (const channel of selected) {
			if (documents.length >= maxDocuments) break;
			let historyCursor: string | undefined;
			for (let page = 0; page < maxPages && documents.length < maxDocuments; page += 1) {
				const url = new URL(`${baseUrl}/conversations.history`);
				url.searchParams.set("channel", channel.id);
				url.searchParams.set("limit", "200");
				if (historyCursor !== undefined) url.searchParams.set("cursor", historyCursor);
				const result = await httpJson(url.toString(), request);
				if (!result.ok) {
					warnings.push(`channel ${channel.name} history failed: ${result.message}`);
					break;
				}
				const envelope = asRecord(result.json);
				if (envelope === undefined || envelope.ok !== true) {
					const error = asString(envelope?.error) ?? "malformed response";
					if (AUTH_ERRORS.has(error)) return { ok: false, reason: "auth", message: `slack: ${error}` };
					if (error === "ratelimited") return { ok: false, reason: "rate-limited", message: "slack: ratelimited" };
					if (CHANNEL_SKIP_ERRORS.has(error)) warnings.push(`channel ${channel.name} skipped: ${error}`);
					else warnings.push(`channel ${channel.name} history failed: ${error}`);
					break;
				}
				for (const raw of asArray(envelope.messages)) {
					if (documents.length >= maxDocuments) break;
					const message = asRecord(raw);
					if (message === undefined) continue;
					const text = asString(message.text);
					const ts = asString(message.ts);
					const subtype = asString(message.subtype);
					if (text === undefined || text.length === 0 || ts === undefined) continue;
					if (subtype === "channel_join") continue;
					const user = asString(message.user) ?? asString(message.username) ?? "unknown";
					documents.push({
						docId: `${channel.id}-${ts.replace(/\./gu, "-")}`,
						hierarchy: ["channels", channel.name],
						title: `#${channel.name}`,
						content: `${user}: ${text}`,
						publishedAt: Math.round(Number.parseFloat(ts) * 1000),
						metadata: { channelId: channel.id, ts, user },
					});
				}
				historyCursor = asString(asRecord(envelope.response_metadata)?.next_cursor);
				if (historyCursor === undefined || historyCursor.length === 0) break;
			}
		}

		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}
