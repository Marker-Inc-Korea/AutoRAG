/**
 * Discord guild connector (issue #1305).
 *
 * Fetches text/announcement channel history through the Discord REST API
 * using a trusted, server-configured bot token. Per-channel permission
 * failures degrade to warnings; a guild-level 429 fails the whole fetch as
 * rate-limited. Never throws; all messages stay path/PII-opaque.
 */

import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asNumber, asRecord, asString, httpJson, parseEpochMs, resolveToken } from "../../http.ts";

export interface DiscordConnectorOptions {
	/** Discord REST base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** Bot token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the bot token. Default `DISCORD_BOT_TOKEN`. */
	readonly tokenEnv?: string;
	/** Guild (server) id to index. Required trusted configuration. */
	readonly guildId?: string;
	/** Restrict indexing to these channel names or ids. */
	readonly channels?: readonly string[];
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxPages?: number;
	readonly maxDocuments?: number;
}

const DEFAULT_BASE_URL = "https://discord.com/api/v10";
const DEFAULT_TOKEN_ENV = "DISCORD_BOT_TOKEN";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_DOCUMENTS = 500;
/** Channel types indexed: 0 = guild text, 5 = announcement. */
const TEXT_CHANNEL_TYPES = new Set([0, 5]);

export class DiscordConnector implements DatasourceConnector {
	private readonly options: DiscordConnectorOptions;

	constructor(options: DiscordConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const token = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		if (token === undefined) return { ok: false, reason: "not-configured", message: "auth token not configured" };
		const guildId = this.options.guildId;
		if (guildId === undefined || guildId.length === 0) {
			return { ok: false, reason: "not-configured", message: "guild id not configured" };
		}
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const request = {
			headers: { Authorization: `Bot ${token}` },
			timeoutMs: this.options.timeoutMs,
			fetchImpl: this.options.fetchImpl,
			signal,
		};
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;

		// 1. Enumerate guild channels; keep text/announcement channels.
		const listResult = await httpJson(`${baseUrl}/guilds/${guildId}/channels`, request);
		if (!listResult.ok) {
			return { ok: false, reason: listResult.reason, message: `channel list failed: ${listResult.message}` };
		}
		const channels: { id: string; name: string }[] = [];
		for (const raw of asArray(listResult.json)) {
			const channel = asRecord(raw);
			const id = asString(channel?.id);
			const name = asString(channel?.name);
			const type = asNumber(channel?.type);
			if (id === undefined || name === undefined || type === undefined) continue;
			if (!TEXT_CHANNEL_TYPES.has(type)) continue;
			channels.push({ id, name });
		}
		const filter = this.options.channels;
		const selected =
			filter === undefined || filter.length === 0
				? channels
				: channels.filter((channel) => filter.includes(channel.id) || filter.includes(channel.name));

		// 2. Page messages per channel (newest first, `before` cursor).
		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		for (const channel of selected) {
			if (documents.length >= maxDocuments) break;
			let before: string | undefined;
			for (let page = 0; page < maxPages && documents.length < maxDocuments; page += 1) {
				const url = new URL(`${baseUrl}/channels/${channel.id}/messages`);
				url.searchParams.set("limit", "100");
				if (before !== undefined) url.searchParams.set("before", before);
				const result = await httpJson(url.toString(), request);
				if (!result.ok) {
					if (result.reason === "rate-limited") {
						return { ok: false, reason: "rate-limited", message: "discord: rate limited" };
					}
					if (result.reason === "permission") {
						warnings.push(`channel ${channel.name} skipped: missing permission`);
					} else {
						warnings.push(`channel ${channel.name} history failed: ${result.message}`);
					}
					break;
				}
				const messages = asArray(result.json);
				if (messages.length === 0) break;
				let lastId: string | undefined;
				for (const raw of messages) {
					const message = asRecord(raw);
					const id = asString(message?.id);
					if (id !== undefined) lastId = id;
					if (documents.length >= maxDocuments) continue;
					const content = asString(message?.content);
					if (message === undefined || id === undefined || content === undefined || content.length === 0) continue;
					const author = asString(asRecord(message.author)?.username) ?? "unknown";
					documents.push({
						docId: `${channel.id}-${id}`,
						hierarchy: ["channels", channel.name],
						title: `#${channel.name}`,
						content: `${author}: ${content}`,
						publishedAt: parseEpochMs(message.timestamp),
						metadata: { channelId: channel.id, messageId: id, author },
					});
				}
				if (lastId === undefined || messages.length < 100) break;
				before = lastId;
			}
		}

		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}
