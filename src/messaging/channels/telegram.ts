import type { ChannelKind } from "../registry.ts";
import { assertEnvName, postJson, requireEnv, scrubSecrets, splitText } from "../support.ts";
import { MessagingConfigError, MessagingError, type OutboundChannel } from "../types.ts";

/** A Telegram bot posting to one chat the user named. The bot token stays in the environment variable `tokenEnv` names. */
export interface TelegramChannelConfig {
	readonly kind: "telegram";
	readonly tokenEnv: string;
	/** Numeric chat id (groups are negative) or `@channelname`. Kept as a string: ids exceed 2^31. */
	readonly chatId: string;
	/** Forum topic to post into. */
	readonly threadId?: number;
}

const TELEGRAM_LIMIT = 4096;
const FIELDS = new Set(["kind", "tokenEnv", "chatId", "threadId"]);
const CHAT_ID = /^(-?\d{1,20}|@[A-Za-z][A-Za-z0-9_]{3,31})$/;

function chatIdOf(value: unknown, path: string): string {
	const text = typeof value === "number" && Number.isSafeInteger(value) ? String(value) : value;
	if (typeof text !== "string" || !CHAT_ID.test(text)) {
		throw new MessagingConfigError(`${path} must be a numeric chat id (groups are negative) or an @channelname`);
	}
	return text;
}

function messageIdOf(reply: unknown): number | undefined {
	if (typeof reply !== "object" || reply === null) return undefined;
	const result = (reply as { result?: unknown }).result;
	if (typeof result !== "object" || result === null) return undefined;
	const id = (result as { message_id?: unknown }).message_id;
	return typeof id === "number" ? id : undefined;
}

export const telegramKind: ChannelKind<TelegramChannelConfig> = {
	kind: "telegram",
	validate(raw, path) {
		for (const key of Object.keys(raw)) {
			if (!FIELDS.has(key)) throw new MessagingConfigError(`${path}.${key} is not a recognized field`);
		}
		const tokenEnv = assertEnvName(raw.tokenEnv, `${path}.tokenEnv`);
		const chatId = chatIdOf(raw.chatId, `${path}.chatId`);
		if (raw.threadId === undefined) return { kind: "telegram", tokenEnv, chatId };
		if (typeof raw.threadId !== "number" || !Number.isSafeInteger(raw.threadId) || raw.threadId < 1) {
			throw new MessagingConfigError(`${path}.threadId must be a positive integer`);
		}
		return { kind: "telegram", tokenEnv, chatId, threadId: raw.threadId };
	},
	create(id, config, runtime): OutboundChannel {
		return {
			id,
			kind: "telegram",
			addressing: "fixed",
			describe: () =>
				`Telegram chat ${config.chatId}${config.threadId === undefined ? "" : ` (topic ${config.threadId})`}`,
			async send(message, signal) {
				const subject = message.subject?.trim();
				const text = subject ? `${subject}\n\n${message.text}` : message.text;
				if (text.trim() === "") throw new MessagingError("The message is empty.");
				const token = requireEnv(runtime.env, config.tokenEnv, "The Telegram bot token");
				const parts = splitText(text, TELEGRAM_LIMIT);
				const ids: string[] = [];
				for (const [index, part] of parts.entries()) {
					try {
						const reply = await postJson(runtime.fetch, {
							label: "Telegram",
							url: `https://api.telegram.org/bot${token}/sendMessage`,
							body: {
								chat_id: config.chatId,
								text: part,
								link_preview_options: { is_disabled: true },
								...(config.threadId === undefined ? {} : { message_thread_id: config.threadId }),
							},
							secrets: [token],
							...(signal === undefined ? {} : { signal }),
						});
						if (typeof reply === "object" && reply !== null && (reply as { ok?: unknown }).ok === false) {
							const why = String((reply as { description?: unknown }).description ?? "no reason given");
							throw new MessagingError(`Telegram rejected the message: ${scrubSecrets(why, [token])}`);
						}
						const messageId = messageIdOf(reply);
						if (messageId === undefined) {
							throw new MessagingError(
								"Telegram answered without a message id, so the send cannot be confirmed.",
							);
						}
						ids.push(String(messageId));
					} catch (error) {
						if (!(error instanceof MessagingError) || parts.length === 1) throw error;
						throw new MessagingError(
							`${error.message} Only ${index} of ${parts.length} parts were sent; do not resend the whole message.`,
						);
					}
				}
				return { destination: `Telegram chat ${config.chatId}`, ids };
			},
		};
	},
};
