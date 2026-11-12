import type { ChannelKind } from "../registry.ts";
import { assertEnvName, postJson, requireEnv, splitText } from "../support.ts";
import { MessagingConfigError, MessagingError, type OutboundChannel } from "../types.ts";

/** A Discord bot posting to one channel the user named. The bot token stays in the environment variable `tokenEnv` names. */
export interface DiscordChannelConfig {
	readonly kind: "discord";
	readonly tokenEnv: string;
	/** Snowflake id of the text channel. Kept as a string: snowflakes exceed 2^53. */
	readonly channelId: string;
}

const DISCORD_LIMIT = 2000;
const SUPPRESS_EMBEDS = 1 << 2;
const FIELDS = new Set(["kind", "tokenEnv", "channelId"]);
const SNOWFLAKE = /^\d{15,25}$/;

function snowflakeOf(value: unknown, path: string): string {
	const text = typeof value === "number" && Number.isSafeInteger(value) ? String(value) : value;
	if (typeof text !== "string" || !SNOWFLAKE.test(text)) {
		throw new MessagingConfigError(`${path} must be the channel's numeric id (Developer Mode > Copy Channel ID)`);
	}
	return text;
}

export const discordKind: ChannelKind<DiscordChannelConfig> = {
	kind: "discord",
	validate(raw, path) {
		for (const key of Object.keys(raw)) {
			if (!FIELDS.has(key)) throw new MessagingConfigError(`${path}.${key} is not a recognized field`);
		}
		return {
			kind: "discord",
			tokenEnv: assertEnvName(raw.tokenEnv, `${path}.tokenEnv`),
			channelId: snowflakeOf(raw.channelId, `${path}.channelId`),
		};
	},
	create(id, config, runtime): OutboundChannel {
		return {
			id,
			kind: "discord",
			addressing: "fixed",
			describe: () => `Discord channel ${config.channelId}`,
			async send(message, signal) {
				const subject = message.subject?.trim();
				const text = subject ? `**${subject}**\n\n${message.text}` : message.text;
				if (text.trim() === "") throw new MessagingError("The message is empty.");
				const token = requireEnv(runtime.env, config.tokenEnv, "The Discord bot token");
				const parts = splitText(text, DISCORD_LIMIT);
				const ids: string[] = [];
				for (const [index, part] of parts.entries()) {
					try {
						const reply = await postJson(runtime.fetch, {
							label: "Discord",
							url: `https://discord.com/api/v10/channels/${config.channelId}/messages`,
							headers: { authorization: `Bot ${token}` },
							// Model text can contain @everyone or a role mention; nothing it writes may notify anyone.
							body: { content: part, allowed_mentions: { parse: [] }, flags: SUPPRESS_EMBEDS },
							secrets: [token],
							...(signal === undefined ? {} : { signal }),
						});
						const messageId =
							typeof reply === "object" && reply !== null ? (reply as { id?: unknown }).id : undefined;
						if (typeof messageId !== "string" || messageId === "") {
							throw new MessagingError(
								"Discord answered without a message id, so the send cannot be confirmed.",
							);
						}
						ids.push(messageId);
					} catch (error) {
						if (!(error instanceof MessagingError) || parts.length === 1) throw error;
						throw new MessagingError(
							`${error.message} Only ${index} of ${parts.length} parts were sent; do not resend the whole message.`,
						);
					}
				}
				return { destination: `Discord channel ${config.channelId}`, ids };
			},
		};
	},
};
