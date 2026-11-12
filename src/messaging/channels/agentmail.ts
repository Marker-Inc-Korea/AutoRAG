import type { ChannelKind } from "../registry.ts";
import { assertEnvName, postJson, requireEnv } from "../support.ts";
import { MessagingConfigError, MessagingError, type OutboundChannel } from "../types.ts";

/**
 * An email address of the agent's own, hosted by AgentMail (agentmail.to): the agent gets an inbox and sends from
 * it, so no mailbox of the user's is ever opened and no SMTP password is stored. The API key stays in the
 * environment variable `apiKeyEnv` names.
 */
export interface AgentMailChannelConfig {
	readonly kind: "agentmail";
	readonly apiKeyEnv: string;
	/** The inbox to send from: its address (`my-agent@agentmail.to`), as AgentMail returns it for `inbox_id`. */
	readonly inboxId: string;
	readonly region: "us" | "eu";
}

const API_HOST: Readonly<Record<AgentMailChannelConfig["region"], string>> = {
	us: "https://api.agentmail.to",
	eu: "https://api.agentmail.eu",
};

/** One send reaches at most this many addresses, so a prompt-injected "email everyone" stays small. */
export const MAX_RECIPIENTS = 10;

const ADDRESS = /^[^\s@<>(),;:"\\]+@[^\s@<>(),;:"\\]+\.[^\s@<>(),;:"\\]+$/;
const FIELDS = new Set(["kind", "apiKeyEnv", "inboxId", "region"]);

function validAddress(value: string): boolean {
	return ADDRESS.test(value);
}

export const agentmailKind: ChannelKind<AgentMailChannelConfig> = {
	kind: "agentmail",
	validate(raw, path) {
		for (const key of Object.keys(raw)) {
			if (!FIELDS.has(key)) throw new MessagingConfigError(`${path}.${key} is not a recognized field`);
		}
		const apiKeyEnv = assertEnvName(raw.apiKeyEnv, `${path}.apiKeyEnv`);
		if (typeof raw.inboxId !== "string" || !validAddress(raw.inboxId)) {
			throw new MessagingConfigError(
				`${path}.inboxId must be the inbox's email address, e.g. my-agent@agentmail.to`,
			);
		}
		const region = raw.region ?? "us";
		if (region !== "us" && region !== "eu") throw new MessagingConfigError(`${path}.region must be "us" or "eu"`);
		return { kind: "agentmail", apiKeyEnv, inboxId: raw.inboxId, region };
	},
	create(id, config, runtime): OutboundChannel {
		return {
			id,
			kind: "agentmail",
			addressing: "recipient",
			describe: () => `email from ${config.inboxId} (AgentMail); needs "to" and "subject"`,
			async send(message, signal) {
				const to = message.to ?? [];
				if (to.length === 0) throw new MessagingError("Email needs at least one recipient in `to`.");
				if (to.length > MAX_RECIPIENTS) {
					throw new MessagingError(
						`Email to ${to.length} recipients refused: at most ${MAX_RECIPIENTS} per message.`,
					);
				}
				const bad = to.find((address) => !validAddress(address));
				if (bad !== undefined)
					throw new MessagingError(`"${bad.replace(/[\r\n]+/g, " ")}" is not an email address.`);
				const subject = message.subject?.trim() ?? "";
				if (subject === "") throw new MessagingError("Email needs a subject.");
				if (/[\r\n]/.test(subject)) throw new MessagingError("An email subject must be one line.");
				const key = requireEnv(runtime.env, config.apiKeyEnv, "The AgentMail API key");
				const reply = await postJson(runtime.fetch, {
					label: "AgentMail",
					url: `${API_HOST[config.region]}/v0/inboxes/${encodeURIComponent(config.inboxId)}/messages/send`,
					headers: { authorization: `Bearer ${key}` },
					body: { to: [...to], subject, text: message.text },
					secrets: [key],
					...(signal === undefined ? {} : { signal }),
				});
				const messageId =
					typeof reply === "object" && reply !== null && "message_id" in reply
						? (reply as { message_id: unknown }).message_id
						: undefined;
				if (typeof messageId !== "string" || messageId === "") {
					throw new MessagingError("AgentMail answered without a message id, so the send cannot be confirmed.");
				}
				return { destination: to.join(", "), ids: [messageId] };
			},
		};
	},
};
