import type { ChannelKind } from "../registry.ts";
import { assertEnvName, postJson, requireEnv, splitText } from "../support.ts";
import { MessagingConfigError, MessagingError, type OutboundChannel } from "../types.ts";

/**
 * A Slack incoming webhook. The webhook URL is itself the credential (it names the workspace, the channel, and
 * the secret), so the config holds only the name of the environment variable that contains it.
 */
export interface SlackChannelConfig {
	readonly kind: "slack";
	readonly webhookEnv: string;
}

const SLACK_PART_LIMIT = 3000;
const FIELDS = new Set(["kind", "webhookEnv"]);
const WEBHOOK_HOSTS = new Set(["hooks.slack.com", "hooks.slack-gov.com"]);

/** Only a Slack webhook gets a POST: a stray or tampered variable must never turn this into a request to anywhere. */
function isSlackWebhook(value: string): boolean {
	let url: URL;
	try {
		url = new URL(value);
	} catch {
		return false;
	}
	return url.protocol === "https:" && WEBHOOK_HOSTS.has(url.hostname) && url.pathname.startsWith("/services/");
}

/** Slack's own escaping: `<!channel>`, `<!here>`, `<@U…>` and links become inert text, and Slack shows them as typed. */
function escapeSlack(text: string): string {
	return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export const slackKind: ChannelKind<SlackChannelConfig> = {
	kind: "slack",
	validate(raw, path) {
		for (const key of Object.keys(raw)) {
			if (!FIELDS.has(key)) throw new MessagingConfigError(`${path}.${key} is not a recognized field`);
		}
		return { kind: "slack", webhookEnv: assertEnvName(raw.webhookEnv, `${path}.webhookEnv`) };
	},
	create(id, config, runtime): OutboundChannel {
		return {
			id,
			kind: "slack",
			addressing: "fixed",
			describe: () => "Slack incoming webhook (the channel the webhook was created for)",
			async send(message, signal) {
				const subject = message.subject?.trim();
				const text = subject ? `${subject}\n\n${message.text}` : message.text;
				if (text.trim() === "") throw new MessagingError("The message is empty.");
				const url = requireEnv(runtime.env, config.webhookEnv, "The Slack incoming webhook URL");
				if (!isSlackWebhook(url)) {
					throw new MessagingError(
						`${config.webhookEnv} does not hold a Slack incoming webhook URL (https://hooks.slack.com/services/...); nothing was sent.`,
					);
				}
				const parts = splitText(escapeSlack(text), SLACK_PART_LIMIT);
				const ids: string[] = [];
				for (const [index, part] of parts.entries()) {
					try {
						const reply = await postJson(runtime.fetch, {
							label: "Slack",
							url,
							// Plain text only: no mrkdwn, no link previews, and `<!channel>` already escaped above.
							body: { text: part, mrkdwn: false, unfurl_links: false, unfurl_media: false },
							secrets: [url],
							...(signal === undefined ? {} : { signal }),
						});
						if (reply !== "ok") throw new MessagingError('Slack did not confirm the message (expected "ok").');
						ids.push("ok");
					} catch (error) {
						if (!(error instanceof MessagingError) || parts.length === 1) throw error;
						throw new MessagingError(
							`${error.message} Only ${index} of ${parts.length} parts were sent; do not resend the whole message.`,
						);
					}
				}
				return { destination: "Slack incoming webhook", ids };
			},
		};
	},
};
