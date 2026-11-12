import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { type Static, Type } from "typebox";
import { MessagingError, type OutboundChannel } from "./types.ts";

export const SEND_MESSAGE_TOOL_NAME = "send_message";

const sendMessageSchema = Type.Object({
	channel: Type.String({ description: "Id of a configured channel (listed in this tool's description)." }),
	text: Type.String({ description: "The message body, plain text. Long text is split by the channel as needed." }),
	to: Type.Optional(
		Type.Array(Type.String(), {
			description: "Recipient addresses. Only for channels that take recipients (email); never for the others.",
		}),
	),
	subject: Type.Optional(
		Type.String({ description: "Subject line. Only for channels that take recipients (email)." }),
	),
});

export type SendMessageInput = Static<typeof sendMessageSchema>;

export interface SendMessageDetails {
	readonly channel?: string;
	readonly destination?: string;
	readonly ids?: readonly string[];
}

function failure(message: string): AgentToolResult<SendMessageDetails> {
	return { content: [{ type: "text", text: message }], details: {}, isError: true };
}

/**
 * The model's way to send a message outside the machine: to a chat, a channel, or an email address the user
 * configured. Every call is approved by the user first (see the permission gate), and nothing is retried.
 */
export function createSendMessageTool(
	channels: ReadonlyMap<string, OutboundChannel>,
): AgentTool<typeof sendMessageSchema, SendMessageDetails> {
	const listing = [...channels.values()].map((channel) => `- ${channel.id}: ${channel.describe()}`).join("\n");
	return {
		name: SEND_MESSAGE_TOOL_NAME,
		label: "Send Message",
		description: `Send a message out of this machine, only when the user asks you to. The user approves each send before it goes. Chat channels deliver to a fixed destination the user configured, so pass only the channel and text. Email channels need "to" (and a "subject"). A send cannot be undone and is never retried: if it fails, tell the user instead of sending again. Never put secrets or private corpus text in a message the user did not ask to share.\nChannels:\n${listing}`,
		parameters: sendMessageSchema,
		async execute(_toolCallId, params, signal): Promise<AgentToolResult<SendMessageDetails>> {
			const channel = channels.get(params.channel);
			if (channel === undefined) {
				return failure(`No channel "${params.channel}". Configured channels: ${[...channels.keys()].join(", ")}.`);
			}
			if (params.text.trim() === "") return failure("The message is empty; nothing was sent.");
			const recipients = params.to ?? [];
			if (channel.addressing === "fixed" && recipients.length > 0) {
				return failure(
					`Channel "${channel.id}" delivers to a fixed destination the user configured; remove "to". Nothing was sent.`,
				);
			}
			if (channel.addressing === "recipient" && recipients.length === 0) {
				return failure(`Channel "${channel.id}" needs at least one address in "to". Nothing was sent.`);
			}
			try {
				const receipt = await channel.send(
					{
						text: params.text,
						...(params.subject === undefined ? {} : { subject: params.subject }),
						...(recipients.length === 0 ? {} : { to: recipients }),
					},
					signal,
				);
				return {
					content: [
						{
							type: "text",
							text: `Sent through ${channel.id} to ${receipt.destination}. Provider ids: ${receipt.ids.join(", ")}.`,
						},
					],
					details: { channel: channel.id, destination: receipt.destination, ids: receipt.ids },
				};
			} catch (error) {
				if (!(error instanceof MessagingError)) throw error;
				return failure(`The message was not delivered through ${channel.id}: ${error.message}`);
			}
		},
	};
}

/** The dialog body for a send: everything that will leave the machine, so the user approves what is really sent. */
export const SEND_SHOWN_TEXT_CHARS = 8000;

export function describeSend(input: Readonly<Record<string, unknown>>): string {
	const text = typeof input.text === "string" ? input.text : "";
	const shown =
		text.length > SEND_SHOWN_TEXT_CHARS
			? `${text.slice(0, SEND_SHOWN_TEXT_CHARS)}\n[... ${text.length - SEND_SHOWN_TEXT_CHARS} more characters are sent but not shown]`
			: text;
	const to = Array.isArray(input.to) && input.to.length > 0 ? input.to.map(String).join(", ") : undefined;
	return [
		`channel: ${String(input.channel)}`,
		...(to === undefined ? [] : [`to: ${to}`]),
		...(typeof input.subject === "string" ? [`subject: ${input.subject}`] : []),
		"message:",
		shown,
	].join("\n");
}
