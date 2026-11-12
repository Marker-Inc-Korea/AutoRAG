/** What the model hands a channel. `to` and `subject` apply only to channels that address a recipient. */
export interface OutboundMessage {
	readonly text: string;
	readonly subject?: string;
	readonly to?: readonly string[];
}

/** Proof a channel accepted the message: where it went and the provider's own ids for it. */
export interface SendReceipt {
	readonly destination: string;
	readonly ids: readonly string[];
}

/**
 * One configured way out. `fixed` channels deliver to a destination written in the user's config (a chat, a
 * channel, a webhook), so the model picks only the channel. `recipient` channels (email) take the destination
 * from the call, which is why every send is approved by the user.
 */
export interface OutboundChannel {
	readonly id: string;
	readonly kind: string;
	readonly addressing: "fixed" | "recipient";
	/** One line for the model and the approval dialog: kind and destination, never a secret. */
	describe(): string;
	send(message: OutboundMessage, signal?: AbortSignal): Promise<SendReceipt>;
}

/** A send failed; the message is safe to show the model and the user (secrets scrubbed). */
export class MessagingError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "MessagingError";
	}
}

/** The `messaging` config section is invalid. */
export class MessagingConfigError extends MessagingError {
	constructor(message: string) {
		super(message);
		this.name = "MessagingConfigError";
	}
}

/** What a channel needs from the host: the environment its secrets are named in, and the HTTP client. */
export interface ChannelRuntime {
	readonly env: Readonly<Record<string, string | undefined>>;
	readonly fetch: typeof fetch;
}
