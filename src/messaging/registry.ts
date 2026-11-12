import { type ChannelRuntime, MessagingConfigError, MessagingError, type OutboundChannel } from "./types.ts";

/** A channel as it appears in `config.json`: a kind plus that kind's own fields. */
export interface ChannelConfig {
	readonly kind: string;
}

export interface MessagingConfig {
	readonly channels: Readonly<Record<string, ChannelConfig>>;
}

/** One way to send. The registry knows nothing about any kind; each kind validates and builds itself. */
export interface ChannelKind<C extends ChannelConfig = ChannelConfig> {
	readonly kind: C["kind"];
	/** Rejects unknown fields and secrets in the wrong place; returns the normalized config. */
	validate(raw: Record<string, unknown>, path: string): C;
	create(id: string, config: C, runtime: ChannelRuntime): OutboundChannel;
}

const kinds = new Map<string, ChannelKind>();

export function registerChannelKind<C extends ChannelConfig>(definition: ChannelKind<C>): void {
	if (kinds.has(definition.kind)) throw new Error(`Channel kind "${definition.kind}" is already registered`);
	kinds.set(definition.kind, definition as unknown as ChannelKind);
}

export function unregisterChannelKind(kind: string): void {
	kinds.delete(kind);
}

export function channelKinds(): string[] {
	return [...kinds.keys()];
}

const CHANNEL_ID = /^[A-Za-z0-9_-]{1,40}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Validates the `messaging` config section. Absent means no channels. */
export function normalizeMessagingConfig(raw: unknown): MessagingConfig {
	if (raw === undefined) return { channels: {} };
	if (!isRecord(raw)) throw new MessagingConfigError("Config field 'messaging' must be an object");
	for (const key of Object.keys(raw)) {
		if (key !== "channels") throw new MessagingConfigError(`messaging.${key} is not a recognized field`);
	}
	if (raw.channels === undefined) return { channels: {} };
	if (!isRecord(raw.channels)) throw new MessagingConfigError("messaging.channels must be an object");
	const channels: Record<string, ChannelConfig> = {};
	for (const [id, entry] of Object.entries(raw.channels)) {
		const path = `messaging.channels.${id}`;
		if (!CHANNEL_ID.test(id)) {
			throw new MessagingConfigError(`${path}: a channel id is 1-40 letters, digits, "_" or "-"`);
		}
		if (!isRecord(entry)) throw new MessagingConfigError(`${path} must be an object`);
		const kind = entry.kind;
		if (typeof kind !== "string")
			throw new MessagingConfigError(`${path}.kind must name a channel kind: ${channelKinds().join(", ")}`);
		const definition = kinds.get(kind);
		if (definition === undefined) {
			throw new MessagingConfigError(
				`${path}.kind "${kind}" is not a channel kind; use one of: ${channelKinds().join(", ")}`,
			);
		}
		channels[id] = definition.validate(entry, path);
	}
	return { channels };
}

/** Builds every configured channel, in config order. Secrets are read lazily, at send time. */
export function createChannels(config: MessagingConfig, runtime: ChannelRuntime): Map<string, OutboundChannel> {
	const out = new Map<string, OutboundChannel>();
	for (const [id, channelConfig] of Object.entries(config.channels)) {
		const definition = kinds.get(channelConfig.kind);
		if (definition === undefined) {
			throw new MessagingError(`Channel "${id}" uses kind "${channelConfig.kind}", which is not registered.`);
		}
		out.set(id, definition.create(id, channelConfig, runtime));
	}
	return out;
}
