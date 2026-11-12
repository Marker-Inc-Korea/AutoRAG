import { registerBuiltinChannelKinds } from "./kinds.ts";
import { createChannels, type MessagingConfig, normalizeMessagingConfig as normalize } from "./registry.ts";
import type { ChannelRuntime, OutboundChannel } from "./types.ts";

export { registerBuiltinChannelKinds } from "./kinds.ts";
export {
	type ChannelConfig,
	type ChannelKind,
	channelKinds,
	type MessagingConfig,
	registerChannelKind,
	unregisterChannelKind,
} from "./registry.ts";
export { assertEnvName, postJson, requireEnv, scrubSecrets, splitText } from "./support.ts";
export {
	createSendMessageTool,
	describeSend,
	SEND_MESSAGE_TOOL_NAME,
	type SendMessageDetails,
} from "./tool.ts";
export {
	type ChannelRuntime,
	MessagingConfigError,
	MessagingError,
	type OutboundChannel,
	type OutboundMessage,
	type SendReceipt,
} from "./types.ts";

/** Validates the `messaging` config section with every built-in kind registered. */
export function normalizeMessagingConfig(raw: unknown): MessagingConfig {
	registerBuiltinChannelKinds();
	return normalize(raw);
}

/** Builds the configured channels with every built-in kind registered. */
export function buildChannels(config: MessagingConfig, runtime: ChannelRuntime): Map<string, OutboundChannel> {
	registerBuiltinChannelKinds();
	return createChannels(config, runtime);
}
