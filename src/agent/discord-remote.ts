import type { ExtensionAPI, ExtensionCommandContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import installDiscordRemote from "pi-discord-remote";

/** Tools the `pi-discord-remote` extension registers; pi's allow-list must name them to keep them active. */
export const DISCORD_REMOTE_TOOL_NAMES = ["discord_ask_user_question", "discord_send_image"] as const;

const DISCORD_COMMAND = "pi-discord-remote";

/** How the AutoRAG agent plugs Discord messages into its own search loop. */
export interface DiscordRemoteHooks {
	/**
	 * Runs one Discord user message as AutoRAG input: a new question when idle,
	 * a steer or follow-up while a search runs, exactly like a typed line.
	 */
	readonly submit: (text: string, pi: ExtensionAPI) => void;
	/** Registers a one-shot callback for when the running search (and its queued follow-ups) has settled. */
	readonly onRunSettled: (callback: () => void) => void;
	/** Whether a search is running. */
	readonly isRunning: () => boolean;
}

type AgentEndHandler = (event: unknown, ctx: unknown) => unknown;
type CommandHandler = (args: string, ctx: ExtensionCommandContext) => Promise<void>;

/**
 * Hosts the community `pi-discord-remote` extension inside AutoRAG's session
 * and starts it on launch.
 *
 * Two behaviors differ from running it as a plain pi package:
 * - Discord messages enter through {@link DiscordRemoteHooks.submit}, so they run
 *   AutoRAG's two-phase search instead of a bare pi turn.
 * - The extension replies on `agent_end`, which AutoRAG's search fires once per
 *   phase. The reply is held until the whole search settles, so Discord gets the
 *   final answer rather than the first phase's.
 */
export function createDiscordRemoteExtension(hooks: DiscordRemoteHooks): ExtensionFactory {
	return (pi) => {
		let command: CommandHandler | undefined;
		let heldAgentEnd: (() => unknown) | undefined;
		const guarded = new Proxy(pi, {
			get(target, property) {
				if (property === "sendUserMessage") {
					return (content: unknown) => {
						hooks.submit(typeof content === "string" ? content : textOfContent(content), target);
					};
				}
				if (property === "registerCommand") {
					return (name: string, options: { handler: CommandHandler }) => {
						if (name === DISCORD_COMMAND) command = options.handler;
						return target.registerCommand(name, options as never);
					};
				}
				if (property === "on") {
					return (event: string, handler: AgentEndHandler) => {
						const on = target.on as unknown as (e: string, h: AgentEndHandler) => unknown;
						if (event !== "agent_end") return on.call(target, event, handler);
						return on.call(target, "agent_end", (agentEnd: unknown, ctx: unknown) => {
							if (!hooks.isRunning()) return handler(agentEnd, ctx);
							// Keep only the latest phase's end: it carries the final answer.
							const first = heldAgentEnd === undefined;
							heldAgentEnd = () => handler(agentEnd, ctx);
							if (first) {
								hooks.onRunSettled(() => {
									const run = heldAgentEnd;
									heldAgentEnd = undefined;
									void run?.();
								});
							}
							return undefined;
						});
					};
				}
				const value = Reflect.get(target, property);
				return typeof value === "function" ? value.bind(target) : value;
			},
		});
		installDiscordRemote(guarded);
		pi.on("session_start", async (event, ctx) => {
			if (event.reason === "reload" || command === undefined) return;
			await command("start", ctx as unknown as ExtensionCommandContext);
		});
	};
}

function textOfContent(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.map((part: { type?: string; text?: string }) => (part.type === "text" ? (part.text ?? "") : ""))
		.join("\n");
}
