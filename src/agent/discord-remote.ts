import type { ExtensionAPI, ExtensionCommandContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { REST, Routes } from "discord.js";
import installDiscordRemote from "pi-discord-remote";
import { loadConfig } from "pi-discord-remote/dist/config.js";
import { splitMessage } from "pi-discord-remote/dist/helpers.js";

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

type Handler = (event: unknown, ctx: unknown) => unknown;
type AgentEndHandler = Handler;

/** The extension tells the model its channel in the system prompt; that is the only place it exposes the id. */
const CHANNEL_ID_HINT = /Active Discord session channel ID: (\d+)/;
type CommandHandler = (args: string, ctx: ExtensionCommandContext) => Promise<void>;

/**
 * Hosts the community `pi-discord-remote` extension inside AutoRAG's session
 * and starts it on launch.
 *
 * Three behaviors differ from running it as a plain pi package:
 * - Discord messages enter through {@link DiscordRemoteHooks.submit}, so they run
 *   AutoRAG's two-phase search instead of a bare pi turn.
 * - The extension replies on `agent_end`, which AutoRAG's search fires once per
 *   phase. The reply is held until the whole search settles, so Discord gets the
 *   final answer rather than the first phase's.
 * - Earlier phases (the fast answer) are posted as soon as the next phase starts,
 *   so Discord sees the fast answer while verification runs. The extension cannot
 *   do this: it posts once per run and then forgets the channel.
 */
export function createDiscordRemoteExtension(hooks: DiscordRemoteHooks): ExtensionFactory {
	return (pi) => {
		let command: CommandHandler | undefined;
		let heldAgentEnd: (() => unknown) | undefined;
		let channelId: string | undefined;
		let discordRun = false;
		let phaseTexts: string[] = [];
		let rest: REST | undefined;
		const postEarlierPhase = async (text: string, notify: (message: string) => void): Promise<void> => {
			try {
				const config = await loadConfig();
				if (config === null || channelId === undefined) return;
				rest ??= new REST({ version: "10" }).setToken(config.token);
				for (const chunk of splitMessage(text)) {
					await rest.post(Routes.channelMessages(channelId), { body: { content: chunk } });
				}
			} catch (error) {
				notify(`Discord remote: could not post the fast answer: ${error instanceof Error ? error.message : error}`);
			}
		};
		const guarded = new Proxy(pi, {
			get(target, property) {
				if (property === "sendUserMessage") {
					return (content: unknown) => {
						discordRun = true;
						phaseTexts = [];
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
						if (event === "before_agent_start") {
							return on.call(target, event, async (hint: unknown, ctx: unknown) => {
								const result = await handler(hint, ctx);
								const match = CHANNEL_ID_HINT.exec(
									(result as { systemPrompt?: string } | undefined)?.systemPrompt ?? "",
								);
								if (match?.[1] !== undefined) channelId = match[1];
								return result;
							});
						}
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
									discordRun = false;
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
		pi.on("message_end", (event) => {
			const message = event.message as { role?: string; content?: { type: string; text?: string }[] };
			if (!discordRun || message.role !== "assistant") return;
			const text = (message.content ?? []).map((part) => (part.type === "text" ? (part.text ?? "") : "")).join("");
			if (text.trim() !== "") phaseTexts.push(text);
		});
		pi.on("agent_start", async (_event, ctx) => {
			// A new phase begins: the previous one's answer is final for Discord only once the run settles, so show it now.
			if (!discordRun || phaseTexts.length === 0) return;
			const text = phaseTexts.join("\n\n").trim();
			phaseTexts = [];
			await postEarlierPhase(text, (message) => ctx.ui.notify(message, "warning"));
		});
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
