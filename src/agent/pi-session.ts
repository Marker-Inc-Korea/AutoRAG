import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, ImageContent, Model } from "@earendil-works/pi-ai";
import { streamSimple as compatStreamSimple } from "@earendil-works/pi-ai/compat";
import {
	type AgentSession,
	type AgentSessionRuntime,
	type CreateAgentSessionRuntimeFactory,
	createAgentSession,
	createAgentSessionFromServices,
	createAgentSessionRuntime,
	createAgentSessionServices,
	DefaultResourceLoader,
	type ExtensionAPI,
	type ExtensionFactory,
	getAgentDir,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	type ToolDefinition,
	type TuiMode,
} from "@earendil-works/pi-coding-agent";
import { getKeybindings, type KeyId, matchesKey } from "@earendil-works/pi-tui";
import { PROVIDER_ID as DAZZI_PROVIDER_ID } from "../cloud/config.ts";
import { registerDazziProvider } from "../cloud/provider.ts";
import { createContextTokenGuardedTransform, withBoundedOutput } from "./context-token-guard.ts";
import { runtimeContextAddendum } from "./environment-prompt.ts";
import { registerIntentTracing } from "./intent-tracing.ts";
import { createAutoRAGMcpClient, mcpPromptAddendum } from "./mcp-client.ts";
import { fitRequestLimits } from "./request-limits.ts";
import { estimateTextTokens } from "./token-count.ts";
import { createWelcomeHeader } from "./welcome-header.ts";
export const PI_BUILTIN_TOOL_NAMES = ["read", "bash", "edit", "write", "grep", "find", "ls"] as const;

/**
 * Terminal UI mode for AutoRAG's interactive session. pi 1.0 switched its own default to
 * fullscreen; AutoRAG shipped on pi's scrollback UI, so that stays the default. A `tuiMode` the
 * user saved in pi's global or project settings still wins, and `InteractiveMode` accepts the
 * result directly (its constructor otherwise falls back to pi's fullscreen default).
 */
export function resolveInteractiveTuiMode(settingsManager: SettingsManager): TuiMode {
	const saved = settingsManager.getProjectSettings().tuiMode ?? settingsManager.getGlobalSettings().tuiMode;
	return saved ?? "regular";
}

export interface AutoRAGPiSessionOptions {
	readonly cwd: string;
	readonly agentDir?: string;
	readonly sessionDir?: string;
	readonly sessionPath?: string;
	readonly persistSession?: boolean;
	readonly model: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
	readonly getSystemPrompt: () => string;
	readonly customTools: readonly AgentTool[];
	/**
	 * Extra pi extensions to load alongside AutoRAG's own prompt/context
	 * extension. Each factory registers its tools, commands, and events with pi.
	 */
	readonly extensionFactories?: readonly ExtensionFactory[];
	/** Extension entry files (e.g. an installed pi package's extension) loaded into the interactive session. */
	readonly extensionPaths?: readonly string[];
	/** CLI-style flag values for the flags those extensions register (`--name` / `--name value`). */
	readonly extensionFlags?: Readonly<Record<string, string | boolean>>;
	/**
	 * Tool names registered by {@link extensionFactories}. pi's `tools`
	 * allow-list keeps only named tools active, so every extension tool that
	 * should be callable must appear here too.
	 */
	readonly extensionToolNames?: readonly string[];
	readonly contextTransform?: (messages: AgentMessage[]) => Promise<AgentMessage[]>;
}
/**
 * How AutoRAG takes one interactive input. `steer` while a pi turn streams
 * returns the text pi should queue as steering (pi delivers it before the
 * turn's next model request); every other outcome is fully handled by AutoRAG.
 */
export type AutoRAGInputDisposition = { readonly kind: "handled" } | { readonly kind: "steer"; readonly text: string };

export interface AutoRAGInteractiveInput {
	readonly text: string;
	readonly images: readonly ImageContent[] | undefined;
	/** pi's delivery for input typed while a turn streams; undefined when no turn streams. */
	readonly streamingBehavior: "steer" | "followUp" | undefined;
}

/** A TUI key AutoRAG acts on itself while a search runs, even when no pi turn streams. */
export type AutoRAGInteractiveKey = "interrupt" | "followUp";

export interface AutoRAGPiInteractiveRuntimeOptions extends Omit<AutoRAGPiSessionOptions, "model"> {
	readonly model?: Model<Api>;
	readonly inactiveToolNames?: readonly string[];
	/**
	 * Routes one interactive input: a new query, a steer of the running search,
	 * or a follow-up queued behind it. Must not wait for a search to finish:
	 * pi's TUI reads the next input only after this returns.
	 */
	readonly onInput: (input: AutoRAGInteractiveInput, pi: ExtensionAPI) => AutoRAGInputDisposition;
	/** Whether a search is running: while it is, every interactive input belongs to it. */
	readonly isSearchRunning?: () => boolean;
	/**
	 * Called for pi's interrupt (Esc) and follow-up (Alt+Enter) keys while a
	 * search runs. pi acts on them only while a turn streams, but a search also
	 * spends time between turns (routing, prefetch, checks). Return true to
	 * consume the key; pi then never sees it.
	 */
	readonly onKey?: (key: AutoRAGInteractiveKey) => boolean;
	/**
	 * Optional best-effort provider for a startup update notice. Resolves to the
	 * notice text, or `undefined` when there is nothing to announce. Called once
	 * per session; failures are swallowed so the check never blocks or fails a
	 * launch.
	 */
	readonly updateNotice?: () => Promise<string | undefined>;
	/**
	 * AutoRAG version shown on the welcome banner. Set only by the TUI: when present,
	 * every session start (launch, `/new`, `/resume`, `/reload`) replaces pi's startup
	 * header with the AutoRAG banner and hides pi's loaded-resource listing.
	 */
	readonly welcomeVersion?: string;
}

export interface AutoRAGPiSession {
	readonly session: AgentSession;
	readonly baseToolNames: readonly string[];
	readonly sessionFile: string | undefined;
	/** Stops what extensions started (MCP server processes), then disposes the session. */
	readonly dispose: () => Promise<void>;
}

/** Adapts a pi-agent-core {@link AgentTool} to pi's extension {@link ToolDefinition}. */
export function toToolDefinition(tool: AgentTool): ToolDefinition {
	return {
		name: tool.name,
		label: tool.label,
		description: tool.description,
		parameters: tool.parameters,
		...(tool.prepareArguments === undefined ? {} : { prepareArguments: tool.prepareArguments }),
		execute: async (toolCallId, params, signal, onUpdate) => {
			// ToolDefinition and AgentTool use the same TypeBox schema; this cast is
			// the adapter boundary between their structurally identical generics.
			return tool.execute(toolCallId, params as never, signal, onUpdate);
		},
	};
}

const EMBEDDED_AUTH_MARKER = "<autorag-embedded-model>";

function modelConfig(model: Model<Api>, configuredApiKey?: string): Parameters<ModelRuntime["registerProvider"]>[1] {
	return {
		baseUrl: model.baseUrl,
		api: model.api,
		...(configuredApiKey === undefined ? {} : { apiKey: configuredApiKey }),
		authHeader: false,
		streamSimple: (requestModel, context, options) =>
			compatStreamSimple(requestModel, context, {
				...options,
				...(configuredApiKey === EMBEDDED_AUTH_MARKER ? { apiKey: undefined } : {}),
			}),
		models: [
			{
				id: model.id,
				name: model.name,
				reasoning: model.reasoning,
				input: [...model.input],
				cost: model.cost,
				contextWindow: model.contextWindow,
				maxTokens: model.maxTokens,
				...(model.inputLimits === undefined ? {} : { inputLimits: model.inputLimits }),
				...(model.thinkingLevelMap === undefined ? {} : { thinkingLevelMap: model.thinkingLevelMap }),
			},
		],
	};
}

async function configureModelRuntime(
	modelRuntime: ModelRuntime,
	model: Model<Api> | undefined,
	apiKey: string | undefined,
	providerApiKeys: Readonly<Record<string, string>> | undefined,
): Promise<void> {
	if (model !== undefined) {
		const configuredKey = apiKey ?? providerApiKeys?.[model.provider];
		if (modelRuntime.getModel(model.provider, model.id) === undefined) {
			modelRuntime.registerProvider(model.provider, modelConfig(model, configuredKey ?? EMBEDDED_AUTH_MARKER));
		}
	}
	const keys = new Map<string, string>();
	if (model !== undefined && apiKey !== undefined) keys.set(model.provider, apiKey);
	for (const [provider, key] of Object.entries(providerApiKeys ?? {})) keys.set(provider, key);
	// Dazzi Models is OAuth-only: its token is the stored `/login` credential, and Pi resolves it from
	// auth.json. A runtime API key would replace that credential with an api_key one the provider has
	// no auth for, so every request then fails with "No API key found for dazzi".
	keys.delete(DAZZI_PROVIDER_ID);
	for (const [provider, key] of keys) await modelRuntime.setRuntimeApiKey(provider, key);
}

/**
 * Keeps every model request within the session model's declared input limits.
 * Registered after AutoRAG's own context transform, so it sizes the final
 * transcript the provider receives.
 */
function enforceRequestLimits(pi: ExtensionAPI): void {
	pi.on("context", (event, ctx) => {
		const messages = fitRequestLimits(event.messages, ctx.model?.inputLimits);
		return messages === undefined ? undefined : { messages };
	});
}

/**
 * Runs AutoRAG's context transform, then keeps the aggregate retrieval payload
 * within the active model's context window by dropping whole lower-ranked
 * candidates. Registered before {@link enforceRequestLimits}.
 */
function guardContext(
	pi: ExtensionAPI,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
): void {
	pi.on("context", async (event, ctx) => {
		// Not part of event.messages, yet sent with every request.
		const activeTools = new Set(pi.getActiveTools());
		const toolDeclarations = pi
			.getAllTools()
			.filter((tool) => activeTools.has(tool.name))
			.map(({ name, description, parameters }) => ({ name, description, parameters }));
		const fixedOverheadTokens =
			estimateTextTokens(ctx.getSystemPrompt()) + estimateTextTokens(JSON.stringify(toolDeclarations));
		return {
			messages: await createContextTokenGuardedTransform(
				ctx.model,
				contextTransform,
				() => fixedOverheadTokens,
			)(event.messages),
		};
	});
}

function createAutoRAGExtension(
	cwd: string,
	getSystemPrompt: () => string,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
): ExtensionFactory {
	return (pi) => {
		pi.on("before_agent_start", (event) => ({
			systemPrompt:
				getSystemPrompt() + mcpPromptAddendum(event.systemPromptOptions.sections) + runtimeContextAddendum(cwd),
		}));
		guardContext(pi, contextTransform);
		enforceRequestLimits(pi);
		registerIntentTracing(pi);
	};
}
/**
 * The AutoRAG key `data` presses: pi's interrupt or follow-up binding. pi's TUI
 * installs its keybindings (user remaps included) as pi-tui's global set; with
 * none installed, pi's defaults apply.
 */
function interactiveKeyOf(data: string): AutoRAGInteractiveKey | undefined {
	const bindings = getKeybindings();
	const keysOf = (action: "app.interrupt" | "app.message.followUp", fallback: KeyId): readonly KeyId[] => {
		const keys = bindings.getKeys(action);
		return keys.length > 0 ? keys : [fallback];
	};
	const windows =
		process.platform === "win32" ||
		(process.platform === "linux" && Boolean(process.env.WSL_DISTRO_NAME || process.env.WSL_INTEROP));
	if (keysOf("app.interrupt", "escape").some((key) => matchesKey(data, key))) return "interrupt";
	if (keysOf("app.message.followUp", windows ? "ctrl+q" : "alt+enter").some((key) => matchesKey(data, key)))
		return "followUp";
	return undefined;
}

/**
 * Prefix `@llblab/pi-telegram` puts on every prompt it injects from a Telegram
 * chat. The bridge hands them to pi through `pi.sendUserMessage`, so pi tags
 * them `source: "extension"`, the same source AutoRAG's own phase prompts use.
 * The prefix is what tells a person typing on a phone from AutoRAG talking to
 * itself.
 */
export const TELEGRAM_INPUT_PREFIX = "[telegram]";

/** `whatsapp-pi` frames every inbound chat message as `Message from <name> (<number>)`. */
export const WHATSAPP_INPUT_PREFIX = "Message from ";

/**
 * The user's own words from a bridged Telegram prompt: the bridge's `[telegram]`
 * prefix and its `[time]` line are envelope, not the question, so a Telegram
 * question reaches the librarian exactly as the same text typed in the TUI.
 * Attachment blocks stay: they are part of what the user sent.
 */
export function stripTelegramEnvelope(text: string): string {
	return text
		.replace(/^\s*\[telegram\]\s?/u, "")
		.replace(/\n*\[time\][^\n]*/gu, "")
		.trim();
}

/**
 * The text of a user input (terminal, or a chat bridge such as Telegram or WhatsApp), or `undefined`
 * for AutoRAG's own phase prompts (`source: "rpc"`) and pi's own extension prompts.
 */
function userInputText(event: { readonly source: string; readonly text: string }): string | undefined {
	if (event.source === "interactive") return event.text.trim();
	if (event.source === "extension" && event.text.trimStart().startsWith(TELEGRAM_INPUT_PREFIX)) {
		return stripTelegramEnvelope(event.text);
	}
	if (event.source === "extension" && event.text.trimStart().startsWith(WHATSAPP_INPUT_PREFIX)) {
		return event.text.trim();
	}
	return undefined;
}

function createAutoRAGInteractiveExtension(
	cwd: string,
	getSystemPrompt: () => string,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
	options: Pick<AutoRAGPiInteractiveRuntimeOptions, "onInput" | "isSearchRunning" | "onKey">,
	updateNotice: (() => Promise<string | undefined>) | undefined,
): ExtensionFactory {
	return (pi) => {
		pi.on("before_agent_start", (event) => ({
			systemPrompt:
				getSystemPrompt() + mcpPromptAddendum(event.systemPromptOptions.sections) + runtimeContextAddendum(cwd),
		}));
		guardContext(pi, contextTransform);
		enforceRequestLimits(pi);
		registerIntentTracing(pi);
		if (updateNotice !== undefined) {
			pi.on("session_start", () => {
				void updateNotice()
					.then((text) => {
						if (text === undefined) return;
						pi.sendMessage({
							customType: "autorag.update",
							content: [{ type: "text", text }],
							display: true,
						});
					})
					.catch(() => undefined);
			});
		}
		const { onInput, isSearchRunning, onKey } = options;
		if (onKey !== undefined) {
			pi.on("session_start", (_event, ctx) => {
				ctx.ui.onTerminalInput((data) => {
					const key = interactiveKeyOf(data);
					return key !== undefined && onKey(key) ? { consume: true } : undefined;
				});
			});
		}
		pi.on("input", (event) => {
			const text = userInputText(event);
			if (text === undefined || text.length === 0) return undefined;
			// A "/" line is pi's (commands, skills, templates), except during a search:
			// handed back to pi then, it would start pi's own prompt on the bound
			// session outside the search. Registered commands never reach this
			// handler; pi runs them first.
			if (text.startsWith("/") && isSearchRunning?.() !== true) return undefined;
			let disposition: AutoRAGInputDisposition;
			try {
				disposition = onInput({ text, images: event.images, streamingBehavior: event.streamingBehavior }, pi);
			} catch (error) {
				// Display-only: while a turn streams, pi's default delivery steers a
				// custom message into it as a user message, and the model would
				// answer this notice instead of the user.
				pi.sendMessage(
					{
						customType: "autorag.error",
						content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
						display: true,
					},
					{ triggerTurn: false },
				);
				return { action: "handled" };
			}
			return disposition.kind === "steer"
				? {
						action: "transform",
						text: disposition.text,
						...(event.images === undefined ? {} : { images: event.images }),
					}
				: { action: "handled" };
		});
	};
}

/** Creates a pi session with AutoRAG's tools and prompt. */
export async function createAutoRAGPiSession(options: AutoRAGPiSessionOptions): Promise<AutoRAGPiSession> {
	const agentDir = options.agentDir ?? getAgentDir();
	const settingsManager = SettingsManager.create(options.cwd, agentDir);
	if (settingsManager.getDefaultTools() === undefined) {
		settingsManager.applyOverrides({ defaultTools: [...PI_BUILTIN_TOOL_NAMES] });
	}
	const modelRuntime = await ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
	});
	await registerDazziProvider(modelRuntime);
	await configureModelRuntime(modelRuntime, options.model, options.apiKey, options.providerApiKeys);
	const customTools = [...options.customTools];
	const extensionToolNames = [...(options.extensionToolNames ?? [])];
	// pi's MCP extension fills the `mcp_servers` prompt section in its own before_agent_start
	// handler; AutoRAG's handler forces the prompt, so it has to run after that one.
	const mcp = createAutoRAGMcpClient(agentDir);
	const resourceLoader = new DefaultResourceLoader({
		cwd: options.cwd,
		agentDir,
		settingsManager,
		extensionFactories: [
			...(mcp?.extensionFactories ?? []),
			createAutoRAGExtension(options.cwd, options.getSystemPrompt, options.contextTransform),
			...(options.extensionFactories ?? []),
		],
		systemPromptOverride: () => options.getSystemPrompt(),
		appendSystemPromptOverride: () => [],
	});
	await resourceLoader.reload();
	const sessionManager = options.sessionPath
		? SessionManager.open(options.sessionPath, options.sessionDir, options.cwd)
		: options.persistSession
			? SessionManager.create(options.cwd, options.sessionDir)
			: SessionManager.inMemory(options.cwd);
	const customToolNames = customTools.map((tool) => tool.name);
	const customToolDefinitions = customTools.map(toToolDefinition);
	const { session } = await createAgentSession({
		cwd: options.cwd,
		agentDir,
		modelRuntime,
		settingsManager,
		resourceLoader,
		sessionManager,
		model: withBoundedOutput(options.model),
		customTools: customToolDefinitions,
		tools: [...PI_BUILTIN_TOOL_NAMES, ...customToolNames, ...extensionToolNames, ...(mcp?.toolNames ?? [])],
		thinkingLevel: "off",
	});
	const initialToolNames = [...new Set([...session.getActiveToolNames(), ...customToolNames, ...extensionToolNames])];
	session.setActiveToolsByName(initialToolNames);
	// pi connects MCP servers on `session_start`, which only a bind emits and no TUI hosts this
	// session. Binding is limited to sessions that have servers so every other headless search
	// keeps running without extension lifecycle events.
	if (mcp !== undefined) await session.bindExtensions({});
	return {
		session,
		baseToolNames: initialToolNames,
		sessionFile: session.sessionFile,
		dispose: async () => {
			try {
				if (mcp !== undefined) await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
			} finally {
				session.dispose();
			}
		},
	};
}

export interface AutoRAGPiInteractiveRuntime {
	readonly runtime: AgentSessionRuntime;
	readonly dispose: () => Promise<void>;
}

export async function createAutoRAGPiInteractiveRuntime(
	options: AutoRAGPiInteractiveRuntimeOptions,
): Promise<AutoRAGPiInteractiveRuntime> {
	const agentDir = options.agentDir ?? getAgentDir();
	const settingsManager = SettingsManager.create(options.cwd, agentDir);
	if (settingsManager.getDefaultTools() === undefined) {
		settingsManager.applyOverrides({ defaultTools: [...PI_BUILTIN_TOOL_NAMES] });
	}
	const modelRuntime = await ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
	});
	await registerDazziProvider(modelRuntime);
	await configureModelRuntime(modelRuntime, options.model, options.apiKey, options.providerApiKeys);
	const sessionManager = options.sessionPath
		? SessionManager.open(options.sessionPath, options.sessionDir, options.cwd)
		: SessionManager.create(options.cwd, options.sessionDir);
	const createRuntime: CreateAgentSessionRuntimeFactory = async ({
		cwd,
		agentDir: runtimeAgentDir,
		sessionManager: runtimeSessionManager,
		sessionStartEvent,
	}) => {
		// Built per runtime: /new, /resume, and fork each start a fresh session whose servers connect anew.
		const mcp = createAutoRAGMcpClient(runtimeAgentDir);
		const services = await createAgentSessionServices({
			cwd,
			agentDir: runtimeAgentDir,
			modelRuntime,
			settingsManager,
			...(options.extensionFlags === undefined
				? {}
				: { extensionFlagValues: new Map(Object.entries(options.extensionFlags)) }),
			resourceLoaderOptions: {
				...(options.extensionPaths === undefined ? {} : { additionalExtensionPaths: [...options.extensionPaths] }),
				extensionFactories: [
					...(mcp?.extensionFactories ?? []),
					...(options.welcomeVersion === undefined
						? []
						: [createWelcomeHeader({ version: options.welcomeVersion, settings: settingsManager })]),
					createAutoRAGInteractiveExtension(
						options.cwd,
						options.getSystemPrompt,
						options.contextTransform,
						{ onInput: options.onInput, isSearchRunning: options.isSearchRunning, onKey: options.onKey },
						options.updateNotice,
					),
					...(options.extensionFactories ?? []),
				],
				systemPrompt: options.getSystemPrompt(),
				appendSystemPrompt: [],
			},
		});
		const result = await createAgentSessionFromServices({
			services,
			sessionManager: runtimeSessionManager,
			sessionStartEvent,
			model: withBoundedOutput(options.model),
			customTools: options.customTools.map(toToolDefinition),
			tools: [
				...PI_BUILTIN_TOOL_NAMES,
				...options.customTools.map((tool) => tool.name),
				...(options.extensionToolNames ?? []),
				...(mcp?.toolNames ?? []),
			],
		});
		if (options.inactiveToolNames !== undefined && options.inactiveToolNames.length > 0) {
			const inactive = new Set(options.inactiveToolNames);
			result.session.setActiveToolsByName(result.session.getActiveToolNames().filter((name) => !inactive.has(name)));
		}
		return { ...result, services, diagnostics: services.diagnostics };
	};
	const runtime = await createAgentSessionRuntime(createRuntime, {
		cwd: options.cwd,
		agentDir,
		sessionManager,
	});
	return { runtime, dispose: () => runtime.dispose() };
}
