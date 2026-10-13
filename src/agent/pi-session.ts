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
import { registerAutoRAGProvider } from "../cloud/provider.ts";
import { fitRequestLimits } from "./request-limits.ts";
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
	/**
	 * Tool names registered by {@link extensionFactories}. pi's `tools`
	 * allow-list keeps only named tools active, so every extension tool that
	 * should be callable must appear here too.
	 */
	readonly extensionToolNames?: readonly string[];
	readonly remoteSession?: boolean;
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
}

export interface AutoRAGPiSession {
	readonly session: AgentSession;
	readonly baseToolNames: readonly string[];
	readonly sessionFile: string | undefined;
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
		name: model.name,
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

function createAutoRAGExtension(
	getSystemPrompt: () => string,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
): ExtensionFactory {
	return (pi) => {
		pi.on("before_agent_start", () => ({ systemPrompt: getSystemPrompt() }));
		if (contextTransform !== undefined) {
			pi.on("context", async (event) => ({ messages: await contextTransform(event.messages) }));
		}
		enforceRequestLimits(pi);
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

function createAutoRAGInteractiveExtension(
	getSystemPrompt: () => string,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
	options: Pick<AutoRAGPiInteractiveRuntimeOptions, "onInput" | "isSearchRunning" | "onKey">,
	updateNotice: (() => Promise<string | undefined>) | undefined,
): ExtensionFactory {
	return (pi) => {
		pi.on("before_agent_start", () => ({ systemPrompt: getSystemPrompt() }));
		if (contextTransform !== undefined) {
			pi.on("context", async (event) => ({ messages: await contextTransform(event.messages) }));
		}
		enforceRequestLimits(pi);
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
			const text = event.text.trim();
			if (event.source !== "interactive" || text.length === 0) return undefined;
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
	await registerAutoRAGProvider(modelRuntime);
	await configureModelRuntime(modelRuntime, options.model, options.apiKey, options.providerApiKeys);
	const customTools = [...options.customTools];
	const extensionToolNames = [...(options.extensionToolNames ?? [])];
	const resourceLoader = new DefaultResourceLoader({
		cwd: options.cwd,
		agentDir,
		settingsManager,
		extensionFactories: [
			createAutoRAGExtension(options.getSystemPrompt, options.contextTransform),
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
		model: options.model,
		customTools: customToolDefinitions,
		tools: [...PI_BUILTIN_TOOL_NAMES, ...customToolNames, ...extensionToolNames],
		excludeTools: options.remoteSession ? ["edit", "write", "powershell"] : undefined,
		thinkingLevel: "off",
	});
	const initialToolNames = [...new Set([...session.getActiveToolNames(), ...customToolNames, ...extensionToolNames])];
	session.setActiveToolsByName(initialToolNames);
	return {
		session,
		baseToolNames: initialToolNames,
		sessionFile: session.sessionFile,
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
	await registerAutoRAGProvider(modelRuntime);
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
		const services = await createAgentSessionServices({
			cwd,
			agentDir: runtimeAgentDir,
			modelRuntime,
			settingsManager,
			resourceLoaderOptions: {
				extensionFactories: [
					createAutoRAGInteractiveExtension(
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
			customTools: options.customTools.map(toToolDefinition),
			tools: [
				...PI_BUILTIN_TOOL_NAMES,
				...options.customTools.map((tool) => tool.name),
				...(options.extensionToolNames ?? []),
			],
			excludeTools: options.remoteSession ? ["edit", "write", "powershell"] : undefined,
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
