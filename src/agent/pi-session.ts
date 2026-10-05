import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, Model } from "@earendil-works/pi-ai";
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
} from "@earendil-works/pi-coding-agent";
export const PI_BUILTIN_TOOL_NAMES = ["read", "bash", "edit", "write", "grep", "find", "ls"] as const;

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
export interface AutoRAGPiInteractiveRuntimeOptions extends Omit<AutoRAGPiSessionOptions, "model"> {
	readonly model?: Model<Api>;
	readonly inactiveToolNames?: readonly string[];
	readonly onQuery: (query: string, pi: ExtensionAPI) => void | Promise<void>;
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

function createAutoRAGExtension(
	getSystemPrompt: () => string,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
): ExtensionFactory {
	return (pi) => {
		pi.on("before_agent_start", () => ({ systemPrompt: getSystemPrompt() }));
		if (contextTransform !== undefined) {
			pi.on("context", async (event) => ({ messages: await contextTransform(event.messages) }));
		}
	};
}
function createAutoRAGInteractiveExtension(
	getSystemPrompt: () => string,
	contextTransform: ((messages: AgentMessage[]) => Promise<AgentMessage[]>) | undefined,
	onQuery: (query: string, pi: ExtensionAPI) => void | Promise<void>,
	updateNotice: (() => Promise<string | undefined>) | undefined,
): ExtensionFactory {
	return (pi) => {
		pi.on("before_agent_start", () => ({ systemPrompt: getSystemPrompt() }));
		if (contextTransform !== undefined) {
			pi.on("context", async (event) => ({ messages: await contextTransform(event.messages) }));
		}
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
		pi.on("input", async (event) => {
			const query = event.text.trim();
			if (event.source !== "interactive" || query.length === 0 || query.startsWith("/")) return undefined;
			try {
				await onQuery(query, pi);
			} catch (error) {
				pi.sendMessage({
					customType: "autorag.error",
					content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
					display: true,
				});
			}
			return { action: "handled" };
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
						options.onQuery,
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
