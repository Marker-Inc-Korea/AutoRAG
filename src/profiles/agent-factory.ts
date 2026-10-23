import type { Api, Model } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import {
	type AgentSession,
	type CreateAgentSessionOptions,
	createAgentSessionFromServices,
	createAgentSessionServices,
	type ModelRuntime,
	SessionManager,
	SettingsManager,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { openPlanRuntime } from "./session-model.ts";

/**
 * Plain pi agent sessions for helper features, one per {@link AgentProfile}.
 *
 * The AutoRAG agent (`AutoRAGAgent`) answers searches and is not touched here.
 * Every other helper agent (smart indexing in the desktop app, file-triggered
 * automations, and whatever follows) is a vanilla pi session: its own system
 * prompt, its own tool set, and its own choice of what pi would otherwise load
 * on its own (AGENTS.md, skills, extensions). What the profiles share is the
 * account: the model the AutoRAG agent runs on, and the credential store under
 * the same agent dir (`auth.json`, `models.json`), so a sign-in or key change
 * reaches them without a second login.
 *
 * A session lives in memory and is gone with `dispose()`: nothing is written
 * under the agent dir or the working directory, and the user's own pi settings
 * (default tools, themes, ...) never apply. A profile session never reads or
 * writes AutoRAG's retrieval memory.
 */

/** How hard the model thinks; the same levels pi's session takes. */
export type ProfileThinking = NonNullable<CreateAgentSessionOptions["thinkingLevel"]>;

/** What pi loads on its own besides the profile's prompt and tools. Everything is off unless a profile turns it on. */
export interface ProfileContext {
	/** AGENTS.md style instruction files from the agent dir and the working directory. */
	readonly projectFiles?: boolean;
	readonly skills?: boolean;
	/** Extensions from the agent dir and the working directory. An extension's tools still have to be named in {@link AgentProfile.tools}. */
	readonly extensions?: boolean;
}

export interface AgentProfile {
	readonly id: string;
	readonly systemPrompt: string;
	/**
	 * Names of the pi tools the model may call (`read`, `ls`, `find`, `grep`, and
	 * `bash`/`edit`/`write` for a profile that may change things; an extension's
	 * tool when {@link ProfileContext.extensions} is on). Tools in
	 * {@link customTools} are always allowed and need no entry here.
	 */
	readonly tools: readonly string[];
	readonly customTools?: readonly ToolDefinition[];
	readonly thinking: ProfileThinking;
	readonly context?: ProfileContext;
}

/** The model a session runs on and the credentials resolved for it; see `resolveSessionModel`. */
export interface ProfileModel {
	readonly model: Model<Api>;
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
}

export interface ProfileSessionOptions {
	readonly profile: AgentProfile;
	/** Working directory of the session: relative paths and project context resolve against it. */
	readonly cwd: string;
	/** pi's agent dir; `auth.json` and `models.json` in it are the shared credential store. */
	readonly agentDir: string;
	readonly env: NodeJS.ProcessEnv;
	readonly model: ProfileModel;
}

/** Stands in for a key when a model has none: pi insists on one to register a provider, the request goes out without it. */
const EMBEDDED_AUTH_MARKER = "<autorag-embedded-model>";

/**
 * A model the runtime's catalog does not list (a custom endpoint, a model id newer
 * than the persisted catalog) is registered from the model itself, the way the
 * AutoRAG agent registers the model of its own session.
 */
function registerUnlistedModel(runtime: ModelRuntime, { model, apiKey, providerApiKeys }: ProfileModel): void {
	if (runtime.getModel(model.provider, model.id) !== undefined) return;
	const configuredKey = apiKey ?? providerApiKeys?.[model.provider];
	runtime.registerProvider(model.provider, {
		baseUrl: model.baseUrl,
		api: model.api,
		apiKey: configuredKey ?? EMBEDDED_AUTH_MARKER,
		authHeader: false,
		streamSimple: (requestModel, context, options) => {
			if (configuredKey !== undefined) return streamSimple(requestModel, context, options);
			// The marker only satisfied registration; the request goes out without a key.
			const { apiKey: _marker, ...withoutKey } = options ?? {};
			return streamSimple(requestModel, context, withoutKey);
		},
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
	});
}

/** Keys the caller resolved are for this session only; a signed-in plan has none and authenticates from its stored credential. */
async function applyRuntimeKeys(
	runtime: ModelRuntime,
	{ model, apiKey, providerApiKeys }: ProfileModel,
): Promise<void> {
	const keys = new Map<string, string>();
	if (apiKey !== undefined) keys.set(model.provider, apiKey);
	for (const [provider, key] of Object.entries(providerApiKeys ?? {})) keys.set(provider, key);
	for (const [provider, key] of keys) await runtime.setRuntimeApiKey(provider, key);
}

/** Builds a ready session for `profile`; the caller prompts it and disposes it. */
export async function createProfileSession(options: ProfileSessionOptions): Promise<AgentSession> {
	const { profile, cwd, agentDir, env, model } = options;
	const runtime = await openPlanRuntime(agentDir, env);
	registerUnlistedModel(runtime, model);
	await applyRuntimeKeys(runtime, model);
	const context = profile.context ?? {};
	const services = await createAgentSessionServices({
		cwd,
		agentDir,
		modelRuntime: runtime,
		settingsManager: SettingsManager.inMemory(),
		resourceLoaderOptions: {
			noContextFiles: context.projectFiles !== true,
			noSkills: context.skills !== true,
			noExtensions: context.extensions !== true,
			noPromptTemplates: true,
			noThemes: true,
			systemPromptOverride: () => profile.systemPrompt,
			appendSystemPromptOverride: () => [],
		},
	});
	const customTools = [...(profile.customTools ?? [])];
	const { session } = await createAgentSessionFromServices({
		services,
		sessionManager: SessionManager.inMemory(cwd),
		model: model.model,
		thinkingLevel: profile.thinking,
		tools: [...profile.tools, ...customTools.map((tool) => tool.name)],
		customTools,
	});
	return session;
}
