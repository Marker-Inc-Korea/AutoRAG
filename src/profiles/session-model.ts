import { join } from "node:path";
import { getModel } from "@earendil-works/pi-ai/compat";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { type CliConfig, resolveAgentModel, resolveConfigReadOnly } from "../cli/config.ts";
import { PROVIDER_ID as DAZZI_PROVIDER_ID } from "../cloud/config.ts";
import { registerDazziProvider } from "../cloud/provider.ts";
import type { ProfileModel } from "./agent-factory.ts";

/**
 * Why the hosted plan's key must not ride along as a session API key.
 *
 * The `dazzi` provider is OAuth-only: its credential is the `/login` entry in
 * Pi's `auth.json`. Handing the resolved key to `ModelRuntime.setRuntimeApiKey`
 * replaces that OAuth credential with an `api_key` one, and an OAuth-only
 * provider accepts no API key, so the session stops with "No API key found for
 * dazzi" even though the user is signed in. That only happens when the
 * session's runtime already lists the model (the persisted catalog snapshot the
 * sign-in writes); a model the runtime does not list is registered from the key
 * instead and works.
 *
 * So for a plan model the runtime lists, the key is dropped and Pi resolves the
 * stored OAuth credential itself. A model it does not list keeps the key.
 */

/** The slice of Pi's `ModelRuntime` this module reads. */
export interface PlanRuntime {
	getModel(provider: string, id: string): unknown;
	refresh(options: {
		readonly allowNetwork: boolean;
		readonly providers: readonly string[];
		readonly force: boolean;
	}): Promise<unknown>;
}

/** A resolved agent model with the credentials a session receives. */
export interface SessionModelAuth {
	readonly model: { readonly provider: string; readonly id: string };
	readonly apiKey?: string;
	readonly providerApiKeys?: Readonly<Record<string, string>>;
}

/** Open Pi's runtime on `agentDir` with the plan provider registered, offline until asked. */
export async function openPlanRuntime(agentDir: string, env: NodeJS.ProcessEnv): Promise<ModelRuntime> {
	const runtime = await ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
		allowModelNetwork: false,
	});
	await registerDazziProvider(runtime, env);
	return runtime;
}

async function planModelListed(runtime: PlanRuntime, model: SessionModelAuth["model"]): Promise<boolean> {
	if (runtime.getModel(model.provider, model.id) !== undefined) return true;
	// A fresh install has no catalog snapshot yet: load it once.
	await runtime.refresh({ allowNetwork: true, providers: [DAZZI_PROVIDER_ID], force: true });
	return runtime.getModel(model.provider, model.id) !== undefined;
}

/**
 * Drop the plan's API key from a resolved model when Pi can authenticate it
 * from the stored OAuth credential. Any other model, or a plan model Pi cannot
 * list (offline, unknown id), is returned unchanged.
 */
export async function preferPlanOAuth<T extends SessionModelAuth>(
	resolved: T,
	openRuntime: () => Promise<PlanRuntime>,
): Promise<T> {
	if (resolved.model.provider !== DAZZI_PROVIDER_ID) return resolved;
	const hasPlanKey = resolved.apiKey !== undefined || resolved.providerApiKeys?.[DAZZI_PROVIDER_ID] !== undefined;
	if (!hasPlanKey) return resolved;
	try {
		if (!(await planModelListed(await openRuntime(), resolved.model))) return resolved;
	} catch {
		return resolved;
	}
	const { apiKey: _apiKey, providerApiKeys, ...rest } = resolved;
	const { [DAZZI_PROVIDER_ID]: _planKey, ...otherKeys } = providerApiKeys ?? {};
	return { ...rest, ...(Object.keys(otherKeys).length > 0 ? { providerApiKeys: otherKeys } : {}) } as T;
}

/**
 * The Dazzi `/v1/models` catalog omits `thinkingLevelMap` for its OpenRouter-backed
 * model ids (e.g. `z-ai/glm-5.3-flash`, `anthropic/claude-haiku-5.5`). Without it,
 * Pi's `clampThinkingLevel(model, "off")` keeps `"off"` and `openai-responses`
 * sends `reasoning: { effort: "none" }`, which mandatory-reasoning endpoints
 * reject with HTTP 400 (`Reasoning is mandatory for this endpoint and cannot be
 * disabled`). Inherit the upstream OpenRouter `thinkingLevelMap` when present so
 * `"off"` clamps to `"low"` for those models.
 */
export function applyDazziThinkingLevels<
	T extends { readonly provider: string; readonly id: string; readonly thinkingLevelMap?: unknown },
>(model: T): T {
	if (model.provider !== DAZZI_PROVIDER_ID || model.thinkingLevelMap !== undefined) return model;
	const upstream = getModel("openrouter", model.id as never);
	if (upstream?.thinkingLevelMap === undefined) return model;
	return { ...model, thinkingLevelMap: upstream.thinkingLevelMap };
}

/**
 * The model every profile agent runs on: the one the shared AutoRAG config names
 * for the AutoRAG agent, with its credentials. AI Search and the profile agents
 * both resolve it, so there is one answer to "which model, which key".
 *
 * Credentials are read fresh on every call, so a login or key change reaches the
 * next session without a restart.
 */
export async function resolveSessionModel(
	config: CliConfig,
	env: NodeJS.ProcessEnv,
	cwd: string,
	agentDir: string,
): Promise<ProfileModel> {
	const resolved = await preferPlanOAuth(await resolveAgentModel(config, { env, cwd }), () =>
		openPlanRuntime(agentDir, env),
	);
	return { ...resolved, model: applyDazziThinkingLevels(resolved.model) };
}

/** {@link resolveSessionModel} for the config file the app and the CLI share. */
export function resolveConfiguredSessionModel(
	env: NodeJS.ProcessEnv,
	cwd: string,
	agentDir: string,
): Promise<ProfileModel> {
	return resolveSessionModel(resolveConfigReadOnly({ flags: {}, env, cwd }), env, cwd, agentDir);
}
