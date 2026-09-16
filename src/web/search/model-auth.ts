/**
 * Model-native search credential resolution.
 *
 * Replaces oh-my-pi's AuthStorage broker for the model-native providers
 * (gemini, anthropic, codex, xai): AutoRAG already resolves model credentials
 * for the agent loop, so web search reuses them instead of asking the user
 * for a second key. Two sources, in priority order:
 *
 * 1. The injected agent model credential — `AutoRAGAgent` installs the
 *    resolved model's provider/apiKey via {@link setModelNativeSearchAuth}
 *    whenever it resolves a session model.
 * 2. The provider's conventional model environment key
 *    (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY,
 *    XAI_API_KEY) — the same variables AutoRAG's model resolution reads.
 *
 * No new signup, no search-specific key: if the user can run the agent with
 * a model from one of these providers, web search rides on that credential.
 */
import { envCredential } from "./credentials.ts";

/** Agent model credential injected for model-native web search. */
export interface ModelNativeSearchAuth {
	/** pi model provider id, e.g. "anthropic" | "openai" | "google" | "xai". */
	readonly provider: string;
	readonly apiKey: string;
	readonly baseUrl?: string;
	readonly modelId?: string;
}

/** Credential view consumed by the model-native search providers. */
export interface ModelNativeCredential {
	readonly apiKey: string;
	readonly baseUrl?: string;
	readonly modelId?: string;
	readonly source: "injected" | "env";
}

/** Model-native search provider ids as they appear in the chain. */
export type ModelNativeSearchProviderId = "gemini" | "anthropic" | "codex" | "xai";

/** pi model provider ids that map onto each model-native search provider. */
const MODEL_PROVIDER_ALIASES: Record<ModelNativeSearchProviderId, readonly string[]> = {
	gemini: ["google", "gemini", "google-gemini-cli"],
	anthropic: ["anthropic"],
	codex: ["openai", "openai-codex"],
	xai: ["xai", "x-ai"],
};

/** Conventional model environment keys per search provider (AutoRAG's own `${PROVIDER}_API_KEY` convention). */
const MODEL_ENV_KEYS: Record<ModelNativeSearchProviderId, readonly string[]> = {
	gemini: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
	anthropic: ["ANTHROPIC_API_KEY"],
	codex: ["OPENAI_API_KEY"],
	xai: ["XAI_API_KEY"],
};

let injectedAuth: ModelNativeSearchAuth | undefined;

/** Install (or clear, with `undefined`) the agent model credential for model-native web search. */
export function setModelNativeSearchAuth(auth: ModelNativeSearchAuth | undefined): void {
	injectedAuth = auth;
}

/** Return the currently injected agent model credential, if any. */
export function getModelNativeSearchAuth(): ModelNativeSearchAuth | undefined {
	return injectedAuth;
}

/**
 * Resolve the credential a model-native search provider should use right now:
 * the injected agent credential when its provider family matches, else the
 * provider's conventional model environment key.
 */
export function resolveModelNativeCredential(
	searchProvider: ModelNativeSearchProviderId,
): ModelNativeCredential | undefined {
	if (injectedAuth && MODEL_PROVIDER_ALIASES[searchProvider].includes(injectedAuth.provider)) {
		return {
			apiKey: injectedAuth.apiKey,
			...(injectedAuth.baseUrl !== undefined ? { baseUrl: injectedAuth.baseUrl } : {}),
			...(injectedAuth.modelId !== undefined ? { modelId: injectedAuth.modelId } : {}),
			source: "injected",
		};
	}
	const envKey = envCredential(...MODEL_ENV_KEYS[searchProvider]);
	if (envKey) return { apiKey: envKey, source: "env" };
	return undefined;
}

/**
 * Map a resolved agent model (provider + apiKey) onto the injectable auth
 * shape. Returns undefined for providers with no model-native search route
 * (e.g. openrouter, local gateways) or when no apiKey was resolved.
 */
export function modelNativeAuthFromAgentModel(resolved: {
	readonly provider: string;
	readonly apiKey?: string;
	readonly baseUrl?: string;
	readonly modelId?: string;
}): ModelNativeSearchAuth | undefined {
	if (!resolved.apiKey) return undefined;
	const isNative = Object.values(MODEL_PROVIDER_ALIASES).some((aliases) => aliases.includes(resolved.provider));
	if (!isNative) return undefined;
	return {
		provider: resolved.provider,
		apiKey: resolved.apiKey,
		...(resolved.baseUrl !== undefined ? { baseUrl: resolved.baseUrl } : {}),
		...(resolved.modelId !== undefined ? { modelId: resolved.modelId } : {}),
	};
}
