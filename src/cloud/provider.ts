/**
 * The hosted AutoRAG (Dazzi) plan as a first-class model provider named
 * `autorag`.
 *
 * The server exposes an OpenAI Responses API surface under `<root>/v1`. Login
 * is OAuth2 authorization-code + PKCE over a loopback redirect (or the headless
 * `AUTORAG_API_KEY`), and the model catalog is discovered live from
 * `GET <root>/v1/models` — no model id is ever hardcoded here. Streaming is
 * delegated to Pi's built-in `openai-responses` API implementation.
 */

import type { RefreshModelsContext } from "@earendil-works/pi-ai";
import type { ModelRuntime, ProviderConfig, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { API_KEY_CONFIG_VALUE, API_PATH, PROVIDER_ID, PROVIDER_NAME, resolveBaseUrl } from "./config.ts";
import { buildStoreEntry, fetchModels, modelsFromStore, resolveCredentialKey } from "./models.ts";
import { loginAutorag, refreshAutoragCredentials } from "./oauth-login.ts";

/** Injection seam for tests and hosts that resolve configuration themselves. */
export type AutoragEnv = Record<string, string | undefined>;

interface Endpoints {
	/** Server root, used for OAuth endpoints. */
	baseUrl: string;
	/** API base (`<root>/v1`), used as the provider `baseUrl` and for model listing. */
	apiBaseUrl: string;
}

/**
 * Resolve the provider's model list.
 *
 * Called by Pi with the credential it resolved for this provider. Without a
 * credential (or without network access) the last persisted catalog, if any,
 * is restored; a failed fetch keeps the previous list instead of blanking it.
 */
export async function refreshAutoragModels(
	context: RefreshModelsContext,
	endpoints: Endpoints,
): Promise<ProviderModelConfig[]> {
	const stored = modelsFromStore(context.stored);
	const apiKey = resolveCredentialKey(context.credential);
	if (!context.allowNetwork || apiKey === undefined) return stored;

	let models: ProviderModelConfig[];
	try {
		models = await fetchModels({ baseUrl: endpoints.baseUrl, apiKey, signal: context.signal });
	} catch {
		// Keep the last known catalog on transient failures and revoked tokens alike.
		return stored;
	}

	if (models.length === 0) return stored;
	try {
		await context.publish({ persist: buildStoreEntry(models, PROVIDER_ID, endpoints.apiBaseUrl) });
	} catch {
		// Persistence is best-effort; the live list below still applies.
	}
	return models;
}

/** Build the `ProviderConfig` handed to `ModelRuntime.registerProvider("autorag", ...)`. */
export function createAutoRAGProviderConfig(env: AutoragEnv = process.env): ProviderConfig {
	const baseUrl = resolveBaseUrl(env);
	const endpoints: Endpoints = { baseUrl, apiBaseUrl: `${baseUrl}${API_PATH}` };
	return {
		name: PROVIDER_NAME,
		baseUrl: endpoints.apiBaseUrl,
		api: "openai-responses",
		apiKey: API_KEY_CONFIG_VALUE,
		oauth: {
			name: PROVIDER_NAME,
			login: (callbacks) => loginAutorag(callbacks, { baseUrl }),
			refreshToken: async (credentials) => refreshAutoragCredentials(credentials),
			getApiKey: (credentials) => credentials.access,
		},
		refreshModels: (context) => refreshAutoragModels(context, endpoints),
	};
}

/**
 * The single registration point: register `autorag` on a Pi `ModelRuntime`.
 *
 * Runs even when the user is not signed in, so `/login` lists AutoRAG. The
 * awaited cache-only refresh loads the persisted catalog snapshot (offline and
 * with `allowModelNetwork:false`); it makes no network call, so a signed-out
 * provider contributes no models and nothing here reaches the network.
 */
export async function registerAutoRAGProvider(runtime: ModelRuntime, env: AutoragEnv = process.env): Promise<void> {
	runtime.registerProvider(PROVIDER_ID, createAutoRAGProviderConfig(env));
	await runtime.refresh({ allowNetwork: false, providers: [PROVIDER_ID] });
}
