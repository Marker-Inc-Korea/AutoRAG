/** Provider identity and endpoint configuration for the hosted AutoRAG (Dazzi) plan. */

export const PROVIDER_ID = "autorag";
export const PROVIDER_NAME = "AutoRAG";

/** Server root, e.g. `https://api.dazziapp.com`. The API lives under `<root>/v1`. */
export const DEFAULT_BASE_URL = "https://api.dazziapp.com";

export const AUTORAG_BASE_URL_ENV = "AUTORAG_BASE_URL";
export const AUTORAG_API_KEY_ENV = "AUTORAG_API_KEY";

/** Server paths appended to the normalized server root. */
export const API_PATH = "/v1";
export const MODELS_PATH = "/v1/models";
export const AUTHORIZE_PATH = "/desktop/authorize";
export const TOKEN_PATH = "/api/desktop/token";

/** OAuth client identifier for the AutoRAG agent (Pi-compatible hosts use `pi`). */
export const OAUTH_CLIENT_ID = "agent";

/**
 * Value used for `ProviderConfig.apiKey`. Pi resolves `$NAME` from the
 * environment, so `AUTORAG_API_KEY` enables a headless, login-free setup.
 */
export const API_KEY_CONFIG_VALUE = `$${AUTORAG_API_KEY_ENV}`;

/** Trim whitespace and trailing slashes so path concatenation is unambiguous. */
export function normalizeBaseUrl(value: string): string {
	return value.trim().replace(/\/+$/, "");
}

/** Server root from `AUTORAG_BASE_URL`, falling back to {@link DEFAULT_BASE_URL}. */
export function resolveBaseUrl(env: Record<string, string | undefined>): string {
	const configured = env[AUTORAG_BASE_URL_ENV];
	const value = typeof configured === "string" && configured.trim() !== "" ? configured : DEFAULT_BASE_URL;
	return normalizeBaseUrl(value);
}
