/** Provider identity and endpoint configuration for the hosted Dazzi Models plan. */

import type { ModelInputLimits } from "@earendil-works/pi-ai";

export const PROVIDER_ID = "dazzi";
export const PROVIDER_NAME = "Dazzi Models";

/**
 * Chat model used when nothing is configured and the user is signed in. Used only
 * if the live or persisted catalog lists it; otherwise the first catalog model.
 */
export const DEFAULT_DAZZI_CHAT_MODEL_ID = "z-ai/glm-5.3-flash";

/**
 * Question-decomposition model used when nothing is configured and the user is
 * signed in. `z-ai/glm-5.3-flash` is not usable here: it requires reasoning, and
 * the decomposition call runs with reasoning off. Same catalog caveat as
 * {@link DEFAULT_DAZZI_CHAT_MODEL_ID}: the model must be in the signed-in catalog.
 */
export const DEFAULT_DAZZI_QUERY_DECOMPOSITION_MODEL_ID = "openai/gpt-6-luna";

/** Server root, e.g. `https://api.dazziapp.com`. The API lives under `<root>/v1`. */
export const DEFAULT_BASE_URL = "https://api.dazziapp.com";

export const AUTORAG_BASE_URL_ENV = "AUTORAG_BASE_URL";

/** Server paths appended to the normalized server root. */
export const API_PATH = "/v1";
export const MODELS_PATH = "/v1/models";
export const AUTHORIZE_PATH = "/desktop/authorize";
export const TOKEN_PATH = "/api/desktop/token";

/**
 * Request limits of the hosted API, declared on every `autorag` model.
 *
 * The API is served from Vercel, which rejects a request body above about
 * 4.49 MB with HTTP 413 before it reaches the model (measured against
 * `POST /v1/responses`). pi resends every image a tool returned on each later
 * turn, so a search that reads a few photos crosses that line. The transcript
 * cap leaves headroom for the tool declarations and request framing the
 * transcript size does not count; the resize cap keeps one image well below it.
 */
export const DAZZI_INPUT_LIMITS = {
	maxRequestBytes: 3_500_000,
	images: { resize: { maxBytes: 1_000_000 } },
} as const satisfies ModelInputLimits;

/** OAuth client identifier for the AutoRAG agent (Pi-compatible hosts use `pi`). */
export const OAUTH_CLIENT_ID = "agent";

/** Trim whitespace and trailing slashes so path concatenation is unambiguous. */
export function normalizeBaseUrl(value: string): string {
	const trimmed = value.trim();
	let end = trimmed.length;
	while (end > 0 && trimmed.charCodeAt(end - 1) === 0x2f /* "/" */) end--;
	return trimmed.slice(0, end);
}

/** Server root from `AUTORAG_BASE_URL`, falling back to {@link DEFAULT_BASE_URL}. */
export function resolveBaseUrl(env: Record<string, string | undefined>): string {
	const configured = env[AUTORAG_BASE_URL_ENV];
	const value = typeof configured === "string" && configured.trim() !== "" ? configured : DEFAULT_BASE_URL;
	return normalizeBaseUrl(value);
}
