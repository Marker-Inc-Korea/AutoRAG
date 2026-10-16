import type { AnyModel, Credential, ModelCost, ModelsStoreEntry } from "@earendil-works/pi-ai";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { DAZZI_INPUT_LIMITS, MODELS_PATH, normalizeBaseUrl } from "./config.ts";
import { DazziHttpError, DazziOAuthError, extractErrorPayload } from "./errors.ts";

/** Used when the server omits `context_window` / `max_output_tokens`. */
export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export interface FetchModelsOptions {
	/** Server root, e.g. `https://api.dazziapp.com`. */
	baseUrl: string;
	/** Bearer token from the stored `/login` credential. */
	apiKey: string;
	signal?: AbortSignal;
	fetchImpl?: typeof fetch;
}

/** Pick the bearer token out of a stored Pi credential. */
export function resolveCredentialKey(credential: Credential | undefined): string | undefined {
	if (credential === undefined) return undefined;
	const value = credential.type === "oauth" ? credential.access : credential.key;
	return typeof value === "string" && value !== "" ? value : undefined;
}

/**
 * `GET <root>/v1/models` -> Pi chat model definitions.
 *
 * Pricing is USD per million tokens, matching Pi's cost model directly.
 */
export async function fetchModels(options: FetchModelsOptions): Promise<ProviderModelConfig[]> {
	const doFetch = options.fetchImpl ?? fetch;
	let response: Response;
	try {
		response = await doFetch(`${normalizeBaseUrl(options.baseUrl)}${MODELS_PATH}`, {
			headers: { authorization: `Bearer ${options.apiKey}`, accept: "application/json" },
			signal: options.signal,
		});
	} catch (cause) {
		if (options.signal?.aborted) {
			throw new DazziOAuthError("Dazzi Models model refresh was cancelled.", { cause });
		}
		throw new DazziOAuthError(`Dazzi Models model listing could not reach ${normalizeBaseUrl(options.baseUrl)}.`, {
			cause,
		});
	}

	let body: unknown;
	try {
		body = await response.json();
	} catch {
		body = undefined;
	}

	if (!response.ok) throw mapModelsError(response.status, body);
	return mapServerModels(body);
}

/** Map a failed `/v1/models` response; 401 means the token is invalid or revoked. */
export function mapModelsError(status: number, body: unknown): DazziHttpError {
	const payload = extractErrorPayload(body);
	if (status === 401) {
		return new DazziHttpError("Dazzi Models rejected the sign-in token (HTTP 401). Run /login again.", {
			status,
			code: payload.code ?? "unauthorized",
			type: payload.type,
		});
	}
	const reason = payload.message ?? "the server returned no error message";
	return new DazziHttpError(`Dazzi Models model listing failed (HTTP ${status}): ${reason}`, {
		status,
		code: payload.code,
		type: payload.type,
	});
}

/** Map an OpenAI-shaped `{ object: "list", data: [...] }` catalog body. */
export function mapServerModels(payload: unknown): ProviderModelConfig[] {
	if (typeof payload !== "object" || payload === null) return [];
	const data = (payload as Record<string, unknown>).data;
	if (!Array.isArray(data)) return [];
	const models: ProviderModelConfig[] = [];
	for (const entry of data) {
		if (typeof entry !== "object" || entry === null) continue;
		const model = mapServerModel(entry as Record<string, unknown>);
		if (model !== undefined) models.push(model);
	}
	return models;
}

/** Map a single Dazzi Models record; returns `undefined` when it has no usable id. */
export function mapServerModel(entry: Record<string, unknown>): ProviderModelConfig | undefined {
	const id = readString(entry.id);
	if (id === undefined) return undefined;
	return {
		type: "chat",
		id,
		name: readString(entry.name) ?? id,
		input: readInputModalities(entry.input_modalities),
		cost: readPricing(entry.pricing),
		reasoning: entry.reasoning === true,
		contextWindow: readPositiveNumber(entry.context_window) ?? DEFAULT_CONTEXT_WINDOW,
		maxTokens: readPositiveNumber(entry.max_output_tokens) ?? DEFAULT_MAX_TOKENS,
		inputLimits: DAZZI_INPUT_LIMITS,
	};
}

/** Snapshot the live catalog for offline startup. */
export function buildStoreEntry(
	models: readonly ProviderModelConfig[],
	providerId: string,
	baseUrl: string,
): ModelsStoreEntry {
	const stored: AnyModel[] = [];
	for (const model of models) {
		if (model.type === "image" || model.type === "classifier") continue;
		stored.push({
			type: "chat",
			id: model.id,
			name: model.name,
			api: model.api ?? "openai-responses",
			provider: providerId,
			baseUrl: model.baseUrl ?? baseUrl,
			input: model.input,
			cost: model.cost,
			reasoning: model.reasoning,
			contextWindow: model.contextWindow,
			maxTokens: model.maxTokens,
		});
	}
	return { models: stored, checkedAt: Date.now() };
}

/** Restore chat model definitions from a persisted catalog snapshot. */
export function modelsFromStore(stored: ModelsStoreEntry | undefined): ProviderModelConfig[] {
	if (stored === undefined) return [];
	const models: ProviderModelConfig[] = [];
	for (const entry of stored.models) {
		if (entry.type === "image" || entry.type === "classifier") continue;
		models.push({
			type: "chat",
			id: entry.id,
			name: entry.name,
			input: entry.input,
			cost: entry.cost,
			reasoning: entry.reasoning,
			contextWindow: entry.contextWindow,
			maxTokens: entry.maxTokens,
			// Limits belong to the server, not the snapshot: older snapshots lack them.
			inputLimits: DAZZI_INPUT_LIMITS,
		});
	}
	return models;
}

function readString(value: unknown): string | undefined {
	return typeof value === "string" && value.trim() !== "" ? value.trim() : undefined;
}

function readPositiveNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

function readInputModalities(value: unknown): ("text" | "image")[] {
	if (!Array.isArray(value)) return ["text"];
	const modalities = value.filter((item): item is "text" | "image" => item === "text" || item === "image");
	const unique = Array.from(new Set(modalities));
	return unique.length > 0 ? unique : ["text"];
}

function readPricing(value: unknown): ModelCost {
	const pricing = typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
	return {
		input: readPositiveNumber(pricing.input) ?? 0,
		output: readPositiveNumber(pricing.output) ?? 0,
		cacheRead: readPositiveNumber(pricing.cache_read) ?? 0,
		cacheWrite: readPositiveNumber(pricing.cache_write) ?? 0,
	};
}
