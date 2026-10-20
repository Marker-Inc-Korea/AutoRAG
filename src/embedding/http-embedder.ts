import type { Embedder, EmbeddingIdentity } from "./embedder.ts";

export type EmbeddingProtocol = "openai" | "tei";

export interface HttpEmbedderOptions {
	readonly protocol: EmbeddingProtocol;
	/** Server root, with or without a trailing `/v1` (openai) and trailing slashes. */
	readonly baseUrl: string;
	readonly model: string;
	/** Expected vector length. Learned from the first response when omitted. */
	readonly dimension?: number;
	/** Reported as `identity().provider`; stores keyed on identity re-embed when it changes. */
	readonly provider: string;
	/** Name of the environment variable holding the bearer token. Read on every call, never stored. */
	readonly apiKeyEnv?: string;
	readonly env?: NodeJS.ProcessEnv;
	/** Inputs per request. */
	readonly batchSize?: number;
	/** TEI only: ask the server to truncate inputs over its context instead of answering 413. */
	readonly truncate?: boolean;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
}

/** The daemon rejects bodies over 32 MiB and more than 512 inputs; remote providers cap lower. */
export const DEFAULT_EMBED_BATCH_SIZE = 32;
const DEFAULT_TIMEOUT_MS = 120_000;

function endpoint(options: HttpEmbedderOptions): string {
	const root = options.baseUrl.replace(/\/+$/, "");
	if (options.protocol === "tei") return `${root}/embed`;
	return `${root.replace(/\/v1$/, "")}/v1/embeddings`;
}

function headers(options: HttpEmbedderOptions): Record<string, string> {
	const result: Record<string, string> = { "Content-Type": "application/json" };
	if (options.apiKeyEnv === undefined) return result;
	const key = (options.env ?? process.env)[options.apiKeyEnv];
	if (typeof key !== "string" || key.length === 0) {
		throw new Error(`embedding API key is not set: environment variable ${options.apiKeyEnv} is empty or missing`);
	}
	result.Authorization = `Bearer ${key}`;
	return result;
}

function requestBody(options: HttpEmbedderOptions, batch: readonly string[]): string {
	if (options.protocol === "tei") {
		return JSON.stringify({ inputs: batch, ...(options.truncate === true ? { truncate: true } : {}) });
	}
	return JSON.stringify({ model: options.model, input: batch });
}

function parseVectors(options: HttpEmbedderOptions, json: unknown, expected: number): number[][] {
	let rows: unknown[];
	if (options.protocol === "tei") {
		rows = Array.isArray(json) ? json : [];
	} else {
		const data = (json as { data?: { index?: number; embedding?: unknown }[] } | null)?.data;
		rows = Array.isArray(data)
			? [...data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0)).map((row) => row.embedding)
			: [];
	}
	if (rows.length !== expected || rows.some((row) => !Array.isArray(row))) {
		throw new Error(`embedding endpoint returned ${rows.length} usable vectors for ${expected} inputs`);
	}
	return rows as number[][];
}

/**
 * Embedder over an HTTP endpoint: a `minsync serve` daemon (either route) or any remote OpenAI-compatible
 * or TEI server. It owns no model and starts nothing.
 */
export function createHttpEmbedder(options: HttpEmbedderOptions): Embedder {
	const fetchImpl = options.fetchImpl ?? fetch;
	const batchSize = Math.max(1, options.batchSize ?? DEFAULT_EMBED_BATCH_SIZE);
	let dimension = options.dimension;
	let probing: Promise<void> | undefined;

	async function request(batch: readonly string[]): Promise<number[][]> {
		const response = await fetchImpl(endpoint(options), {
			method: "POST",
			headers: headers(options),
			body: requestBody(options, batch),
			signal: AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS),
		});
		if (!response.ok) {
			const detail = await response.text().catch(() => "");
			throw new Error(`embedding endpoint ${endpoint(options)} returned HTTP ${response.status}: ${detail}`);
		}
		const vectors = parseVectors(options, await response.json(), batch.length);
		dimension ??= vectors[0]?.length;
		if (vectors.some((vector) => vector.length !== dimension)) {
			throw new Error(`embedding endpoint returned a vector whose dimension differs from the expected ${dimension}`);
		}
		return vectors;
	}

	async function learnDimension(): Promise<void> {
		if (dimension !== undefined) return;
		probing ??= request(["dimension probe"]).then(() => undefined);
		await probing;
	}

	return {
		async identity(): Promise<EmbeddingIdentity> {
			await learnDimension();
			return { provider: options.provider, model: options.model, dimension: dimension as number };
		},
		async embed(texts) {
			if (texts.length === 0) return [];
			const vectors: number[][] = [];
			for (let offset = 0; offset < texts.length; offset += batchSize) {
				vectors.push(...(await request(texts.slice(offset, offset + batchSize))));
			}
			return vectors;
		},
	};
}
