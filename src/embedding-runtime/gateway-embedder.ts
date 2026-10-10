import type { ProfileId } from "./types.ts";

export interface EmbeddingIdentity {
	readonly provider: string;
	readonly model: string;
	readonly dimension: number;
}

/** Embedder contract: identity is async because the gateway resolves it lazily. */
export interface Embedder {
	identity(): Promise<EmbeddingIdentity>;
	embed(texts: readonly string[]): Promise<readonly (readonly number[])[]>;
}

export interface GatewayEmbedderOptions {
	/** Runtime resolver; defaults to the shared embedding-runtime ensureRuntime. */
	readonly runtime?: {
		ensureRuntime(input?: { profileId?: ProfileId; cachedOnly?: boolean }): Promise<{
			baseUrl: string;
			identity: { provider: string; model: string; dimension: number };
		}>;
	};
	readonly profileId?: ProfileId;
	readonly fetchImpl?: typeof fetch;
}

/**
 * Texts per `/v1/embeddings` call. The llama.cpp upstream rejects very large
 * batches (HTTP 413 on a full-archive sync), so embedding fans out in
 * bounded sequential batches while preserving input order.
 */
export const EMBED_BATCH_SIZE = 32;

/**
 * Production embedder backed by the loopback autorag-gateway. The runtime is
 * ensured lazily with `cachedOnly` so embedding never triggers a model
 * download; an unprimed cache surfaces as a caller-side degrade.
 */
export function createGatewayEmbedder(options: GatewayEmbedderOptions = {}): Embedder {
	let ensured: { baseUrl: string; identity: EmbeddingIdentity } | undefined;
	async function ensure(): Promise<{ baseUrl: string; identity: EmbeddingIdentity }> {
		if (ensured !== undefined) return ensured;
		let runtime = options.runtime;
		if (runtime === undefined) {
			// Lazy: index.ts constructs the shared runtime at module load, so a static
			// import would eagerly pull the whole supervisor/cache graph for consumers
			// that only touch the embedder contract.
			const module = await import("./index.ts");
			runtime = { ensureRuntime: module.ensureRuntime };
		}
		const result = await runtime.ensureRuntime({ profileId: options.profileId, cachedOnly: true });
		ensured = {
			baseUrl: result.baseUrl,
			identity: {
				provider: result.identity.provider,
				model: result.identity.model,
				dimension: result.identity.dimension,
			},
		};
		return ensured;
	}
	return {
		identity: async () => (await ensure()).identity,
		async embed(texts) {
			if (texts.length === 0) return [];
			const { baseUrl, identity } = await ensure();
			const fetchImpl = options.fetchImpl ?? fetch;
			const embeddings: number[][] = [];
			for (let offset = 0; offset < texts.length; offset += EMBED_BATCH_SIZE) {
				const batch = texts.slice(offset, offset + EMBED_BATCH_SIZE);
				const response = await fetchImpl(`${baseUrl}/v1/embeddings`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ model: identity.model, input: [...batch] }),
				});
				if (!response.ok) throw new Error(`gateway /v1/embeddings returned HTTP ${response.status}`);
				const json = (await response.json()) as { data?: { index?: number; embedding?: number[] }[] };
				const rows = [...(json.data ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
				const batchEmbeddings = rows.map((row) => row.embedding ?? []);
				if (
					batchEmbeddings.length !== batch.length ||
					batchEmbeddings.some((row) => row.length !== identity.dimension)
				) {
					throw new Error("gateway /v1/embeddings returned an invalid embedding batch");
				}
				embeddings.push(...batchEmbeddings);
			}
			return embeddings;
		},
	};
}
