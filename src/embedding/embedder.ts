/**
 * The contract every semantic consumer in AutoRAG codes against. AutoRAG does not run an embedding model:
 * vectors come from MinSync (the `minsync serve` daemon, or MinSync's in-process engine) or from a
 * remote endpoint the operator configured.
 */

export interface EmbeddingIdentity {
	readonly provider: string;
	readonly model: string;
	readonly dimension: number;
}

export interface Embedder {
	identity(): Promise<EmbeddingIdentity>;
	embed(texts: readonly string[]): Promise<readonly (readonly number[])[]>;
}

/** Where a {@link ManagedEmbedder} is sending text right now. `none` means nothing can embed. */
export type EmbeddingBackend = "daemon" | "in-process" | "remote" | "none";

/** Whether the embedding backend answers right now. Probed on every call; nothing is cached. */
export interface EmbedderHealth {
	readonly backend: EmbeddingBackend;
	readonly alive: boolean;
	/** The answering server's address when alive, or why it is not. */
	readonly detail?: string;
}

/** Where a {@link ManagedEmbedder} currently sends text, for external CLIs that need a URL. */
export interface EmbedderEndpoint {
	/** Server root; the OpenAI route is under `/v1`, TEI at `/embed`. */
	readonly baseUrl: string;
	readonly protocol: "openai" | "tei";
	readonly model: string;
	readonly dimension: number | undefined;
	/** Name of the environment variable that holds the bearer token; remote endpoints only. */
	readonly apiKeyEnv?: string;
	readonly backend: "daemon" | "remote";
}

/**
 * An {@link Embedder} that reports where it embeds. AutoRAG never starts or stops a server: the operator
 * runs `minsync serve` or configures a remote endpoint, and `health` / `describeEndpoint` only look.
 */
export interface ManagedEmbedder extends Embedder {
	/** Probes the backend now: the daemon's or the remote's `GET /health`. Never starts anything. */
	health(): Promise<EmbedderHealth>;
	/** The HTTP endpoint behind this embedder right now, or undefined while it would embed in-process. */
	describeEndpoint(): Promise<EmbedderEndpoint | undefined>;
	close(): Promise<void>;
}
