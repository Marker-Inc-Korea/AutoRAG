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

/** Where a {@link ManagedEmbedder} is sending text right now. */
export type EmbeddingBackend = "daemon" | "in-process" | "remote";

export interface EmbedderReadiness {
	readonly backend: EmbeddingBackend;
	/** True when this call started the `minsync serve` daemon. */
	readonly daemonStarted: boolean;
	/** Pid of a daemon this call started; the daemon outlives the process and stops itself when idle. */
	readonly daemonPid?: number;
	/** Why the daemon was not used, verbatim, when the in-process engine took over. */
	readonly daemonError?: string;
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
 * An {@link Embedder} that can be brought up explicitly. `embed` and `identity` never start a process,
 * so answering a question never spawns a server; `ensureReady` is what `refresh`/`watch` call first.
 */
export interface ManagedEmbedder extends Embedder {
	ensureReady(): Promise<EmbedderReadiness>;
	/**
	 * The HTTP endpoint behind this embedder right now, or undefined while it would embed in-process.
	 * Probes only: it never starts a daemon.
	 */
	describeEndpoint(): Promise<EmbedderEndpoint | undefined>;
	close(): Promise<void>;
}
