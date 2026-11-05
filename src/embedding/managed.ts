import { DEFAULT_DAEMON_MODEL, type MinSyncDaemon, probeMinSyncDaemon } from "./daemon.ts";
import type { Embedder, EmbedderHealth, EmbeddingIdentity, EmbedOptions, ManagedEmbedder } from "./embedder.ts";
import { createHttpEmbedder, type EmbeddingProtocol } from "./http-embedder.ts";
import { type InProcessEmbedder, loadNativeEmbedder } from "./in-process.ts";
import type { EmbeddingPrefixes } from "./prefixes.ts";

/** An operator-configured remote endpoint. When present, AutoRAG never looks for MinSync's daemon. */
export interface RemoteEmbedderSettings {
	readonly baseUrl: string;
	readonly model: string;
	readonly protocol: EmbeddingProtocol;
	readonly dimension?: number;
	readonly apiKeyEnv?: string;
	readonly batchSize?: number;
	readonly timeoutMs?: number;
	/** Reported as the identity provider; defaults to the protocol name. */
	readonly provider?: string;
}

export interface ManagedEmbedderOptions {
	readonly remote?: RemoteEmbedderSettings;
	/**
	 * Prepended to every input by role: `query` for `role: "query"`, `passage` otherwise. Applied here, on the
	 * client, for every backend, because neither the daemon nor a remote server is asked to prefix.
	 */
	readonly prefixes?: EmbeddingPrefixes;
	/** User home that holds MinSync's server state. */
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	/** Model of the daemon to use. Defaults to MinSync's default Qwen3 embedder. */
	readonly model?: string;
	/** Loads MinSync's in-process engine; only called when no daemon answers. */
	readonly inProcess?: () => Promise<InProcessEmbedder>;
	readonly fetchImpl?: typeof fetch;
}

const PROVIDER = "minsync";
const REMOTE_HEALTH_TIMEOUT_MS = 2_000;

function message(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/**
 * `GET /health` of a remote endpoint's server root. TEI and `minsync serve` answer it, and so do most
 * OpenAI-compatible servers (vLLM, llama.cpp); a hosted API that has no such route answers 404, which still
 * proves the host is reachable, so only a network failure or a 5xx counts as down.
 */
async function probeRemote(remote: RemoteEmbedderSettings, fetchImpl: typeof fetch): Promise<EmbedderHealth> {
	const root = remote.baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
	try {
		const response = await fetchImpl(`${root}/health`, { signal: AbortSignal.timeout(REMOTE_HEALTH_TIMEOUT_MS) });
		if (response.status >= 500) return { backend: "remote", alive: false, detail: `HTTP ${response.status}` };
		return { backend: "remote", alive: true, detail: root };
	} catch (error) {
		return { backend: "remote", alive: false, detail: message(error) };
	}
}

/**
 * The one embedder every AutoRAG consumer shares. AutoRAG owns no embedding model and never starts or stops
 * a process: the operator runs `minsync serve` (see docs/minsync-setup.md) or points at a remote endpoint.
 *
 * - **remote**: a configured endpoint is used as is; nothing local is looked for or loaded.
 * - **daemon**: a running `minsync serve` is preferred, because it keeps one warm copy of the model for every
 *   process. It is looked up again on every call, so one started or stopped while AutoRAG runs is noticed.
 * - **in-process**: MinSync's own engine, used when no daemon answers. Daemon and in-process vectors are
 *   identical for the same model and device, so both report one identity and stores never re-embed
 *   because the daemon came or went.
 */
export function createManagedEmbedder(options: ManagedEmbedderOptions): ManagedEmbedder {
	const fetchImpl = options.fetchImpl ?? fetch;
	const wantedModel = options.model ?? DEFAULT_DAEMON_MODEL;
	const prefixes = options.prefixes;
	const prefixed = (texts: readonly string[], embedOptions: EmbedOptions | undefined): readonly string[] => {
		const prefix = embedOptions?.role === "query" ? prefixes?.query : prefixes?.passage;
		return prefix === undefined || prefix === "" ? texts : texts.map((text) => `${prefix}${text}`);
	};
	const prefixFields = prefixes === undefined ? {} : { queryPrefix: prefixes.query, passagePrefix: prefixes.passage };

	if (options.remote !== undefined) {
		const remoteSettings = options.remote;
		const remote = createHttpEmbedder({
			...remoteSettings,
			provider: remoteSettings.provider ?? remoteSettings.protocol,
			...(options.env === undefined ? {} : { env: options.env }),
			fetchImpl,
		});
		return {
			identity: () => remote.identity(),
			embed: (texts, embedOptions) => remote.embed(prefixed(texts, embedOptions)),
			health: () => probeRemote(remoteSettings, fetchImpl),
			describeEndpoint: async () => ({
				baseUrl: remoteSettings.baseUrl,
				protocol: remoteSettings.protocol,
				model: remoteSettings.model,
				dimension: remoteSettings.dimension,
				...(remoteSettings.apiKeyEnv === undefined ? {} : { apiKeyEnv: remoteSettings.apiKeyEnv }),
				...prefixFields,
				backend: "remote" as const,
			}),
			close: async () => undefined,
		};
	}

	let daemon: { readonly info: MinSyncDaemon; readonly embedder: Embedder } | undefined;
	let inProcess: Promise<InProcessEmbedder> | undefined;

	const adopt = (info: MinSyncDaemon): Embedder => {
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl: info.baseUrl,
			model: info.model,
			dimension: info.dimension,
			provider: PROVIDER,
			fetchImpl,
		});
		daemon = { info, embedder };
		return embedder;
	};

	const probe = async (): Promise<MinSyncDaemon | undefined> =>
		probeMinSyncDaemon({
			model: wantedModel,
			...(options.home === undefined ? {} : { home: options.home }),
			fetchImpl,
		});

	const loadInProcess = (): Promise<InProcessEmbedder> => {
		inProcess ??= (options.inProcess ?? (() => loadNativeEmbedder()))();
		// A failed load is not cached: the addon may be installed while a long-lived host runs.
		inProcess.catch(() => {
			inProcess = undefined;
		});
		return inProcess;
	};

	const normalized = async (source: Embedder): Promise<EmbeddingIdentity> => {
		const identity = await source.identity();
		return { provider: PROVIDER, model: identity.model, dimension: identity.dimension };
	};

	async function active(): Promise<{ readonly source: Embedder; readonly viaDaemon: boolean }> {
		const found = await probe();
		if (found !== undefined)
			return { source: daemon?.info.baseUrl === found.baseUrl ? daemon.embedder : adopt(found), viaDaemon: true };
		daemon = undefined;
		try {
			return { source: await loadInProcess(), viaDaemon: false };
		} catch (error) {
			throw new Error(
				`no embedding daemon is running and no in-process embedder is installed: ${message(error)}. Start \`minsync serve\` (see docs/minsync-setup.md) or set minSync.embedder to a remote endpoint.`,
				{ cause: error },
			);
		}
	}

	return {
		async identity() {
			return normalized((await active()).source);
		},
		async embed(texts, embedOptions) {
			if (texts.length === 0) return [];
			const inputs = prefixed(texts, embedOptions);
			const { source, viaDaemon } = await active();
			try {
				return await source.embed(inputs);
			} catch (error) {
				if (!viaDaemon) throw error;
				// The daemon exited (idle timeout, crash) since it was found: forget it and use what is left.
				daemon = undefined;
				const next = await active();
				if (next.viaDaemon) throw error;
				return next.source.embed(inputs);
			}
		},
		async health() {
			const found = await probe();
			if (found !== undefined) return { backend: "daemon", alive: true, detail: found.baseUrl };
			// No daemon answers. In-process is only a real backend when the addon is loaded or loadable.
			try {
				await loadInProcess();
				return { backend: "in-process", alive: true };
			} catch (error) {
				return { backend: "none", alive: false, detail: message(error) };
			}
		},
		async describeEndpoint() {
			// Probes only, so a daemon that has exited is not reported.
			const found = await probe();
			if (found === undefined) {
				daemon = undefined;
				return undefined;
			}
			adopt(found);
			return {
				baseUrl: found.baseUrl,
				protocol: "openai" as const,
				model: found.model,
				dimension: found.dimension,
				...prefixFields,
				backend: "daemon" as const,
			};
		},
		async close() {
			const loaded = inProcess;
			inProcess = undefined;
			if (loaded !== undefined) await (await loaded.catch(() => undefined))?.close();
		},
	};
}
