import { DEFAULT_DAEMON_MODEL, ensureMinSyncDaemon, type MinSyncDaemon, probeMinSyncDaemon } from "./daemon.ts";
import type { Embedder, EmbedderEndpoint, EmbedderReadiness, EmbeddingIdentity, ManagedEmbedder } from "./embedder.ts";
import { createHttpEmbedder, type EmbeddingProtocol } from "./http-embedder.ts";
import { type InProcessEmbedder, loadNativeEmbedder } from "./in-process.ts";

/** An operator-configured remote endpoint. When present, AutoRAG never starts MinSync's daemon for it. */
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
	/** User home that holds MinSync's server state. */
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	/** Model of the daemon to use. Defaults to MinSync's default Qwen3 embedder. */
	readonly model?: string;
	/** Working directory for a daemon this embedder starts, so it reads the workspace's `[embedder]` settings. */
	readonly cwd?: string;
	/** Locates the `minsync` executable; only called when a daemon has to be started. */
	readonly resolveBinary?: () => Promise<string>;
	/** Loads MinSync's in-process engine; only called when no daemon answers. */
	readonly inProcess?: () => Promise<InProcessEmbedder>;
	readonly readyTimeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
}

const PROVIDER = "minsync";

function message(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/**
 * The one embedder every AutoRAG consumer shares. AutoRAG owns no embedding model:
 *
 * - **remote** — a configured endpoint is used as is; nothing local is started or loaded.
 * - **daemon** — a running `minsync serve` is preferred, because it keeps one warm copy of the model for
 *   every process. `ensureReady` (refresh/watch only) starts it when none answers; `embed` never does.
 * - **in-process** — MinSync's own engine, used when no daemon answers. Daemon and in-process vectors are
 *   identical for the same model and device, so both report one identity and stores never re-embed
 *   because the daemon came or went.
 */
export function createManagedEmbedder(options: ManagedEmbedderOptions): ManagedEmbedder {
	const fetchImpl = options.fetchImpl;
	const wantedModel = options.model ?? DEFAULT_DAEMON_MODEL;

	if (options.remote !== undefined) {
		const remote = createHttpEmbedder({
			...options.remote,
			provider: options.remote.provider ?? options.remote.protocol,
			...(options.env === undefined ? {} : { env: options.env }),
			...(fetchImpl === undefined ? {} : { fetchImpl }),
		});
		return {
			identity: () => remote.identity(),
			embed: (texts) => remote.embed(texts),
			ensureReady: async () => ({ backend: "remote", daemonStarted: false }),
			describeEndpoint: async () => ({
				baseUrl: options.remote?.baseUrl as string,
				protocol: options.remote?.protocol as EmbeddingProtocol,
				model: options.remote?.model as string,
				dimension: options.remote?.dimension,
				...(options.remote?.apiKeyEnv === undefined ? {} : { apiKeyEnv: options.remote.apiKeyEnv }),
				backend: "remote" as const,
			}),
			close: async () => undefined,
		};
	}

	let daemon: { readonly info: MinSyncDaemon; readonly embedder: Embedder } | undefined;
	let daemonError: string | undefined;
	let starting: Promise<EmbedderReadiness> | undefined;
	let inProcess: Promise<InProcessEmbedder> | undefined;

	const adopt = (info: MinSyncDaemon): Embedder => {
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl: info.baseUrl,
			model: info.model,
			dimension: info.dimension,
			provider: PROVIDER,
			...(fetchImpl === undefined ? {} : { fetchImpl }),
		});
		daemon = { info, embedder };
		return embedder;
	};

	const probe = async (): Promise<Embedder | undefined> => {
		const found = await probeMinSyncDaemon({
			model: wantedModel,
			...(options.home === undefined ? {} : { home: options.home }),
			...(fetchImpl === undefined ? {} : { fetchImpl }),
		});
		return found === undefined ? undefined : adopt(found);
	};

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

	async function startDaemon(): Promise<EmbedderReadiness> {
		try {
			const binaryPath = await (options.resolveBinary ?? rejectNoBinary)();
			const started = await ensureMinSyncDaemon({
				binaryPath,
				model: wantedModel,
				...(options.home === undefined ? {} : { home: options.home }),
				...(options.env === undefined ? {} : { env: options.env }),
				...(options.cwd === undefined ? {} : { cwd: options.cwd }),
				...(options.readyTimeoutMs === undefined ? {} : { readyTimeoutMs: options.readyTimeoutMs }),
				...(fetchImpl === undefined ? {} : { fetchImpl }),
			});
			adopt(started);
			return {
				backend: "daemon",
				daemonStarted: started.started,
				...(started.pid === undefined ? {} : { daemonPid: started.pid }),
			};
		} catch (error) {
			daemonError = message(error);
			return { backend: "in-process", daemonStarted: false, daemonError };
		}
	}

	async function active(): Promise<{ readonly source: Embedder; readonly viaDaemon: boolean }> {
		const source = daemon?.embedder ?? (await probe());
		if (source !== undefined) return { source, viaDaemon: true };
		try {
			return { source: await loadInProcess(), viaDaemon: false };
		} catch (error) {
			throw new Error(
				[
					daemonError === undefined ? undefined : `no embedding daemon: ${daemonError}`,
					`no in-process embedder: ${message(error)}`,
				]
					.filter((line) => line !== undefined)
					.join("; "),
				{ cause: error },
			);
		}
	}

	return {
		async identity() {
			return normalized((await active()).source);
		},
		async embed(texts) {
			if (texts.length === 0) return [];
			const { source, viaDaemon } = await active();
			try {
				return await source.embed(texts);
			} catch (error) {
				if (!viaDaemon) throw error;
				// The daemon exited (idle timeout, crash) since it was found: forget it and use what is left.
				daemon = undefined;
				const next = await active();
				if (next.viaDaemon) throw error;
				return next.source.embed(texts);
			}
		},
		async ensureReady() {
			// A daemon stops itself when idle, so a past success proves nothing: look again every time.
			if ((await probe()) !== undefined) return { backend: "daemon", daemonStarted: false };
			// A failed start is remembered for the life of this embedder instead of retried on every call.
			if (daemonError !== undefined) return { backend: "in-process", daemonStarted: false, daemonError };
			starting ??= startDaemon().finally(() => {
				starting = undefined;
			});
			return starting;
		},
		async describeEndpoint() {
			// Probes (never starts) so a daemon that has exited is not reported.
			const found = await probeMinSyncDaemon({
				model: wantedModel,
				...(options.home === undefined ? {} : { home: options.home }),
				...(fetchImpl === undefined ? {} : { fetchImpl }),
			});
			if (found !== undefined) adopt(found);
			else daemon = undefined;
			return found === undefined
				? undefined
				: {
						baseUrl: found.baseUrl,
						protocol: "openai" as const,
						model: found.model,
						dimension: found.dimension,
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

async function rejectNoBinary(): Promise<string> {
	throw new Error("no `minsync` executable resolver was provided");
}
