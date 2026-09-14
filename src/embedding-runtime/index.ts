import { basename } from "node:path";
import { resolveAutoRAGHome } from "../config/home.ts";
import {
	type CacheOptions,
	downloadAsset as downloadCacheAsset,
	importAsset as importCacheAsset,
	verifyCacheEntry as verifyCache,
} from "./cache.ts";
import { startEmbeddingGateway } from "./gateway.ts";
import { MODEL_ASSETS, resolveProfile, selectPlatformAsset } from "./manifest.ts";
import { EmbeddingRuntimeSupervisor as RuntimeSupervisor, type SupervisorStatus } from "./supervisor.ts";
import type { BackendKind, HealthStatus, ProfileId, RuntimeProfile } from "./types.ts";

export interface EmbeddingRuntimeCache {
	downloadAsset(
		asset: {
			id: string;
			url: string;
			filename: string;
			sha256: string;
			kind?: "model" | "runtime";
			archiveMembers?: readonly string[];
		},
		options?: CacheOptions,
	): Promise<string>;
	importAsset(
		sourcePath: string,
		asset: {
			id: string;
			url: string;
			filename: string;
			sha256: string;
			kind?: "model" | "runtime";
			archiveMembers?: readonly string[];
		},
		options?: CacheOptions,
	): Promise<string>;
	verifyCacheEntry(path: string, expectedSha256: string): Promise<string>;
}

export interface EmbeddingRuntimeSupervisor {
	ensureRunning(): Promise<SupervisorStatus>;
	shutdown(): Promise<void>;
	status(): SupervisorStatus;
}

export interface EmbeddingRuntimeGateway {
	readonly url: string;
	close(): Promise<void>;
}

export interface EmbeddingRuntimeOptions {
	cacheRoot?: string;
	backend?: BackendKind;
	offline?: boolean;
	platform?: "darwin-arm64-metal" | "win-x64-cpu" | "win-x64-vulkan";
	fetch?: typeof fetch;
	supervisor?: EmbeddingRuntimeSupervisor;
	cache?: EmbeddingRuntimeCache;
	gatewayFactory?: (options: Parameters<typeof startEmbeddingGateway>[0]) => Promise<EmbeddingRuntimeGateway>;
}

export interface EnsureRuntimeOptions extends EmbeddingRuntimeOptions {
	profileId?: ProfileId;
}

export interface RuntimeIdentity {
	readonly profileId: ProfileId;
	readonly provider: string;
	readonly model: string;
	readonly dimension: number;
	readonly runtimeBuild: string;
	readonly modelRevision: string;
	readonly artifactSha256: string;
}

export interface EnsuredRuntime {
	readonly baseUrl: string;
	readonly profile: RuntimeProfile;
	readonly identity: RuntimeIdentity;
	readonly supervisor: SupervisorStatus;
}

export interface RuntimeStatus {
	readonly state: SupervisorStatus["state"];
	readonly backend: BackendKind;
	readonly model: string;
	readonly pid?: number;
	readonly port?: number;
	readonly uptimeMs?: number;
	readonly lastError?: string;
	readonly profileId?: ProfileId;
	readonly baseUrl?: string;
	readonly health: HealthStatus;
}

function modelAsset(profile: RuntimeProfile) {
	return profile.profileId === "qwen3-embedding-0.6b"
		? { id: profile.profileId, filename: basename(profile.model), ...MODEL_ASSETS.qwen3 }
		: { id: profile.profileId, filename: basename(profile.model), ...MODEL_ASSETS.embeddinggemma };
}
function runtimeAsset(platform: EmbeddingRuntimeOptions["platform"] | undefined, backend: BackendKind) {
	const selected = platform ?? "darwin-arm64-metal";
	return selectPlatformAsset(selected, backend, false);
}
function identity(profile: RuntimeProfile): RuntimeIdentity {
	return {
		profileId: profile.profileId,
		provider: profile.provider,
		model: profile.model,
		dimension: profile.dimension,
		runtimeBuild: profile.runtimeBuild,
		modelRevision: profile.modelRevision,
		artifactSha256: profile.artifactSha256,
	};
}

export function createEmbeddingRuntime(options: EmbeddingRuntimeOptions = {}) {
	const root = options.cacheRoot ?? resolveAutoRAGHome();
	const cache: EmbeddingRuntimeCache = options.cache ?? {
		downloadAsset: downloadCacheAsset,
		importAsset: importCacheAsset,
		verifyCacheEntry: verifyCache,
	};
	let supervisor: EmbeddingRuntimeSupervisor = options.supervisor ?? new RuntimeSupervisor({ cacheRoot: root });
	let gateway: EmbeddingRuntimeGateway | undefined;
	let activeProfile: RuntimeProfile | undefined;
	let stoppedByService = false;

	function profileOf(profileId?: ProfileId): RuntimeProfile {
		return resolveProfile(profileId ?? "qwen3-embedding-0.6b");
	}
	async function cacheRuntime(profile: RuntimeProfile): Promise<{ modelPath: string; runtimePath: string }> {
		const cacheOptions = { cacheRoot: root, offline: options.offline, fetch: options.fetch };
		const modelPath = await cache.downloadAsset(modelAsset(profile), cacheOptions);
		const runtimePath = await cache.downloadAsset(
			runtimeAsset(options.platform, options.backend ?? profile.backend),
			cacheOptions,
		);
		return { modelPath, runtimePath };
	}
	async function ensureRuntime(input: EnsureRuntimeOptions = {}): Promise<EnsuredRuntime> {
		const profile = profileOf(input.profileId);
		stoppedByService = false;
		const selectedSupervisor = input.supervisor ?? options.supervisor;
		if (selectedSupervisor) supervisor = selectedSupervisor;
		const paths = await cacheRuntime(profile);
		if (supervisor instanceof RuntimeSupervisor) {
			// The default supervisor is constructed with the selected profile/cache paths
			// so model and runtime resolution remain deterministic.
			supervisor = new RuntimeSupervisor({
				cacheRoot: root,
				profileId: profile.profileId,
				backend: input.backend ?? options.backend ?? profile.backend,
				modelPath: paths.modelPath,
				executablePath: paths.runtimePath,
				fetch: input.fetch ?? options.fetch,
			});
		}
		const status = await supervisor.ensureRunning();
		if (!status.port) throw new Error("Embedding runtime did not expose a loopback port.");
		if (gateway) await gateway.close();
		const factory = input.gatewayFactory ?? options.gatewayFactory ?? startEmbeddingGateway;
		gateway = await factory({
			profile,
			upstreamUrl: `http://127.0.0.1:${status.port}`,
			fetch: input.fetch ?? options.fetch,
		});
		activeProfile = profile;
		return { baseUrl: gateway.url, profile, identity: identity(profile), supervisor: status };
	}
	async function stopRuntime(): Promise<void> {
		if (gateway) {
			await gateway.close();
			gateway = undefined;
		}
		await supervisor.shutdown();
		stoppedByService = true;
		activeProfile = undefined;
	}
	async function runtimeStatus(): Promise<RuntimeStatus> {
		const status = supervisor.status();
		const state = stoppedByService ? "stopped" : status.state;
		const base = {
			state,
			backend: status.backend,
			model: status.model,
			...(status.pid === undefined ? {} : { pid: status.pid }),
			...(status.port === undefined ? {} : { port: status.port }),
			...(status.uptimeMs === undefined ? {} : { uptimeMs: status.uptimeMs }),
			...(status.lastError === undefined ? {} : { lastError: status.lastError }),
		};
		if (!gateway || status.state !== "ready")
			return {
				...base,
				...(activeProfile ? { profileId: activeProfile.profileId } : {}),
				health: {
					ok: false,
					code: "unavailable",
					message: status.state === "stopped" ? "Gateway is stopped." : "Gateway is unavailable.",
					retryable: true,
				},
			};
		try {
			const response = await (options.fetch ?? fetch)(`${gateway.url}/healthz`);
			if (!response.ok) throw new Error(`Gateway health returned HTTP ${response.status}.`);
			const health = (await response.json()) as HealthStatus;
			return { ...base, profileId: activeProfile?.profileId, baseUrl: gateway.url, health };
		} catch {
			return {
				...base,
				profileId: activeProfile?.profileId,
				baseUrl: gateway.url,
				health: { ok: false, code: "unavailable", message: "Gateway health check failed.", retryable: true },
			};
		}
	}
	return {
		resolveRuntimeProfile: (profileId?: ProfileId) => profileOf(profileId),
		ensureRuntime,
		stopRuntime,
		runtimeStatus,
		prefetchModel: async (profileId?: ProfileId) => {
			const profile = profileOf(profileId);
			const path = await cache.downloadAsset(modelAsset(profile), {
				cacheRoot: root,
				offline: options.offline,
				fetch: options.fetch,
			});
			return { profileId: profile.profileId, path };
		},
		importModel: async (profileId: ProfileId | undefined, sourcePath: string) => {
			const profile = profileOf(profileId);
			const path = await cache.importAsset(sourcePath, modelAsset(profile), { cacheRoot: root });
			return { profileId: profile.profileId, path };
		},
		verifyModel: async (profileId?: ProfileId) => {
			const profile = profileOf(profileId);
			const path = await cache.downloadAsset(modelAsset(profile), { cacheRoot: root, offline: true });
			const hash = await cache.verifyCacheEntry(path, profile.artifactSha256);
			return { profileId: profile.profileId, path, hash };
		},
	};
}

const defaultRuntime = createEmbeddingRuntime();
export const resolveRuntimeProfile = defaultRuntime.resolveRuntimeProfile;
export const ensureRuntime = defaultRuntime.ensureRuntime;
export const stopRuntime = defaultRuntime.stopRuntime;
export const runtimeStatus = defaultRuntime.runtimeStatus;
export const prefetchModel = defaultRuntime.prefetchModel;
export const importModel = defaultRuntime.importModel;
export const verifyModel = defaultRuntime.verifyModel;
