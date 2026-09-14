export type BackendKind = "auto" | "cpu" | "vulkan";
export type PlatformId = "darwin-arm64-metal" | "win-x64-cpu" | "win-x64-vulkan";
export type ProfileId = "qwen3-embedding-0.6b" | "embeddinggemma-300m";

export interface ProtocolCapabilities {
	readonly healthz: boolean;
	readonly embed: boolean;
	readonly openaiCompatible: boolean;
	readonly ollamaCompatible: boolean;
}

export interface RuntimeProfile {
	readonly profileId: ProfileId;
	readonly provider: string;
	readonly model: string;
	readonly dimension: number;
	readonly queryPrefix: string;
	readonly passagePrefix: string;
	readonly runtimeBuild: string;
	readonly modelRevision: string;
	readonly artifactSha256: string;
	readonly backend: BackendKind;
	readonly protocolCapabilities: ProtocolCapabilities;
}

export interface HealthResult {
	readonly ok: true;
	readonly profileId: ProfileId;
	readonly dimension: number;
	readonly runtimeBuild: string;
}

export interface HealthFailure {
	readonly ok: false;
	readonly code: "unavailable" | "starting" | "incompatible" | "failed";
	readonly message: string;
	readonly retryable: boolean;
}

export type HealthStatus = HealthResult | HealthFailure;

export interface FailureResult {
	readonly ok: false;
	readonly code: string;
	readonly message: string;
	readonly cause?: unknown;
}

export class CacheError extends Error {
	readonly code:
		| "invalid-hash"
		| "offline-missing"
		| "download"
		| "hash-mismatch"
		| "io"
		| "extraction"
		| "missing-member";
	readonly path?: string;

	constructor(
		code: CacheError["code"],
		message: string,
		options: { readonly path?: string; readonly cause?: unknown } = {},
	) {
		super(message, { cause: options.cause });
		this.name = "CacheError";
		this.code = code;
		this.path = options.path;
	}
}

export interface ModelAsset {
	readonly url: string;
	readonly revision: string;
	readonly sha256: string;
	readonly license: string;
	readonly noticeReference: string;
}

export interface RuntimeAsset {
	readonly id: string;
	readonly url: string;
	readonly filename: string;
	readonly sha256: string;
	readonly version: string;
	readonly platform: PlatformId;
	readonly archiveMembers: readonly string[];
	readonly kind: "runtime";
}
