import { execFileSync } from "node:child_process";
import { cpus, totalmem } from "node:os";
import {
	AUTO_RETRIEVER_ID,
	DEFAULT_MINSYNC_BM25_LANGUAGE,
	DEFAULT_MINSYNC_EMBEDDER_ID,
	DEFAULT_MINSYNC_SPARSE_ID,
	NONE_SPARSE_ID,
	SPARSE_ONLY_EMBEDDER_ID,
} from "./embedder-config.ts";
import type { MinSyncBm25Config, MinSyncEmbedderConfig, MinSyncSparseConfig } from "./types.ts";

const GIB = 1024 ** 3;

/**
 * Hardware thresholds empirically calibrated in desktop VM experiments (8 / 16 / 24 GiB RAM
 * with ~30 Chrome tabs + IDE + chat/editor running) and aligned with MinSync's `src/hardware.rs`:
 *
 * - RAM < 16 GiB (or CPU-only with < 4 cores) -> `bm25-only` (zero ML inference)
 * - RAM >= 16 GiB (with Apple Silicon, CUDA GPU, or >= 4 CPU cores) -> `sparse+bm25`
 * - Apple Silicon with >= 24 GiB unified RAM, OR NVIDIA CUDA >= 8 GB VRAM + >= 16 GiB RAM -> `full`
 */
export const MINSYNC_DENSE_MIN_APPLE_RAM_GIB = 24;
export const MINSYNC_DENSE_MIN_CUDA_HOST_RAM_GIB = 16;
export const MINSYNC_DENSE_MIN_CUDA_VRAM_MIB = 8_000;
export const MINSYNC_SPLADE_MIN_RAM_GIB = 16;
export const MINSYNC_SPLADE_MIN_CPU_CORES = 4;

export const AUTORAG_HARDWARE_OVERRIDE_ENV = "AUTORAG_HARDWARE_OVERRIDE";
export const MINSYNC_HARDWARE_OVERRIDE_ENV = "MINSYNC_HARDWARE_OVERRIDE";

export interface CudaGpuInfo {
	readonly name: string;
	readonly vramMib: number;
}

export interface MinSyncHardwareProfile {
	readonly platform: string;
	readonly arch: string;
	readonly cores: number;
	readonly totalRamBytes: number;
	readonly cudaGpus: readonly CudaGpuInfo[];
}

export type MinSyncIndexingTier = "bm25-only" | "sparse+bm25" | "full";

export interface MinSyncHardwareRecommendation {
	readonly tier: MinSyncIndexingTier;
	readonly embedderId: string;
	readonly sparseId: string;
	readonly bm25Enabled: boolean;
	readonly reason: string;
}

export interface ResolvedMinSyncRetrieverPlan {
	readonly embedder: MinSyncEmbedderConfig | undefined;
	readonly sparse: MinSyncSparseConfig | undefined;
	readonly bm25: MinSyncBm25Config;
	readonly tier: MinSyncIndexingTier;
	readonly hardware: MinSyncHardwareProfile;
	readonly autoApplied: boolean;
}

/** Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` output. */
export function parseNvidiaSmiOutput(output: string): readonly CudaGpuInfo[] {
	const gpus: CudaGpuInfo[] = [];
	for (const line of output.split(/\r?\n/u)) {
		const cut = line.lastIndexOf(",");
		if (cut < 0) continue;
		const name = line.slice(0, cut).trim();
		const vramMib = Number.parseInt(line.slice(cut + 1).trim(), 10);
		if (name.length > 0 && Number.isFinite(vramMib) && vramMib > 0) {
			gpus.push({ name, vramMib });
		}
	}
	return gpus;
}

function probeCudaGpus(platform: string): readonly CudaGpuInfo[] {
	const candidates =
		platform === "win32"
			? ["nvidia-smi", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"]
			: platform === "linux"
				? ["nvidia-smi"]
				: [];
	for (const cmd of candidates) {
		try {
			const stdout = execFileSync(cmd, ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], {
				encoding: "utf8",
				timeout: 5_000,
				stdio: ["ignore", "pipe", "ignore"],
			});
			const parsed = parseNvidiaSmiOutput(stdout);
			if (parsed.length > 0) return parsed;
		} catch {
			// Driver absent or nvidia-smi unavailable; try next candidate.
		}
	}
	return [];
}

function parseOverride(raw: string | undefined): MinSyncHardwareProfile | undefined {
	const trimmed = raw?.trim();
	if (!trimmed) return undefined;
	try {
		const obj = JSON.parse(trimmed) as Record<string, unknown>;
		if (typeof obj !== "object" || obj === null) return undefined;
		// Accept both AutoRAG camelCase and MinSync Rust snake_case JSON overrides.
		const platformRaw = typeof obj.platform === "string" ? obj.platform : typeof obj.os === "string" ? obj.os : "";
		const platform = platformRaw === "macos" ? "darwin" : platformRaw === "windows" ? "win32" : platformRaw;
		const archRaw = typeof obj.arch === "string" ? obj.arch : "";
		const arch = archRaw === "aarch64" ? "arm64" : archRaw === "x86_64" ? "x64" : archRaw;
		const cores = typeof obj.cores === "number" && obj.cores >= 1 ? Math.floor(obj.cores) : 1;
		const totalRamBytes =
			typeof obj.totalRamBytes === "number"
				? obj.totalRamBytes
				: typeof obj.total_ram_bytes === "number"
					? obj.total_ram_bytes
					: typeof obj.memoryBytes === "number"
						? obj.memoryBytes
						: 0;
		const rawGpus = Array.isArray(obj.cudaGpus)
			? obj.cudaGpus
			: Array.isArray(obj.cuda_gpus)
				? obj.cuda_gpus
				: Array.isArray(obj.gpus)
					? obj.gpus
					: [];
		const cudaGpus: CudaGpuInfo[] = [];
		for (const gpu of rawGpus) {
			if (typeof gpu !== "object" || gpu === null) continue;
			const g = gpu as Record<string, unknown>;
			const name = typeof g.name === "string" ? g.name : "";
			const vramMib =
				typeof g.vramMib === "number"
					? g.vramMib
					: typeof g.vram_mib === "number"
						? g.vram_mib
						: typeof g.vramMiB === "number"
							? g.vramMiB
							: 0;
			if (name.length > 0 && vramMib > 0) cudaGpus.push({ name, vramMib });
		}
		if (platform.length === 0 || arch.length === 0 || !Number.isFinite(totalRamBytes) || totalRamBytes <= 0) {
			return undefined;
		}
		return { platform, arch, cores, totalRamBytes, cudaGpus };
	} catch {
		return undefined;
	}
}

let cachedHostProfile: MinSyncHardwareProfile | undefined;

/** Probe the current host's RAM, CPU cores, and CUDA GPUs (honoring env override for tests). */
export function probeMinSyncHardware(env: NodeJS.ProcessEnv = process.env): MinSyncHardwareProfile {
	const override = parseOverride(env[AUTORAG_HARDWARE_OVERRIDE_ENV] ?? env[MINSYNC_HARDWARE_OVERRIDE_ENV]);
	if (override !== undefined) return override;
	if (cachedHostProfile !== undefined) return cachedHostProfile;
	const platform = process.platform;
	cachedHostProfile = {
		platform,
		arch: process.arch,
		cores: Math.max(1, cpus().length),
		totalRamBytes: totalmem(),
		cudaGpus: platform === "win32" || platform === "linux" ? probeCudaGpus(platform) : [],
	};
	return cachedHostProfile;
}

/** Pure hardware-to-indexing-tier evaluator matching MinSync's `Hardware::recommend`. */
export function evaluateMinSyncHardwareTier(hardware: MinSyncHardwareProfile): MinSyncHardwareRecommendation {
	const isAppleSilicon = hardware.platform === "darwin" && hardware.arch === "arm64";
	const maxCudaVram = hardware.cudaGpus.reduce((best, gpu) => Math.max(best, gpu.vramMib), 0);
	const hasAccelerator = isAppleSilicon || maxCudaVram > 0;

	if (isAppleSilicon && hardware.totalRamBytes >= MINSYNC_DENSE_MIN_APPLE_RAM_GIB * GIB) {
		return {
			tier: "full",
			embedderId: DEFAULT_MINSYNC_EMBEDDER_ID,
			sparseId: DEFAULT_MINSYNC_SPARSE_ID,
			bm25Enabled: true,
			reason: `Apple Silicon with ${Math.floor(hardware.totalRamBytes / GIB)} GiB unified RAM supports dense + SPLADE + BM25`,
		};
	}
	if (
		maxCudaVram >= MINSYNC_DENSE_MIN_CUDA_VRAM_MIB &&
		hardware.totalRamBytes >= MINSYNC_DENSE_MIN_CUDA_HOST_RAM_GIB * GIB
	) {
		return {
			tier: "full",
			embedderId: DEFAULT_MINSYNC_EMBEDDER_ID,
			sparseId: DEFAULT_MINSYNC_SPARSE_ID,
			bm25Enabled: true,
			reason: `CUDA GPU (${Math.floor(maxCudaVram / 1024)} GiB VRAM) + ${Math.floor(hardware.totalRamBytes / GIB)} GiB RAM supports dense + SPLADE + BM25`,
		};
	}
	if (
		hardware.totalRamBytes >= MINSYNC_SPLADE_MIN_RAM_GIB * GIB &&
		(hasAccelerator || hardware.cores >= MINSYNC_SPLADE_MIN_CPU_CORES)
	) {
		return {
			tier: "sparse+bm25",
			embedderId: SPARSE_ONLY_EMBEDDER_ID,
			sparseId: DEFAULT_MINSYNC_SPARSE_ID,
			bm25Enabled: true,
			reason: isAppleSilicon
				? `RAM < ${MINSYNC_DENSE_MIN_APPLE_RAM_GIB} GiB; running SPLADE + BM25 without local dense model`
				: `No >= ${MINSYNC_DENSE_MIN_CUDA_VRAM_MIB / 1000} GB CUDA GPU; running SPLADE + BM25 without local dense model`,
		};
	}
	return {
		tier: "bm25-only",
		embedderId: SPARSE_ONLY_EMBEDDER_ID,
		sparseId: NONE_SPARSE_ID,
		bm25Enabled: true,
		reason:
			hardware.totalRamBytes < MINSYNC_SPLADE_MIN_RAM_GIB * GIB
				? `RAM < ${MINSYNC_SPLADE_MIN_RAM_GIB} GiB; running zero-ML BM25 only to avoid desktop memory pressure`
				: `CPU-only with < ${MINSYNC_SPLADE_MIN_CPU_CORES} cores; running zero-ML BM25 only`,
	};
}

/**
 * Resolve `embedder`, `sparse`, and `bm25` configs, expanding `"auto"` via hardware probing while
 * preserving every explicit override (`none`, `native:...`, `tei:...`, `openai:...`).
 */
export function resolveMinSyncRetrieverPlan(input: {
	readonly embedder?: MinSyncEmbedderConfig;
	readonly sparse?: MinSyncSparseConfig;
	readonly bm25?: MinSyncBm25Config;
	readonly hardware?: MinSyncHardwareProfile;
}): ResolvedMinSyncRetrieverPlan {
	const hardware = input.hardware ?? probeMinSyncHardware();
	const recommendation = evaluateMinSyncHardwareTier(hardware);
	const embedderIsAuto = input.embedder?.id?.toLowerCase() === AUTO_RETRIEVER_ID;
	const sparseIsAuto = input.sparse?.id?.toLowerCase() === AUTO_RETRIEVER_ID;
	const autoApplied = embedderIsAuto || sparseIsAuto;

	const embedder: MinSyncEmbedderConfig | undefined = embedderIsAuto
		? recommendation.embedderId === SPARSE_ONLY_EMBEDDER_ID
			? { id: SPARSE_ONLY_EMBEDDER_ID }
			: { ...input.embedder, id: recommendation.embedderId }
		: input.embedder;

	const sparse: MinSyncSparseConfig | undefined = sparseIsAuto
		? recommendation.sparseId === NONE_SPARSE_ID
			? { id: NONE_SPARSE_ID }
			: { ...input.sparse, id: recommendation.sparseId }
		: input.sparse;

	const bm25: MinSyncBm25Config = {
		enabled: input.bm25?.enabled ?? true,
		language: input.bm25?.language ?? DEFAULT_MINSYNC_BM25_LANGUAGE,
	};

	const denseOn = embedder?.id !== SPARSE_ONLY_EMBEDDER_ID;
	const sparseOn = sparse?.id !== NONE_SPARSE_ID;
	const tier: MinSyncIndexingTier = denseOn ? "full" : sparseOn ? "sparse+bm25" : "bm25-only";

	return { embedder, sparse, bm25, tier, hardware, autoApplied };
}
