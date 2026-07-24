import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import {
	MinSyncVectorMethod,
	type MinSyncVectorMethodOptions,
} from "../../src/minsync/method.ts";
import type {
	MinSyncEmbedderConfig,
	MinSyncSyncResult,
} from "../../src/minsync/types.ts";
import {
	BM25Method,
	type BM25MethodOptions,
	type BM25SyncResult,
} from "../../src/retrieval/methods/bm25.ts";
import { ParallelRetriever, ResultMerger } from "../../src/retrieval/merger.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";
import {
	assertBenchmarkDirectoryIdentity,
	snapshotBenchmarkDirectory,
} from "./workspace.ts";
import { validateBenchmarkWorkspace } from "./run.ts";
import type { BenchmarkMethod } from "./types.ts";

const API_KEY_ENV_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const EMBEDDER_FIELDS = new Set([
	"id",
	"baseUrl",
	"apiKeyEnv",
	"dimension",
	"queryPrefix",
	"passagePrefix",
	"timeoutMs",
	"batchSize",
	"maxRetries",
	"maxConcurrent",
]);
const CONFIG_FIELDS = new Set(["binaryPath", "autoInstall", "embedder"]);
const POSITIVE_INTEGER_FIELDS = [
	"dimension",
	"timeoutMs",
	"batchSize",
	"maxRetries",
	"maxConcurrent",
] as const;
const RETRIEVAL_LIMIT = 100;

export interface BenchmarkMinSyncConfig {
	readonly binaryPath?: string;
	readonly autoInstall?: boolean;
	readonly embedder: MinSyncEmbedderConfig;
}

export interface SanitizedMethodConfig {
	readonly embedderId?: string;
	readonly endpointKind: "local" | "remote";
	readonly apiKeyEnv?: string;
	readonly dimension: number;
}

export interface BenchmarkMinSyncMethod extends RetrievalMethod {
	sync(): Promise<MinSyncSyncResult>;
}

export type BenchmarkMinSyncFactory = (
	options: MinSyncVectorMethodOptions,
) => BenchmarkMinSyncMethod;

export interface BenchmarkBM25Method extends RetrievalMethod {
	sync(): Promise<BM25SyncResult>;
}

export type BenchmarkBM25Factory = (
	options: BM25MethodOptions,
) => BenchmarkBM25Method;

export interface CreateBenchmarkMethodsOptions {
	readonly names: readonly BenchmarkMethod[];
	readonly root: string;
	readonly documentBySource?: ReadonlyMap<string, string>;
	readonly config: BenchmarkMinSyncConfig | undefined;
	readonly env?: NodeJS.ProcessEnv;
	readonly now?: () => number;
	readonly createBm25?: BenchmarkBM25Factory;
	readonly createMinSync?: BenchmarkMinSyncFactory;
}

export interface CreatedBenchmarkMethods {
	readonly methods: ReadonlyMap<BenchmarkMethod, RetrievalMethod>;
	readonly indexingLatencyMs: Readonly<
		Partial<Record<"bm25" | "minsync", number>>
	>;
	readonly reportConfig?: SanitizedMethodConfig;
}

/**
 * Load the benchmark-only MinSync JSON config. The returned object contains
 * only values needed to construct MinSync; reporting must use
 * {@link sanitizeMethodConfig}.
 */
export function loadBenchmarkConfig(
	path: string,
	env: NodeJS.ProcessEnv = process.env,
): BenchmarkMinSyncConfig {
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(path, "utf8"));
	} catch {
		throw new Error("Unable to load MinSync benchmark configuration");
	}
	return normalizeMethodConfig(parsed, env);
}

/**
 * Return the deliberately small method-config shape permitted in benchmark
 * reports. API key values and literal endpoint URLs are never copied.
 */
export function sanitizeMethodConfig(
	config: Pick<BenchmarkMinSyncConfig, "embedder">,
): SanitizedMethodConfig {
	const embedder = config.embedder;
	const sanitized: {
		embedderId?: string;
		endpointKind: "local" | "remote";
		apiKeyEnv?: string;
		dimension: number;
	} = {
		endpointKind: endpointKind(embedder.baseUrl),
		dimension: requirePositiveSafeInteger(
			embedder.dimension,
			"embedder.dimension",
		),
	};
	if (embedder.id !== undefined) sanitized.embedderId = embedder.id;
	if (embedder.apiKeyEnv !== undefined) sanitized.apiKeyEnv = embedder.apiKeyEnv;
	return sanitized;
}

/**
 * Construct and synchronize the real production retrieval methods needed for
 * a benchmark run. Validation is synchronous so malformed configuration and
 * workspace identities fail before the returned promise or any index write.
 */
export function createBenchmarkMethods(
	options: CreateBenchmarkMethodsOptions,
): Promise<CreatedBenchmarkMethods> {
	const names = validateMethodNames(options.names);
	const needsBm25 = names.has("bm25") || names.has("hybrid");
	const needsMinSync = names.has("minsync") || names.has("hybrid");
	let config: BenchmarkMinSyncConfig | undefined;
	if (needsMinSync) {
		if (options.config?.embedder === undefined) {
			throw new Error("MinSync benchmark requires an embedder configuration");
		}
		config = normalizeMethodConfig(options.config, options.env ?? process.env);
	}
	if (options.documentBySource === undefined) {
		throw new Error("MIRACL benchmark requires an exact document source map");
	}
	const root = validateBenchmarkWorkspace(options.root, options.documentBySource);
	const identity = snapshotBenchmarkDirectory(root);
	return constructAndSyncMethods({
		...options,
		names,
		needsBm25,
		needsMinSync,
		config,
		root,
		identity,
	});
}

interface ConstructOptions
	extends Omit<CreateBenchmarkMethodsOptions, "names" | "config" | "root"> {
	readonly names: ReadonlySet<BenchmarkMethod>;
	readonly needsBm25: boolean;
	readonly needsMinSync: boolean;
	readonly config: BenchmarkMinSyncConfig | undefined;
	readonly root: string;
	readonly identity: ReturnType<typeof snapshotBenchmarkDirectory>;
}

async function constructAndSyncMethods(
	options: ConstructOptions,
): Promise<CreatedBenchmarkMethods> {
	const now = options.now ?? (() => performance.now());
	const indexingLatencyMs: Partial<Record<"bm25" | "minsync", number>> = {};
	const createBm25 =
		options.createBm25 ??
		((methodOptions: BM25MethodOptions) => new BM25Method(methodOptions));
	const createMinSync =
		options.createMinSync ??
		((methodOptions: MinSyncVectorMethodOptions) =>
			new MinSyncVectorMethod(methodOptions));
	let bm25: BenchmarkBM25Method | undefined;
	let minsync: BenchmarkMinSyncMethod | undefined;

	if (options.needsBm25) {
		bm25 = createBm25({ root: options.root });
		const startedAt = now();
		let result: BM25SyncResult;
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		try {
			result = await bm25.sync();
		} catch {
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			now();
			throw new Error("BM25 benchmark indexing failed");
		}
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		indexingLatencyMs.bm25 = elapsedMilliseconds(startedAt, now());
		if (
			(result.readiness !== "ready" &&
				result.readiness !== "degraded_fallback") ||
			result.engine === "none" ||
			result.indexedChunks < 1
		) {
			throw new Error("BM25 benchmark indexing failed");
		}
	}

	if (options.needsMinSync) {
		const config = options.config as BenchmarkMinSyncConfig;
		minsync = createMinSync({
			root: options.root,
			binaryPath: config.binaryPath,
			autoInstall: config.autoInstall ?? false,
			embedder: config.embedder,
		});
		const startedAt = now();
		let result: MinSyncSyncResult;
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		try {
			result = await minsync.sync();
		} catch {
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			now();
			throw new Error("MinSync benchmark indexing failed");
		}
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		indexingLatencyMs.minsync = elapsedMilliseconds(startedAt, now());
		if (!result.ok) {
			throw new Error("MinSync benchmark indexing failed");
		}
	}

	const methods = new Map<BenchmarkMethod, RetrievalMethod>();
	for (const name of options.names) {
		if (name === "bm25" && bm25 !== undefined) methods.set(name, bm25);
		if (name === "minsync" && minsync !== undefined) methods.set(name, minsync);
		if (name === "hybrid" && bm25 !== undefined && minsync !== undefined) {
			methods.set(name, new HybridBenchmarkMethod(bm25, minsync));
		}
	}
	return {
		methods,
		indexingLatencyMs,
		reportConfig:
			options.config === undefined
				? undefined
				: sanitizeMethodConfig(options.config),
	};
}

/**
 * Retrieve hybrid candidates through the production diagnostic fan-out and
 * production result merger. Any component diagnostic invalidates the query.
 */
export async function retrieveHybrid(
	methods: readonly RetrievalMethod[],
	query: string,
	topK: number,
): Promise<RetrievalResult[]> {
	validateHybridMethods(methods);
	validateTopK(topK);
	const retriever = new ParallelRetriever();
	const merger = new ResultMerger();
	const outcome = await retriever.retrieveWithDiagnostics([...methods], query, {
		topK,
	});
	if (outcome.diagnostics.length > 0) {
		throw new Error("Hybrid benchmark retrieval failed");
	}
	return merger.merge(outcome.results, { topK, dedup: true });
}

class HybridBenchmarkMethod implements RetrievalMethod {
	private readonly methods: readonly [RetrievalMethod, RetrievalMethod];

	constructor(bm25: RetrievalMethod, minsync: RetrievalMethod) {
		this.methods = [bm25, minsync];
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "hybrid",
			type: "hybrid",
			description: "Production BM25 and MinSync benchmark fusion",
			status: "active",
			capabilities: ["lexical", "semantic", "deduplicated"],
		};
	}

	retrieve(
		query: string,
		options: RetrievalOptions,
	): Promise<RetrievalResult[]> {
		return retrieveHybrid(
			this.methods,
			query,
			options.topK ?? RETRIEVAL_LIMIT,
		);
	}
}

function normalizeMethodConfig(
	raw: unknown,
	env: NodeJS.ProcessEnv,
): BenchmarkMinSyncConfig {
	const record = requireRecord(raw, "MinSync benchmark configuration");
	rejectUnknownFields(record, CONFIG_FIELDS, "MinSync benchmark configuration");
	const embedderRecord = requireRecord(
		record.embedder,
		"MinSync benchmark embedder",
	);
	rejectUnknownFields(
		embedderRecord,
		EMBEDDER_FIELDS,
		"MinSync benchmark embedder",
	);

	const embedder: {
		id?: string;
		baseUrl?: string;
		apiKeyEnv?: string;
		dimension: number;
		queryPrefix?: string;
		passagePrefix?: string;
		timeoutMs?: number;
		batchSize?: number;
		maxRetries?: number;
		maxConcurrent?: number;
	} = {
		dimension: requirePositiveSafeInteger(
			embedderRecord.dimension,
			"embedder.dimension",
		),
	};
	for (const field of ["id", "baseUrl"] as const) {
		const value = embedderRecord[field];
		if (value === undefined) continue;
		if (typeof value !== "string" || value.trim().length === 0) {
			throw new Error(`MinSync benchmark embedder.${field} must be a non-empty string`);
		}
		embedder[field] = value;
	}
	if (embedder.baseUrl !== undefined) validateBaseUrl(embedder.baseUrl);
	for (const field of ["queryPrefix", "passagePrefix"] as const) {
		const value = embedderRecord[field];
		if (value === undefined) continue;
		if (typeof value !== "string") {
			throw new Error(`MinSync benchmark embedder.${field} must be a string`);
		}
		embedder[field] = value;
	}
	for (const field of POSITIVE_INTEGER_FIELDS) {
		if (field === "dimension") continue;
		const value = embedderRecord[field];
		if (value !== undefined) {
			embedder[field] = requirePositiveSafeInteger(
				value,
				`embedder.${field}`,
			);
		}
	}
	if (embedderRecord.apiKeyEnv !== undefined) {
		if (
			typeof embedderRecord.apiKeyEnv !== "string" ||
			!API_KEY_ENV_PATTERN.test(embedderRecord.apiKeyEnv)
		) {
			throw new Error(
				"MinSync benchmark embedder.apiKeyEnv must be a valid environment-variable name",
			);
		}
		const apiKeyEnv = embedderRecord.apiKeyEnv;
		if (
			!Object.hasOwn(env, apiKeyEnv) ||
			typeof env[apiKeyEnv] !== "string" ||
			env[apiKeyEnv]?.length === 0
		) {
			throw new Error(
				`MinSync benchmark requires environment variable ${apiKeyEnv}`,
			);
		}
		embedder.apiKeyEnv = apiKeyEnv;
	}

	const normalized: {
		binaryPath?: string;
		autoInstall?: boolean;
		embedder: MinSyncEmbedderConfig;
	} = { embedder };
	if (record.binaryPath !== undefined) {
		if (
			typeof record.binaryPath !== "string" ||
			record.binaryPath.trim().length === 0
		) {
			throw new Error(
				"MinSync benchmark binaryPath must be a non-empty string",
			);
		}
		normalized.binaryPath = record.binaryPath;
	}
	if (record.autoInstall !== undefined) {
		if (typeof record.autoInstall !== "boolean") {
			throw new Error("MinSync benchmark autoInstall must be a boolean");
		}
		normalized.autoInstall = record.autoInstall;
	}
	return normalized;
}

function validateMethodNames(
	names: readonly BenchmarkMethod[],
): ReadonlySet<BenchmarkMethod> {
	if (names.length === 0) {
		throw new Error("MIRACL benchmark requires at least one retrieval method");
	}
	const result = new Set<BenchmarkMethod>();
	for (const name of names) {
		if (name !== "bm25" && name !== "minsync" && name !== "hybrid") {
			throw new Error("MIRACL benchmark method is not recognized");
		}
		if (result.has(name)) {
			throw new Error("MIRACL benchmark methods must be unique");
		}
		result.add(name);
	}
	return result;
}

function validateHybridMethods(methods: readonly RetrievalMethod[]): void {
	if (methods.length !== 2) {
		throw new Error("Hybrid benchmark requires BM25 and MinSync methods");
	}
	const names = new Set(methods.map((method) => method.describe().name));
	if (
		names.size !== 2 ||
		!names.has("bm25") ||
		!names.has("minsync")
	) {
		throw new Error("Hybrid benchmark requires BM25 and MinSync methods");
	}
}

function validateTopK(topK: number): void {
	if (
		!Number.isSafeInteger(topK) ||
		topK < 1 ||
		topK > RETRIEVAL_LIMIT
	) {
		throw new Error(
			`topK must be a safe integer between 1 and ${RETRIEVAL_LIMIT}`,
		);
	}
}

function requireRecord(
	value: unknown,
	label: string,
): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`${label} must be an object`);
	}
	return value as Record<string, unknown>;
}

function rejectUnknownFields(
	record: Record<string, unknown>,
	allowlist: ReadonlySet<string>,
	label: string,
): void {
	for (const field of Object.keys(record)) {
		if (!allowlist.has(field)) {
			throw new Error(`${label}.${field} is not a recognized field`);
		}
	}
}

function requirePositiveSafeInteger(value: unknown, field: string): number {
	if (
		typeof value !== "number" ||
		!Number.isSafeInteger(value) ||
		value < 1
	) {
		throw new Error(
			`MinSync benchmark ${field} must be a positive safe integer`,
		);
	}
	return value;
}

function validateBaseUrl(baseUrl: string): void {
	try {
		const parsed = new URL(baseUrl);
		if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
			throw new Error("unsupported protocol");
		}
	} catch {
		throw new Error(
			"MinSync benchmark embedder.baseUrl must be a valid HTTP endpoint",
		);
	}
}

function endpointKind(baseUrl: string | undefined): "local" | "remote" {
	if (baseUrl === undefined) return "local";
	validateBaseUrl(baseUrl);
	const hostname = new URL(baseUrl).hostname.toLowerCase();
	if (
		hostname === "localhost" ||
		hostname === "[::1]" ||
		hostname === "::1" ||
		hostname === "0.0.0.0" ||
		hostname.startsWith("127.")
	) {
		return "local";
	}
	return "remote";
}

function elapsedMilliseconds(startedAt: number, finishedAt: number): number {
	const elapsed = finishedAt - startedAt;
	return Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0;
}
