import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { accessSync, constants, createReadStream, lstatSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve, sep } from "node:path";
import { performance } from "node:perf_hooks";
import { parse } from "smol-toml";
import { rewriteEmbedderConfig } from "../../src/minsync/embedder-config.ts";
import { minSyncWorkspaceRoot } from "../../src/minsync/paths.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncResult } from "../../src/minsync/types.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "../../src/minsync/workspace.ts";
import { ParallelRetriever, ResultMerger } from "../../src/retrieval/merger.ts";
import { BM25Method, type BM25SyncResult } from "../../src/retrieval/methods/bm25.ts";
import { matchesVirtualPathScope } from "../../src/retrieval/scope.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";
import { FullCorpusBm25Method } from "./full-bm25.ts";
import type { JsonLinesAttestation } from "./jsonl.ts";
import { type BenchmarkRetrievalLifecycle, validateBenchmarkWorkspace } from "./run.ts";
import type { BenchmarkMethod } from "./types.ts";
import { assertBenchmarkDirectoryIdentity, snapshotBenchmarkDirectory } from "./workspace.ts";

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
const POSITIVE_INTEGER_FIELDS = ["dimension", "timeoutMs", "batchSize", "maxRetries", "maxConcurrent"] as const;
const RETRIEVAL_LIMIT = 100;
const RETRIEVAL_OVERFETCH_LIMIT = 1_600;
const HYBRID_METHOD_ORDER = ["bm25", "minsync"] as const;
const MINSYNC_MAX_STDOUT_BYTES = 16 * 1024 * 1024;
const MINSYNC_MAX_STDERR_BYTES = 1024 * 1024;

export interface BoundedMinSyncProcessOptions {
	readonly timeoutMs: number;
	readonly maxStdoutBytes: number;
	readonly maxStderrBytes: number;
}

export interface BoundedMinSyncProcessResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	readonly timedOut?: true;
	readonly outputExceeded?: true;
}

export function runBoundedMinSyncProcess(
	command: string,
	args: readonly string[],
	cwd: string,
	options: BoundedMinSyncProcessOptions,
): Promise<BoundedMinSyncProcessResult> {
	for (const [label, value] of [
		["timeoutMs", options.timeoutMs],
		["maxStdoutBytes", options.maxStdoutBytes],
		["maxStderrBytes", options.maxStderrBytes],
	] as const) {
		if (!Number.isSafeInteger(value) || value < 1) {
			throw new Error(`MinSync benchmark process ${label} must be a positive safe integer`);
		}
	}
	return new Promise((resolvePromise) => {
		const child = spawn(command, [...args], { cwd, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
		const stdout: Buffer[] = [];
		const stderr: Buffer[] = [];
		let stdoutBytes = 0;
		let stderrBytes = 0;
		let timedOut = false;
		let outputExceeded = false;
		let settled = false;
		const append = (chunks: Buffer[], chunk: Buffer, currentBytes: number, maximum: number): number => {
			const remaining = Math.max(maximum - currentBytes, 0);
			if (remaining > 0) chunks.push(chunk.subarray(0, remaining));
			if (chunk.byteLength > remaining) {
				outputExceeded = true;
				child.kill("SIGKILL");
			}
			return Math.min(currentBytes + chunk.byteLength, maximum);
		};
		child.stdout.on("data", (chunk: Buffer) => {
			stdoutBytes = append(stdout, chunk, stdoutBytes, options.maxStdoutBytes);
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderrBytes = append(stderr, chunk, stderrBytes, options.maxStderrBytes);
		});
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill("SIGKILL");
		}, options.timeoutMs);
		const finish = (code: number | null, processError?: Error) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			const stdoutText = Buffer.concat(stdout, stdoutBytes).toString("utf8");
			const stderrText =
				processError === undefined ? Buffer.concat(stderr, stderrBytes).toString("utf8") : processError.message;
			resolvePromise({
				ok: !timedOut && !outputExceeded && processError === undefined && code === 0,
				stdout: stdoutText,
				stderr: stderrText,
				code,
				...(timedOut ? { timedOut: true as const } : {}),
				...(outputExceeded ? { outputExceeded: true as const } : {}),
			});
		};
		child.once("error", (error) => finish(null, error));
		child.once("close", (code) => finish(code));
	});
}

export interface BenchmarkMinSyncConfig {
	readonly binaryPath: string;
	readonly autoInstall: false;
	readonly embedder: MinSyncEmbedderConfig & {
		readonly id: string;
		readonly dimension: number;
		readonly timeoutMs: number;
	};
}

export interface SanitizedMinSyncConfig {
	readonly executableSha256: string;
	readonly modelId: string;
	readonly endpointKind: "local" | "remote";
	readonly authentication: "environment" | "none";
	readonly dimension: number;
	readonly timeoutMs: number;
	readonly maxStdoutBytes: number;
	readonly maxStderrBytes: number;
	readonly queryPrefixSha256: string;
	readonly passagePrefixSha256: string;
	readonly batchSize?: number;
	readonly maxRetries?: number;
	readonly maxConcurrent?: number;
}

export interface SanitizedMethodConfig {
	readonly bm25?: {
		readonly engine: "tantivy" | "typescript-fallback";
	};
	readonly minsync?: SanitizedMinSyncConfig;
}

export interface CreateBenchmarkMethodsOptions {
	readonly names: readonly BenchmarkMethod[];
	readonly root: string;
	readonly documentBySource?: ReadonlyMap<string, string>;
	readonly fullCorpus?: {
		readonly path: string;
		readonly attestation: JsonLinesAttestation;
	};
	readonly config: BenchmarkMinSyncConfig | undefined;
	readonly now?: () => number;
}

export interface CreatedBenchmarkMethods {
	readonly methods: ReadonlyMap<BenchmarkMethod, RetrievalMethod>;
	readonly indexingLatencyMs: Readonly<Partial<Record<"bm25" | "minsync", number>>>;
	readonly reportConfig?: SanitizedMethodConfig;
}

/**
 * Load the benchmark-only MinSync JSON config. The returned object contains
 * only values needed to construct MinSync; reporting must use
 * {@link sanitizeMethodConfig}.
 */
export function loadBenchmarkConfig(path: string): BenchmarkMinSyncConfig {
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(path, "utf8"));
	} catch {
		throw new Error("Unable to load MinSync benchmark configuration");
	}
	return normalizeMethodConfig(parsed, process.env);
}

/**
 * Return the deliberately small method-config shape permitted in benchmark
 * reports. API key values and literal endpoint URLs are never copied.
 */
export function sanitizeMethodConfig(config: BenchmarkMinSyncConfig, executableSha256: string): SanitizedMinSyncConfig {
	const embedder = config.embedder;
	if (!/^[0-9a-f]{64}$/u.test(executableSha256)) {
		throw new Error("MinSync benchmark executable SHA-256 is invalid");
	}
	const sanitized: SanitizedMinSyncConfig = {
		executableSha256,
		modelId: requirePortableEmbedderId(embedder.id),
		endpointKind: endpointKind(embedder.baseUrl),
		authentication: embedder.apiKeyEnv === undefined ? "none" : "environment",
		dimension: requirePositiveSafeInteger(embedder.dimension, "embedder.dimension"),
		timeoutMs: requirePositiveSafeInteger(embedder.timeoutMs, "embedder.timeoutMs"),
		maxStdoutBytes: MINSYNC_MAX_STDOUT_BYTES,
		maxStderrBytes: MINSYNC_MAX_STDERR_BYTES,
		queryPrefixSha256: createHash("sha256")
			.update(embedder.queryPrefix ?? "", "utf8")
			.digest("hex"),
		passagePrefixSha256: createHash("sha256")
			.update(embedder.passagePrefix ?? "", "utf8")
			.digest("hex"),
	};
	return {
		...sanitized,
		...(embedder.batchSize === undefined ? {} : { batchSize: embedder.batchSize }),
		...(embedder.maxRetries === undefined ? {} : { maxRetries: embedder.maxRetries }),
		...(embedder.maxConcurrent === undefined ? {} : { maxConcurrent: embedder.maxConcurrent }),
	};
}

/**
 * Construct and synchronize the real production retrieval methods needed for
 * a benchmark run. Validation is synchronous so malformed configuration and
 * workspace identities fail before the returned promise or any index write.
 */
export function createBenchmarkMethods(options: CreateBenchmarkMethodsOptions): Promise<CreatedBenchmarkMethods> {
	const names = validateMethodNames(options.names);
	const needsBm25 = names.has("bm25") || names.has("hybrid");
	const needsMinSync = names.has("minsync") || names.has("hybrid");
	if (options.fullCorpus !== undefined && (names.size !== 1 || !names.has("bm25"))) {
		throw new Error("full profile supports only bm25");
	}
	let config: BenchmarkMinSyncConfig | undefined;
	if (needsMinSync) {
		if (options.config?.embedder === undefined) {
			throw new Error("MinSync benchmark requires an embedder configuration");
		}
		config = normalizeMethodConfig(options.config, process.env);
	}
	if (options.documentBySource === undefined && options.fullCorpus === undefined) {
		throw new Error("MIRACL benchmark requires an exact document source map");
	}
	const root =
		options.fullCorpus === undefined
			? validateBenchmarkWorkspace(options.root, options.documentBySource as ReadonlyMap<string, string>)
			: realpathSync(options.root);
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

interface ConstructOptions extends Omit<CreateBenchmarkMethodsOptions, "names" | "config" | "root"> {
	readonly names: ReadonlySet<BenchmarkMethod>;
	readonly needsBm25: boolean;
	readonly needsMinSync: boolean;
	readonly config: BenchmarkMinSyncConfig | undefined;
	readonly root: string;
	readonly identity: ReturnType<typeof snapshotBenchmarkDirectory>;
}

async function constructAndSyncMethods(options: ConstructOptions): Promise<CreatedBenchmarkMethods> {
	const now = options.now ?? (() => performance.now());
	const indexingLatencyMs: Partial<Record<"bm25" | "minsync", number>> = {};
	let bm25: RetrievalMethod | undefined;
	let bm25Engine: "tantivy" | "typescript-fallback" | undefined;
	let minsync: RetrievalMethod | undefined;
	let minSyncReportConfig: SanitizedMinSyncConfig | undefined;

	if (options.needsBm25) {
		if (options.fullCorpus !== undefined) {
			const streamingBm25 = new FullCorpusBm25Method({
				root: options.root,
				corpusPath: options.fullCorpus.path,
				attestation: options.fullCorpus.attestation,
			});
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			const startedAt = now();
			const result = await streamingBm25.sync();
			indexingLatencyMs.bm25 = elapsedMilliseconds(startedAt, now());
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			if (result.indexedDocuments !== options.fullCorpus.attestation.records || result.indexedChunks < 1) {
				throw new Error("BM25 benchmark indexing failed");
			}
			bm25 = streamingBm25;
			bm25Engine = "tantivy";
		} else {
			const productionBm25 = new BM25Method({ root: options.root });
			let result: BM25SyncResult;
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			const startedAt = now();
			try {
				result = await productionBm25.sync();
			} catch {
				const indexingLatencyMsValue = elapsedMilliseconds(startedAt, now());
				assertBenchmarkDirectoryIdentity(options.root, options.identity);
				indexingLatencyMs.bm25 = indexingLatencyMsValue;
				throw new Error("BM25 benchmark indexing failed");
			}
			indexingLatencyMs.bm25 = elapsedMilliseconds(startedAt, now());
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			if (
				(result.readiness !== "ready" && result.readiness !== "degraded_fallback") ||
				result.engine === "none" ||
				result.indexedChunks < 1
			) {
				throw new Error("BM25 benchmark indexing failed");
			}
			bm25 = productionBm25;
			bm25Engine = result.engine as "tantivy" | "typescript-fallback";
		}
	}

	if (options.needsMinSync) {
		const config = options.config as BenchmarkMinSyncConfig;
		let executable: ExecutableIdentity;
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		try {
			executable = await resolveBenchmarkExecutable(options.root, config);
		} catch {
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			throw new Error("MinSync benchmark executable is unavailable");
		}
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		let result: MinSyncSyncResult;
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		assertExecutableMetadata(executable);
		const startedAt = now();
		try {
			result = await syncBenchmarkMinSync(options.root, executable.path, config.embedder);
		} catch {
			const indexingLatencyMsValue = elapsedMilliseconds(startedAt, now());
			assertBenchmarkDirectoryIdentity(options.root, options.identity);
			assertExecutableIdentity(executable);
			indexingLatencyMs.minsync = indexingLatencyMsValue;
			throw new Error("MinSync benchmark indexing failed");
		}
		indexingLatencyMs.minsync = elapsedMilliseconds(startedAt, now());
		assertBenchmarkDirectoryIdentity(options.root, options.identity);
		assertExecutableIdentity(executable);
		const expectedWorkspacePath = minSyncWorkspaceRoot(options.root);
		if (
			!result.ok ||
			!Number.isSafeInteger(result.synced) ||
			result.synced < 1 ||
			result.workspacePath !== expectedWorkspacePath
		) {
			throw new Error("MinSync benchmark indexing failed");
		}
		const workspaceIdentity = await snapshotMinSyncWorkspace(options.root, expectedWorkspacePath);
		minsync = new CheckedMinSyncMethod({
			root: options.root,
			rootIdentity: options.identity,
			executable,
			workspacePath: expectedWorkspacePath,
			workspaceIdentity,
			embedder: config.embedder,
			pathMap: buildMinSyncPathMap(options.root, expectedWorkspacePath),
		});
		minSyncReportConfig = sanitizeMethodConfig(config, executable.sha256);
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
		reportConfig: {
			...(bm25Engine === undefined ? {} : { bm25: { engine: bm25Engine } }),
			...(minSyncReportConfig === undefined ? {} : { minsync: minSyncReportConfig }),
		},
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
	const deterministicResults = new Map<string, RetrievalResult[]>();
	for (const name of HYBRID_METHOD_ORDER) {
		const results = outcome.results.get(name);
		if (results === undefined) {
			throw new Error("Hybrid benchmark retrieval failed");
		}
		deterministicResults.set(name, [...results].sort(compareRetrievalResults));
	}
	return merger.merge(deterministicResults, { topK, dedup: true });
}

class HybridBenchmarkMethod implements RetrievalMethod, BenchmarkRetrievalLifecycle {
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

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveHybrid(this.methods, query, options.topK ?? RETRIEVAL_LIMIT);
	}

	beforeBenchmarkBatch(): Promise<void> {
		return runLifecycleHook(this.methods, "beforeBenchmarkBatch");
	}

	beforeBenchmarkQuery(): Promise<void> {
		return runLifecycleHook(this.methods, "beforeBenchmarkQuery");
	}

	afterBenchmarkQuery(): Promise<void> {
		return runLifecycleHook(this.methods, "afterBenchmarkQuery");
	}

	afterBenchmarkBatch(): Promise<void> {
		return runLifecycleHook(this.methods, "afterBenchmarkBatch");
	}
}

type LifecycleHookName =
	| "beforeBenchmarkBatch"
	| "beforeBenchmarkQuery"
	| "afterBenchmarkQuery"
	| "afterBenchmarkBatch";

async function runLifecycleHook(methods: readonly RetrievalMethod[], hookName: LifecycleHookName): Promise<void> {
	for (const method of methods) {
		const lifecycle = method as RetrievalMethod & BenchmarkRetrievalLifecycle;
		await lifecycle[hookName]?.call(lifecycle);
	}
}

interface ExecutableIdentity {
	readonly path: string;
	readonly device: number;
	readonly inode: number;
	readonly size: number;
	readonly modifiedAtMs: number;
	readonly sha256: string;
}

interface PathIdentity {
	readonly path: string;
	readonly device: number;
	readonly inode: number;
	readonly size: number;
	readonly modifiedAtMs: number;
	readonly kind: "directory" | "file";
}

interface FileIntegrity extends PathIdentity {
	readonly kind: "file";
	readonly sha256: string;
}

interface MinSyncWorkspaceIdentity {
	readonly benchmarkRoot: string;
	readonly workspace: PathIdentity;
	readonly stateDirectory: PathIdentity;
	readonly config: FileIntegrity;
	readonly manifest: FileIntegrity;
	readonly cursor: FileIntegrity;
	readonly collection: PathIdentity;
	readonly collectionTree: readonly PathIdentity[];
	readonly collectionTreeSha256: string;
}

interface CheckedMinSyncMethodOptions {
	readonly root: string;
	readonly rootIdentity: ReturnType<typeof snapshotBenchmarkDirectory>;
	readonly executable: ExecutableIdentity;
	readonly workspacePath: string;
	readonly workspaceIdentity: MinSyncWorkspaceIdentity;
	readonly embedder: MinSyncEmbedderConfig;
	readonly pathMap: ReturnType<typeof buildMinSyncPathMap>;
}

class CheckedMinSyncMethod implements RetrievalMethod, BenchmarkRetrievalLifecycle {
	private readonly root: string;
	private readonly rootIdentity: ReturnType<typeof snapshotBenchmarkDirectory>;
	private readonly executable: ExecutableIdentity;
	private readonly workspacePath: string;
	private readonly workspaceIdentity: MinSyncWorkspaceIdentity;
	private readonly embedder: MinSyncEmbedderConfig;
	private readonly pathMap: ReturnType<typeof buildMinSyncPathMap>;

	constructor(options: CheckedMinSyncMethodOptions) {
		this.root = options.root;
		this.rootIdentity = options.rootIdentity;
		this.executable = options.executable;
		this.workspacePath = options.workspacePath;
		this.workspaceIdentity = options.workspaceIdentity;
		this.embedder = options.embedder;
		this.pathMap = options.pathMap;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "minsync",
			type: "vector",
			description: "Checked MinSync benchmark retrieval over parsed markdown mirrors",
			status: "active",
			capabilities: ["semantic", "parsed-mirrors", "virtual-paths", "checked-process"],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const topK = options.topK ?? 20;
		validateTopK(topK);
		const queryK = options.scope ? Math.min(Math.max(topK * 5, topK + 20), RETRIEVAL_OVERFETCH_LIMIT) : topK;
		assertBenchmarkDirectoryIdentity(this.root, this.rootIdentity);
		assertExecutableMetadata(this.executable);
		assertMinSyncQueryBoundaryMetadata(this.workspaceIdentity);
		let processResult: BoundedMinSyncProcessResult;
		try {
			processResult = await runBoundedMinSyncProcess(
				this.executable.path,
				["query", "--format", "json", "-k", String(queryK), query],
				this.workspacePath,
				{
					timeoutMs: this.embedder.timeoutMs as number,
					maxStdoutBytes: MINSYNC_MAX_STDOUT_BYTES,
					maxStderrBytes: MINSYNC_MAX_STDERR_BYTES,
				},
			);
		} catch {
			throw new Error("MinSync benchmark retrieval failed");
		} finally {
			assertBenchmarkDirectoryIdentity(this.root, this.rootIdentity);
			assertExecutableMetadata(this.executable);
			assertMinSyncQueryBoundaryMetadata(this.workspaceIdentity);
		}
		if (!processResult.ok) {
			throw new Error("MinSync benchmark retrieval failed");
		}
		const hits = parseCheckedMinSyncHits(processResult.stdout);
		const results: RetrievalResult[] = [];
		for (const hit of hits) {
			const entry = this.pathMap.get(hit.path);
			if (!entry) {
				throw new Error("MinSync benchmark retrieval failed");
			}
			if (!matchesVirtualPathScope(entry.virtualPath, options.scope)) continue;
			results.push({
				id: `minsync:${entry.virtualPath}:${basename(hit.path)}`,
				content: hit.text,
				source: entry.virtualPath,
				score: hit.score,
				metadata: { method: "minsync" },
			});
			if (results.length >= topK) break;
		}
		return results;
	}

	async beforeBenchmarkBatch(): Promise<void> {
		assertBenchmarkDirectoryIdentity(this.root, this.rootIdentity);
		assertExecutableIdentity(this.executable);
		await assertMinSyncWorkspaceIntegrity(this.workspaceIdentity);
	}

	beforeBenchmarkQuery(): void {
		assertBenchmarkDirectoryIdentity(this.root, this.rootIdentity);
		assertExecutableMetadata(this.executable);
		assertMinSyncWorkspaceMetadata(this.workspaceIdentity);
	}

	afterBenchmarkQuery(): void {
		assertBenchmarkDirectoryIdentity(this.root, this.rootIdentity);
		assertExecutableMetadata(this.executable);
		assertMinSyncWorkspaceMetadata(this.workspaceIdentity);
	}

	async afterBenchmarkBatch(): Promise<void> {
		assertBenchmarkDirectoryIdentity(this.root, this.rootIdentity);
		assertExecutableIdentity(this.executable);
		await assertMinSyncWorkspaceIntegrity(this.workspaceIdentity);
	}
}

async function syncBenchmarkMinSync(
	root: string,
	binaryPath: string,
	embedder: MinSyncEmbedderConfig,
): Promise<MinSyncSyncResult> {
	const workspacePath = minSyncWorkspaceRoot(root);
	syncMinSyncWorkspace(root, { workspacePath });
	const processOptions = {
		timeoutMs: embedder.timeoutMs as number,
		maxStdoutBytes: MINSYNC_MAX_STDOUT_BYTES,
		maxStderrBytes: MINSYNC_MAX_STDERR_BYTES,
	};
	const init = await runBoundedMinSyncProcess(
		binaryPath,
		["init", "--format", "json", "--embedder", embedder.id as string],
		workspacePath,
		processOptions,
	);
	if (!init.ok) return { ok: false, synced: 0, workspacePath, reason: "init-failed" };
	rewriteEmbedderConfig(workspacePath, embedder);
	const sync = await runBoundedMinSyncProcess(binaryPath, ["sync", "--format", "json"], workspacePath, processOptions);
	if (!sync.ok) return { ok: false, synced: 0, workspacePath, reason: "sync-failed" };
	return { ok: true, synced: readSyncedCount(sync.stdout), workspacePath };
}

function readSyncedCount(stdout: string): number {
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		return 0;
	}
	if (!isRecord(parsed)) return 0;
	for (const field of ["files_processed", "synced"]) {
		const value = parsed[field];
		if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) return value;
	}
	return 0;
}

async function resolveBenchmarkExecutable(root: string, config: BenchmarkMinSyncConfig): Promise<ExecutableIdentity> {
	void root;
	if (config.binaryPath === undefined || config.autoInstall !== false) {
		throw new Error("MinSync benchmark executable is unavailable");
	}
	return snapshotExecutable(config.binaryPath);
}

function snapshotExecutable(path: string): ExecutableIdentity {
	const metadata = snapshotExecutableMetadata(path);
	try {
		return {
			...metadata,
			sha256: createHash("sha256").update(readFileSync(metadata.path)).digest("hex"),
		};
	} catch {
		throw new Error("MinSync benchmark executable is unavailable");
	}
}

function snapshotExecutableMetadata(path: string): Omit<ExecutableIdentity, "sha256"> {
	try {
		const canonicalPath = realpathSync(path);
		const stats = lstatSync(canonicalPath);
		if (!stats.isFile() || stats.isSymbolicLink()) throw new Error("not a real file");
		accessSync(canonicalPath, constants.X_OK);
		return {
			path: canonicalPath,
			device: stats.dev,
			inode: stats.ino,
			size: stats.size,
			modifiedAtMs: stats.mtimeMs,
		};
	} catch {
		throw new Error("MinSync benchmark executable is unavailable");
	}
}

function assertExecutableMetadata(expected: ExecutableIdentity): void {
	let actual: Omit<ExecutableIdentity, "sha256">;
	try {
		actual = snapshotExecutableMetadata(expected.path);
	} catch {
		throw new Error("MinSync benchmark executable changed");
	}
	if (
		actual.path !== expected.path ||
		actual.device !== expected.device ||
		actual.inode !== expected.inode ||
		actual.size !== expected.size ||
		actual.modifiedAtMs !== expected.modifiedAtMs
	) {
		throw new Error("MinSync benchmark executable changed");
	}
}

function assertExecutableIdentity(expected: ExecutableIdentity): void {
	assertExecutableMetadata(expected);
	let digest: string;
	try {
		digest = createHash("sha256").update(readFileSync(expected.path)).digest("hex");
	} catch {
		throw new Error("MinSync benchmark executable changed");
	}
	if (digest !== expected.sha256) {
		throw new Error("MinSync benchmark executable changed");
	}
}

async function snapshotMinSyncWorkspace(
	benchmarkRoot: string,
	workspacePath: string,
): Promise<MinSyncWorkspaceIdentity> {
	try {
		const workspace = snapshotPathIdentity(workspacePath, "directory", benchmarkRoot);
		const stateDirectory = snapshotPathIdentity(join(workspace.path, ".minsync"), "directory", workspace.path);
		const config = snapshotFileIntegrity(join(stateDirectory.path, "config.toml"), stateDirectory.path);
		const manifest = snapshotFileIntegrity(join(stateDirectory.path, "manifest.json"), stateDirectory.path);
		const cursor = snapshotFileIntegrity(join(stateDirectory.path, "cursor.json"), stateDirectory.path);
		const collectionPath = readCollectionPath(config.path);
		if (isAbsolute(collectionPath) || collectionPath.length === 0 || collectionPath.split(/[\\/]/u).includes("..")) {
			throw new Error("unsafe collection path");
		}
		const collection = snapshotPathIdentity(
			resolve(stateDirectory.path, collectionPath),
			"directory",
			stateDirectory.path,
		);
		const collectionTree = snapshotDirectoryTreeMetadata(collection.path);
		return {
			benchmarkRoot,
			workspace,
			stateDirectory,
			config,
			manifest,
			cursor,
			collection,
			collectionTree,
			collectionTreeSha256: await hashDirectoryTree(collection.path, collectionTree),
		};
	} catch {
		throw new Error("MinSync benchmark workspace changed");
	}
}

function assertMinSyncQueryBoundaryMetadata(expected: MinSyncWorkspaceIdentity): void {
	for (const identity of [expected.workspace, expected.stateDirectory]) {
		let actual: PathIdentity;
		try {
			actual = snapshotPathIdentity(identity.path, identity.kind, expected.benchmarkRoot);
		} catch {
			throw new Error("MinSync benchmark workspace changed");
		}
		if (!samePathIdentity(actual, identity)) {
			throw new Error("MinSync benchmark workspace changed");
		}
	}
}

function assertMinSyncWorkspaceMetadata(expected: MinSyncWorkspaceIdentity): void {
	for (const identity of [
		expected.workspace,
		expected.stateDirectory,
		expected.config,
		expected.manifest,
		expected.cursor,
		expected.collection,
	]) {
		let actual: PathIdentity;
		try {
			actual = snapshotPathIdentity(identity.path, identity.kind, expected.benchmarkRoot);
		} catch {
			throw new Error("MinSync benchmark workspace changed");
		}
		if (!samePathIdentity(actual, identity)) {
			throw new Error("MinSync benchmark workspace changed");
		}
	}
	let actualCollectionTree: readonly PathIdentity[];
	try {
		actualCollectionTree = snapshotDirectoryTreeMetadata(expected.collection.path);
	} catch {
		throw new Error("MinSync benchmark workspace changed");
	}
	if (!samePathIdentityList(actualCollectionTree, expected.collectionTree)) {
		throw new Error("MinSync benchmark workspace changed");
	}
}

async function assertMinSyncWorkspaceIntegrity(expected: MinSyncWorkspaceIdentity): Promise<void> {
	const actual = await snapshotMinSyncWorkspace(expected.benchmarkRoot, expected.workspace.path);
	if (
		!samePathIdentity(actual.workspace, expected.workspace) ||
		!samePathIdentity(actual.stateDirectory, expected.stateDirectory) ||
		!sameFileIntegrity(actual.config, expected.config) ||
		!sameFileIntegrity(actual.manifest, expected.manifest) ||
		!sameFileIntegrity(actual.cursor, expected.cursor) ||
		!samePathIdentity(actual.collection, expected.collection) ||
		!samePathIdentityList(actual.collectionTree, expected.collectionTree) ||
		actual.collectionTreeSha256 !== expected.collectionTreeSha256
	) {
		throw new Error("MinSync benchmark workspace changed");
	}
}

function snapshotPathIdentity(path: string, kind: PathIdentity["kind"], container: string): PathIdentity {
	const stats = lstatSync(path);
	if (stats.isSymbolicLink() || (kind === "directory" ? !stats.isDirectory() : !stats.isFile())) {
		throw new Error("invalid filesystem object");
	}
	const canonicalPath = realpathSync(path);
	if (canonicalPath !== path || !isContainedPath(canonicalPath, container)) {
		throw new Error("filesystem object escaped container");
	}
	return {
		path: canonicalPath,
		device: stats.dev,
		inode: stats.ino,
		size: stats.size,
		modifiedAtMs: stats.mtimeMs,
		kind,
	};
}

function snapshotFileIntegrity(path: string, container: string): FileIntegrity {
	const identity = snapshotPathIdentity(path, "file", container);
	return {
		...identity,
		kind: "file",
		sha256: createHash("sha256").update(readFileSync(identity.path)).digest("hex"),
	};
}

function readCollectionPath(configPath: string): string {
	const config = parse(readFileSync(configPath, "utf8")) as {
		collection?: { path?: unknown };
	};
	const path = config.collection?.path;
	if (typeof path !== "string") {
		throw new Error("missing collection path");
	}
	return path;
}

function snapshotDirectoryTreeMetadata(root: string): readonly PathIdentity[] {
	const identities: PathIdentity[] = [];
	const visit = (directory: string): void => {
		for (const name of readdirSync(directory).sort(compareCodePoints)) {
			const path = join(directory, name);
			const identity = snapshotPathIdentity(path, lstatSync(path).isDirectory() ? "directory" : "file", root);
			identities.push(identity);
			if (identity.kind === "directory") {
				visit(identity.path);
			}
		}
	};
	visit(root);
	return identities;
}

async function hashDirectoryTree(root: string, identities: readonly PathIdentity[]): Promise<string> {
	const hash = createHash("sha256");
	for (const identity of identities) {
		const relativePath = relative(root, identity.path);
		hash.update(
			`${identity.kind}\0${relativePath}\0${identity.device}\0${identity.inode}\0${identity.size}\0${identity.modifiedAtMs}\0`,
		);
		if (identity.kind !== "file") continue;
		const stream = createReadStream(identity.path, { highWaterMark: 64 * 1024 });
		for await (const chunk of stream) {
			hash.update(chunk);
		}
	}
	return hash.digest("hex");
}

function samePathIdentity(left: PathIdentity, right: PathIdentity): boolean {
	return (
		left.path === right.path &&
		left.device === right.device &&
		left.inode === right.inode &&
		left.size === right.size &&
		left.modifiedAtMs === right.modifiedAtMs &&
		left.kind === right.kind
	);
}

function sameFileIntegrity(left: FileIntegrity, right: FileIntegrity): boolean {
	return samePathIdentity(left, right) && left.sha256 === right.sha256;
}

function samePathIdentityList(left: readonly PathIdentity[], right: readonly PathIdentity[]): boolean {
	return (
		left.length === right.length &&
		left.every((identity, index) => {
			const expected = right[index];
			return expected !== undefined && samePathIdentity(identity, expected);
		})
	);
}

function isContainedPath(path: string, root: string): boolean {
	const descendant = relative(root, path);
	return descendant !== ".." && !descendant.startsWith(`..${sep}`) && !isAbsolute(descendant);
}

function parseCheckedMinSyncHits(stdout: string): readonly MinSyncQueryHit[] {
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		throw new Error("MinSync benchmark retrieval failed");
	}
	const candidates = isRecord(parsed) ? parsed.results : parsed;
	if (!Array.isArray(candidates) || !candidates.every(isMinSyncQueryHit)) {
		throw new Error("MinSync benchmark retrieval failed");
	}
	return candidates;
}

function isMinSyncQueryHit(value: unknown): value is MinSyncQueryHit {
	if (!isRecord(value)) return false;
	return (
		typeof value.path === "string" &&
		typeof value.score === "number" &&
		Number.isFinite(value.score) &&
		typeof value.text === "string"
	);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function compareRetrievalResults(left: RetrievalResult, right: RetrievalResult): number {
	if (!Number.isFinite(left.score) || !Number.isFinite(right.score)) {
		throw new Error("Hybrid benchmark retrieval failed");
	}
	return (
		right.score - left.score || compareCodePoints(left.source, right.source) || compareCodePoints(left.id, right.id)
	);
}

function compareCodePoints(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}

function normalizeMethodConfig(raw: unknown, env: NodeJS.ProcessEnv): BenchmarkMinSyncConfig {
	const record = requireRecord(raw, "MinSync benchmark configuration");
	rejectUnknownFields(record, CONFIG_FIELDS, "MinSync benchmark configuration");
	if (record.binaryPath === undefined) {
		throw new Error("MinSync benchmark binaryPath is required");
	}
	if (typeof record.binaryPath !== "string" || record.binaryPath.trim().length === 0) {
		throw new Error("MinSync benchmark binaryPath must be a non-empty string");
	}
	if (record.autoInstall !== false) {
		throw new Error("MinSync benchmark autoInstall must be false");
	}
	const embedderRecord = requireRecord(record.embedder, "MinSync benchmark embedder");
	rejectUnknownFields(embedderRecord, EMBEDDER_FIELDS, "MinSync benchmark embedder");

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
		dimension: requirePositiveSafeInteger(embedderRecord.dimension, "embedder.dimension"),
	};
	const embedderId = embedderRecord.id;
	if (embedderId === undefined) {
		throw new Error("MinSync benchmark embedder.id is required");
	}
	if (typeof embedderId !== "string" || embedderId.trim().length === 0) {
		throw new Error("MinSync benchmark embedder.id must be a non-empty string");
	}
	embedder.id = requirePortableEmbedderId(embedderId);
	const baseUrl = embedderRecord.baseUrl;
	if (baseUrl !== undefined) {
		if (typeof baseUrl !== "string" || baseUrl.trim().length === 0) {
			throw new Error("MinSync benchmark embedder.baseUrl must be a non-empty string");
		}
		embedder.baseUrl = baseUrl;
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
			embedder[field] = requirePositiveSafeInteger(value, `embedder.${field}`);
		}
	}
	if (embedder.timeoutMs === undefined) {
		throw new Error("MinSync benchmark embedder.timeoutMs is required");
	}
	if (embedderRecord.apiKeyEnv !== undefined) {
		if (typeof embedderRecord.apiKeyEnv !== "string" || !API_KEY_ENV_PATTERN.test(embedderRecord.apiKeyEnv)) {
			throw new Error("MinSync benchmark embedder.apiKeyEnv must be a valid environment-variable name");
		}
		const apiKeyEnv = embedderRecord.apiKeyEnv;
		if (!Object.hasOwn(env, apiKeyEnv) || typeof env[apiKeyEnv] !== "string" || env[apiKeyEnv]?.length === 0) {
			throw new Error(`MinSync benchmark requires environment variable ${apiKeyEnv}`);
		}
		embedder.apiKeyEnv = apiKeyEnv;
	}

	const normalized: {
		binaryPath: string;
		autoInstall: false;
		embedder: MinSyncEmbedderConfig;
	} = { binaryPath: record.binaryPath, autoInstall: false, embedder };
	return normalized as BenchmarkMinSyncConfig;
}

function requirePortableEmbedderId(value: string): string {
	if (/^(?:[\\/]|[A-Za-z]:[\\/]|file:)/iu.test(value)) {
		throw new Error("embedder.id must not be an absolute filesystem path");
	}
	return value;
}

function validateMethodNames(names: readonly BenchmarkMethod[]): ReadonlySet<BenchmarkMethod> {
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
	if (names.size !== 2 || !names.has("bm25") || !names.has("minsync")) {
		throw new Error("Hybrid benchmark requires BM25 and MinSync methods");
	}
}

function validateTopK(topK: number): void {
	if (!Number.isSafeInteger(topK) || topK < 1 || topK > RETRIEVAL_OVERFETCH_LIMIT) {
		throw new Error(`topK must be a safe integer between 1 and ${RETRIEVAL_OVERFETCH_LIMIT}`);
	}
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`${label} must be an object`);
	}
	return value as Record<string, unknown>;
}

function rejectUnknownFields(record: Record<string, unknown>, allowlist: ReadonlySet<string>, label: string): void {
	for (const field of Object.keys(record)) {
		if (!allowlist.has(field)) {
			throw new Error(`${label}.${field} is not a recognized field`);
		}
	}
}

function requirePositiveSafeInteger(value: unknown, field: string): number {
	if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
		throw new Error(`MinSync benchmark ${field} must be a positive safe integer`);
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
		throw new Error("MinSync benchmark embedder.baseUrl must be a valid HTTP endpoint");
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
