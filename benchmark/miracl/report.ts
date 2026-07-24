import { randomUUID } from "node:crypto";
import type { FileHandle } from "node:fs/promises";
import { chmod, lstat, mkdir, open, realpath, rename, rm, unlink } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import type { SanitizedMethodConfig } from "./methods.ts";
import type { MethodMetrics } from "./metrics.ts";
import type { BenchmarkMethod, BenchmarkProfile, QueryRunRecord, RankedHit } from "./types.ts";

const REPORT_FILES = ["manifest.json", "results.jsonl", "metrics.json", "summary.md"] as const;
const METHOD_NAMES = new Set<BenchmarkMethod>(["bm25", "minsync", "hybrid"]);
const ENV_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

export interface RunDatasetManifest {
	readonly normalizationVersion: number;
	readonly revisions: {
		readonly topics: string;
		readonly corpus: string;
	};
	readonly seed?: number;
	readonly counts: Readonly<Record<string, number>>;
}

export interface RunEnvironmentManifest {
	readonly autoRagCommit: string;
	readonly platform: string;
	readonly architecture: string;
	readonly node: string;
	readonly bun?: string;
	readonly measuredAt: string;
}

export interface RunManifestInput {
	readonly profile: BenchmarkProfile;
	readonly preparedDirectory: string;
	readonly dataset: RunDatasetManifest;
	readonly methods: readonly BenchmarkMethod[];
	readonly methodConfig?: SanitizedMethodConfig;
	readonly environment: RunEnvironmentManifest;
}

export interface RunManifestV1 extends RunManifestInput {
	readonly schemaVersion: 1;
	readonly methods: readonly BenchmarkMethod[];
	readonly methodConfig?: SanitizedMethodConfig;
}

export interface RunMetricsV1 {
	readonly schemaVersion: 1;
	readonly methods: readonly MethodMetrics[];
	readonly indexingLatencyMs: Readonly<Partial<Record<"bm25" | "minsync", number>>>;
	readonly peakRssBytes?: number;
}

export interface WriteRunReportOptions {
	readonly directory: string;
	readonly manifest: RunManifestInput;
	readonly records: readonly QueryRunRecord[];
	readonly metrics: readonly MethodMetrics[];
	readonly indexingLatencyMs?: Readonly<Partial<Record<"bm25" | "minsync", number>>>;
	readonly peakRssBytes?: number;
}

interface DirectoryIdentity {
	readonly device: number;
	readonly inode: number;
}

interface PublicationPaths {
	readonly parent: string;
	readonly destination: string;
	readonly staging: string;
	readonly lock: string;
}

export async function writeRunReport(options: WriteRunReportOptions): Promise<void> {
	const paths = await resolvePublicationPaths(options.directory);
	const manifest = normalizeRunManifest(options.manifest);
	const records = [...options.records]
		.map((record) => validateQueryRunRecord(record))
		.sort(compareRecords)
		.map(stabilizeRecord);
	const metrics = normalizeRunMetrics({
		schemaVersion: 1,
		methods: options.metrics,
		indexingLatencyMs: options.indexingLatencyMs ?? {},
		peakRssBytes: options.peakRssBytes,
	});
	validateReportCoherence(manifest, records, metrics);
	const contents = new Map<(typeof REPORT_FILES)[number], string>([
		["manifest.json", `${JSON.stringify(manifest)}\n`],
		["results.jsonl", records.map((record) => JSON.stringify(record)).join("\n") + (records.length > 0 ? "\n" : "")],
		["metrics.json", `${JSON.stringify(metrics)}\n`],
		["summary.md", renderSummary(manifest, metrics)],
	]);

	let lockHandle: FileHandle | undefined;
	let stagingIdentity: DirectoryIdentity | undefined;
	let published = false;
	try {
		lockHandle = await acquirePublicationLock(paths);
		await assertDestinationAbsent(paths.destination);
		await mkdir(paths.staging, { mode: 0o700 });
		await chmod(paths.staging, 0o700);
		stagingIdentity = await snapshotDirectory(paths.staging);
		for (const name of REPORT_FILES) {
			await assertDirectoryIdentity(paths.staging, stagingIdentity);
			await writeDurablePrivateFile(join(paths.staging, name), contents.get(name) as string);
			await assertDirectoryIdentity(paths.staging, stagingIdentity);
		}
		await fsyncDirectory(paths.staging);
		await assertDirectoryIdentity(paths.staging, stagingIdentity);
		await assertDestinationAbsent(paths.destination);
		await renameDirectoryNoReplace(paths.staging, paths.destination);
		await assertDirectoryIdentity(paths.destination, stagingIdentity);
		published = true;
		await fsyncDirectory(paths.parent);
	} finally {
		if (!published && stagingIdentity !== undefined) {
			await removeOwnedDirectory(paths.staging, stagingIdentity);
		}
		if (lockHandle !== undefined) {
			await releasePublicationLock(paths.lock, lockHandle);
		}
	}
}

export function normalizeRunManifest(value: RunManifestInput): RunManifestV1 {
	if (value.profile !== "smoke" && value.profile !== "full") {
		throw new Error("run manifest profile must be smoke or full");
	}
	const preparedDirectory = requireNonBlank(value.preparedDirectory, "run manifest preparedDirectory");
	const normalizationVersion = requireNonNegativeSafeInteger(
		value.dataset?.normalizationVersion,
		"run manifest normalizationVersion",
	);
	const revisions = {
		topics: requireNonBlank(value.dataset?.revisions?.topics, "run manifest topics revision"),
		corpus: requireNonBlank(value.dataset?.revisions?.corpus, "run manifest corpus revision"),
	};
	const counts: Record<string, number> = {};
	for (const key of [
		"queries",
		"qrels",
		"positiveQrels",
		"corpus",
		"judgedDocuments",
		...(value.profile === "smoke" ? ["distractors"] : []),
	]) {
		counts[key] = requireNonNegativeSafeInteger(value.dataset?.counts?.[key], `run manifest counts.${key}`);
	}
	const methods = normalizeMethodNames(value.methods);
	const environment = normalizeEnvironment(value.environment);
	const dataset: {
		normalizationVersion: number;
		revisions: { topics: string; corpus: string };
		seed?: number;
		counts: Record<string, number>;
	} = { normalizationVersion, revisions, counts };
	if (value.profile === "smoke") {
		dataset.seed = requireSafeInteger(value.dataset.seed, "run manifest seed");
	} else if (value.dataset.seed !== undefined) {
		throw new Error("full run manifest must not contain a seed");
	}
	const methodConfig = value.methodConfig === undefined ? undefined : normalizeMethodConfig(value.methodConfig);
	return {
		schemaVersion: 1,
		profile: value.profile,
		preparedDirectory,
		dataset,
		methods,
		environment,
		...(methodConfig === undefined ? {} : { methodConfig }),
	};
}

export function normalizeRunMetrics(value: RunMetricsV1): RunMetricsV1 {
	if (value.schemaVersion !== 1) {
		throw new Error("metrics schemaVersion must be 1");
	}
	if (!isRecord(value.indexingLatencyMs)) {
		throw new Error("indexingLatencyMs must be an object");
	}
	assertExactKeys(
		value.indexingLatencyMs,
		new Set(Object.keys(value.indexingLatencyMs).filter((key) => key === "bm25" || key === "minsync")),
		"indexingLatencyMs",
	);
	for (const key of Object.keys(value.indexingLatencyMs)) {
		if (key !== "bm25" && key !== "minsync") {
			throw new Error(`indexingLatencyMs has unknown field ${key}`);
		}
	}
	const methods = [...value.methods]
		.map(validateMethodMetrics)
		.sort((left, right) => compareCodePoints(left.method, right.method));
	const methodNames = new Set<BenchmarkMethod>();
	for (const method of methods) {
		if (methodNames.has(method.method)) {
			throw new Error(`duplicate metrics method ${method.method}`);
		}
		methodNames.add(method.method);
	}
	const indexingLatencyMs: Partial<Record<"bm25" | "minsync", number>> = {};
	for (const method of ["bm25", "minsync"] as const) {
		const valueForMethod = value.indexingLatencyMs[method];
		if (valueForMethod !== undefined) {
			indexingLatencyMs[method] = requireFiniteNonNegative(valueForMethod, `indexingLatencyMs.${method}`);
		}
	}
	const normalized: {
		schemaVersion: 1;
		methods: readonly MethodMetrics[];
		indexingLatencyMs: Partial<Record<"bm25" | "minsync", number>>;
		peakRssBytes?: number;
	} = { schemaVersion: 1, methods, indexingLatencyMs };
	if (value.peakRssBytes !== undefined) {
		normalized.peakRssBytes = requirePositiveSafeInteger(value.peakRssBytes, "peakRssBytes");
	}
	return normalized;
}

export function validateQueryRunRecord(value: unknown): QueryRunRecord {
	if (!isRecord(value)) throw new Error("run record must be an object");
	const allowed = new Set(["schemaVersion", "method", "queryId", "latencyMs", "hits", "errorCode"]);
	assertExactKeys(value, allowed, "run record");
	if (value.schemaVersion !== 1) {
		throw new Error("record schemaVersion must be 1");
	}
	const method = requireMethod(value.method, "record method");
	const queryId = requireNonBlank(value.queryId, "record queryId");
	const latencyMs = requireFiniteNonNegative(value.latencyMs, "record latencyMs");
	if (!Array.isArray(value.hits)) throw new Error("record hits must be an array");
	const hits = value.hits.map((hit, index) => validateRankedHit(hit, index));
	const documentIds = new Set<string>();
	const ranks = new Set<number>();
	for (const hit of hits) {
		if (documentIds.has(hit.documentId)) {
			throw new Error(`duplicate hit document ${hit.documentId}`);
		}
		if (ranks.has(hit.rank)) throw new Error(`duplicate hit rank ${hit.rank}`);
		documentIds.add(hit.documentId);
		ranks.add(hit.rank);
	}
	if (value.errorCode !== undefined && value.errorCode !== "retrieval-failed") {
		throw new Error("record errorCode is invalid");
	}
	if (value.errorCode === "retrieval-failed" && hits.length !== 0) {
		throw new Error("failed run record must not contain hits");
	}
	return {
		schemaVersion: 1,
		method,
		queryId,
		latencyMs,
		hits,
		...(value.errorCode === "retrieval-failed" ? { errorCode: "retrieval-failed" as const } : {}),
	};
}

export function validateMethodMetrics(value: unknown): MethodMetrics {
	if (!isRecord(value)) throw new Error("method metrics must be an object");
	assertExactKeys(
		value,
		new Set(["method", "queryCount", "failureCount", "recallAt", "mrrAt10", "successAt", "ndcgAt10", "latencyMs"]),
		"method metrics",
	);
	const method = requireMethod(value.method, "metrics method");
	const queryCount = requireNonNegativeSafeInteger(value.queryCount, "metrics queryCount");
	const failureCount = requireNonNegativeSafeInteger(value.failureCount, "metrics failureCount");
	if (failureCount > queryCount) {
		throw new Error("metrics failureCount exceeds queryCount");
	}
	const recallAt = normalizeCutoffMap(value.recallAt, ["5", "10", "100"], "recallAt");
	const successAt = normalizeCutoffMap(value.successAt, ["1", "5"], "successAt");
	if (!isRecord(value.latencyMs)) {
		throw new Error("metrics latencyMs must be an object");
	}
	assertExactKeys(value.latencyMs, new Set(["mean", "p50", "p95"]), "metrics latencyMs");
	return {
		method,
		queryCount,
		failureCount,
		recallAt,
		mrrAt10: requireUnitInterval(value.mrrAt10, "metrics mrrAt10"),
		successAt,
		ndcgAt10: requireUnitInterval(value.ndcgAt10, "metrics ndcgAt10"),
		latencyMs: {
			mean: requireFiniteNonNegative(value.latencyMs.mean, "metrics latency mean"),
			p50: requireFiniteNonNegative(value.latencyMs.p50, "metrics latency p50"),
			p95: requireFiniteNonNegative(value.latencyMs.p95, "metrics latency p95"),
		},
	};
}

function validateRankedHit(value: unknown, index: number): RankedHit {
	if (!isRecord(value)) throw new Error(`record hit ${index} must be an object`);
	assertExactKeys(value, new Set(["documentId", "score", "rank"]), `record hit ${index}`);
	return {
		documentId: requireNonBlank(value.documentId, `record hit ${index} documentId`),
		score: requireFinite(value.score, `record hit ${index} score`),
		rank: requirePositiveSafeInteger(value.rank, `record hit ${index} rank`),
	};
}

function normalizeCutoffMap<K extends string>(value: unknown, keys: readonly K[], label: string): Record<K, number> {
	if (!isRecord(value)) throw new Error(`metrics ${label} must be an object`);
	assertExactKeys(value, new Set(keys), `metrics ${label}`);
	return Object.fromEntries(
		keys.map((key) => [key, requireUnitInterval(value[key], `metrics ${label}.${key}`)]),
	) as Record<K, number>;
}

function normalizeMethodNames(values: readonly BenchmarkMethod[]): BenchmarkMethod[] {
	if (!Array.isArray(values) || values.length === 0) {
		throw new Error("run manifest methods must be a non-empty array");
	}
	const names = values.map((value) => requireMethod(value, "run manifest method"));
	if (new Set(names).size !== names.length) {
		throw new Error("run manifest methods must not contain duplicates");
	}
	return names.sort(compareCodePoints);
}

function normalizeMethodConfig(value: SanitizedMethodConfig): SanitizedMethodConfig {
	if (!isRecord(value)) throw new Error("methodConfig must be an object");
	if (value.endpointKind !== "local" && value.endpointKind !== "remote") {
		throw new Error("methodConfig endpointKind must be local or remote");
	}
	const normalized: {
		embedderId?: string;
		endpointKind: "local" | "remote";
		apiKeyEnv?: string;
		dimension: number;
	} = {
		endpointKind: value.endpointKind,
		dimension: requirePositiveSafeInteger(value.dimension, "methodConfig dimension"),
	};
	if (value.embedderId !== undefined) {
		normalized.embedderId = requireOpaqueDisclosure(value.embedderId, "methodConfig embedderId");
	}
	if (value.apiKeyEnv !== undefined) {
		const apiKeyEnv = requireNonBlank(value.apiKeyEnv, "methodConfig apiKeyEnv");
		if (!ENV_NAME_PATTERN.test(apiKeyEnv)) {
			throw new Error("methodConfig apiKeyEnv must be an environment variable name");
		}
		normalized.apiKeyEnv = apiKeyEnv;
	}
	return normalized;
}

function normalizeEnvironment(value: RunEnvironmentManifest): RunEnvironmentManifest {
	if (!isRecord(value)) throw new Error("run manifest environment must be an object");
	const normalized: {
		autoRagCommit: string;
		platform: string;
		architecture: string;
		node: string;
		bun?: string;
		measuredAt: string;
	} = {
		autoRagCommit: requireOpaqueDisclosure(value.autoRagCommit, "environment autoRagCommit"),
		platform: requireOpaqueDisclosure(value.platform, "environment platform"),
		architecture: requireOpaqueDisclosure(value.architecture, "environment architecture"),
		node: requireOpaqueDisclosure(value.node, "environment node"),
		measuredAt: requireNonBlank(value.measuredAt, "environment measuredAt"),
	};
	if (!Number.isFinite(Date.parse(normalized.measuredAt))) {
		throw new Error("environment measuredAt must be an ISO timestamp");
	}
	if (value.bun !== undefined) {
		normalized.bun = requireOpaqueDisclosure(value.bun, "environment bun");
	}
	return normalized;
}

function validateReportCoherence(
	manifest: RunManifestV1,
	records: readonly QueryRunRecord[],
	metrics: RunMetricsV1,
): void {
	const manifestMethods = new Set(manifest.methods);
	const recordMethods = new Set(records.map((record) => record.method));
	const metricMethods = new Set(metrics.methods.map((metric) => metric.method));
	for (const method of manifestMethods) {
		if (!recordMethods.has(method)) {
			throw new Error(`run report has no records for method ${method}`);
		}
		if (!metricMethods.has(method)) {
			throw new Error(`run report has no metrics for method ${method}`);
		}
	}
	for (const method of [...recordMethods, ...metricMethods]) {
		if (!manifestMethods.has(method)) {
			throw new Error(`run report contains undeclared method ${method}`);
		}
	}
	const pairs = new Set<string>();
	for (const record of records) {
		const pair = `${record.method}\0${record.queryId}`;
		if (pairs.has(pair)) {
			throw new Error(`duplicate query-method record for ${record.method}/${record.queryId}`);
		}
		pairs.add(pair);
	}
	for (const metric of metrics.methods) {
		const methodRecords = records.filter((record) => record.method === metric.method);
		const failures = methodRecords.filter((record) => record.errorCode !== undefined).length;
		if (metric.queryCount !== methodRecords.length || metric.failureCount !== failures) {
			throw new Error(`metrics counts do not match records for ${metric.method}`);
		}
	}
}

function stabilizeRecord(record: QueryRunRecord): QueryRunRecord {
	return {
		...record,
		hits: [...record.hits].sort(
			(left, right) => left.rank - right.rank || compareCodePoints(left.documentId, right.documentId),
		),
	};
}

function renderSummary(manifest: RunManifestV1, metrics: RunMetricsV1): string {
	const lines = [
		"# MIRACL Korean Retrieval Benchmark",
		"",
		"## Dataset",
		"",
		`- Profile: ${markdown(manifest.profile)}`,
		`- Topics revision: ${markdown(manifest.dataset.revisions.topics)}`,
		`- Corpus revision: ${markdown(manifest.dataset.revisions.corpus)}`,
		`- Queries: ${manifest.dataset.counts.queries}`,
		`- Corpus passages: ${manifest.dataset.counts.corpus}`,
		"",
		"## Method configuration",
		"",
		"| Field | Value |",
		"| --- | --- |",
	];
	if (manifest.methodConfig === undefined) {
		lines.push("| MinSync embedder | Not used |");
	} else {
		lines.push(
			`| Embedder ID | ${markdown(manifest.methodConfig.embedderId ?? "not disclosed")} |`,
			`| Endpoint kind | ${manifest.methodConfig.endpointKind} |`,
			`| API key environment variable | ${markdown(manifest.methodConfig.apiKeyEnv ?? "none")} |`,
			`| Dimension | ${manifest.methodConfig.dimension} |`,
		);
	}
	lines.push(
		"",
		"## Quality",
		"",
		"| Method | nDCG@10 | Recall@5 | Recall@10 | Recall@100 | MRR@10 | Success@1 | Success@5 | Failed queries |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	);
	for (const metric of metrics.methods) {
		lines.push(
			`| ${metric.method} | ${formatMetric(metric.ndcgAt10)} | ${formatMetric(metric.recallAt["5"])} | ${formatMetric(metric.recallAt["10"])} | ${formatMetric(metric.recallAt["100"])} | ${formatMetric(metric.mrrAt10)} | ${formatMetric(metric.successAt["1"])} | ${formatMetric(metric.successAt["5"])} | ${metric.failureCount} |`,
		);
	}
	lines.push(
		"",
		"## Performance",
		"",
		"| Method | Indexing ms | Query mean ms | Query p50 ms | Query p95 ms |",
		"| --- | ---: | ---: | ---: | ---: |",
	);
	for (const metric of metrics.methods) {
		const indexing =
			metric.method === "hybrid"
				? hybridIndexingLatency(metrics.indexingLatencyMs)
				: metrics.indexingLatencyMs[metric.method];
		lines.push(
			`| ${metric.method} | ${indexing === undefined ? "n/a" : formatMilliseconds(indexing)} | ${formatMilliseconds(metric.latencyMs.mean)} | ${formatMilliseconds(metric.latencyMs.p50)} | ${formatMilliseconds(metric.latencyMs.p95)} |`,
		);
	}
	lines.push(
		"",
		`Peak process RSS: ${metrics.peakRssBytes === undefined ? "unavailable" : `${metrics.peakRssBytes} bytes`}.`,
		"",
		"## Limitations",
		"",
		"- Query failures are scored as zero for quality metrics and excluded from latency statistics.",
		"- Peak RSS is process-wide and is disclosed only when the runtime reports a reliable maximum.",
		"- MinSync integrity checks run outside measured query intervals and may affect filesystem cache state.",
		"- Results from different embedders or endpoint kinds are not directly comparable.",
		"",
	);
	return `${lines.join("\n")}\n`;
}

function hybridIndexingLatency(values: Readonly<Partial<Record<"bm25" | "minsync", number>>>): number | undefined {
	if (values.bm25 === undefined || values.minsync === undefined) return undefined;
	const total = values.bm25 + values.minsync;
	return Number.isFinite(total) ? total : undefined;
}

async function resolvePublicationPaths(directory: string): Promise<PublicationPaths> {
	const requested = requireNonBlank(directory, "report directory");
	const absolute = resolve(requested);
	const requestedParent = dirname(absolute);
	let parent: string;
	try {
		parent = await realpath(requestedParent);
	} catch {
		throw new Error("report parent directory must already exist");
	}
	const parentStats = await lstat(parent);
	if (!parentStats.isDirectory() || parentStats.isSymbolicLink()) {
		throw new Error("report parent must be a real directory");
	}
	const destination = join(parent, basename(absolute));
	return {
		parent,
		destination,
		staging: `${destination}.staging-${process.pid}-${randomUUID()}`,
		lock: `${destination}.publish.lock`,
	};
}

async function acquirePublicationLock(paths: PublicationPaths): Promise<FileHandle> {
	try {
		const handle = await open(paths.lock, "wx", 0o600);
		await handle.sync();
		return handle;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "EEXIST") {
			throw new Error(`report directory already exists or is being published: ${paths.destination}`);
		}
		throw error;
	}
}

async function releasePublicationLock(path: string, handle: FileHandle): Promise<void> {
	const owned = await handle.stat();
	await handle.close();
	const quarantine = `${path}.cleanup-${process.pid}-${randomUUID()}`;
	try {
		await rename(path, quarantine);
		const current = await lstat(quarantine);
		if (current.isFile() && !current.isSymbolicLink() && current.dev === owned.dev && current.ino === owned.ino) {
			await unlink(quarantine);
			return;
		}
		try {
			await rename(quarantine, path);
		} catch {
			// Preserve a replacement under the quarantine name when the lock
			// pathname was concurrently claimed.
		}
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
}

async function writeDurablePrivateFile(path: string, contents: string): Promise<void> {
	const handle = await open(path, "wx", 0o600);
	try {
		await handle.writeFile(contents, "utf8");
		await handle.sync();
	} finally {
		await handle.close();
	}
	await chmod(path, 0o600);
	const stats = await lstat(path);
	if (!stats.isFile() || stats.isSymbolicLink()) {
		throw new Error("report file changed during staging");
	}
}

async function fsyncDirectory(path: string): Promise<void> {
	const handle = await open(path, "r");
	try {
		await handle.sync();
	} finally {
		await handle.close();
	}
}

interface BunFfiModule {
	readonly dlopen: (
		path: string,
		symbols: Record<string, { readonly args: readonly string[]; readonly returns: string }>,
	) => {
		readonly symbols: Record<string, (...args: unknown[]) => number>;
		close(): void;
	};
	readonly ptr: (buffer: Uint8Array) => unknown;
}

async function renameDirectoryNoReplace(source: string, destination: string): Promise<void> {
	if (process.versions.bun === undefined || (process.platform !== "darwin" && process.platform !== "linux")) {
		// The benchmark CLI runs under Bun. Node's Windows directory rename is
		// already non-replacing; Node-based unit tests use the adjacent lock to
		// serialize conforming publishers.
		await rename(source, destination);
		return;
	}
	const runtimeImport = (specifier: string): Promise<unknown> => import(specifier);
	const ffi = (await runtimeImport("bun:ffi")) as BunFfiModule;
	const sourceBytes = Buffer.from(`${source}\0`, "utf8");
	const destinationBytes = Buffer.from(`${destination}\0`, "utf8");
	let result: number;
	if (process.platform === "darwin") {
		const library = ffi.dlopen("/usr/lib/libSystem.B.dylib", {
			renamex_np: { args: ["ptr", "ptr", "u32"], returns: "i32" },
		});
		try {
			result = library.symbols.renamex_np!(ffi.ptr(sourceBytes), ffi.ptr(destinationBytes), 4);
		} finally {
			library.close();
		}
	} else {
		const library = ffi.dlopen("libc.so.6", {
			renameat2: {
				args: ["i32", "ptr", "i32", "ptr", "u32"],
				returns: "i32",
			},
		});
		try {
			result = library.symbols.renameat2!(-100, ffi.ptr(sourceBytes), -100, ffi.ptr(destinationBytes), 1);
		} finally {
			library.close();
		}
	}
	if (result !== 0) {
		try {
			await lstat(destination);
			throw new Error(`report directory already exists: ${destination}`);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		}
		throw new Error("atomic no-replace report publication failed");
	}
}

async function assertDestinationAbsent(path: string): Promise<void> {
	try {
		await lstat(path);
		throw new Error(`report directory already exists: ${path}`);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
}

async function snapshotDirectory(path: string): Promise<DirectoryIdentity> {
	const stats = await lstat(path);
	if (!stats.isDirectory() || stats.isSymbolicLink()) {
		throw new Error("report staging directory changed");
	}
	const canonical = await realpath(path);
	if (canonical !== path) throw new Error("report staging directory escaped");
	return { device: stats.dev, inode: stats.ino };
}

async function assertDirectoryIdentity(path: string, identity: DirectoryIdentity): Promise<void> {
	const current = await snapshotDirectory(path);
	if (current.device !== identity.device || current.inode !== identity.inode) {
		throw new Error("report staging directory changed");
	}
}

async function removeOwnedDirectory(path: string, identity: DirectoryIdentity): Promise<void> {
	try {
		await assertDirectoryIdentity(path, identity);
	} catch {
		return;
	}
	const quarantine = `${path}.cleanup-${process.pid}-${randomUUID()}`;
	await rename(path, quarantine);
	let moved: DirectoryIdentity;
	try {
		moved = await snapshotDirectory(quarantine);
	} catch {
		return;
	}
	if (moved.device === identity.device && moved.inode === identity.inode) {
		await rm(quarantine, { recursive: true, force: true });
		return;
	}
	try {
		await rename(quarantine, path);
	} catch {
		// Preserve a replacement under the quarantine name when the original
		// pathname was concurrently claimed.
	}
}

function compareRecords(left: QueryRunRecord, right: QueryRunRecord): number {
	return compareCodePoints(left.method, right.method) || compareCodePoints(left.queryId, right.queryId);
}

function requireMethod(value: unknown, label: string): BenchmarkMethod {
	if (typeof value !== "string" || !METHOD_NAMES.has(value as BenchmarkMethod)) {
		throw new Error(`${label} is invalid`);
	}
	return value as BenchmarkMethod;
}

function requireNonBlank(value: unknown, label: string): string {
	if (typeof value !== "string" || value.trim().length === 0) {
		throw new Error(`${label} must be non-blank`);
	}
	return value;
}

function requireOpaqueDisclosure(value: unknown, label: string): string {
	const text = requireNonBlank(value, label);
	if (text.includes("://") || /[\r\n|]/u.test(text)) {
		throw new Error(`${label} must be an opaque value`);
	}
	return text;
}

function requireSafeInteger(value: unknown, label: string): number {
	if (!Number.isSafeInteger(value)) throw new Error(`${label} must be a safe integer`);
	return value as number;
}

function requireNonNegativeSafeInteger(value: unknown, label: string): number {
	const number = requireSafeInteger(value, label);
	if (number < 0) throw new Error(`${label} must be non-negative`);
	return number;
}

function requirePositiveSafeInteger(value: unknown, label: string): number {
	const number = requireSafeInteger(value, label);
	if (number < 1) throw new Error(`${label} must be positive`);
	return number;
}

function requireFinite(value: unknown, label: string): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		throw new Error(`${label} must be finite`);
	}
	return value;
}

function requireFiniteNonNegative(value: unknown, label: string): number {
	const number = requireFinite(value, label);
	if (number < 0) throw new Error(`${label} must be non-negative`);
	return number;
}

function requireUnitInterval(value: unknown, label: string): number {
	const number = requireFiniteNonNegative(value, label);
	if (number > 1) throw new Error(`${label} must be at most 1`);
	return number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertExactKeys(value: Record<string, unknown>, allowed: ReadonlySet<string>, label: string): void {
	for (const key of Object.keys(value)) {
		if (!allowed.has(key)) throw new Error(`${label} has unknown field ${key}`);
	}
	for (const key of allowed) {
		if (key === "errorCode") continue;
		if (!(key in value)) throw new Error(`${label} is missing field ${key}`);
	}
}

function markdown(value: string): string {
	return value.replaceAll("|", "\\|").replaceAll("\r", " ").replaceAll("\n", " ");
}

function formatMetric(value: number): string {
	return value.toFixed(6);
}

function formatMilliseconds(value: number): string {
	return value.toFixed(3);
}

function compareCodePoints(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}
