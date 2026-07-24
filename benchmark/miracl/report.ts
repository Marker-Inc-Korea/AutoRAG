import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { FileHandle } from "node:fs/promises";
import { lstat, mkdir, open, realpath } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import type { SanitizedMethodConfig } from "./methods.ts";
import { evaluateRun, type MethodMetrics } from "./metrics.ts";
import {
	MIRACL_FULL_CORPUS_PASSAGES,
	MIRACL_NORMALIZATION_VERSION,
	MIRACL_SOURCES,
} from "./profiles.ts";
import type { BenchmarkMethod, Qrel, QueryRunRecord, RankedHit } from "./types.ts";

const REPORT_FILES = ["manifest.json", "results.jsonl", "metrics.json", "summary.md"] as const;
const METHOD_NAMES = new Set<BenchmarkMethod>(["bm25", "minsync", "hybrid"]);
const ENV_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const MAX_EMBEDDED_QRELS = 10_000;
const MAX_DISCLOSED_ID_BYTES = 128;
const MAX_MANIFEST_CANONICAL_BYTES = 4 * 1_024 * 1_024;

export interface RunFileAttestation {
	readonly sha256: string;
	readonly bytes: number;
}

export interface RunNormalizedAttestation extends RunFileAttestation {
	readonly records: number;
}

export interface RunEvaluationV1 {
	readonly schemaVersion: 1;
	readonly qrels: readonly Qrel[];
}

interface RunDatasetManifestBase {
	readonly normalizationVersion: number;
	readonly revisions: {
		readonly topics: string;
		readonly corpus: string;
	};
	readonly input: {
		readonly topics: RunFileAttestation;
		readonly qrels: RunFileAttestation;
		readonly corpus: readonly RunFileAttestation[];
	};
	readonly normalized: {
		readonly queries: RunNormalizedAttestation;
		readonly qrels: RunNormalizedAttestation;
		readonly corpus: RunNormalizedAttestation;
	};
	readonly evaluation: RunEvaluationV1;
}

export interface SmokeRunDatasetManifest extends RunDatasetManifestBase {
	readonly seed: number;
	readonly counts: {
		readonly queries: number;
		readonly qrels: number;
		readonly positiveQrels: number;
		readonly corpus: number;
		readonly judgedDocuments: number;
		readonly distractors: number;
	};
}

export interface FullRunDatasetManifest extends RunDatasetManifestBase {
	readonly counts: {
		readonly queries: number;
		readonly qrels: number;
		readonly positiveQrels: number;
		readonly corpus: number;
		readonly judgedDocuments: number;
	};
}

export type RunDatasetManifest = SmokeRunDatasetManifest | FullRunDatasetManifest;

export interface RunEnvironmentManifest {
	readonly autoRagCommit: string;
	readonly platform: string;
	readonly architecture: string;
	readonly node: string;
	readonly bun?: string;
	readonly measuredAt: string;
}

interface RunManifestInputBase {
	readonly schemaVersion: 1;
	readonly methods: readonly BenchmarkMethod[];
	readonly methodConfig?: SanitizedMethodConfig;
	readonly environment: RunEnvironmentManifest;
}

export type RunManifestInput =
	| (RunManifestInputBase & {
			readonly profile: "smoke";
			readonly dataset: SmokeRunDatasetManifest;
	  })
	| (RunManifestInputBase & {
			readonly profile: "full";
			readonly dataset: FullRunDatasetManifest;
	  });

export type RunManifestV1 = RunManifestInput;

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

interface FileIdentity extends DirectoryIdentity {
	readonly size: number;
}

interface PublicationPaths {
	readonly parent: string;
	readonly destination: string;
	readonly staging: string;
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
	validateCompleteGridAndMetrics(manifest, records, metrics);
	const contents = new Map<(typeof REPORT_FILES)[number], string>([
		["manifest.json", `${JSON.stringify(manifest)}\n`],
		["results.jsonl", records.map((record) => JSON.stringify(record)).join("\n") + (records.length > 0 ? "\n" : "")],
		["metrics.json", `${JSON.stringify(metrics)}\n`],
		["summary.md", renderSummary(manifest, metrics)],
	]);

	let stagingIdentity: DirectoryIdentity;
	const children = new Map<(typeof REPORT_FILES)[number], FileIdentity>();
	let published = false;
	await assertDestinationAbsent(paths.destination);
	await mkdir(paths.staging, { mode: 0o700 });
	stagingIdentity = await snapshotDirectory(paths.staging);
	for (const name of REPORT_FILES) {
		await assertDirectoryIdentity(paths.staging, stagingIdentity);
		await assertOwnedChildren(paths.staging, children);
		const identity = await writeDurablePrivateFile(
			join(paths.staging, name),
			contents.get(name) as string,
			(initial) => children.set(name, initial),
		);
		children.set(name, identity);
		await assertFileIdentity(join(paths.staging, name), identity);
		await assertDirectoryIdentity(paths.staging, stagingIdentity);
	}
	await assertOwnedChildren(paths.staging, children);
	await fsyncDirectory(paths.staging, stagingIdentity);
	await assertDirectoryIdentity(paths.staging, stagingIdentity);
	await renameDirectoryNoReplace(paths.staging, paths.destination);
	published = true;
	try {
		await assertDirectoryIdentity(paths.destination, stagingIdentity);
		await assertOwnedChildren(paths.destination, children);
		await fsyncDirectory(paths.parent);
	} catch (error) {
		throw new Error(
			"published report failed identity validation and was left in place as potentially corrupt",
			{ cause: error },
		);
	}
	if (!published) throw new Error("report publication failed");
}

export function normalizeRunManifest(value: unknown): RunManifestV1 {
	const manifest = requireExactShape(
		value,
		["schemaVersion", "profile", "dataset", "methods", "environment"],
		["methodConfig"],
		"run manifest",
	);
	if (manifest.schemaVersion !== 1) {
		throw new Error("run manifest schemaVersion must be 1");
	}
	if (manifest.profile !== "smoke" && manifest.profile !== "full") {
		throw new Error("run manifest profile must be smoke or full");
	}
	const profile = manifest.profile;
	const datasetValue = requireExactShape(
		manifest.dataset,
		[
			"normalizationVersion",
			"revisions",
			"counts",
			"input",
			"normalized",
			"evaluation",
			...(profile === "smoke" ? ["seed"] : []),
		],
		[],
		"run manifest dataset",
	);
	const normalizationVersion = requireNonNegativeSafeInteger(
		datasetValue.normalizationVersion,
		"run manifest normalizationVersion",
	);
	if (normalizationVersion !== MIRACL_NORMALIZATION_VERSION) {
		throw new Error(`run manifest normalizationVersion must be ${MIRACL_NORMALIZATION_VERSION}`);
	}
	const revisionsValue = requireExactShape(
		datasetValue.revisions,
		["topics", "corpus"],
		[],
		"run manifest dataset revisions",
	);
	const revisions = {
		topics: requireOpaqueDisclosure(revisionsValue.topics, "run manifest topics revision"),
		corpus: requireOpaqueDisclosure(revisionsValue.corpus, "run manifest corpus revision"),
	};
	if (revisions.topics !== MIRACL_SOURCES.topics.revision) {
		throw new Error("run manifest topics revision must match the pinned MIRACL source");
	}
	if (revisions.corpus !== MIRACL_SOURCES.corpus.revision) {
		throw new Error("run manifest corpus revision must match the pinned MIRACL source");
	}
	const countKeys = [
		"queries",
		"qrels",
		"positiveQrels",
		"corpus",
		"judgedDocuments",
		...(profile === "smoke" ? ["distractors"] : []),
	];
	const countValues = requireExactShape(datasetValue.counts, countKeys, [], "run manifest dataset counts");
	const counts: Record<string, number> = {};
	for (const key of countKeys) {
		counts[key] = requireNonNegativeSafeInteger(countValues[key], `run manifest counts.${key}`);
	}
	const inputValue = requireExactShape(
		datasetValue.input,
		["topics", "qrels", "corpus"],
		[],
		"run manifest dataset input",
	);
	if (!Array.isArray(inputValue.corpus) || inputValue.corpus.length !== 3) {
		throw new Error("run manifest dataset input corpus must contain three attestations");
	}
	const input = {
		topics: normalizeFileAttestation(inputValue.topics, "run manifest dataset input topics"),
		qrels: normalizeFileAttestation(inputValue.qrels, "run manifest dataset input qrels"),
		corpus: inputValue.corpus.map((entry, index) =>
			normalizeFileAttestation(entry, `run manifest dataset input corpus ${index}`),
		),
	};
	const normalizedValue = requireExactShape(
		datasetValue.normalized,
		["queries", "qrels", "corpus"],
		[],
		"run manifest dataset normalized",
	);
	const normalized = {
		queries: normalizeFileAttestation(
			normalizedValue.queries,
			"run manifest dataset normalized queries",
			counts.queries,
		),
		qrels: normalizeFileAttestation(normalizedValue.qrels, "run manifest dataset normalized qrels", counts.qrels),
		corpus: normalizeFileAttestation(normalizedValue.corpus, "run manifest dataset normalized corpus", counts.corpus),
	};
	const evaluation = normalizeEvaluation(datasetValue.evaluation, counts);
	validateDatasetCounts(profile, counts);
	const methods = normalizeMethodNames(manifest.methods as readonly BenchmarkMethod[]);
	const environment = normalizeEnvironment(manifest.environment as RunEnvironmentManifest);
	const methodConfig =
		manifest.methodConfig === undefined
			? undefined
			: normalizeMethodConfig(manifest.methodConfig as SanitizedMethodConfig);
	if (profile === "smoke") {
		const dataset: SmokeRunDatasetManifest = {
			normalizationVersion,
			revisions,
			seed: requireSafeInteger(datasetValue.seed, "run manifest seed"),
			counts: counts as unknown as SmokeRunDatasetManifest["counts"],
			input,
			normalized,
			evaluation,
		};
		return assertCanonicalManifestSize({
			schemaVersion: 1,
			profile,
			dataset,
			methods,
			environment,
			...(methodConfig === undefined ? {} : { methodConfig }),
		});
	}
	const dataset: FullRunDatasetManifest = {
		normalizationVersion,
		revisions,
		counts: counts as unknown as FullRunDatasetManifest["counts"],
		input,
		normalized,
		evaluation,
	};
	return assertCanonicalManifestSize({
		schemaVersion: 1,
		profile,
		dataset,
		methods,
		environment,
		...(methodConfig === undefined ? {} : { methodConfig }),
	});
}

export function normalizeRunMetrics(value: unknown): RunMetricsV1 {
	const metricsValue = requireExactShape(
		value,
		["schemaVersion", "methods", "indexingLatencyMs"],
		["peakRssBytes"],
		"metrics",
	);
	if (metricsValue.schemaVersion !== 1) {
		throw new Error("metrics schemaVersion must be 1");
	}
	if (!isRecord(metricsValue.indexingLatencyMs)) {
		throw new Error("indexingLatencyMs must be an object");
	}
	for (const key of Object.keys(metricsValue.indexingLatencyMs)) {
		if (key !== "bm25" && key !== "minsync") {
			throw new Error(`indexingLatencyMs has unknown field ${key}`);
		}
	}
	if (!Array.isArray(metricsValue.methods)) throw new Error("metrics methods must be an array");
	const methods = [...metricsValue.methods]
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
		const valueForMethod = metricsValue.indexingLatencyMs[method];
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
	if (metricsValue.peakRssBytes !== undefined) {
		normalized.peakRssBytes = requirePositiveSafeInteger(metricsValue.peakRssBytes, "peakRssBytes");
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
	if (value.hits.length > 100) throw new Error("record hits must contain at most 100 entries");
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
	for (let rank = 1; rank <= hits.length; rank += 1) {
		if (!ranks.has(rank)) throw new Error("record hit ranks must be contiguous from 1");
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
		rank: requireAtMost(
			requirePositiveSafeInteger(value.rank, `record hit ${index} rank`),
			100,
			`record hit ${index} rank`,
		),
	};
}

function normalizeFileAttestation(value: unknown, label: string): RunFileAttestation;
function normalizeFileAttestation(value: unknown, label: string, expectedRecords: number): RunNormalizedAttestation;
function normalizeFileAttestation(
	value: unknown,
	label: string,
	expectedRecords?: number,
): RunFileAttestation | RunNormalizedAttestation {
	const record = requireExactShape(
		value,
		["sha256", "bytes", ...(expectedRecords === undefined ? [] : ["records"])],
		[],
		label,
	);
	if (typeof record.sha256 !== "string" || !SHA256_PATTERN.test(record.sha256)) {
		throw new Error(`${label} sha256 is invalid`);
	}
	const normalized: { sha256: string; bytes: number; records?: number } = {
		sha256: record.sha256,
		bytes: requirePositiveSafeInteger(record.bytes, `${label} bytes`),
	};
	if (expectedRecords !== undefined) {
		const records = requireNonNegativeSafeInteger(record.records, `${label} records`);
		if (records !== expectedRecords) throw new Error(`${label} records do not match dataset counts`);
		normalized.records = records;
	}
	return normalized as RunFileAttestation | RunNormalizedAttestation;
}

function normalizeEvaluation(value: unknown, counts: Readonly<Record<string, number>>): RunEvaluationV1 {
	const evaluation = requireExactShape(value, ["schemaVersion", "qrels"], [], "run manifest dataset evaluation");
	if (evaluation.schemaVersion !== 1) throw new Error("run manifest dataset evaluation schemaVersion must be 1");
	if (!Array.isArray(evaluation.qrels)) throw new Error("run manifest dataset evaluation qrels must be an array");
	if (evaluation.qrels.length > MAX_EMBEDDED_QRELS) {
		throw new Error(`run manifest dataset evaluation qrels must contain at most ${MAX_EMBEDDED_QRELS} entries`);
	}
	if (evaluation.qrels.length !== counts.qrels) {
		throw new Error("run manifest dataset evaluation qrel count does not match dataset counts");
	}
	const pairs = new Set<string>();
	const qrels = evaluation.qrels.map((value, index): Qrel => {
		const qrel = requireExactShape(
			value,
			["queryId", "documentId", "relevance"],
			[],
			`run manifest dataset evaluation qrel ${index}`,
		);
		const queryId = requireBoundedId(qrel.queryId, `run manifest dataset evaluation qrel ${index} queryId`);
		const documentId = requireBoundedId(qrel.documentId, `run manifest dataset evaluation qrel ${index} documentId`);
		const relevance = requireNonNegativeSafeInteger(
			qrel.relevance,
			`run manifest dataset evaluation qrel ${index} relevance`,
		);
		const pair = `${queryId}\0${documentId}`;
		if (pairs.has(pair)) throw new Error(`duplicate run manifest evaluation qrel ${queryId}/${documentId}`);
		pairs.add(pair);
		return { queryId, documentId, relevance };
	});
	qrels.sort(
		(left, right) =>
			compareCodePoints(left.queryId, right.queryId) || compareCodePoints(left.documentId, right.documentId),
	);
	const queryIds = new Set(qrels.map((qrel) => qrel.queryId));
	const documentIds = new Set(qrels.map((qrel) => qrel.documentId));
	const positives = qrels.filter((qrel) => qrel.relevance > 0).length;
	if (queryIds.size !== counts.queries)
		throw new Error("run manifest evaluation query count does not match dataset counts");
	if (documentIds.size !== counts.judgedDocuments) {
		throw new Error("run manifest evaluation judged document count does not match dataset counts");
	}
	if (positives !== counts.positiveQrels) {
		throw new Error("run manifest evaluation positive qrel count does not match dataset counts");
	}
	for (const queryId of queryIds) {
		if (!qrels.some((qrel) => qrel.queryId === queryId && qrel.relevance > 0)) {
			throw new Error(`run manifest evaluation query ${queryId} has no positive qrel`);
		}
	}
	return { schemaVersion: 1, qrels };
}

function validateDatasetCounts(profile: "smoke" | "full", counts: Readonly<Record<string, number>>): void {
	if (counts.queries < 1) throw new Error("run manifest dataset must contain at least one query");
	if (counts.qrels < 1) throw new Error("run manifest dataset must contain at least one qrel");
	if (counts.positiveQrels > counts.qrels) throw new Error("run manifest positive qrels exceed qrels");
	if (counts.judgedDocuments < 1 || counts.judgedDocuments > counts.qrels) {
		throw new Error("run manifest judged documents are inconsistent with qrels");
	}
	if (counts.judgedDocuments > counts.corpus) {
		throw new Error("run manifest judged documents exceed corpus");
	}
	if (profile === "smoke" && counts.distractors !== counts.corpus - counts.judgedDocuments) {
		throw new Error("run manifest distractors do not match corpus minus judged documents");
	}
	if (profile === "full" && counts.corpus !== MIRACL_FULL_CORPUS_PASSAGES) {
		throw new Error(`full corpus must contain exactly ${MIRACL_FULL_CORPUS_PASSAGES} passages`);
	}
}

function assertCanonicalManifestSize(manifest: RunManifestV1): RunManifestV1 {
	const bytes = Buffer.byteLength(JSON.stringify(manifest), "utf8");
	if (bytes > MAX_MANIFEST_CANONICAL_BYTES) {
		throw new Error(`run manifest exceeds ${MAX_MANIFEST_CANONICAL_BYTES} canonical bytes`);
	}
	return manifest;
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
	const config = requireExactShape(value, ["endpointKind", "dimension"], ["embedderId", "apiKeyEnv"], "methodConfig");
	if (config.endpointKind !== "local" && config.endpointKind !== "remote") {
		throw new Error("methodConfig endpointKind must be local or remote");
	}
	const normalized: {
		embedderId?: string;
		endpointKind: "local" | "remote";
		apiKeyEnv?: string;
		dimension: number;
	} = {
		endpointKind: config.endpointKind,
		dimension: requirePositiveSafeInteger(config.dimension, "methodConfig dimension"),
	};
	if (config.embedderId !== undefined) {
		normalized.embedderId = requirePortableEmbedderId(config.embedderId, "methodConfig embedderId");
	}
	if (config.apiKeyEnv !== undefined) {
		const apiKeyEnv = requireNonBlank(config.apiKeyEnv, "methodConfig apiKeyEnv");
		if (!ENV_NAME_PATTERN.test(apiKeyEnv)) {
			throw new Error("methodConfig apiKeyEnv must be an environment variable name");
		}
		normalized.apiKeyEnv = apiKeyEnv;
	}
	return normalized;
}

function normalizeEnvironment(value: RunEnvironmentManifest): RunEnvironmentManifest {
	const environment = requireExactShape(
		value,
		["autoRagCommit", "platform", "architecture", "node", "measuredAt"],
		["bun"],
		"run manifest environment",
	);
	const normalized: {
		autoRagCommit: string;
		platform: string;
		architecture: string;
		node: string;
		bun?: string;
		measuredAt: string;
	} = {
		autoRagCommit: requireOpaqueDisclosure(environment.autoRagCommit, "environment autoRagCommit"),
		platform: requireOpaqueDisclosure(environment.platform, "environment platform"),
		architecture: requireOpaqueDisclosure(environment.architecture, "environment architecture"),
		node: requireOpaqueDisclosure(environment.node, "environment node"),
		measuredAt: requireNonBlank(environment.measuredAt, "environment measuredAt"),
	};
	if (!Number.isFinite(Date.parse(normalized.measuredAt))) {
		throw new Error("environment measuredAt must be an ISO timestamp");
	}
	if (environment.bun !== undefined) {
		normalized.bun = requireOpaqueDisclosure(environment.bun, "environment bun");
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

function validateCompleteGridAndMetrics(
	manifest: RunManifestV1,
	records: readonly QueryRunRecord[],
	metrics: RunMetricsV1,
): void {
	const queryIds = new Set(manifest.dataset.evaluation.qrels.map((qrel) => qrel.queryId));
	const expectedPairs = new Set<string>();
	for (const method of manifest.methods) {
		for (const queryId of queryIds) expectedPairs.add(`${method}\0${queryId}`);
	}
	const actualPairs = new Set(records.map((record) => `${record.method}\0${record.queryId}`));
	if (
		actualPairs.size !== expectedPairs.size ||
		[...expectedPairs].some((pair) => !actualPairs.has(pair))
	) {
		throw new Error("run report has an incomplete query-method grid");
	}
	const recomputed = evaluateRun(records, manifest.dataset.evaluation.qrels);
	if (JSON.stringify(recomputed) !== JSON.stringify(metrics.methods)) {
		throw new Error("run report metrics do not match recomputed metrics");
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
		"- Full MinSync executable and index-content hashes run outside measured query intervals; cheap O(1) device, inode, size, and mtime checks run at query boundaries.",
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
	};
}

async function writeDurablePrivateFile(
	path: string,
	contents: string,
	onInitialIdentity: (identity: FileIdentity) => void,
): Promise<FileIdentity> {
	const handle = await open(path, "wx", 0o600);
	let identity: FileIdentity;
	try {
		identity = await snapshotOpenRegularFile(handle, "report file");
		onInitialIdentity(identity);
		await handle.writeFile(contents, "utf8");
		await handle.sync();
		const written = await snapshotOpenRegularFile(handle, "report file");
		if (written.device !== identity.device || written.inode !== identity.inode) {
			throw new Error("report file changed during staging");
		}
		identity = written;
	} finally {
		await handle.close();
	}
	return identity;
}

async function fsyncDirectory(path: string, expected?: DirectoryIdentity): Promise<void> {
	const handle = await open(path, "r");
	try {
		const stats = await handle.stat();
		if (
			!stats.isDirectory() ||
			(expected !== undefined && (stats.dev !== expected.device || stats.ino !== expected.inode))
		) {
			throw new Error("report staging directory changed");
		}
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
	if (process.platform !== "darwin" && process.platform !== "linux") {
		throw new Error("atomic no-replace report publication is unavailable on this platform");
	}
	if (process.versions.bun === undefined) {
		await renameDirectoryNoReplaceWithBun(source, destination);
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

const BUN_NO_REPLACE_HELPER = String.raw`
import { dlopen, ptr } from "bun:ffi";
const source = Buffer.from(process.argv[1] + "\0");
const destination = Buffer.from(process.argv[2] + "\0");
let result;
if (process.platform === "darwin") {
  const library = dlopen("/usr/lib/libSystem.B.dylib", {
    renamex_np: { args: ["ptr", "ptr", "u32"], returns: "i32" },
  });
  try { result = library.symbols.renamex_np(ptr(source), ptr(destination), 4); }
  finally { library.close(); }
} else if (process.platform === "linux") {
  const library = dlopen("libc.so.6", {
    renameat2: { args: ["i32", "ptr", "i32", "ptr", "u32"], returns: "i32" },
  });
  try { result = library.symbols.renameat2(-100, ptr(source), -100, ptr(destination), 1); }
  finally { library.close(); }
} else {
  process.exit(72);
}
process.exit(result === 0 ? 0 : 73);
`;

async function renameDirectoryNoReplaceWithBun(source: string, destination: string): Promise<void> {
	try {
		await new Promise<void>((resolvePromise, rejectPromise) => {
			execFile(
				"bun",
				["-e", BUN_NO_REPLACE_HELPER, source, destination],
				{ timeout: 5_000, windowsHide: true },
				(error) => {
					if (error === null) resolvePromise();
					else rejectPromise(error);
				},
			);
		});
	} catch {
		try {
			await lstat(destination);
			throw new Error(`report directory already exists: ${destination}`);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		}
		throw new Error("atomic no-replace report publication failed: no-replace runtime unavailable or failed");
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
	if ((stats.mode & 0o077) !== 0) throw new Error("report staging directory permissions are not private");
	return { device: stats.dev, inode: stats.ino };
}

async function assertDirectoryIdentity(path: string, identity: DirectoryIdentity): Promise<void> {
	const current = await snapshotDirectory(path);
	if (current.device !== identity.device || current.inode !== identity.inode) {
		throw new Error("report staging directory changed");
	}
}

async function snapshotOpenRegularFile(handle: FileHandle, label: string): Promise<FileIdentity> {
	const stats = await handle.stat();
	if (!stats.isFile()) throw new Error(`${label} must be a regular file`);
	if ((stats.mode & 0o077) !== 0) throw new Error(`${label} permissions are not private`);
	return { device: stats.dev, inode: stats.ino, size: stats.size };
}

async function assertFileIdentity(path: string, identity: FileIdentity, requireExactSize = true): Promise<void> {
	const stats = await lstat(path);
	if (
		!stats.isFile() ||
		stats.isSymbolicLink() ||
		stats.dev !== identity.device ||
		stats.ino !== identity.inode ||
		(requireExactSize && stats.size !== identity.size)
	) {
		throw new Error("report file changed during staging");
	}
}

async function assertOwnedChildren(
	root: string,
	children: ReadonlyMap<(typeof REPORT_FILES)[number], FileIdentity>,
): Promise<void> {
	for (const [name, identity] of children) {
		await assertFileIdentity(join(root, name), identity);
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

function requireBoundedId(value: unknown, label: string): string {
	const id = requireNonBlank(value, label);
	if (Buffer.byteLength(id, "utf8") > MAX_DISCLOSED_ID_BYTES) {
		throw new Error(`${label} exceeds ${MAX_DISCLOSED_ID_BYTES} bytes`);
	}
	return id;
}

function requireOpaqueDisclosure(value: unknown, label: string): string {
	const text = requireNonBlank(value, label);
	if (text.includes("://") || /[\r\n|]/u.test(text)) {
		throw new Error(`${label} must be an opaque value`);
	}
	return text;
}

function requirePortableEmbedderId(value: unknown, label: string): string {
	const text = requireNonBlank(value, label);
	if (/^(?:[\\/]|[A-Za-z]:[\\/]|file:)/iu.test(text)) {
		throw new Error(`${label} must not be an absolute filesystem path`);
	}
	return requireOpaqueDisclosure(text, label);
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

function requireAtMost(value: number, maximum: number, label: string): number {
	if (value > maximum) throw new Error(`${label} must be at most ${maximum}`);
	return value;
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

function requireExactShape(
	value: unknown,
	requiredKeys: readonly string[],
	optionalKeys: readonly string[],
	label: string,
): Record<string, unknown> {
	if (!isRecord(value)) throw new Error(`${label} must be an object`);
	const allowed = new Set([...requiredKeys, ...optionalKeys]);
	for (const key of Object.keys(value)) {
		if (!allowed.has(key)) throw new Error(`${label} has unknown field ${key}`);
	}
	for (const key of requiredKeys) {
		if (!(key in value)) throw new Error(`${label} is missing field ${key}`);
	}
	return value;
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
