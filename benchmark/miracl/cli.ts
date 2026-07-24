#!/usr/bin/env bun

import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile, realpath, rename, rm } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { readJsonLines } from "./jsonl.ts";
import {
	type CreateBenchmarkMethodsOptions,
	type CreatedBenchmarkMethods,
	createBenchmarkMethods,
	loadBenchmarkConfig,
} from "./methods.ts";
import { evaluateRun, type MethodMetrics } from "./metrics.ts";
import { type PreparedManifest, type PrepareOptions, prepareMiracl, validatePreparedManifest } from "./prepare.ts";
import { MIRACL_FULL_CORPUS_PASSAGES } from "./profiles.ts";
import {
	normalizeRunManifest,
	normalizeRunMetrics,
	type RunManifestInput,
	type RunManifestV1,
	type RunMetricsV1,
	validateQueryRunRecord,
	type WriteRunReportOptions,
	writeRunReport,
} from "./report.ts";
import { runMethodQueries } from "./run.ts";
import type {
	BenchmarkMethod,
	BenchmarkProfile,
	BenchmarkQuery,
	CorpusDocument,
	Qrel,
	QueryRunRecord,
} from "./types.ts";
import {
	assertBenchmarkPathOutsideAutorag,
	type BenchmarkDirectoryIdentity,
	materializeBenchmarkWorkspace,
	snapshotBenchmarkDirectory,
} from "./workspace.ts";

const METHODS = new Set<BenchmarkMethod>(["bm25", "minsync", "hybrid"]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export interface CliDependencies {
	readonly prepareMiracl: (options: PrepareOptions) => Promise<PreparedManifest>;
	readonly createBenchmarkMethods: (options: CreateBenchmarkMethodsOptions) => Promise<CreatedBenchmarkMethods>;
	readonly writeRunReport: (options: WriteRunReportOptions) => Promise<void>;
	readonly writeStdout: (line: string) => void;
	readonly now: () => Date;
	readonly autoRagCommit: () => string;
	readonly peakRssBytes: () => number | undefined;
}

type CliDependencyOverrides = Partial<CliDependencies>;

interface PrepareCommand {
	readonly command: "prepare";
	readonly profile: BenchmarkProfile;
	readonly output: string;
	readonly confirmFull: boolean;
}

interface RunCommand {
	readonly command: "run";
	readonly profile: BenchmarkProfile;
	readonly prepared: string;
	readonly output: string;
	readonly methods: readonly BenchmarkMethod[];
	readonly config?: string;
}

interface EvaluateCommand {
	readonly command: "evaluate";
	readonly run: string;
}

type Command = PrepareCommand | RunCommand | EvaluateCommand;

interface LoadedPrepared {
	readonly directory: string;
	readonly manifest: PreparedManifest;
	readonly queries: readonly BenchmarkQuery[];
	readonly qrels: readonly Qrel[];
	readonly corpus?: readonly CorpusDocument[];
}

interface LoadedRun {
	readonly directory: string;
	readonly manifest: RunManifestV1;
	readonly records: readonly QueryRunRecord[];
	readonly metrics: RunMetricsV1;
}

export async function runCli(args: readonly string[], overrides: CliDependencyOverrides = {}): Promise<number> {
	const dependencies = dependenciesWithDefaults(overrides);
	const command = parseCommand(args);
	if (command.command === "prepare") {
		return runPrepareCommand(command, dependencies);
	}
	if (command.command === "run") {
		return runBenchmarkCommand(command, dependencies);
	}
	return runEvaluateCommand(command, dependencies);
}

async function runPrepareCommand(command: PrepareCommand, dependencies: CliDependencies): Promise<number> {
	assertBenchmarkPathOutsideAutorag(command.output);
	if (command.profile === "full") {
		dependencies.writeStdout(
			`Full MIRACL Korean preparation will normalize ${MIRACL_FULL_CORPUS_PASSAGES.toLocaleString("en-US")} passages into ${resolve(command.output)}.`,
		);
		if (!command.confirmFull) {
			throw new Error("full preparation requires --confirm-full before any download or write");
		}
		await dependencies.prepareMiracl({
			profile: "full",
			outputDir: command.output,
		});
	} else {
		await dependencies.prepareMiracl({
			profile: "smoke",
			outputDir: command.output,
		});
	}
	dependencies.writeStdout(`Prepared MIRACL Korean ${command.profile} profile at ${resolve(command.output)}.`);
	return 0;
}

async function runBenchmarkCommand(command: RunCommand, dependencies: CliDependencies): Promise<number> {
	assertBenchmarkPathOutsideAutorag(command.prepared);
	assertBenchmarkPathOutsideAutorag(command.output);
	await assertPathAbsent(command.output, "run output");
	const prepared = await loadPrepared(command.prepared, true);
	if (prepared.manifest.profile !== command.profile) {
		throw new Error(`--profile ${command.profile} does not match prepared profile ${prepared.manifest.profile}`);
	}
	const corpus = prepared.corpus as readonly CorpusDocument[];
	const workspacePath = await allocateWorkspacePath(command.output);
	let workspaceIdentity: BenchmarkDirectoryIdentity | undefined;
	try {
		const workspace = materializeBenchmarkWorkspace(workspacePath, corpus);
		workspaceIdentity = snapshotBenchmarkDirectory(workspace.root);
		const needsMinSync = command.methods.some((method) => method === "minsync" || method === "hybrid");
		const config =
			command.config === undefined
				? undefined
				: loadBenchmarkConfig(await canonicalRegularFile(command.config, "benchmark config"));
		if (needsMinSync && config === undefined) {
			throw new Error("--config is required for MinSync or hybrid methods");
		}
		const created = await dependencies.createBenchmarkMethods({
			names: command.methods,
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config,
		});
		const records: QueryRunRecord[] = [];
		for (const method of [...command.methods].sort(compareCodePoints)) {
			const retrieval = created.methods.get(method);
			if (retrieval === undefined) {
				throw new Error(`benchmark method factory did not create ${method}`);
			}
			records.push(
				...(await runMethodQueries({
					method,
					retrieval,
					queries: prepared.queries,
					documentBySource: workspace.documentBySource,
					topK: 100,
				})),
			);
		}
		assertCompleteRunRecords(
			records,
			command.methods,
			prepared.queries.map((query) => query.queryId),
		);
		const metrics = evaluateRun(records, prepared.qrels);
		const reportManifest = createRunManifest(command, prepared, created, dependencies);
		await dependencies.writeRunReport({
			directory: command.output,
			manifest: reportManifest,
			records,
			metrics,
			indexingLatencyMs: created.indexingLatencyMs,
			peakRssBytes: dependencies.peakRssBytes(),
		});
		const failures = records.filter((record) => record.errorCode !== undefined).length;
		dependencies.writeStdout(
			`Wrote ${records.length} MIRACL query-method records to ${resolve(command.output)} (${failures} failed).`,
		);
		return failures === 0 ? 0 : 1;
	} finally {
		if (workspaceIdentity !== undefined) {
			await removeOwnedWorkspace(workspacePath, workspaceIdentity);
		}
	}
}

async function runEvaluateCommand(command: EvaluateCommand, dependencies: CliDependencies): Promise<number> {
	const run = await loadRun(command.run);
	const prepared = await loadPrepared(run.manifest.preparedDirectory, false);
	if (prepared.manifest.profile !== run.manifest.profile) {
		throw new Error("run profile does not match its prepared dataset");
	}
	assertDatasetMatchesRunManifest(prepared.manifest, run.manifest);
	assertCompleteRunRecords(
		run.records,
		run.manifest.methods,
		prepared.queries.map((query) => query.queryId),
	);
	const evaluated = evaluateRun(run.records, prepared.qrels);
	assertPersistedMetricsMatch(evaluated, run.metrics);
	dependencies.writeStdout(JSON.stringify({ schemaVersion: 1, methods: evaluated }));
	return run.records.some((record) => record.errorCode !== undefined) ? 1 : 0;
}

function parseCommand(args: readonly string[]): Command {
	const [command, ...rest] = args;
	if (command === undefined) {
		throw new Error("Command is required: prepare, run, or evaluate");
	}
	if (command === "prepare") {
		const options = parseOptions(rest, new Set(["--profile", "--output"]), new Set(["--confirm-full"]));
		const profile = parseProfile(requireOption(options, "--profile"));
		const output = requireOption(options, "--output");
		const confirmFull = options.get("--confirm-full") === true;
		if (profile === "smoke" && confirmFull) {
			throw new Error("--confirm-full conflicts with --profile smoke");
		}
		return { command, profile, output, confirmFull };
	}
	if (command === "run") {
		const options = parseOptions(
			rest,
			new Set(["--profile", "--prepared", "--output", "--methods", "--config"]),
			new Set(),
		);
		const profile = parseProfile(requireOption(options, "--profile"));
		const prepared = requireOption(options, "--prepared");
		const output = requireOption(options, "--output");
		const methods = parseMethods(requireOption(options, "--methods"));
		const configValue = options.get("--config");
		const config = typeof configValue === "string" ? configValue : undefined;
		const needsMinSync = methods.some((method) => method === "minsync" || method === "hybrid");
		if (needsMinSync && config === undefined) {
			throw new Error("--config is required for MinSync or hybrid methods");
		}
		if (!needsMinSync && config !== undefined) {
			throw new Error("--config conflicts with a BM25-only run");
		}
		return { command, profile, prepared, output, methods, ...(config ? { config } : {}) };
	}
	if (command === "evaluate") {
		const options = parseOptions(rest, new Set(["--run"]), new Set());
		return { command, run: requireOption(options, "--run") };
	}
	throw new Error(`Unknown command: ${command}`);
}

function parseOptions(
	args: readonly string[],
	valueOptions: ReadonlySet<string>,
	booleanOptions: ReadonlySet<string>,
): ReadonlyMap<string, string | true> {
	const parsed = new Map<string, string | true>();
	for (let index = 0; index < args.length; index += 1) {
		const token = args[index] as string;
		if (!token.startsWith("--")) {
			throw new Error(`Unexpected argument: ${token}`);
		}
		if (!valueOptions.has(token) && !booleanOptions.has(token)) {
			throw new Error(`Unknown option: ${token}`);
		}
		if (parsed.has(token)) {
			throw new Error(`Duplicate option: ${token}`);
		}
		if (booleanOptions.has(token)) {
			parsed.set(token, true);
			continue;
		}
		const value = args[index + 1];
		if (value === undefined || value.startsWith("--")) {
			throw new Error(`${token} requires a value`);
		}
		parsed.set(token, value);
		index += 1;
	}
	return parsed;
}

function requireOption(options: ReadonlyMap<string, string | true>, name: string): string {
	const value = options.get(name);
	if (typeof value !== "string" || value.trim().length === 0) {
		throw new Error(`${name} is required`);
	}
	return value;
}

function parseProfile(value: string): BenchmarkProfile {
	if (value !== "smoke" && value !== "full") {
		throw new Error("--profile must be smoke or full");
	}
	return value;
}

function parseMethods(value: string): BenchmarkMethod[] {
	const parts = value.split(",");
	if (parts.length === 0 || parts.some((part) => part.length === 0 || part.trim() !== part)) {
		throw new Error("--methods must be a comma-separated method list");
	}
	const methods = parts.map((part) => {
		if (!METHODS.has(part as BenchmarkMethod)) {
			throw new Error(`Unknown benchmark method: ${part}`);
		}
		return part as BenchmarkMethod;
	});
	if (new Set(methods).size !== methods.length) {
		throw new Error("--methods must not contain duplicates");
	}
	return methods;
}

async function loadPrepared(directory: string, includeCorpus: boolean): Promise<LoadedPrepared> {
	const canonicalDirectory = await canonicalDirectoryPath(directory, "prepared directory");
	const manifestPath = await containedRegularFile(canonicalDirectory, "prepared-manifest.json");
	let rawManifest: unknown;
	try {
		rawManifest = JSON.parse(await readFile(manifestPath, "utf8"));
	} catch {
		throw new Error("prepared manifest is not valid JSON");
	}
	const manifest = validatePreparedManifest(rawManifest);
	const queriesPath = await containedRegularFile(canonicalDirectory, manifest.files.queries);
	const qrelsPath = await containedRegularFile(canonicalDirectory, manifest.files.qrels);
	const queries = validateQueries(await readJsonLines<unknown>(queriesPath));
	const qrels = validateQrels(await readJsonLines<unknown>(qrelsPath), new Set(queries.map((query) => query.queryId)));
	let corpus: CorpusDocument[] | undefined;
	let corpusPath: string | undefined;
	if (includeCorpus) {
		corpusPath = await containedRegularFile(canonicalDirectory, manifest.files.corpus);
		corpus = validateCorpus(await readJsonLines<unknown>(corpusPath));
	}
	validatePreparedContents(manifest, queries, qrels, corpus);
	if (manifest.profile === "full") {
		await assertNormalizedFile(queriesPath, manifest.normalized.queries, queries.length, "queries");
		await assertNormalizedFile(qrelsPath, manifest.normalized.qrels, qrels.length, "qrels");
		if (includeCorpus && corpusPath !== undefined && corpus !== undefined) {
			await assertNormalizedFile(corpusPath, manifest.normalized.corpus, corpus.length, "corpus");
		}
	}
	return {
		directory: canonicalDirectory,
		manifest,
		queries,
		qrels,
		...(corpus === undefined ? {} : { corpus }),
	};
}

function validatePreparedContents(
	manifest: PreparedManifest,
	queries: readonly BenchmarkQuery[],
	qrels: readonly Qrel[],
	corpus: readonly CorpusDocument[] | undefined,
): void {
	if (queries.length !== manifest.counts.queries) {
		throw new Error("prepared query count does not match manifest");
	}
	if (qrels.length !== manifest.counts.qrels) {
		throw new Error("prepared qrel count does not match manifest");
	}
	if (qrels.filter((qrel) => qrel.relevance > 0).length !== manifest.counts.positiveQrels) {
		throw new Error("prepared positive qrel count does not match manifest");
	}
	const judgedDocuments = new Set(qrels.map((qrel) => qrel.documentId));
	if (judgedDocuments.size !== manifest.counts.judgedDocuments) {
		throw new Error("prepared judged document count does not match manifest");
	}
	if (manifest.profile === "smoke") {
		if (queries.some((query, index) => query.queryId !== manifest.selectedIds.queryIds[index])) {
			throw new Error("prepared query IDs do not match smoke manifest");
		}
	}
	if (manifest.profile === "smoke") {
		const positiveQueries = new Set(qrels.filter((qrel) => qrel.relevance > 0).map((qrel) => qrel.queryId));
		for (const query of queries) {
			if (!positiveQueries.has(query.queryId)) {
				throw new Error(`prepared query ${query.queryId} has no positive qrel`);
			}
		}
	}
	if (corpus === undefined) return;
	if (corpus.length !== manifest.counts.corpus) {
		throw new Error("prepared corpus count does not match manifest");
	}
	const corpusIds = new Set(corpus.map((document) => document.documentId));
	for (const documentId of judgedDocuments) {
		if (!corpusIds.has(documentId)) {
			throw new Error(`qrel references missing prepared document ${documentId}`);
		}
	}
	if (
		manifest.profile === "smoke" &&
		corpus.some((document, index) => document.documentId !== manifest.selectedIds.documentIds[index])
	) {
		throw new Error("prepared corpus IDs do not match smoke manifest");
	}
}

function validateQueries(values: readonly unknown[]): BenchmarkQuery[] {
	const ids = new Set<string>();
	return values.map((value, index) => {
		const record = requireExactRecord(value, new Set(["queryId", "text"]), `query ${index}`);
		const queryId = requireNonBlank(record.queryId, `query ${index} queryId`);
		if (ids.has(queryId)) throw new Error(`duplicate prepared query ${queryId}`);
		ids.add(queryId);
		if (typeof record.text !== "string") {
			throw new Error(`query ${index} text must be a string`);
		}
		return { queryId, text: record.text };
	});
}

function validateQrels(values: readonly unknown[], queryIds: ReadonlySet<string>): Qrel[] {
	const pairs = new Set<string>();
	return values.map((value, index) => {
		const record = requireExactRecord(value, new Set(["queryId", "documentId", "relevance"]), `qrel ${index}`);
		const queryId = requireNonBlank(record.queryId, `qrel ${index} queryId`);
		const documentId = requireNonBlank(record.documentId, `qrel ${index} documentId`);
		if (!queryIds.has(queryId)) {
			throw new Error(`qrel ${index} references unknown query ${queryId}`);
		}
		if (!Number.isSafeInteger(record.relevance) || (record.relevance as number) < 0) {
			throw new Error(`qrel ${index} relevance must be a non-negative safe integer`);
		}
		const pair = `${queryId}\0${documentId}`;
		if (pairs.has(pair)) {
			throw new Error(`duplicate prepared qrel ${queryId}/${documentId}`);
		}
		pairs.add(pair);
		return {
			queryId,
			documentId,
			relevance: record.relevance as number,
		};
	});
}

function validateCorpus(values: readonly unknown[]): CorpusDocument[] {
	const ids = new Set<string>();
	return values.map((value, index) => {
		const record = requireExactRecord(value, new Set(["documentId", "title", "text"]), `corpus record ${index}`);
		const documentId = requireNonBlank(record.documentId, `corpus record ${index} documentId`);
		if (ids.has(documentId)) {
			throw new Error(`duplicate prepared document ${documentId}`);
		}
		ids.add(documentId);
		if (typeof record.title !== "string" || typeof record.text !== "string") {
			throw new Error(`corpus record ${index} title and text must be strings`);
		}
		return { documentId, title: record.title, text: record.text };
	});
}

async function loadRun(directory: string): Promise<LoadedRun> {
	const canonicalDirectory = await canonicalDirectoryPath(directory, "run directory");
	const manifestPath = await containedRegularFile(canonicalDirectory, "manifest.json");
	let rawManifest: unknown;
	try {
		rawManifest = JSON.parse(await readFile(manifestPath, "utf8"));
	} catch {
		throw new Error("run manifest is not valid JSON");
	}
	const manifestRecord = requireExactRecord(
		rawManifest,
		new Set([
			"schemaVersion",
			"profile",
			"preparedDirectory",
			"dataset",
			"methods",
			"environment",
			...((rawManifest as Record<string, unknown>)?.methodConfig === undefined ? [] : ["methodConfig"]),
		]),
		"run manifest",
	);
	if (manifestRecord.schemaVersion !== 1) {
		throw new Error("run manifest schemaVersion must be 1");
	}
	validatePersistedRunManifestShape(manifestRecord);
	const manifest = normalizeRunManifest(manifestRecord as unknown as RunManifestInput);
	const records = (await readJsonLines<unknown>(await containedRegularFile(canonicalDirectory, "results.jsonl"))).map(
		validateQueryRunRecord,
	);
	const metricsPath = await containedRegularFile(canonicalDirectory, "metrics.json");
	let rawMetrics: unknown;
	try {
		rawMetrics = JSON.parse(await readFile(metricsPath, "utf8"));
	} catch {
		throw new Error("run metrics are not valid JSON");
	}
	const metricsRecord = requireExactRecord(
		rawMetrics,
		new Set([
			"schemaVersion",
			"methods",
			"indexingLatencyMs",
			...((rawMetrics as Record<string, unknown>)?.peakRssBytes === undefined ? [] : ["peakRssBytes"]),
		]),
		"run metrics",
	);
	requireRecordShape(
		metricsRecord.indexingLatencyMs,
		new Set(),
		new Set(["bm25", "minsync"]),
		"run metrics indexingLatencyMs",
	);
	const metrics = normalizeRunMetrics(metricsRecord as unknown as RunMetricsV1);
	return { directory: canonicalDirectory, manifest, records, metrics };
}

function validatePersistedRunManifestShape(manifest: Record<string, unknown>): void {
	const profile = manifest.profile;
	if (profile !== "smoke" && profile !== "full") {
		throw new Error("run manifest profile must be smoke or full");
	}
	const dataset = requireRecordShape(
		manifest.dataset,
		new Set(["normalizationVersion", "revisions", "counts", ...(profile === "smoke" ? ["seed"] : [])]),
		new Set(["normalizationVersion", "revisions", "counts", ...(profile === "smoke" ? ["seed"] : [])]),
		"run manifest dataset",
	);
	requireExactRecord(dataset.revisions, new Set(["topics", "corpus"]), "run manifest dataset revisions");
	requireExactRecord(
		dataset.counts,
		new Set([
			"queries",
			"qrels",
			"positiveQrels",
			"corpus",
			"judgedDocuments",
			...(profile === "smoke" ? ["distractors"] : []),
		]),
		"run manifest dataset counts",
	);
	requireRecordShape(
		manifest.environment,
		new Set(["autoRagCommit", "platform", "architecture", "node", "measuredAt"]),
		new Set(["autoRagCommit", "platform", "architecture", "node", "bun", "measuredAt"]),
		"run manifest environment",
	);
	if (manifest.methodConfig !== undefined) {
		requireRecordShape(
			manifest.methodConfig,
			new Set(["endpointKind", "dimension"]),
			new Set(["endpointKind", "dimension", "embedderId", "apiKeyEnv"]),
			"run manifest methodConfig",
		);
	}
}

function assertCompleteRunRecords(
	records: readonly QueryRunRecord[],
	methods: readonly BenchmarkMethod[],
	queryIds: readonly string[],
): void {
	const expectedMethods = new Set(methods);
	const expectedQueries = new Set(queryIds);
	if (expectedMethods.size !== methods.length || expectedQueries.size !== queryIds.length) {
		throw new Error("run declaration contains duplicate methods or queries");
	}
	const pairs = new Set<string>();
	for (const record of records) {
		if (!expectedMethods.has(record.method)) {
			throw new Error(`run record has undeclared method ${record.method}`);
		}
		if (!expectedQueries.has(record.queryId)) {
			throw new Error(`run record has undeclared query ${record.queryId}`);
		}
		const pair = `${record.method}\0${record.queryId}`;
		if (pairs.has(pair)) {
			throw new Error(`duplicate query-method record for ${record.method}/${record.queryId}`);
		}
		pairs.add(pair);
	}
	const expectedCount = expectedMethods.size * expectedQueries.size;
	if (records.length !== expectedCount) {
		throw new Error(`run records are incomplete: expected ${expectedCount}, got ${records.length}`);
	}
}

function createRunManifest(
	command: RunCommand,
	prepared: LoadedPrepared,
	created: CreatedBenchmarkMethods,
	dependencies: CliDependencies,
): RunManifestInput {
	const environment: {
		autoRagCommit: string;
		platform: string;
		architecture: string;
		node: string;
		bun?: string;
		measuredAt: string;
	} = {
		autoRagCommit: dependencies.autoRagCommit(),
		platform: process.platform,
		architecture: process.arch,
		node: process.version,
		measuredAt: dependencies.now().toISOString(),
	};
	const bunVersion = process.versions.bun;
	if (bunVersion !== undefined) environment.bun = bunVersion;
	const dataset: RunManifestInput["dataset"] = {
		normalizationVersion: prepared.manifest.normalizationVersion,
		revisions: { ...prepared.manifest.revisions },
		counts: { ...prepared.manifest.counts },
		...(prepared.manifest.profile === "smoke" ? { seed: prepared.manifest.seed } : {}),
	};
	return {
		profile: command.profile,
		preparedDirectory: prepared.directory,
		dataset,
		methods: command.methods,
		environment,
		...(created.reportConfig === undefined ? {} : { methodConfig: created.reportConfig }),
	};
}

function assertDatasetMatchesRunManifest(prepared: PreparedManifest, run: RunManifestV1): void {
	const expected = {
		normalizationVersion: prepared.normalizationVersion,
		revisions: prepared.revisions,
		counts: prepared.counts,
		...(prepared.profile === "smoke" ? { seed: prepared.seed } : {}),
	};
	if (JSON.stringify(run.dataset) !== JSON.stringify(expected)) {
		throw new Error("run dataset disclosure does not match prepared manifest");
	}
}

function assertPersistedMetricsMatch(evaluated: readonly MethodMetrics[], persisted: RunMetricsV1): void {
	const normalizedEvaluated = [...evaluated].sort((left, right) => compareCodePoints(left.method, right.method));
	if (JSON.stringify(normalizedEvaluated) !== JSON.stringify(persisted.methods)) {
		throw new Error("persisted metrics do not match evaluated run records");
	}
}

async function assertNormalizedFile(
	path: string,
	expected: { readonly sha256: string; readonly bytes: number; readonly records: number },
	actualRecords: number,
	label: string,
): Promise<void> {
	if (!SHA256_PATTERN.test(expected.sha256)) {
		throw new Error(`prepared normalized ${label} hash is invalid`);
	}
	const hash = createHash("sha256");
	let bytes = 0;
	for await (const chunk of createReadStream(path)) {
		hash.update(chunk);
		bytes += chunk.byteLength;
	}
	if (hash.digest("hex") !== expected.sha256 || bytes !== expected.bytes || actualRecords !== expected.records) {
		throw new Error(`prepared normalized ${label} identity does not match manifest`);
	}
}

async function canonicalDirectoryPath(path: string, label: string): Promise<string> {
	let stats: Awaited<ReturnType<typeof lstat>>;
	try {
		stats = await lstat(path);
	} catch {
		throw new Error(`${label} does not exist`);
	}
	if (!stats.isDirectory() || stats.isSymbolicLink()) {
		throw new Error(`${label} must be a real directory`);
	}
	const canonical = await realpath(path);
	const canonicalStats = await lstat(canonical);
	if (!canonicalStats.isDirectory() || canonicalStats.isSymbolicLink()) {
		throw new Error(`${label} must be a real directory`);
	}
	return canonical;
}

async function canonicalRegularFile(path: string, label: string): Promise<string> {
	let stats: Awaited<ReturnType<typeof lstat>>;
	try {
		stats = await lstat(path);
	} catch {
		throw new Error(`${label} does not exist`);
	}
	if (!stats.isFile() || stats.isSymbolicLink()) {
		throw new Error(`${label} must be a real file`);
	}
	return realpath(path);
}

async function containedRegularFile(root: string, name: string): Promise<string> {
	if (basename(name) !== name || name === "." || name === "..") {
		throw new Error("manifest file name must be a simple relative name");
	}
	const path = join(root, name);
	const canonical = await canonicalRegularFile(path, `manifest file ${name}`);
	if (!isContained(canonical, root)) {
		throw new Error(`manifest file ${name} escapes its directory`);
	}
	return canonical;
}

async function allocateWorkspacePath(output: string): Promise<string> {
	const absoluteOutput = resolve(output);
	const parent = await realpath(dirname(absoluteOutput));
	const workspace = join(parent, `.${basename(absoluteOutput)}.workspace-${process.pid}-${randomUUID()}`);
	await assertPathAbsent(workspace, "benchmark workspace");
	return workspace;
}

async function assertPathAbsent(path: string, label: string): Promise<void> {
	try {
		await lstat(path);
		throw new Error(`${label} already exists: ${path}`);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
	}
}

async function removeOwnedWorkspace(path: string, identity: BenchmarkDirectoryIdentity): Promise<void> {
	try {
		const current = snapshotBenchmarkDirectory(path);
		if (current.device !== identity.device || current.inode !== identity.inode) return;
	} catch {
		return;
	}
	const quarantine = `${path}.cleanup-${process.pid}-${randomUUID()}`;
	await rename(path, quarantine);
	let moved: BenchmarkDirectoryIdentity;
	try {
		moved = snapshotBenchmarkDirectory(quarantine);
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

function requireExactRecord(value: unknown, keys: ReadonlySet<string>, label: string): Record<string, unknown> {
	return requireRecordShape(value, keys, keys, label);
}

function requireRecordShape(
	value: unknown,
	requiredKeys: ReadonlySet<string>,
	allowedKeys: ReadonlySet<string>,
	label: string,
): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`${label} must be an object`);
	}
	const record = value as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!allowedKeys.has(key)) throw new Error(`${label} has unknown field ${key}`);
	}
	for (const key of requiredKeys) {
		if (!(key in record)) throw new Error(`${label} is missing field ${key}`);
	}
	return record;
}

function requireNonBlank(value: unknown, label: string): string {
	if (typeof value !== "string" || value.trim().length === 0) {
		throw new Error(`${label} must be non-blank`);
	}
	return value;
}

function isContained(path: string, root: string): boolean {
	const descendant = relative(root, path);
	return descendant !== ".." && !descendant.startsWith(`..${sep}`) && !isAbsolute(descendant);
}

function dependenciesWithDefaults(overrides: CliDependencyOverrides): CliDependencies {
	return {
		prepareMiracl: (options) => prepareMiracl(options as never),
		createBenchmarkMethods,
		writeRunReport,
		writeStdout: (line) => console.log(line),
		now: () => new Date(),
		autoRagCommit: readAutoRagCommit,
		peakRssBytes: readPeakRssBytes,
		...overrides,
	};
}

function readAutoRagCommit(): string {
	try {
		const value = execFileSync("git", ["rev-parse", "HEAD"], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
			timeout: 5_000,
		}).trim();
		return /^[0-9a-f]{40}$/u.test(value) ? value : "unknown";
	} catch {
		return "unknown";
	}
}

function readPeakRssBytes(): number | undefined {
	if (!["darwin", "linux", "freebsd"].includes(process.platform)) return undefined;
	const maxRssKiB = process.resourceUsage().maxRSS;
	if (!Number.isSafeInteger(maxRssKiB) || maxRssKiB <= 0) return undefined;
	const bytes = maxRssKiB * 1024;
	return Number.isSafeInteger(bytes) ? bytes : undefined;
}

function compareCodePoints(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}

const entryPath = process.argv[1];
if (entryPath !== undefined && import.meta.url === pathToFileURL(resolve(entryPath)).href) {
	runCli(process.argv.slice(2))
		.then((exitCode) => {
			process.exitCode = exitCode;
		})
		.catch((error: unknown) => {
			const message = error instanceof Error ? error.message : String(error);
			console.error(message);
			process.exitCode = 1;
		});
}
