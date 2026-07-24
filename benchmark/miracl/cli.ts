#!/usr/bin/env bun

import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, readdir, realpath, rmdir, unlink } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { streamFullCorpus } from "./full-bm25.ts";
import { forEachJsonLine, readJsonLines } from "./jsonl.ts";
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
	type RunFileAttestation,
	type RunManifestInput,
	type RunManifestV1,
	type RunMetricsV1,
	type RunNormalizedAttestation,
	renderRunSummary,
	validateQueryRunRecord,
	validateRunReportCoherence,
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
	materializeEmptyBenchmarkWorkspace,
	snapshotBenchmarkDirectory,
} from "./workspace.ts";

const METHODS = new Set<BenchmarkMethod>(["bm25", "minsync", "hybrid"]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const MAX_RUN_MANIFEST_BYTES = 4 * 1024 * 1024;
const MAX_RUN_METRICS_BYTES = 256 * 1024;
const MAX_RUN_RESULTS_BYTES = 64 * 1024 * 1024;
const MAX_RUN_RESULT_RECORDS = 30_000;
const MAX_RUN_SUMMARY_BYTES = 256 * 1024;
const RUN_REPORT_FILES = ["manifest.json", "metrics.json", "results.jsonl", "summary.md"] as const;
const MAX_PREPARED_MANIFEST_BYTES = 1024 * 1024;
const MAX_PREPARED_QUERIES_BYTES = 32 * 1024 * 1024;
const MAX_PREPARED_QRELS_BYTES = 256 * 1024 * 1024;
const MAX_PREPARED_SMOKE_CORPUS_BYTES = 512 * 1024 * 1024;
const MAX_PREPARED_FULL_CORPUS_BYTES = 12 * 1024 * 1024 * 1024;
const MAX_PREPARED_QUERIES = 10_000;
const MAX_PREPARED_QRELS = 100_000;
const MAX_PREPARED_SMOKE_CORPUS_RECORDS = 100_000;

class BoundedArtifactReadError extends Error {}

export interface CliDependencies {
	readonly prepareMiracl: (options: PrepareOptions) => Promise<PreparedManifest>;
	readonly createBenchmarkMethods: (options: CreateBenchmarkMethodsOptions) => Promise<CreatedBenchmarkMethods>;
	readonly writeRunReport: (options: WriteRunReportOptions) => Promise<void>;
	readonly writeStdout: (line: string) => void;
	readonly now: () => Date;
	readonly autoRagCommit: () => string;
	readonly peakRssBeforeReportBytes: () => number | undefined;
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
	readonly corpusPath?: string;
	readonly normalized: {
		readonly queries: RunNormalizedAttestation;
		readonly qrels: RunNormalizedAttestation;
		readonly corpus?: RunNormalizedAttestation;
	};
}

interface LoadedRun {
	readonly directory: string;
	readonly manifest: RunManifestV1;
	readonly records: readonly QueryRunRecord[];
	readonly metrics: RunMetricsV1;
}

interface WorkspaceEntryIdentity {
	readonly device: number;
	readonly inode: number;
}

interface WorkspaceTreeOwnership {
	readonly root: BenchmarkDirectoryIdentity;
	readonly files: ReadonlyMap<string, WorkspaceEntryIdentity>;
	readonly directories: ReadonlyMap<string, WorkspaceEntryIdentity>;
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
	if (command.profile === "full" && (command.methods.length !== 1 || command.methods[0] !== "bm25")) {
		throw new Error(
			"full profile supports only bm25; production MinSync requires a file per passage and is not claimed as scalable",
		);
	}
	await assertPathAbsent(command.output, "run output");
	const prepared = await loadPrepared(command.prepared, true);
	if (prepared.manifest.profile !== command.profile) {
		throw new Error(`--profile ${command.profile} does not match prepared profile ${prepared.manifest.profile}`);
	}
	const workspacePath = await allocateWorkspacePath(command.output);
	let workspaceIdentity: BenchmarkDirectoryIdentity | undefined;
	let workspaceOwnership: WorkspaceTreeOwnership | undefined;
	try {
		const workspace =
			prepared.manifest.profile === "full"
				? {
						root: materializeEmptyBenchmarkWorkspace(workspacePath),
						documentBySource: undefined,
					}
				: materializeBenchmarkWorkspace(workspacePath, prepared.corpus as readonly CorpusDocument[]);
		workspaceIdentity = snapshotBenchmarkDirectory(workspace.root);
		workspaceOwnership = await snapshotOwnedWorkspace(workspace.root, workspaceIdentity);
		const needsMinSync = command.methods.some((method) => method === "minsync" || method === "hybrid");
		const config =
			command.config === undefined
				? undefined
				: loadBenchmarkConfig(await canonicalRegularFile(command.config, "benchmark config"));
		if (needsMinSync && config === undefined) {
			throw new Error("--config is required for MinSync or hybrid methods");
		}
		let created: CreatedBenchmarkMethods;
		try {
			created = await dependencies.createBenchmarkMethods({
				names: command.methods,
				root: workspace.root,
				documentBySource: workspace.documentBySource,
				...(prepared.manifest.profile === "full"
					? {
							fullCorpus: {
								path: prepared.corpusPath as string,
								attestation: prepared.manifest.normalized.corpus,
							},
						}
					: {}),
				config,
			});
		} catch (error) {
			try {
				workspaceOwnership = await snapshotOwnedWorkspace(workspace.root, workspaceIdentity);
			} catch {
				// The previous exact snapshot remains the only cleanup authority.
			}
			throw error;
		}
		workspaceOwnership = await snapshotOwnedWorkspace(workspace.root, workspaceIdentity);
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
					documentBySource: workspace.documentBySource ?? miraclDocumentIdFromSource,
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
			peakRssBeforeReportBytes: dependencies.peakRssBeforeReportBytes(),
		});
		const failures = records.filter((record) => record.errorCode !== undefined).length;
		dependencies.writeStdout(
			`Wrote ${records.length} MIRACL query-method records to ${resolve(command.output)} (${failures} failed).`,
		);
		return failures === 0 ? 0 : 1;
	} finally {
		if (workspaceOwnership !== undefined) {
			await removeOwnedWorkspace(workspacePath, workspaceOwnership);
		}
	}
}

async function runEvaluateCommand(command: EvaluateCommand, dependencies: CliDependencies): Promise<number> {
	const run = await loadRun(command.run);
	const qrels = run.manifest.dataset.evaluation.qrels;
	assertCompleteRunRecords(run.records, run.manifest.methods, [...new Set(qrels.map((qrel) => qrel.queryId))]);
	const evaluated = evaluateRun(run.records, qrels);
	assertPersistedMetricsMatch(evaluated, run.metrics);
	const output = normalizeRunMetrics({
		schemaVersion: 1,
		methods: evaluated,
		indexingLatencyMs: run.metrics.indexingLatencyMs,
		...(run.metrics.peakRssBeforeReportBytes === undefined
			? {}
			: { peakRssBeforeReportBytes: run.metrics.peakRssBeforeReportBytes }),
	});
	dependencies.writeStdout(JSON.stringify(output));
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
		rawManifest = JSON.parse(
			await readBoundedPrivateUtf8(manifestPath, MAX_PREPARED_MANIFEST_BYTES, "prepared manifest"),
		);
	} catch (error) {
		throwPreservingBoundedReadError(error, "prepared manifest is not valid JSON");
	}
	const manifest = validatePreparedManifest(rawManifest);
	assertPreparedBounds(manifest);
	const queriesPath = await containedRegularFile(canonicalDirectory, manifest.files.queries);
	const qrelsPath = await containedRegularFile(canonicalDirectory, manifest.files.qrels);
	const queryValues: unknown[] = [];
	const normalizedQueries = await forEachJsonLine<unknown>(
		queriesPath,
		(value) => {
			queryValues.push(value);
		},
		{
			totalBytes: manifest.normalized.queries.bytes,
			maxRecords: manifest.normalized.queries.records,
			requirePrivateFile: true,
			label: "prepared queries",
		},
	);
	const queries = validateQueries(queryValues);
	const qrelValues: unknown[] = [];
	const normalizedQrels = await forEachJsonLine<unknown>(
		qrelsPath,
		(value) => {
			qrelValues.push(value);
		},
		{
			totalBytes: manifest.normalized.qrels.bytes,
			maxRecords: manifest.normalized.qrels.records,
			requirePrivateFile: true,
			label: "prepared qrels",
		},
	);
	const qrels = validateQrels(qrelValues, new Set(queries.map((query) => query.queryId)));
	let corpus: CorpusDocument[] | undefined;
	let corpusPath: string | undefined;
	let normalizedCorpus: RunNormalizedAttestation | undefined;
	if (includeCorpus) {
		corpusPath = await containedRegularFile(canonicalDirectory, manifest.files.corpus);
		if (manifest.profile === "full") {
			normalizedCorpus = await streamFullCorpus({
				path: corpusPath,
				attestation: manifest.normalized.corpus,
				requiredDocumentIds: new Set(qrels.map((qrel) => qrel.documentId)),
			});
		} else {
			const corpusValues: unknown[] = [];
			normalizedCorpus = await forEachJsonLine<unknown>(
				corpusPath,
				(value) => {
					corpusValues.push(value);
				},
				{
					totalBytes: manifest.normalized.corpus.bytes,
					maxRecords: manifest.normalized.corpus.records,
					requirePrivateFile: true,
					label: "prepared corpus",
				},
			);
			corpus = validateCorpus(corpusValues);
		}
	}
	validatePreparedContents(manifest, queries, qrels, corpus);
	assertNormalizedIdentity(normalizedQueries, manifest.normalized.queries, "queries");
	assertNormalizedIdentity(normalizedQrels, manifest.normalized.qrels, "qrels");
	if (includeCorpus && normalizedCorpus !== undefined) {
		assertNormalizedIdentity(normalizedCorpus, manifest.normalized.corpus, "corpus");
	}
	return {
		directory: canonicalDirectory,
		manifest,
		queries,
		qrels,
		...(corpus === undefined ? {} : { corpus }),
		...(corpusPath === undefined ? {} : { corpusPath }),
		normalized: {
			queries: normalizedQueries,
			qrels: normalizedQrels,
			...(normalizedCorpus === undefined ? {} : { corpus: normalizedCorpus }),
		},
	};
}

function miraclDocumentIdFromSource(source: string): string | undefined {
	const match = /^\/miracl\/([^/]+)\.md$/u.exec(source);
	if (match === null) return undefined;
	try {
		const documentId = decodeURIComponent(match[1] as string);
		return `/miracl/${encodeURIComponent(documentId)}.md` === source && documentId.trim().length > 0
			? documentId
			: undefined;
	} catch {
		return undefined;
	}
}

function assertPreparedBounds(manifest: PreparedManifest): void {
	const declarations = [
		["queries", manifest.normalized.queries, MAX_PREPARED_QUERIES_BYTES, MAX_PREPARED_QUERIES],
		["qrels", manifest.normalized.qrels, MAX_PREPARED_QRELS_BYTES, MAX_PREPARED_QRELS],
		[
			"corpus",
			manifest.normalized.corpus,
			manifest.profile === "full" ? MAX_PREPARED_FULL_CORPUS_BYTES : MAX_PREPARED_SMOKE_CORPUS_BYTES,
			manifest.profile === "full" ? MIRACL_FULL_CORPUS_PASSAGES : MAX_PREPARED_SMOKE_CORPUS_RECORDS,
		],
	] as const;
	for (const [label, declaration, maxBytes, maxRecords] of declarations) {
		if (declaration.bytes > maxBytes) {
			throw new Error(`prepared ${label} declaration exceeds ${maxBytes} bytes`);
		}
		if (declaration.records > maxRecords) {
			throw new Error(`prepared ${label} declaration exceeds ${maxRecords} records`);
		}
	}
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
	const children = (await readdir(canonicalDirectory)).sort(compareCodePoints);
	if (
		children.length !== RUN_REPORT_FILES.length ||
		children.some((name, index) => name !== RUN_REPORT_FILES[index])
	) {
		throw new Error("run directory must contain exactly manifest.json, results.jsonl, metrics.json, and summary.md");
	}
	const manifestPath = await containedRegularFile(canonicalDirectory, "manifest.json");
	let rawManifest: unknown;
	try {
		rawManifest = JSON.parse(await readBoundedPrivateUtf8(manifestPath, MAX_RUN_MANIFEST_BYTES, "run manifest"));
	} catch (error) {
		throwPreservingBoundedReadError(error, "run manifest is not valid JSON");
	}
	const manifest = normalizeRunManifest(rawManifest);
	const queryCount = new Set(manifest.dataset.evaluation.qrels.map((qrel) => qrel.queryId)).size;
	const expectedRecords = queryCount * manifest.methods.length;
	if (!Number.isSafeInteger(expectedRecords) || expectedRecords < 1 || expectedRecords > MAX_RUN_RESULT_RECORDS) {
		throw new Error(`run results declaration exceeds ${MAX_RUN_RESULT_RECORDS} records`);
	}
	const records = (
		await readJsonLines<unknown>(await containedRegularFile(canonicalDirectory, "results.jsonl"), {
			totalBytes: MAX_RUN_RESULTS_BYTES,
			maxRecords: expectedRecords,
			requirePrivateFile: true,
			label: "results JSONL",
		})
	).map(validateQueryRunRecord);
	const metricsPath = await containedRegularFile(canonicalDirectory, "metrics.json");
	let rawMetrics: unknown;
	try {
		rawMetrics = JSON.parse(await readBoundedPrivateUtf8(metricsPath, MAX_RUN_METRICS_BYTES, "run metrics"));
	} catch (error) {
		throwPreservingBoundedReadError(error, "run metrics are not valid JSON");
	}
	const metrics = normalizeRunMetrics(rawMetrics);
	validateRunReportCoherence(manifest, records, metrics);
	const summaryPath = await containedRegularFile(canonicalDirectory, "summary.md");
	const summary = await readBoundedPrivateUtf8(summaryPath, MAX_RUN_SUMMARY_BYTES, "run summary");
	if (summary !== renderRunSummary(manifest, metrics)) {
		throw new Error("summary.md does not match the canonical report");
	}
	return { directory: canonicalDirectory, manifest, records, metrics };
}

function throwPreservingBoundedReadError(error: unknown, fallback: string): never {
	if (error instanceof BoundedArtifactReadError) throw error;
	throw new Error(fallback);
}

async function readBoundedPrivateUtf8(path: string, maxBytes: number, label: string): Promise<string> {
	const pathStats = await lstat(path);
	assertPrivateBoundedRegularFile(pathStats, maxBytes, label);
	const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
	try {
		const openStats = await handle.stat();
		assertPrivateBoundedRegularFile(openStats, maxBytes, label);
		if (openStats.dev !== pathStats.dev || openStats.ino !== pathStats.ino || openStats.size !== pathStats.size) {
			throw new BoundedArtifactReadError(`${label} changed before reading`);
		}
		const bytes = Buffer.alloc(openStats.size);
		let offset = 0;
		while (offset < bytes.length) {
			const result = await handle.read(bytes, offset, bytes.length - offset, offset);
			if (result.bytesRead === 0) throw new BoundedArtifactReadError(`${label} changed while reading`);
			offset += result.bytesRead;
		}
		const finalStats = await handle.stat();
		if (finalStats.dev !== openStats.dev || finalStats.ino !== openStats.ino || finalStats.size !== openStats.size) {
			throw new BoundedArtifactReadError(`${label} changed while reading`);
		}
		return bytes.toString("utf8");
	} finally {
		await handle.close();
	}
}

function assertPrivateBoundedRegularFile(
	stats: Awaited<ReturnType<typeof lstat>>,
	maxBytes: number,
	label: string,
): void {
	if (!stats.isFile() || stats.isSymbolicLink()) {
		throw new BoundedArtifactReadError(`${label} must be a real file`);
	}
	if ((Number(stats.mode) & 0o077) !== 0) {
		throw new BoundedArtifactReadError(`${label} must be private`);
	}
	if (stats.size > maxBytes) {
		throw new BoundedArtifactReadError(`${label} exceeds ${maxBytes} bytes`);
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
	const corpusNormalized = prepared.normalized.corpus;
	if (corpusNormalized === undefined) throw new Error("prepared corpus attestation is unavailable");
	const identity = {
		normalizationVersion: prepared.manifest.normalizationVersion,
		revisions: { ...prepared.manifest.revisions },
		input: {
			topics: sourceAttestation(prepared.manifest.sources.topics),
			qrels: sourceAttestation(prepared.manifest.sources.qrels),
			corpus: prepared.manifest.sources.corpus.map(sourceAttestation),
		},
		normalized: {
			queries: { ...prepared.normalized.queries },
			qrels: { ...prepared.normalized.qrels },
			corpus: { ...corpusNormalized },
		},
		evaluation: {
			schemaVersion: 1 as const,
			qrels: prepared.qrels.map((qrel) => ({ ...qrel })),
		},
	};
	const common = {
		schemaVersion: 1 as const,
		methods: command.methods,
		environment,
		methodConfig:
			created.reportConfig ??
			(() => {
				throw new Error("benchmark method provenance is unavailable");
			})(),
	};
	if (prepared.manifest.profile === "smoke") {
		return {
			...common,
			profile: "smoke",
			dataset: {
				...identity,
				seed: prepared.manifest.seed,
				counts: { ...prepared.manifest.counts },
			},
		};
	}
	return {
		...common,
		profile: "full",
		dataset: {
			...identity,
			counts: { ...prepared.manifest.counts },
		},
	};
}

function assertPersistedMetricsMatch(evaluated: readonly MethodMetrics[], persisted: RunMetricsV1): void {
	const normalizedEvaluated = [...evaluated].sort((left, right) => compareCodePoints(left.method, right.method));
	if (JSON.stringify(normalizedEvaluated) !== JSON.stringify(persisted.methods)) {
		throw new Error("persisted metrics do not match evaluated run records");
	}
}

function sourceAttestation(source: { readonly sha256: string; readonly bytes: number }): RunFileAttestation {
	return { sha256: source.sha256, bytes: source.bytes };
}

function assertNormalizedIdentity(
	actual: RunNormalizedAttestation,
	expected: { readonly sha256: string; readonly bytes: number; readonly records: number },
	label: string,
): void {
	if (!SHA256_PATTERN.test(expected.sha256)) {
		throw new Error(`prepared normalized ${label} hash is invalid`);
	}
	if (actual.sha256 !== expected.sha256 || actual.bytes !== expected.bytes || actual.records !== expected.records) {
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

async function snapshotOwnedWorkspace(
	path: string,
	identity: BenchmarkDirectoryIdentity,
): Promise<WorkspaceTreeOwnership> {
	const root = snapshotBenchmarkDirectory(path);
	if (root.device !== identity.device || root.inode !== identity.inode) {
		throw new Error("benchmark workspace changed");
	}
	const files = new Map<string, WorkspaceEntryIdentity>();
	const directories = new Map<string, WorkspaceEntryIdentity>();
	const visit = async (directory: string, relativeDirectory: string): Promise<void> => {
		for (const name of await readdir(directory)) {
			const relativePath = relativeDirectory.length === 0 ? name : join(relativeDirectory, name);
			const childPath = join(path, relativePath);
			const stats = await lstat(childPath);
			if (stats.isSymbolicLink()) continue;
			const entryIdentity = { device: stats.dev, inode: stats.ino };
			if (stats.isFile()) {
				files.set(relativePath, entryIdentity);
			} else if (stats.isDirectory()) {
				directories.set(relativePath, entryIdentity);
				await visit(childPath, relativePath);
			}
		}
	};
	await visit(path, "");
	return { root, files, directories };
}

async function removeOwnedWorkspace(path: string, ownership: WorkspaceTreeOwnership): Promise<void> {
	try {
		const current = snapshotBenchmarkDirectory(path);
		if (current.device !== ownership.root.device || current.inode !== ownership.root.inode) return;
	} catch {
		return;
	}
	for (const [relativePath, identity] of ownership.files) {
		const child = join(path, relativePath);
		try {
			const stats = await lstat(child);
			if (
				stats.isFile() &&
				!stats.isSymbolicLink() &&
				stats.dev === identity.device &&
				stats.ino === identity.inode
			) {
				await unlink(child);
			}
		} catch {
			// Replaced or missing children are preserved.
		}
	}
	const directories = [...ownership.directories].sort(
		([left], [right]) => right.split(sep).length - left.split(sep).length,
	);
	for (const [relativePath, identity] of directories) {
		const child = join(path, relativePath);
		try {
			const stats = await lstat(child);
			if (
				stats.isDirectory() &&
				!stats.isSymbolicLink() &&
				stats.dev === identity.device &&
				stats.ino === identity.inode
			) {
				await rmdir(child);
			}
		} catch {
			// Non-empty, replaced, or missing directories are preserved.
		}
	}
	try {
		const current = snapshotBenchmarkDirectory(path);
		if (current.device === ownership.root.device && current.inode === ownership.root.inode) {
			await rmdir(path);
		}
	} catch {
		// Non-empty or replaced roots are preserved.
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
		peakRssBeforeReportBytes: readPeakRssBytes,
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
	return normalizeMaxRssBytes(process.resourceUsage().maxRSS, process.versions.bun);
}

export function normalizeMaxRssBytes(maxRss: number, bunVersion: string | undefined): number | undefined {
	if (!Number.isSafeInteger(maxRss) || maxRss <= 0) return undefined;
	const bytes = bunVersion === undefined ? maxRss * 1024 : maxRss;
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
