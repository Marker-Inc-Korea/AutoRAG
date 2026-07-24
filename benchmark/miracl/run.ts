import { lstatSync, realpathSync } from "node:fs";
import { isAbsolute, relative, sep } from "node:path";
import { performance } from "node:perf_hooks";
import { loadMirrorIndex } from "../../src/mirror/index-store.ts";
import { parsedMirrorIndexPath, parsedOutputPath } from "../../src/mirror/paths.ts";
import {
	BM25Method,
	type BM25MethodOptions,
	type BM25SyncResult,
} from "../../src/retrieval/methods/bm25.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";
import {
	assertBenchmarkDirectoryIdentity,
	assertBenchmarkPathOutsideAutorag,
	snapshotBenchmarkDirectory,
} from "./workspace.ts";
import type {
	BenchmarkMethod,
	BenchmarkQuery,
	QueryRunRecord,
	RankedHit,
} from "./types.ts";

const RETRIEVAL_CANDIDATE_LIMIT = 100;

export interface RunBm25QueriesOptions {
	readonly root: string;
	readonly queries: readonly BenchmarkQuery[];
	readonly documentBySource: ReadonlyMap<string, string>;
	readonly topK: number;
	readonly now?: () => number;
	readonly bm25?: Omit<BM25MethodOptions, "root" | "indexPath">;
}

export interface BM25QueryRunResult {
	readonly indexingLatencyMs: number;
	readonly records: readonly QueryRunRecord[];
}

export interface BenchmarkBM25Method extends RetrievalMethod {
	sync(): Promise<BM25SyncResult>;
}

export type BenchmarkBM25Factory = (options: BM25MethodOptions) => BenchmarkBM25Method;

export interface RunMethodQueriesOptions {
	readonly method: BenchmarkMethod;
	readonly retrieval: RetrievalMethod;
	readonly queries: readonly BenchmarkQuery[];
	readonly documentBySource: ReadonlyMap<string, string>;
	readonly topK: number;
	readonly now?: () => number;
}

export async function runBm25Queries(
	options: RunBm25QueriesOptions,
	createBm25: BenchmarkBM25Factory = (methodOptions) => new BM25Method(methodOptions),
): Promise<BM25QueryRunResult> {
	validateTopK(options.topK);
	const root = validateBenchmarkWorkspace(options.root, options.documentBySource);
	const rootIdentity = snapshotBenchmarkDirectory(root);
	const now = options.now ?? (() => performance.now());
	const retrieval = createBm25({
		root,
		enabled: options.bm25?.enabled,
		fallback: options.bm25?.fallback,
		forceEngine: options.bm25?.forceEngine,
		importBinding: options.bm25?.importBinding,
	});
	const indexingStartedAt = now();
	let syncResult: BM25SyncResult;
	assertBenchmarkDirectoryIdentity(root, rootIdentity);
	try {
		syncResult = await retrieval.sync();
	} catch {
		assertBenchmarkDirectoryIdentity(root, rootIdentity);
		now();
		throw new Error("BM25 benchmark indexing failed");
	}
	assertBenchmarkDirectoryIdentity(root, rootIdentity);
	const indexingLatencyMs = elapsedMilliseconds(indexingStartedAt, now());
	if (
		(syncResult.readiness !== "ready" && syncResult.readiness !== "degraded_fallback") ||
		syncResult.engine === "none" ||
		syncResult.indexedChunks < 1
	) {
		throw new Error("BM25 benchmark indexing failed");
	}

	const records = await runMethodQueries({
		method: "bm25",
		retrieval,
		queries: options.queries,
		documentBySource: options.documentBySource,
		topK: options.topK,
		now,
	});
	return { indexingLatencyMs, records };
}

export async function runMethodQueries(
	options: RunMethodQueriesOptions,
): Promise<QueryRunRecord[]> {
	validateTopK(options.topK);
	validateDocumentBySource(options.documentBySource);
	const now = options.now ?? (() => performance.now());
	const records: QueryRunRecord[] = [];

	for (const query of options.queries) {
		const startedAt = now();
		try {
			const results = await options.retrieval.retrieve(query.text, {
				topK: RETRIEVAL_CANDIDATE_LIMIT,
			});
			const hits = rankDocumentHits(results, options.documentBySource, options.topK);
			records.push({
				schemaVersion: 1,
				method: options.method,
				queryId: query.queryId,
				latencyMs: elapsedMilliseconds(startedAt, now()),
				hits,
			});
		} catch {
			records.push({
				schemaVersion: 1,
				method: options.method,
				queryId: query.queryId,
				latencyMs: elapsedMilliseconds(startedAt, now()),
				hits: [],
				errorCode: "retrieval-failed",
			});
		}
	}

	return records;
}

export function validateBenchmarkWorkspace(
	root: string,
	documentBySource: ReadonlyMap<string, string>,
): string {
	assertBenchmarkPathOutsideAutorag(root);
	let canonicalRoot: string;
	try {
		const rootStats = lstatSync(root);
		if (!rootStats.isDirectory() || rootStats.isSymbolicLink()) {
			throw new Error("not a real directory");
		}
		canonicalRoot = realpathSync(root);
		const indexStats = lstatSync(parsedMirrorIndexPath(canonicalRoot));
		if (!indexStats.isFile() || indexStats.isSymbolicLink()) {
			throw new Error("not a real mirror index");
		}
	} catch {
		throw new Error("BM25 benchmark requires a valid parsed mirror workspace");
	}

	let index: ReturnType<typeof loadMirrorIndex>;
	try {
		index = loadMirrorIndex(canonicalRoot);
	} catch {
		throw new Error("BM25 benchmark requires a valid parsed mirror workspace");
	}
	const entries = Object.values(index.entries);
	if (entries.length === 0 || entries.length !== documentBySource.size) {
		throw new Error("BM25 benchmark requires a valid parsed mirror workspace");
	}

	for (const entry of entries) {
		if (
			!documentBySource.has(entry.virtualPath) ||
			!entry.virtualPath.startsWith("/miracl/") ||
			entry.sourcePath !== entry.virtualPath ||
			entry.parserName !== "miracl-benchmark"
		) {
			throw new Error("BM25 benchmark requires a valid parsed mirror workspace");
		}
		const expectedOutput = parsedOutputPath(canonicalRoot, entry.virtualPath);
		if (entry.outputPath !== expectedOutput) {
			throw new Error("BM25 benchmark parsed mirror is stale");
		}
		try {
			const outputStats = lstatSync(entry.outputPath);
			if (!outputStats.isFile() || outputStats.isSymbolicLink()) {
				throw new Error("not a real mirror file");
			}
			const realOutput = realpathSync(entry.outputPath);
			if (realOutput !== entry.outputPath || !isContainedPath(realOutput, canonicalRoot)) {
				throw new Error("mirror file escaped workspace");
			}
		} catch {
			throw new Error("BM25 benchmark parsed mirror is stale");
		}
	}
	validateDocumentBySource(documentBySource);
	return canonicalRoot;
}

function validateDocumentBySource(documentBySource: ReadonlyMap<string, string>): void {
	const documentIds = new Set<string>();
	for (const [source, documentId] of documentBySource) {
		if (typeof documentId !== "string" || documentId.trim().length === 0 || documentIds.has(documentId)) {
			throw new Error("BM25 benchmark document source map must be bijective");
		}
		documentIds.add(documentId);
		let expectedSource: string;
		try {
			expectedSource = `/miracl/${encodeURIComponent(documentId)}.md`;
		} catch {
			throw new Error("BM25 benchmark document source map must be bijective");
		}
		if (source !== expectedSource) {
			throw new Error("BM25 benchmark document source map must be bijective");
		}
	}
}

function rankDocumentHits(
	results: readonly RetrievalResult[],
	documentBySource: ReadonlyMap<string, string>,
	topK: number,
): RankedHit[] {
	const scoreByDocument = new Map<string, number>();
	for (const result of results) {
		const documentId = documentBySource.get(result.source);
		if (documentId === undefined) {
			throw new Error("retrieval returned a source outside the benchmark corpus");
		}
		if (!Number.isFinite(result.score)) {
			throw new Error("retrieval returned a non-finite score");
		}
		const previousScore = scoreByDocument.get(documentId);
		if (previousScore === undefined || result.score > previousScore) {
			scoreByDocument.set(documentId, result.score);
		}
	}

	return [...scoreByDocument]
		.sort(
			([leftId, leftScore], [rightId, rightScore]) =>
				rightScore - leftScore || compareCodePoints(leftId, rightId),
		)
		.slice(0, topK)
		.map(([documentId, score], index) => ({
			documentId,
			score,
			rank: index + 1,
		}));
}

function validateTopK(topK: number): void {
	if (!Number.isSafeInteger(topK) || topK < 1 || topK > RETRIEVAL_CANDIDATE_LIMIT) {
		throw new Error(`topK must be a safe integer between 1 and ${RETRIEVAL_CANDIDATE_LIMIT}`);
	}
}

function elapsedMilliseconds(startedAt: number, finishedAt: number): number {
	const elapsed = finishedAt - startedAt;
	return Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : 0;
}

function compareCodePoints(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}

function isContainedPath(path: string, root: string): boolean {
	const descendant = relative(root, path);
	return descendant !== ".." && !descendant.startsWith(`..${sep}`) && !isAbsolute(descendant);
}
