import { performance } from "node:perf_hooks";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";
import type {
	BenchmarkMethod,
	BenchmarkQuery,
	QueryRunRecord,
	RankedHit,
} from "./types.ts";

const RETRIEVAL_CANDIDATE_LIMIT = 100;

export interface RunMethodQueriesOptions {
	readonly method: BenchmarkMethod;
	readonly retrieval: RetrievalMethod;
	readonly queries: readonly BenchmarkQuery[];
	readonly documentBySource: ReadonlyMap<string, string>;
	readonly topK: number;
	readonly now?: () => number;
}

export async function runMethodQueries(
	options: RunMethodQueriesOptions,
): Promise<QueryRunRecord[]> {
	validateTopK(options.topK);
	if (options.retrieval.describe().status !== "active") {
		throw new Error("benchmark retrieval method must be ready before queries run");
	}
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
