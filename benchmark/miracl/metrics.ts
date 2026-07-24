import type { BenchmarkMethod, Qrel, QueryRunRecord, RankedHit } from "./types.ts";

const RECALL_CUTOFFS = [5, 10, 100] as const;
const SUCCESS_CUTOFFS = [1, 5, 10] as const;
const NDCG_CUTOFF = 10;
const MRR_CUTOFF = 10;

type RecallCutoff = (typeof RECALL_CUTOFFS)[number];
type SuccessCutoff = (typeof SUCCESS_CUTOFFS)[number];

export interface LatencyMetrics {
	readonly p50: number;
	readonly p95: number;
	readonly p99: number;
}

export interface MethodMetrics {
	readonly method: BenchmarkMethod;
	readonly queryCount: number;
	readonly failureCount: number;
	readonly recallAt: Readonly<Record<`${RecallCutoff}`, number>>;
	readonly mrrAt10: number;
	readonly successAt: Readonly<Record<`${SuccessCutoff}`, number>>;
	readonly ndcgAt10: number;
	readonly latencyMs: LatencyMetrics;
}

interface QueryMetrics {
	readonly recallAt: Record<`${RecallCutoff}`, number>;
	readonly mrrAt10: number;
	readonly successAt: Record<`${SuccessCutoff}`, number>;
	readonly ndcgAt10: number;
}

/**
 * Evaluates MIRACL retrieval records using the query-method records as the
 * macro-average population. Unjudged retrieved documents contribute zero
 * relevance, as in standard qrel-based retrieval evaluation.
 */
export function evaluateRun(
	records: readonly QueryRunRecord[],
	qrels: readonly Qrel[],
): MethodMetrics[] {
	const qrelsByQuery = groupQrels(qrels);
	const recordsByMethod = groupRecords(records, qrelsByQuery);

	return [...recordsByMethod.entries()]
		.sort(([left], [right]) => compareCodePoints(left, right))
		.map(([method, methodRecords]) => evaluateMethod(method, methodRecords, qrelsByQuery));
}

function groupQrels(qrels: readonly Qrel[]): ReadonlyMap<string, ReadonlyMap<string, number>> {
	const qrelsByQuery = new Map<string, Map<string, number>>();
	for (const qrel of qrels) {
		assertNonBlankId(qrel.queryId, "qrel query id");
		assertNonBlankId(qrel.documentId, "qrel document id");
		assertRelevance(qrel.relevance);

		const documents = qrelsByQuery.get(qrel.queryId) ?? new Map<string, number>();
		if (documents.has(qrel.documentId)) {
			throw new Error(`duplicate qrel for ${qrel.queryId}/${qrel.documentId}`);
		}
		documents.set(qrel.documentId, qrel.relevance);
		qrelsByQuery.set(qrel.queryId, documents);
	}
	return qrelsByQuery;
}

function groupRecords(
	records: readonly QueryRunRecord[],
	qrelsByQuery: ReadonlyMap<string, ReadonlyMap<string, number>>,
): ReadonlyMap<BenchmarkMethod, readonly QueryRunRecord[]> {
	const recordsByMethod = new Map<BenchmarkMethod, QueryRunRecord[]>();
	const methodQueryPairs = new Set<string>();
	for (const record of records) {
		assertNonBlankId(record.queryId, "record query id");
		assertFiniteNonNegative(record.latencyMs, "latencyMs");
		validateHits(record.hits);
		if (!qrelsByQuery.has(record.queryId)) {
			throw new Error(`query ${record.queryId} has no qrels`);
		}

		const pair = `${record.method}\u0000${record.queryId}`;
		if (methodQueryPairs.has(pair)) {
			throw new Error(`duplicate query-method record for ${record.method}/${record.queryId}`);
		}
		methodQueryPairs.add(pair);
		const methodRecords = recordsByMethod.get(record.method) ?? [];
		methodRecords.push(record);
		recordsByMethod.set(record.method, methodRecords);
	}
	return recordsByMethod;
}

function evaluateMethod(
	method: BenchmarkMethod,
	records: readonly QueryRunRecord[],
	qrelsByQuery: ReadonlyMap<string, ReadonlyMap<string, number>>,
): MethodMetrics {
	const recallTotals = zeroRecallTotals();
	const successTotals = zeroSuccessTotals();
	let mrrTotal = 0;
	let ndcgTotal = 0;
	let failureCount = 0;
	const successfulLatencies: number[] = [];

	for (const record of records) {
		if (record.errorCode !== undefined) {
			failureCount += 1;
			continue;
		}
		successfulLatencies.push(record.latencyMs);
		const queryMetrics = evaluateQuery(record.hits, qrelsByQuery.get(record.queryId)!);
		for (const cutoff of RECALL_CUTOFFS) {
			recallTotals[`${cutoff}`] += queryMetrics.recallAt[`${cutoff}`];
		}
		for (const cutoff of SUCCESS_CUTOFFS) {
			successTotals[`${cutoff}`] += queryMetrics.successAt[`${cutoff}`];
		}
		mrrTotal += queryMetrics.mrrAt10;
		ndcgTotal += queryMetrics.ndcgAt10;
	}

	const denominator = records.length;
	return {
		method,
		queryCount: denominator,
		failureCount,
		recallAt: divideRecallTotals(recallTotals, denominator),
		mrrAt10: mrrTotal / denominator,
		successAt: divideSuccessTotals(successTotals, denominator),
		ndcgAt10: ndcgTotal / denominator,
		latencyMs: percentileMetrics(successfulLatencies),
	};
}

function evaluateQuery(hits: readonly RankedHit[], qrels: ReadonlyMap<string, number>): QueryMetrics {
	const rankedHits = [...hits].sort((left, right) => left.rank - right.rank);
	const positiveDocumentCount = [...qrels.values()].filter((relevance) => relevance > 0).length;
	const recallAt = zeroRecallTotals();
	const successAt = zeroSuccessTotals();

	for (const cutoff of RECALL_CUTOFFS) {
		const retrievedRelevantCount = rankedHits.filter(
			(hit) => hit.rank <= cutoff && (qrels.get(hit.documentId) ?? 0) > 0,
		).length;
		recallAt[`${cutoff}`] = positiveDocumentCount === 0 ? 0 : retrievedRelevantCount / positiveDocumentCount;
	}
	for (const cutoff of SUCCESS_CUTOFFS) {
		successAt[`${cutoff}`] = rankedHits.some(
			(hit) => hit.rank <= cutoff && (qrels.get(hit.documentId) ?? 0) > 0,
		)
			? 1
			: 0;
	}

	const firstRelevant = rankedHits.find(
		(hit) => hit.rank <= MRR_CUTOFF && (qrels.get(hit.documentId) ?? 0) > 0,
	);
	const dcg = discountedCumulativeGain(rankedHits, qrels, NDCG_CUTOFF);
	const idealDcg = idealDiscountedCumulativeGain(qrels, NDCG_CUTOFF);
	return {
		recallAt,
		mrrAt10: firstRelevant === undefined ? 0 : 1 / firstRelevant.rank,
		successAt,
		ndcgAt10: idealDcg === 0 ? 0 : dcg / idealDcg,
	};
}

function discountedCumulativeGain(
	hits: readonly RankedHit[],
	qrels: ReadonlyMap<string, number>,
	cutoff: number,
): number {
	return hits.reduce((total, hit) => {
		if (hit.rank > cutoff) return total;
		return total + gain(qrels.get(hit.documentId) ?? 0) / discount(hit.rank);
	}, 0);
}

function idealDiscountedCumulativeGain(qrels: ReadonlyMap<string, number>, cutoff: number): number {
	return [...qrels.values()]
		.sort((left, right) => right - left)
		.slice(0, cutoff)
		.reduce((total, relevance, index) => total + gain(relevance) / discount(index + 1), 0);
}

function gain(relevance: number): number {
	return 2 ** relevance - 1;
}

function discount(rank: number): number {
	return Math.log2(rank + 1);
}

function percentileMetrics(latencies: readonly number[]): LatencyMetrics {
	return {
		p50: nearestRankPercentile(latencies, 50),
		p95: nearestRankPercentile(latencies, 95),
		p99: nearestRankPercentile(latencies, 99),
	};
}

function nearestRankPercentile(values: readonly number[], percentile: number): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((left, right) => left - right);
	return sorted[Math.ceil((percentile / 100) * sorted.length) - 1]!;
}

function validateHits(hits: readonly RankedHit[]): void {
	const documentIds = new Set<string>();
	const ranks = new Set<number>();
	for (const hit of hits) {
		assertNonBlankId(hit.documentId, "hit document id");
		assertFinite(hit.score, "hit score");
		if (!Number.isSafeInteger(hit.rank) || hit.rank < 1) {
			throw new Error("hit rank must be a positive safe integer");
		}
		if (documentIds.has(hit.documentId)) {
			throw new Error(`duplicate hit document ${hit.documentId}`);
		}
		if (ranks.has(hit.rank)) {
			throw new Error(`duplicate hit rank ${hit.rank}`);
		}
		documentIds.add(hit.documentId);
		ranks.add(hit.rank);
	}
}

function assertRelevance(relevance: number): void {
	if (!Number.isSafeInteger(relevance) || relevance < 0 || !Number.isFinite(gain(relevance))) {
		throw new Error("qrel relevance must be a non-negative safe integer with a finite gain");
	}
}

function assertFiniteNonNegative(value: number, field: string): void {
	if (!Number.isFinite(value) || value < 0) {
		throw new Error(`${field} must be finite and non-negative`);
	}
}

function assertFinite(value: number, field: string): void {
	if (!Number.isFinite(value)) {
		throw new Error(`${field} must be finite`);
	}
}

function assertNonBlankId(value: string, field: string): void {
	if (typeof value !== "string" || value.trim().length === 0) {
		throw new Error(`${field} must be non-blank`);
	}
}

function zeroRecallTotals(): Record<`${RecallCutoff}`, number> {
	return { "5": 0, "10": 0, "100": 0 };
}

function zeroSuccessTotals(): Record<`${SuccessCutoff}`, number> {
	return { "1": 0, "5": 0, "10": 0 };
}

function divideRecallTotals(
	totals: Record<`${RecallCutoff}`, number>,
	denominator: number,
): Record<`${RecallCutoff}`, number> {
	return { "5": totals["5"] / denominator, "10": totals["10"] / denominator, "100": totals["100"] / denominator };
}

function divideSuccessTotals(
	totals: Record<`${SuccessCutoff}`, number>,
	denominator: number,
): Record<`${SuccessCutoff}`, number> {
	return { "1": totals["1"] / denominator, "5": totals["5"] / denominator, "10": totals["10"] / denominator };
}

function compareCodePoints(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}
