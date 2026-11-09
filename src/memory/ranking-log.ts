import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { z } from "zod";
import { bm25Scores, tokenize } from "../retrieval/bm25.ts";
import { evidenceChunkKey } from "../retrieval/evidence-id.ts";
import type { RetrievalResult } from "../retrieval/types.ts";
import type { JudgedEvidenceRecord } from "./judged-evidence.ts";
import type { SimilarQuestion } from "./similar-queries.ts";

/**
 * Learning-to-rank training log.
 *
 * Retrieval memory keeps only the evidence Jev judged to support an answer: the positives, with no record of what
 * they beat. A ranker learns from comparisons inside one query, so this log keeps, per search run, the whole
 * candidate pool every retrieval channel produced, the raw per-channel ranks and scores exactly as they were at
 * serve time, where each candidate was shown to the model, and the outcome labels the run produced (cited by the
 * final answer, Jev's citation relevance and evidence-support probabilities). Features are frozen when the run is
 * logged and never recomputed later, because corpus statistics, memory, and indexes move on. Chunk text is never
 * stored: only its source, a hash of its identity, its length, and the text signals computed from it at log time.
 */

/** Bump on any change to what a record means; an export refuses to mix versions. */
export const RANKING_LOG_SCHEMA_VERSION = 1;

/** The ranking log lives beside the memory file it extends. */
export function rankingLogPath(memoryPath: string): string {
	return `${memoryPath}.ranking.db`;
}

const observationSchema = z.object({
	/** `prefetch`: the baseline pool before the first answer. `tool`: a retrieval tool the model called. */
	stage: z.enum(["prefetch", "tool"]),
	/** `minsync`, `datasource:<id>`, `web`, or the tool name. */
	surface: z.string(),
	/** Retrieval method that produced the hit (backend metadata), else the surface. */
	method: z.string(),
	/** The search query this hit answered. */
	query: z.string(),
	/** 1-based rank within this surface's list for this query. */
	rank: z.number().int().positive(),
	/** The surface's own score, unnormalized across surfaces. */
	score: z.number(),
	rerankScore: z.number().optional(),
	/** Distinct retrieval methods the cross-method merger saw backing this source. */
	retrievalMethodCount: z.number().optional(),
	sourceChunkCount: z.number().optional(),
	duplicateHitCount: z.number().optional(),
});

const exposureSchema = z.object({
	/** `baseline` (the first-answer prompt) or the tool whose result showed the chunk. */
	surface: z.string(),
	query: z.string(),
	/** 1-based position in what the model was shown. */
	position: z.number().int().positive(),
});

const labelsSchema = z.object({
	/** The final answer cites this chunk (after the relevance filter). */
	citedFinal: z.boolean(),
	/** Any answer of the run (first or final) cited this chunk, before the relevance filter. */
	modelCited: z.boolean(),
	/** Jev's P(relevant to the sentence citing it or to the question); present only when Jev judged it. */
	relevance: z.number().optional(),
	/** Jev's P(supports the sentence and is needed); present only when Jev judged it. */
	support: z.number().optional(),
});

const candidateSchema = z.object({
	/** sha256 of the chunk identity (source + stored text), 24 hex chars. */
	chunkId: z.string(),
	source: z.string(),
	/** Where the evidence lives (`local files`, a datasource id, `web`); absent when unrecognized. */
	origin: z.string().optional(),
	contentChars: z.number().int().nonnegative(),
	observations: z.array(observationSchema),
	rerank: z.object({ rank: z.number().int().positive(), score: z.number() }).optional(),
	exposures: z.array(exposureSchema),
	text: z.object({
		/** BM25 of the chunk against the user's question, over this run's candidate pool. */
		bm25: z.number(),
		/** Share of the question's distinct terms the chunk contains. */
		questionCoverage: z.number(),
	}),
	memory: z.object({
		/** Judged evidence from this source stored before the run started. */
		priorSupport: z.number().int().nonnegative(),
		/** Judged evidence from this source among the similar past questions; null when memory was not consulted. */
		similarQuestionSupport: z.number().int().nonnegative().nullable(),
	}),
	labels: labelsSchema,
});

const judgmentStateSchema = z.enum(["judged", "unavailable", "not-run"]);

const runSchema = z.object({
	schemaVersion: z.literal(RANKING_LOG_SCHEMA_VERSION),
	runId: z.string(),
	conversationId: z.string(),
	question: z.string(),
	startedAt: z.number(),
	finishedAt: z.number(),
	searchQueries: z.array(z.string()),
	datasources: z.array(z.string()),
	web: z.boolean(),
	reranker: z.object({ provider: z.string(), model: z.string() }).optional(),
	judgments: z.object({ citationRelevance: judgmentStateSchema, support: judgmentStateSchema }),
	candidates: z.array(candidateSchema),
});

export type RankingObservation = z.infer<typeof observationSchema>;
export type RankingExposure = z.infer<typeof exposureSchema>;
export type RankingLabels = z.infer<typeof labelsSchema>;
export type RankingCandidate = z.infer<typeof candidateSchema>;
export type RankingRunRecord = z.infer<typeof runSchema>;
export type RankingJudgmentState = z.infer<typeof judgmentStateSchema>;

interface PendingCandidate {
	readonly key: string;
	readonly source: string;
	readonly content: string;
	readonly contentChars: number;
	readonly observations: RankingObservation[];
	readonly exposures: RankingExposure[];
	rerank?: { readonly rank: number; readonly score: number };
}

export interface RankingRunStart {
	readonly runId: string;
	readonly conversationId: string;
	readonly question: string;
	readonly startedAt: number;
}

export interface RankingRunFinish {
	/** Chunk keys ({@link evidenceChunkKey}) the final answer cites. */
	readonly finalCitationKeys: readonly string[];
	readonly finishedAt: number;
	/** Retrieval memory's judged evidence; only records created before the run started count. */
	readonly priorEvidence?: readonly JudgedEvidenceRecord[];
	/** The similar past questions memory found for this run; omitted when memory was not consulted. */
	readonly similarQuestions?: readonly SimilarQuestion[];
	/** Where a piece of evidence lives, or undefined when unrecognized. */
	readonly origin?: (evidence: { readonly source: string; readonly method: string }) => string | undefined;
}

function metadataNumber(metadata: Record<string, unknown>, key: string): number | undefined {
	const value = metadata[key];
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * Collects one search run's candidates, exposures, and outcome labels while the run executes, then freezes them
 * into a {@link RankingRunRecord}. Every call is cheap and synchronous; nothing is written until the agent
 * persists the finished record.
 */
export class RankingRunRecorder {
	private readonly start: RankingRunStart;
	private readonly candidates = new Map<string, PendingCandidate>();
	private searchQueries: readonly string[];
	private datasources: readonly string[] = [];
	private web = false;
	private reranker: { provider: string; model: string } | undefined;
	private readonly modelCited = new Set<string>();
	private readonly relevance = new Map<string, number>();
	private relevanceState: RankingJudgmentState = "not-run";
	private support: ReadonlyMap<string, number> | undefined;
	private supportState: RankingJudgmentState = "not-run";

	constructor(start: RankingRunStart) {
		this.start = start;
		this.searchQueries = [start.question];
	}

	/** What the query planner decided: the search queries, the selected datasources, and whether the web is searched. */
	plan(plan: {
		readonly searchQueries: readonly string[];
		readonly datasources: readonly string[];
		readonly web: boolean;
	}): void {
		this.searchQueries = [...plan.searchQueries];
		this.datasources = [...plan.datasources];
		this.web = plan.web;
	}

	/** One surface's ranked list for one query in the baseline prefetch, before any rerank or cut. */
	retrieved(event: {
		readonly surface: string;
		readonly query: string;
		readonly results: readonly RetrievalResult[];
	}): void {
		this.observe("prefetch", event.surface, event.query, event.results);
	}

	/** A retrieval tool's output: both a ranked list the run retrieved and what the model was shown. */
	tool(event: {
		readonly surface: string;
		readonly query: string;
		readonly results: readonly RetrievalResult[];
	}): void {
		this.observe("tool", event.surface, event.query, event.results);
		this.exposed(event);
	}

	/** The reranker's ordering of the baseline pool; candidates it cut get no rerank entry. */
	reranked(event: {
		readonly provider: string;
		readonly model: string;
		readonly results: readonly RetrievalResult[];
	}): void {
		this.reranker = { provider: event.provider, model: event.model };
		for (const [index, result] of event.results.entries()) {
			const candidate = this.candidate(result);
			candidate.rerank ??= { rank: index + 1, score: result.score };
		}
	}

	/** What the model was shown, in order. */
	exposed(event: {
		readonly surface: string;
		readonly query: string;
		readonly results: readonly RetrievalResult[];
	}): void {
		for (const [index, result] of event.results.entries()) {
			this.candidate(result).exposures.push({ surface: event.surface, query: event.query, position: index + 1 });
		}
	}

	/**
	 * Chunks one answer of the run cited, before the relevance filter, with Jev's P(relevant) per chunk key. An
	 * undefined `relevance` means Jev could not judge this answer's citations.
	 */
	cited(keys: readonly string[], relevance: ReadonlyMap<string, number> | undefined): void {
		for (const key of keys) this.modelCited.add(key);
		if (relevance === undefined) {
			if (this.relevanceState === "not-run") this.relevanceState = "unavailable";
			return;
		}
		this.relevanceState = "judged";
		for (const [key, probability] of relevance) this.relevance.set(key, probability);
	}

	/** Jev's P(supports) per chunk key from the memory gate; undefined when the judgment fell back. */
	supported(support: ReadonlyMap<string, number> | undefined): void {
		this.supportState = support === undefined ? "unavailable" : "judged";
		this.support = support;
	}

	finish(finish: RankingRunFinish): RankingRunRecord {
		const final = new Set(finish.finalCitationKeys);
		const pending = [...this.candidates.values()];
		const bm25 = bm25Scores(
			this.start.question,
			pending.map((candidate) => candidate.content),
		);
		const questionTerms = [...new Set(tokenize(this.start.question))];
		const before = (record: JudgedEvidenceRecord): boolean => record.createdAt < this.start.startedAt;
		const priorBySource = new Map<string, number>();
		for (const record of finish.priorEvidence ?? []) {
			if (before(record)) priorBySource.set(record.source, (priorBySource.get(record.source) ?? 0) + 1);
		}
		let similarBySource: Map<string, number> | undefined;
		if (finish.similarQuestions !== undefined) {
			similarBySource = new Map();
			for (const question of finish.similarQuestions) {
				for (const record of question.records) {
					if (before(record)) similarBySource.set(record.source, (similarBySource.get(record.source) ?? 0) + 1);
				}
			}
		}
		const candidates = pending.map((candidate, index): RankingCandidate => {
			const tokens = tokenize(candidate.content);
			const covered = questionTerms.filter((term) =>
				tokens.some((token) => token === term || (term.length >= 2 && token.startsWith(term))),
			).length;
			const relevance = this.relevance.get(candidate.key);
			const support = this.support?.get(candidate.key);
			const citedFinal = final.has(candidate.key);
			const origin = finish.origin?.({
				source: candidate.source,
				method: candidate.observations[0]?.method ?? candidate.exposures[0]?.surface ?? "",
			});
			return {
				chunkId: createHash("sha256").update(candidate.key).digest("hex").slice(0, 24),
				source: candidate.source,
				...(origin !== undefined ? { origin } : {}),
				contentChars: candidate.contentChars,
				observations: candidate.observations,
				...(candidate.rerank !== undefined ? { rerank: candidate.rerank } : {}),
				exposures: candidate.exposures,
				text: {
					bm25: bm25[index] ?? 0,
					questionCoverage: questionTerms.length === 0 ? 0 : covered / questionTerms.length,
				},
				memory: {
					priorSupport: priorBySource.get(candidate.source) ?? 0,
					similarQuestionSupport:
						similarBySource === undefined ? null : (similarBySource.get(candidate.source) ?? 0),
				},
				labels: {
					citedFinal,
					modelCited: citedFinal || this.modelCited.has(candidate.key),
					...(relevance !== undefined ? { relevance } : {}),
					...(support !== undefined ? { support } : {}),
				},
			};
		});
		return {
			schemaVersion: RANKING_LOG_SCHEMA_VERSION,
			runId: this.start.runId,
			conversationId: this.start.conversationId,
			question: this.start.question,
			startedAt: this.start.startedAt,
			finishedAt: finish.finishedAt,
			searchQueries: [...this.searchQueries],
			datasources: [...this.datasources],
			web: this.web,
			...(this.reranker !== undefined ? { reranker: this.reranker } : {}),
			judgments: { citationRelevance: this.relevanceState, support: this.supportState },
			candidates,
		};
	}

	private observe(
		stage: RankingObservation["stage"],
		surface: string,
		query: string,
		results: readonly RetrievalResult[],
	): void {
		for (const [index, result] of results.entries()) {
			const metadata = result.metadata;
			const method = typeof metadata.method === "string" ? metadata.method : surface;
			const rerankScore = metadataNumber(metadata, "rerankScore");
			const methods = metadata.retrievalMethods;
			const sourceChunkCount = metadataNumber(metadata, "sourceChunkCount");
			const duplicateHitCount = metadataNumber(metadata, "duplicateHitCount");
			this.candidate(result).observations.push({
				stage,
				surface,
				method,
				query,
				rank: index + 1,
				score: result.score,
				...(rerankScore !== undefined ? { rerankScore } : {}),
				...(Array.isArray(methods) ? { retrievalMethodCount: methods.length } : {}),
				...(sourceChunkCount !== undefined ? { sourceChunkCount } : {}),
				...(duplicateHitCount !== undefined ? { duplicateHitCount } : {}),
			});
		}
	}

	private candidate(result: RetrievalResult): PendingCandidate {
		const key = evidenceChunkKey(result.source, result.content);
		let candidate = this.candidates.get(key);
		if (candidate === undefined) {
			candidate = {
				key,
				source: result.source,
				content: result.content,
				contentChars: result.content.length,
				observations: [],
				exposures: [],
			};
			this.candidates.set(key, candidate);
		}
		return candidate;
	}
}

export interface RankingLogListing {
	/** Valid runs, oldest first. */
	readonly runs: readonly RankingRunRecord[];
	/** Rows that are not a valid record of this schema version. */
	readonly skipped: number;
}

/**
 * Append-only SQLite store of finished runs, one JSON row per run keyed by its id. Several agent processes share
 * one memory directory, so every run is its own row and a write never rewrites another process's runs. Unlike the
 * question-vector cache this is not derived data: a file that cannot be opened is reported, never replaced.
 */
export class RankingLogStore {
	private readonly db: DatabaseSync;

	constructor(path: string) {
		mkdirSync(dirname(path), { recursive: true });
		const db = new DatabaseSync(path);
		try {
			db.exec(`PRAGMA busy_timeout = 5000;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at INTEGER NOT NULL, record TEXT NOT NULL) WITHOUT ROWID;`);
		} catch (error) {
			db.close();
			throw new Error(
				`Ranking log ${path} could not be opened: ${error instanceof Error ? error.message : String(error)}`,
			);
		}
		this.db = db;
	}

	/** Store one finished run; a run id already stored is left as it was. */
	append(record: RankingRunRecord): void {
		this.db
			.prepare("INSERT OR IGNORE INTO runs (run_id, started_at, record) VALUES (?, ?, ?)")
			.run(record.runId, record.startedAt, JSON.stringify(record));
	}

	list(): RankingLogListing {
		const rows = this.db.prepare("SELECT record FROM runs ORDER BY started_at, run_id").all() as {
			record: string;
		}[];
		const runs: RankingRunRecord[] = [];
		let skipped = 0;
		for (const row of rows) {
			let raw: unknown;
			try {
				raw = JSON.parse(row.record);
			} catch {
				skipped += 1;
				continue;
			}
			const parsed = runSchema.safeParse(raw);
			if (parsed.success) runs.push(parsed.data);
			else skipped += 1;
		}
		return { runs, skipped };
	}

	close(): void {
		this.db.close();
	}
}
