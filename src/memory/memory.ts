import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { z } from "zod";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import {
	isPathOpaqueIdentifier,
	type NormalizedEvidenceRef,
	normalizeEvidenceRef,
	normalizeEvidenceText,
} from "../retrieval/evidence-id.ts";
import type { JudgedEvidenceRecord } from "./judged-evidence.ts";

export interface EvidenceContext {
	readonly retrieverMix?: readonly string[];
	readonly parserType?: string;
	readonly documentType?: string;
	readonly documentArea?: string;
	readonly evidenceType?: string;
	readonly evidenceLocation?: string;
	readonly confidence?: number;
}

export interface EvidenceChunkRecord extends NormalizedEvidenceRef, EvidenceContext {
	readonly excerptHash: string;
	readonly firstSeenAt: number;
	readonly lastSeenAt: number;
	readonly metadata?: Record<string, unknown>;
}

export interface CuratedResultRecord {
	readonly resultId: string;
	readonly sessionId: string;
	readonly number: number;
	readonly query: string;
	readonly title: string;
	readonly summary: string;
	readonly resultHash: string;
	readonly evidenceIds: readonly string[];
	readonly confidence?: number;
	readonly createdAt: number;
}

/** A long-term lesson summarized from a batch of judged evidence about one kind of question. */
export interface RetrievalInsight {
	readonly id: string;
	readonly clusterKey: string;
	readonly domain: string;
	readonly recommendedSources: string[];
	readonly recommendedMethods: string[];
	readonly rationale: string;
	supportingEvidenceCount: number;
	confidence: number;
	readonly createdAt: number;
	updatedAt: number;
}

/** Summarizes one batch of judged evidence into insights. */
export type InsightExtractor = (records: readonly JudgedEvidenceRecord[], now: number) => RetrievalInsight[];

export interface MemoryWarning {
	readonly code: string;
	readonly message: string;
	readonly timestamp: number;
}

export interface MemorySchema {
	readonly version: 5;
	curatedResults: CuratedResultRecord[];
	evidenceChunks: EvidenceChunkRecord[];
	/** Every piece of evidence Jev judged to support the question it was cited for. Never evicted. */
	judgedEvidence: JudgedEvidenceRecord[];
	warnings: MemoryWarning[];
	insights: RetrievalInsight[];
	/** Ids of judged evidence not yet summarized into insights; summarized 100 at a time. */
	pendingInsightEntries: string[];
}

export interface RetrievalMemoryOptions {
	storagePath: string;
	insightExtractor?: InsightExtractor;
}

export interface SessionEvidenceRef extends NormalizedEvidenceRef, EvidenceContext {
	readonly metadata?: Record<string, unknown>;
}

export interface SessionCuratedResultInput {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	readonly content: string;
	readonly method: string;
	readonly source: string;
	readonly confidence?: number;
	readonly evidenceRefs: readonly SessionEvidenceRef[];
}

export interface SessionRecordInput {
	readonly sessionId: string;
	readonly query: string;
	readonly results: readonly SessionCuratedResultInput[];
}

const MAX_WARNINGS = 50;
const INSIGHT_BATCH_SIZE = 100;
/** A cluster needs this many judged evidence records, from at least {@link MIN_INSIGHT_RUNS} search runs, to become an insight. */
const MIN_INSIGHT_SUPPORT = 5;
const MIN_INSIGHT_RUNS = 2;
const MAX_CONTEXT_LABEL_LENGTH = 120;
const MAX_RETRIEVER_LABEL_LENGTH = 64;
const MAX_RETRIEVER_MIX = 8;
const INSIGHT_WARNING = "[AutoRAG] Retrieval memory insight extraction failed; continuing without insights";
const RESET_WARNING = "[AutoRAG] Retrieval memory is not v4/v5-compatible; starting fresh";
const LOCK_WAIT_TIMEOUT_MS = 5_000;
const LOCK_STALE_MS = 30_000;
const LOCK_RETRY_MS = 10;

class RetrievalMemoryLockTimeoutError extends Error {
	constructor() {
		super("Timed out waiting for retrieval memory lock");
		this.name = "RetrievalMemoryLockTimeoutError";
	}
}

function emptyMemory(): MemorySchema {
	return {
		version: 5,
		curatedResults: [],
		evidenceChunks: [],
		judgedEvidence: [],
		warnings: [],
		insights: [],
		pendingInsightEntries: [],
	};
}

function cloneMemory(data: MemorySchema): MemorySchema {
	return structuredClone(data);
}

function recordsEqual(left: unknown, right: unknown): boolean {
	return JSON.stringify(left) === JSON.stringify(right);
}

function warningKey(warning: MemoryWarning): string {
	return `${warning.timestamp}\0${warning.code}\0${warning.message}`;
}

const warningSchema = z.object({ code: z.string(), message: z.string(), timestamp: z.number() });

// v4 also stored the user's verdict and a remote-peer tag on each result; parsing drops both.
const curatedResultSchema = z.object({
	resultId: z.string(),
	sessionId: z.string(),
	number: z.number(),
	query: z.string(),
	title: z.string(),
	summary: z.string(),
	resultHash: z.string(),
	evidenceIds: z.array(z.string()),
	confidence: z.number().optional(),
	createdAt: z.number(),
});

const evidenceChunkSchema = z
	.object({
		stableEvidenceId: z.string(),
		method: z.string(),
		source: z.string(),
		excerptHash: z.string(),
		firstSeenAt: z.number(),
		lastSeenAt: z.number(),
	})
	.passthrough();

const judgedEvidenceSchema = z.object({
	id: z.string(),
	sessionId: z.string(),
	conversationId: z.string(),
	question: z.string(),
	searchQuery: z.string(),
	method: z.string(),
	source: z.string(),
	stableEvidenceId: z.string(),
	resultNumber: z.number(),
	title: z.string(),
	excerpt: z.string(),
	probability: z.number(),
	createdAt: z.number(),
});

// v4 insights counted `supportingSignalCount`; they carry over as evidence counts.
const insightSchema = z
	.object({
		id: z.string(),
		clusterKey: z.string(),
		domain: z.string(),
		recommendedSources: z.array(z.string()),
		recommendedMethods: z.array(z.string()),
		rationale: z.string(),
		supportingEvidenceCount: z.number().optional(),
		supportingSignalCount: z.number().optional(),
		confidence: z.number(),
		createdAt: z.number(),
		updatedAt: z.number(),
	})
	.transform(({ supportingSignalCount, supportingEvidenceCount, ...rest }, ctx): RetrievalInsight => {
		const count = supportingEvidenceCount ?? supportingSignalCount;
		if (count === undefined) {
			ctx.addIssue({ code: "custom", message: "insight has no supporting count" });
			return z.NEVER;
		}
		return { ...rest, supportingEvidenceCount: count };
	});

const persistedSchema = z.object({
	version: z.union([z.literal(4), z.literal(5)]),
	curatedResults: z.array(z.unknown()),
	evidenceChunks: z.array(z.unknown()),
	warnings: z.array(z.unknown()),
	insights: z.array(z.unknown()).optional(),
	judgedEvidence: z.array(z.unknown()).optional(),
	pendingInsightEntries: z.array(z.unknown()).optional(),
});

/** The items of a persisted list that match `schema`; a malformed item is dropped, never the whole file. */
function parseItems<T>(schema: z.ZodType<T>, items: readonly unknown[] = []): T[] {
	return items.flatMap((item) => {
		const parsed = schema.safeParse(item);
		return parsed.success ? [parsed.data] : [];
	});
}

/** Parse a memory file of either version into the current schema, or undefined when it is neither. */
function parsePersisted(raw: unknown): MemorySchema | undefined {
	const file = persistedSchema.safeParse(raw);
	if (!file.success) return undefined;
	const data = file.data;
	return {
		version: 5,
		curatedResults: parseItems(curatedResultSchema, data.curatedResults),
		evidenceChunks: parseItems(evidenceChunkSchema, data.evidenceChunks).map((chunk) =>
			normalizeEvidenceChunkRecord(chunk as unknown as EvidenceChunkRecord),
		),
		judgedEvidence: parseItems(judgedEvidenceSchema, data.judgedEvidence),
		warnings: parseItems(warningSchema, data.warnings),
		insights: parseItems(insightSchema, data.insights),
		pendingInsightEntries: parseItems(z.string(), data.pendingInsightEntries),
	};
}

function normalizeContextLabel(value: string | undefined, maxLength = MAX_CONTEXT_LABEL_LENGTH): string | undefined {
	if (value === undefined) return undefined;
	let withoutControls = "";
	for (const character of value) {
		const code = character.charCodeAt(0);
		withoutControls += code < 32 || code === 127 ? " " : character;
	}
	const normalized = withoutControls.replace(/\s+/gu, " ").trim().slice(0, maxLength);
	if (!normalized || !isPathOpaqueIdentifier(normalized)) return undefined;
	return normalized;
}

function normalizeEvidenceContext(context: EvidenceContext): EvidenceContext {
	const retrieverMix = Array.from(
		new Set(
			(context.retrieverMix ?? [])
				.map((value) => normalizeContextLabel(value, MAX_RETRIEVER_LABEL_LENGTH))
				.filter((value): value is string => value !== undefined),
		),
	).slice(0, MAX_RETRIEVER_MIX);
	const parserType = normalizeContextLabel(context.parserType);
	const documentType = normalizeContextLabel(context.documentType);
	const documentArea = normalizeContextLabel(context.documentArea);
	const evidenceType = normalizeContextLabel(context.evidenceType);
	const evidenceLocation = normalizeContextLabel(context.evidenceLocation);
	const confidence =
		context.confidence !== undefined && Number.isFinite(context.confidence)
			? Math.max(0, Math.min(1, context.confidence))
			: undefined;
	return {
		...(retrieverMix.length > 0 ? { retrieverMix } : {}),
		...(parserType !== undefined ? { parserType } : {}),
		...(documentType !== undefined ? { documentType } : {}),
		...(documentArea !== undefined ? { documentArea } : {}),
		...(evidenceType !== undefined ? { evidenceType } : {}),
		...(evidenceLocation !== undefined ? { evidenceLocation } : {}),
		...(confidence !== undefined ? { confidence } : {}),
	};
}

function normalizeEvidenceChunkRecord(record: EvidenceChunkRecord): EvidenceChunkRecord {
	const {
		retrieverMix: _retrieverMix,
		parserType: _parserType,
		documentType: _documentType,
		documentArea: _documentArea,
		evidenceType: _evidenceType,
		evidenceLocation: _evidenceLocation,
		confidence: _confidence,
		...base
	} = record;
	return { ...base, ...normalizeEvidenceContext(record) };
}

function hashText(value: string): string {
	return createHash("sha256").update(value).digest("hex");
}

function resultId(sessionId: string, number: number): string {
	return `${sessionId}:${number}`;
}

function resultHash(query: string, title: string, summary: string, evidenceIds: readonly string[]): string {
	return hashText([query, title, summary, ...evidenceIds].join("\0"));
}

function normalizeInsightDomain(query: string): string {
	return query
		.toLowerCase()
		.replace(/[^\p{L}\p{N}]+/gu, " ")
		.trim()
		.split(/\s+/u)
		.filter((token) => token.length > 1)
		.slice(0, 5)
		.join(" ");
}

function insightMatches(insight: RetrievalInsight, query: string): boolean {
	const domain = normalizeInsightDomain(query);
	if (domain.length === 0) return false;
	if (insight.clusterKey === domain || insight.domain === domain) return true;
	const insightTokens = new Set(insight.clusterKey.split(" ").filter(Boolean));
	const queryTokens = domain.split(" ").filter(Boolean);
	if (insightTokens.size < 2 || queryTokens.length < 2) return false;
	let overlap = 0;
	for (const token of queryTokens) if (insightTokens.has(token)) overlap++;
	return overlap >= Math.min(2, insightTokens.size) && overlap / insightTokens.size >= 0.6;
}

/**
 * Groups a batch of judged evidence by question topic and keeps the topics
 * that recur: enough evidence, from more than one search run, so one lucky
 * search does not become a lesson. The lesson names the methods and sources
 * that keep supplying evidence for that topic; it is advisory only.
 */
function defaultInsightExtractor(records: readonly JudgedEvidenceRecord[], now: number): RetrievalInsight[] {
	const clusters = new Map<
		string,
		{
			support: number;
			probabilitySum: number;
			runs: Set<string>;
			methods: Map<string, number>;
			sources: Map<string, number>;
		}
	>();
	for (const record of records) {
		const domain = normalizeInsightDomain(record.question);
		if (domain.length === 0) continue;
		const cluster = clusters.get(domain) ?? {
			support: 0,
			probabilitySum: 0,
			runs: new Set<string>(),
			methods: new Map<string, number>(),
			sources: new Map<string, number>(),
		};
		cluster.support++;
		cluster.probabilitySum += record.probability;
		cluster.runs.add(record.sessionId);
		cluster.methods.set(record.method, (cluster.methods.get(record.method) ?? 0) + 1);
		cluster.sources.set(record.source, (cluster.sources.get(record.source) ?? 0) + 1);
		clusters.set(domain, cluster);
	}
	const byCount = (a: [string, number], b: [string, number]): number => b[1] - a[1] || a[0].localeCompare(b[0]);
	return Array.from(clusters.entries())
		.filter(([, cluster]) => cluster.support >= MIN_INSIGHT_SUPPORT && cluster.runs.size >= MIN_INSIGHT_RUNS)
		.map(([domain, cluster]): RetrievalInsight => {
			const methods = Array.from(cluster.methods.entries()).sort(byCount);
			const sources = Array.from(cluster.sources.entries()).sort(byCount);
			const methodConsistency = (methods[0]?.[1] ?? 0) / cluster.support;
			return {
				id: `insight:${hashText(domain).slice(0, 24)}`,
				clusterKey: domain,
				domain,
				recommendedSources: sources.slice(0, 3).map(([source]) => source),
				recommendedMethods: methods.slice(0, 3).map(([method]) => method),
				rationale: `${cluster.support} judged evidence record(s) from ${cluster.runs.size} search run(s) mostly came from ${methods[0]?.[0] ?? "unknown"}; advisory only, not a method disable rule`,
				supportingEvidenceCount: cluster.support,
				confidence: Math.min(1, (cluster.probabilitySum / cluster.support) * methodConsistency),
				createdAt: now,
				updatedAt: now,
			};
		})
		.sort(
			(a, b) =>
				b.confidence - a.confidence ||
				b.supportingEvidenceCount - a.supportingEvidenceCount ||
				a.domain.localeCompare(b.domain),
		);
}

/**
 * Long-term retrieval memory: the evidence Jev judged to genuinely support the
 * question it was cited for, plus the curated results and evidence chunks of
 * past searches. Nothing is evicted; every 100 judged records are summarized
 * into long-term insights.
 */
export class RetrievalMemory {
	private readonly storagePath: string;
	private readonly insightExtractor: InsightExtractor;
	private data: MemorySchema = emptyMemory();
	private persistedData: MemorySchema = emptyMemory();

	constructor(options: RetrievalMemoryOptions) {
		this.storagePath = options.storagePath;
		this.insightExtractor = options.insightExtractor ?? defaultInsightExtractor;
	}

	load(): void {
		this.data = this.readPersistedData();
		this.persistedData = cloneMemory(this.data);
	}

	save(): void {
		const dir = dirname(this.storagePath);
		if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
		const lock = this.acquireLock();
		const localData = this.data;
		let tmpPath: string | undefined;
		try {
			this.data = this.mergeWithPersisted(this.readPersistedData());
			this.summarizeCompleteBatches();
			this.data.warnings = this.data.warnings.slice(-MAX_WARNINGS);
			tmpPath = `${this.storagePath}.${randomUUID()}.tmp`;
			writeFileSync(tmpPath, `${JSON.stringify(this.data, null, 2)}\n`, "utf-8");
			lock.assertOwned();
			renameSync(tmpPath, this.storagePath);
			this.persistedData = cloneMemory(this.data);
		} catch (error) {
			this.data = localData;
			throw error;
		} finally {
			try {
				if (tmpPath && existsSync(tmpPath)) unlinkSync(tmpPath);
			} finally {
				lock.release();
			}
		}
	}

	getSchema(): MemorySchema {
		return this.data;
	}

	getJudgedEvidence(): readonly JudgedEvidenceRecord[] {
		return this.data.judgedEvidence;
	}

	/** Everything judged during one conversation, oldest first. */
	getConversationEvidence(conversationId: string): JudgedEvidenceRecord[] {
		return this.data.judgedEvidence
			.filter((record) => record.conversationId === conversationId)
			.sort((a, b) => a.createdAt - b.createdAt);
	}

	recordCuratedResultsSession(input: SessionRecordInput): void {
		const now = Date.now();
		for (const result of input.results) {
			const evidenceIds: string[] = [];
			for (const ref of result.evidenceRefs) {
				this.upsertEvidence(ref, now);
				evidenceIds.push(ref.stableEvidenceId);
			}
			const id = resultId(input.sessionId, result.number);
			const record: CuratedResultRecord = {
				resultId: id,
				sessionId: input.sessionId,
				number: result.number,
				query: input.query,
				title: result.title,
				summary: result.summary,
				resultHash: resultHash(input.query, result.title, result.summary, evidenceIds),
				evidenceIds,
				...(result.confidence !== undefined ? { confidence: result.confidence } : {}),
				createdAt: now,
			};
			const existingIndex = this.data.curatedResults.findIndex((entry) => entry.resultId === id);
			if (existingIndex >= 0) this.data.curatedResults[existingIndex] = record;
			else this.data.curatedResults.push(record);
		}
	}

	/** Remember judged evidence. A record already stored under the same id is left as it was. */
	recordJudgedEvidence(records: readonly JudgedEvidenceRecord[]): void {
		const known = new Set(this.data.judgedEvidence.map((record) => record.id));
		for (const record of records) {
			if (known.has(record.id)) continue;
			known.add(record.id);
			this.data.judgedEvidence.push(record);
			this.data.pendingInsightEntries.push(record.id);
		}
	}

	getInsights(query: string): RetrievalInsight[] {
		return this.data.insights
			.filter((insight) => insightMatches(insight, query))
			.sort(
				(a, b) =>
					b.confidence - a.confidence ||
					b.supportingEvidenceCount - a.supportingEvidenceCount ||
					b.updatedAt - a.updatedAt ||
					a.domain.localeCompare(b.domain),
			);
	}

	private readPersistedData(): MemorySchema {
		if (!existsSync(this.storagePath)) return emptyMemory();
		try {
			const parsed: unknown = JSON.parse(readFileSync(this.storagePath, "utf-8"));
			const persisted = parsePersisted(parsed);
			if (persisted !== undefined) return persisted;
		} catch (error) {
			if (!(error instanceof Error)) throw error;
		}
		return this.incompatibleMemory();
	}

	private incompatibleMemory(): MemorySchema {
		console.warn(RESET_WARNING);
		const data = emptyMemory();
		data.warnings.push({
			code: "memory-reset",
			message: "Retrieval memory was reset because it was not v4/v5-compatible",
			timestamp: Date.now(),
		});
		return data;
	}

	private acquireLock(): FileLockHandle {
		return acquireFileLock(`${this.storagePath}.lock`, {
			timeoutMs: LOCK_WAIT_TIMEOUT_MS,
			staleMs: LOCK_STALE_MS,
			retryMs: LOCK_RETRY_MS,
			timeoutError: () => new RetrievalMemoryLockTimeoutError(),
		});
	}

	/**
	 * Fold what this instance added since it last loaded or saved into the file
	 * as it is now, so processes saving independently never overwrite each other.
	 */
	private mergeWithPersisted(persisted: MemorySchema): MemorySchema {
		const merged = cloneMemory(persisted);
		const baselineResults = new Map(this.persistedData.curatedResults.map((record) => [record.resultId, record]));
		for (const record of this.data.curatedResults) {
			const baseline = baselineResults.get(record.resultId);
			if (baseline && recordsEqual(record, baseline)) continue;
			const existingIndex = merged.curatedResults.findIndex((item) => item.resultId === record.resultId);
			if (existingIndex < 0) merged.curatedResults.push(record);
			else if (record.createdAt >= merged.curatedResults[existingIndex].createdAt) {
				merged.curatedResults[existingIndex] = record;
			}
		}

		const baselineEvidence = new Map(
			this.persistedData.evidenceChunks.map((record) => [record.stableEvidenceId, record]),
		);
		for (const record of this.data.evidenceChunks) {
			const baseline = baselineEvidence.get(record.stableEvidenceId);
			if (baseline && recordsEqual(record, baseline)) continue;
			const existingIndex = merged.evidenceChunks.findIndex(
				(item) => item.stableEvidenceId === record.stableEvidenceId,
			);
			if (existingIndex < 0) {
				merged.evidenceChunks.push(record);
				continue;
			}
			const existing = merged.evidenceChunks[existingIndex];
			const latest = record.lastSeenAt >= existing.lastSeenAt ? record : existing;
			merged.evidenceChunks[existingIndex] = {
				...latest,
				firstSeenAt: Math.min(existing.firstSeenAt, record.firstSeenAt),
				lastSeenAt: Math.max(existing.lastSeenAt, record.lastSeenAt),
			};
		}

		const baselineJudgedIds = new Set(this.persistedData.judgedEvidence.map((record) => record.id));
		const judgedIds = new Set(merged.judgedEvidence.map((record) => record.id));
		for (const record of this.data.judgedEvidence) {
			if (baselineJudgedIds.has(record.id) || judgedIds.has(record.id)) continue;
			judgedIds.add(record.id);
			merged.judgedEvidence.push(record);
		}

		const baselinePending = new Set(this.persistedData.pendingInsightEntries);
		const pending = new Set(merged.pendingInsightEntries);
		for (const id of this.data.pendingInsightEntries) {
			if (baselinePending.has(id) || pending.has(id)) continue;
			pending.add(id);
			merged.pendingInsightEntries.push(id);
		}

		const baselineWarningKeys = new Set(this.persistedData.warnings.map(warningKey));
		const warningKeys = new Set(merged.warnings.map(warningKey));
		for (const warning of this.data.warnings) {
			const key = warningKey(warning);
			if (baselineWarningKeys.has(key) || warningKeys.has(key)) continue;
			warningKeys.add(key);
			merged.warnings.push(warning);
		}

		const baselineInsights = new Map(this.persistedData.insights.map((insight) => [insight.clusterKey, insight]));
		for (const insight of this.data.insights) {
			const baseline = baselineInsights.get(insight.clusterKey);
			if (baseline && recordsEqual(insight, baseline)) continue;
			const existingIndex = merged.insights.findIndex((item) => item.clusterKey === insight.clusterKey);
			if (existingIndex < 0) {
				merged.insights.push(insight);
				continue;
			}
			const existing = merged.insights[existingIndex];
			const supportDelta = baseline
				? Math.max(0, insight.supportingEvidenceCount - baseline.supportingEvidenceCount)
				: insight.supportingEvidenceCount;
			const support = existing.supportingEvidenceCount + supportDelta;
			merged.insights[existingIndex] = {
				...existing,
				recommendedSources: Array.from(
					new Set([...existing.recommendedSources, ...insight.recommendedSources]),
				).slice(0, 3),
				recommendedMethods: Array.from(
					new Set([...existing.recommendedMethods, ...insight.recommendedMethods]),
				).slice(0, 3),
				rationale: insight.updatedAt >= existing.updatedAt ? insight.rationale : existing.rationale,
				supportingEvidenceCount: support,
				confidence: Math.max(existing.confidence, insight.confidence, Math.min(1, support / 100)),
				createdAt: Math.min(existing.createdAt, insight.createdAt),
				updatedAt: Math.max(existing.updatedAt, insight.updatedAt),
			};
		}

		return merged;
	}

	/**
	 * Every full group of 100 pending judged records is summarized into
	 * insights; the remainder waits for the next batch. A failing extractor never
	 * blocks the save, and the batch it failed on is not retried.
	 */
	private summarizeCompleteBatches(): void {
		const pending = this.data.pendingInsightEntries;
		const completeBatchCount = Math.floor(pending.length / INSIGHT_BATCH_SIZE);
		if (completeBatchCount === 0) return;
		this.data.pendingInsightEntries = pending.slice(completeBatchCount * INSIGHT_BATCH_SIZE);
		const recordsById = new Map(this.data.judgedEvidence.map((record) => [record.id, record]));
		const now = Date.now();
		try {
			for (let index = 0; index < completeBatchCount; index++) {
				const batch = pending
					.slice(index * INSIGHT_BATCH_SIZE, (index + 1) * INSIGHT_BATCH_SIZE)
					.map((id) => recordsById.get(id))
					.filter((record): record is JudgedEvidenceRecord => record !== undefined);
				this.mergeInsights(this.insightExtractor(batch, now));
			}
		} catch (error) {
			console.warn(INSIGHT_WARNING);
			this.data.warnings.push({
				code: "insight-extraction-failed",
				message: `Retrieval insight extraction failed; memory save continued without insights: ${error instanceof Error ? error.message : String(error)}`,
				timestamp: now,
			});
		}
	}

	private mergeInsights(insights: readonly RetrievalInsight[]): void {
		for (const insight of insights) {
			const existingIndex = this.data.insights.findIndex((entry) => entry.clusterKey === insight.clusterKey);
			if (existingIndex < 0) {
				this.data.insights.push(insight);
				continue;
			}
			const existing = this.data.insights[existingIndex];
			const sources = Array.from(new Set([...existing.recommendedSources, ...insight.recommendedSources])).slice(
				0,
				3,
			);
			const methods = Array.from(new Set([...existing.recommendedMethods, ...insight.recommendedMethods])).slice(
				0,
				3,
			);
			const support = existing.supportingEvidenceCount + insight.supportingEvidenceCount;
			this.data.insights[existingIndex] = {
				...existing,
				recommendedSources: sources,
				recommendedMethods: methods,
				rationale: insight.rationale,
				supportingEvidenceCount: support,
				confidence: Math.max(existing.confidence, insight.confidence, Math.min(1, support / 100)),
				updatedAt: Math.max(existing.updatedAt, insight.updatedAt),
			};
		}
	}

	private upsertEvidence(ref: SessionEvidenceRef, timestamp: number): void {
		const excerpt = ref.excerpt ?? ref.content ?? "";
		const record: EvidenceChunkRecord = {
			...ref,
			excerptHash: hashText(normalizeEvidenceText(excerpt)),
			firstSeenAt: timestamp,
			lastSeenAt: timestamp,
		};
		const existingIndex = this.data.evidenceChunks.findIndex(
			(entry) => entry.stableEvidenceId === ref.stableEvidenceId,
		);
		if (existingIndex >= 0) {
			const existing = this.data.evidenceChunks[existingIndex];
			this.data.evidenceChunks[existingIndex] = {
				...record,
				firstSeenAt: existing.firstSeenAt,
				lastSeenAt: timestamp,
			};
		} else {
			this.data.evidenceChunks.push(record);
		}
	}
}

export function normalizeSessionEvidenceRef(
	input: Omit<SessionEvidenceRef, "stableEvidenceId"> & { readonly stableEvidenceId?: string },
): SessionEvidenceRef {
	const context = normalizeEvidenceContext(input);
	const {
		retrieverMix: _retrieverMix,
		parserType: _parserType,
		documentType: _documentType,
		documentArea: _documentArea,
		evidenceType: _evidenceType,
		evidenceLocation: _evidenceLocation,
		confidence: _confidence,
		...base
	} = input;
	if (base.stableEvidenceId && isPathOpaqueIdentifier(base.stableEvidenceId)) {
		return { ...base, stableEvidenceId: base.stableEvidenceId, ...context };
	}
	return { ...normalizeEvidenceRef(base), ...context };
}
