import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import {
	isPathOpaqueIdentifier,
	type NormalizedEvidenceRef,
	normalizeEvidenceRef,
	normalizeEvidenceText,
} from "../retrieval/evidence-id.ts";

export type FeedbackOutcome = "pending" | "useful" | "not_useful";
export type FeedbackSentiment = "useful" | "not_useful";
export type FeedbackSignalSource = "explicit" | "followup" | "retry";

export interface SignalDefaults {
	readonly explicitWeight: number;
	readonly followupWeight: number;
	readonly retryWeight: number;
	readonly implicitCap: number;
	readonly decayHalfLifeMs?: number;
}

export interface EvidenceChunkRecord extends NormalizedEvidenceRef {
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
	readonly createdAt: number;
}

export interface FeedbackSignal {
	readonly id: string;
	readonly target:
		| { readonly type: "curated_result"; readonly resultId: string }
		| { readonly type: "evidence_chunk"; readonly stableEvidenceId: string }
		| { readonly type: "method"; readonly method: string };
	readonly query: string;
	readonly method?: string;
	readonly sentiment: FeedbackSentiment;
	readonly source: FeedbackSignalSource;
	readonly weight: number;
	readonly confidenceCap: number;
	readonly eventId: string;
	readonly timestamp: number;
}

export interface MethodHint {
	readonly method: string;
	readonly score: number;
	readonly confidence: number;
	readonly reason: string;
}

export interface RetrievalInsight {
	readonly id: string;
	readonly clusterKey: string;
	readonly domain: string;
	readonly recommendedSources: string[];
	readonly recommendedMethods: string[];
	readonly rationale: string;
	supportingSignalCount: number;
	confidence: number;
	readonly createdAt: number;
	updatedAt: number;
}

export interface InsightExtractionSignal {
	readonly signal: FeedbackSignal;
	readonly method?: string;
	readonly source?: string;
}

export type InsightExtractor = (signals: readonly InsightExtractionSignal[], now: number) => RetrievalInsight[];

export interface MemoryWarning {
	readonly code: string;
	readonly message: string;
	readonly timestamp: number;
}

export interface MemorySchemaV4 {
	readonly version: 4;
	curatedResults: CuratedResultRecord[];
	evidenceChunks: EvidenceChunkRecord[];
	feedbackSignals: FeedbackSignal[];
	readonly signalDefaults: SignalDefaults;
	warnings: MemoryWarning[];
	insights: RetrievalInsight[];
	pendingInsightSignals: InsightExtractionSignal[];
}

export interface MemoryEntry {
	id: string;
	query: string;
	method: string;
	outcome: FeedbackOutcome;
	timestamp: number;
	metadata?: { resultCount?: number };
}

export interface SearchAttempt {
	id: string;
	query: string;
	method: string;
	sources: string[];
	timestamp: number;
}

export interface ResultFeedback {
	source: string;
	useful: boolean;
}

export interface SessionEvidenceRef extends NormalizedEvidenceRef {
	readonly metadata?: Record<string, unknown>;
}

export interface SessionCuratedResultInput {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	readonly content: string;
	readonly method: string;
	readonly source: string;
	readonly evidenceRefs: readonly SessionEvidenceRef[];
}

export interface SessionRecordInput {
	readonly sessionId: string;
	readonly query: string;
	readonly results: readonly SessionCuratedResultInput[];
}

export interface NumberedFeedbackInput {
	readonly sessionId: string;
	readonly query: string;
	readonly feedback: readonly { readonly number: number; readonly useful: boolean }[];
}

export interface RetrievalMemoryOptions {
	storagePath: string;
	insightExtractor?: InsightExtractor;
}

const DEFAULT_SIGNAL_DEFAULTS: SignalDefaults = {
	explicitWeight: 1,
	followupWeight: 0.25,
	retryWeight: -0.25,
	implicitCap: 0.5,
};
const MAX_RECORDS = 500;
const MAX_WARNINGS = 50;
const INSIGHT_BATCH_SIZE = 100;
const MAX_INSIGHTS = 200;
const MIN_INSIGHT_SUPPORT = 5;
const MIN_INSIGHT_SCORE = 3;
const INSIGHT_WARNING = "[AutoRAG] Retrieval memory insight extraction failed; continuing without insights";
const RESET_WARNING = "[AutoRAG] Retrieval memory is not v4-compatible; starting fresh";
const LOCK_WAIT_TIMEOUT_MS = 5_000;
const LOCK_STALE_MS = 30_000;
const LOCK_RETRY_MS = 10;

class RetrievalMemoryLockTimeoutError extends Error {
	constructor() {
		super("Timed out waiting for retrieval memory lock");
		this.name = "RetrievalMemoryLockTimeoutError";
	}
}

function emptyMemoryV4(): MemorySchemaV4 {
	return {
		version: 4,
		curatedResults: [],
		evidenceChunks: [],
		feedbackSignals: [],
		signalDefaults: DEFAULT_SIGNAL_DEFAULTS,
		warnings: [],
		insights: [],
		pendingInsightSignals: [],
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function cloneMemory(data: MemorySchemaV4): MemorySchemaV4 {
	return structuredClone(data);
}

function recordsEqual(left: unknown, right: unknown): boolean {
	return JSON.stringify(left) === JSON.stringify(right);
}

function feedbackTargetKey(target: FeedbackSignal["target"]): string {
	if (target.type === "curated_result") return `curated_result:${target.resultId}`;
	if (target.type === "evidence_chunk") return `evidence_chunk:${target.stableEvidenceId}`;
	return `method:${target.method}`;
}

function feedbackSignalKey(signal: FeedbackSignal): string {
	return `${signal.eventId}\0${feedbackTargetKey(signal.target)}`;
}

function warningKey(warning: MemoryWarning): string {
	return `${warning.timestamp}\0${warning.code}\0${warning.message}`;
}

function isV4(data: unknown): data is Omit<MemorySchemaV4, "insights" | "pendingInsightSignals"> & {
	insights?: unknown;
	pendingInsightSignals?: unknown;
} {
	return (
		isRecord(data) &&
		data.version === 4 &&
		Array.isArray(data.curatedResults) &&
		Array.isArray(data.evidenceChunks) &&
		Array.isArray(data.feedbackSignals) &&
		isRecord(data.signalDefaults) &&
		typeof data.signalDefaults.explicitWeight === "number" &&
		Array.isArray(data.warnings) &&
		(data.insights === undefined || Array.isArray(data.insights)) &&
		(data.pendingInsightSignals === undefined || Array.isArray(data.pendingInsightSignals))
	);
}

function isInsight(value: unknown): value is RetrievalInsight {
	return (
		isRecord(value) &&
		typeof value.id === "string" &&
		typeof value.clusterKey === "string" &&
		typeof value.domain === "string" &&
		Array.isArray(value.recommendedSources) &&
		Array.isArray(value.recommendedMethods) &&
		typeof value.rationale === "string" &&
		typeof value.supportingSignalCount === "number" &&
		typeof value.confidence === "number" &&
		typeof value.createdAt === "number" &&
		typeof value.updatedAt === "number"
	);
}

function isInsightExtractionSignal(value: unknown): value is InsightExtractionSignal {
	return isRecord(value) && isRecord(value.signal) && typeof value.signal.query === "string";
}

function normalizeV4(
	data: Omit<MemorySchemaV4, "insights" | "pendingInsightSignals"> & {
		insights?: unknown;
		pendingInsightSignals?: unknown;
	},
): MemorySchemaV4 {
	return {
		version: 4,
		curatedResults: data.curatedResults,
		evidenceChunks: data.evidenceChunks,
		feedbackSignals: data.feedbackSignals,
		signalDefaults: data.signalDefaults,
		warnings: data.warnings,
		insights: Array.isArray(data.insights) ? data.insights.filter(isInsight) : [],
		pendingInsightSignals: Array.isArray(data.pendingInsightSignals)
			? data.pendingInsightSignals.filter(isInsightExtractionSignal).slice(-INSIGHT_BATCH_SIZE + 1)
			: [],
	};
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

function queryMatches(entryQuery: string, query: string): boolean {
	const a = entryQuery.toLowerCase();
	const b = query.toLowerCase();
	return a === b || a.includes(b) || b.includes(a);
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

function defaultInsightExtractor(signals: readonly InsightExtractionSignal[], now: number): RetrievalInsight[] {
	const clusters = new Map<
		string,
		{
			domain: string;
			score: number;
			support: number;
			explicitSupport: number;
			methods: Map<string, number>;
			sources: Map<string, number>;
			firstSeenAt: number;
			lastSeenAt: number;
		}
	>();
	for (const item of signals) {
		const method = item.method;
		if (!method) continue;
		const domain = normalizeInsightDomain(item.signal.query);
		if (domain.length === 0) continue;
		const current = clusters.get(domain) ?? {
			domain,
			score: 0,
			support: 0,
			explicitSupport: 0,
			methods: new Map<string, number>(),
			sources: new Map<string, number>(),
			firstSeenAt: item.signal.timestamp,
			lastSeenAt: item.signal.timestamp,
		};
		current.score += item.signal.weight;
		current.support++;
		if (item.signal.source === "explicit") current.explicitSupport++;
		current.methods.set(method, (current.methods.get(method) ?? 0) + 1);
		if (item.source) current.sources.set(item.source, (current.sources.get(item.source) ?? 0) + 1);
		current.firstSeenAt = Math.min(current.firstSeenAt, item.signal.timestamp);
		current.lastSeenAt = Math.max(current.lastSeenAt, item.signal.timestamp);
		clusters.set(domain, current);
	}
	return Array.from(clusters.values())
		.filter(
			(cluster) =>
				cluster.support >= MIN_INSIGHT_SUPPORT &&
				cluster.explicitSupport > 0 &&
				cluster.score >= MIN_INSIGHT_SCORE &&
				cluster.methods.size > 0,
		)
		.map((cluster) => {
			const methods = Array.from(cluster.methods.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
			const sources = Array.from(cluster.sources.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
			const methodConsistency = methods[0][1] / cluster.support;
			const confidence = Math.min(1, Math.max(0, (cluster.score / cluster.support) * methodConsistency));
			return {
				id: `insight:${hashText(cluster.domain).slice(0, 24)}`,
				clusterKey: cluster.domain,
				domain: cluster.domain,
				recommendedSources: sources.slice(0, 3).map(([source]) => source),
				recommendedMethods: methods.slice(0, 3).map(([method]) => method),
				rationale: `${cluster.support} evicted feedback signal(s) consistently supported ${methods[0][0]}; advisory only, not a method disable rule`,
				supportingSignalCount: cluster.support,
				confidence,
				createdAt: now,
				updatedAt: now,
			};
		})
		.sort(
			(a, b) =>
				b.confidence - a.confidence ||
				b.supportingSignalCount - a.supportingSignalCount ||
				a.domain.localeCompare(b.domain),
		);
}

export class RetrievalMemory {
	private readonly storagePath: string;
	private readonly insightExtractor: InsightExtractor;
	private data: MemorySchemaV4 = emptyMemoryV4();
	private persistedData: MemorySchemaV4 = emptyMemoryV4();
	private legacyEntries = new Map<string, MemoryEntry>();
	private legacySourceToAttemptId = new Map<string, string>();

	constructor(options: RetrievalMemoryOptions) {
		this.storagePath = options.storagePath;
		this.insightExtractor = options.insightExtractor ?? defaultInsightExtractor;
	}

	load(): void {
		this.legacyEntries = new Map();
		this.legacySourceToAttemptId = new Map();
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
			this.capData();
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

	getSchema(): MemorySchemaV4 {
		return this.data;
	}

	getSignalCount(): number {
		return this.data.feedbackSignals.length;
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
				createdAt: now,
			};
			const existingIndex = this.data.curatedResults.findIndex((entry) => entry.resultId === id);
			if (existingIndex >= 0) this.data.curatedResults[existingIndex] = record;
			else this.data.curatedResults.push(record);
		}
	}

	recordNumberedFeedback(input: NumberedFeedbackInput): boolean {
		let changed = false;
		for (const item of input.feedback) {
			const curated = this.data.curatedResults.find(
				(result) => result.sessionId === input.sessionId && result.number === item.number,
			);
			if (!curated) continue;
			const sentiment: FeedbackSentiment = item.useful ? "useful" : "not_useful";
			const eventId = `${input.sessionId}:${item.number}:${sentiment}`;
			if (this.data.feedbackSignals.some((signal) => signal.eventId === eventId)) continue;
			const sign = item.useful ? 1 : -1;
			const explicitWeight = this.data.signalDefaults.explicitWeight * sign;
			this.data.feedbackSignals.push({
				id: randomUUID(),
				target: { type: "curated_result", resultId: curated.resultId },
				query: curated.query,
				sentiment,
				source: "explicit",
				weight: explicitWeight,
				confidenceCap: 1,
				eventId,
				timestamp: Date.now(),
			});
			const evidenceWeight = curated.evidenceIds.length > 0 ? explicitWeight / curated.evidenceIds.length : 0;
			for (const stableEvidenceId of curated.evidenceIds) {
				const evidence = this.data.evidenceChunks.find((chunk) => chunk.stableEvidenceId === stableEvidenceId);
				this.data.feedbackSignals.push({
					id: randomUUID(),
					target: { type: "evidence_chunk", stableEvidenceId },
					query: curated.query,
					method: evidence?.method,
					sentiment,
					source: "explicit",
					weight: evidenceWeight,
					confidenceCap: 1,
					eventId,
					timestamp: Date.now(),
				});
			}
			changed = true;
		}
		return changed;
	}

	recordWeakSignal(query: string, method: string, source: "followup" | "retry"): void {
		const rawWeight =
			source === "followup" ? this.data.signalDefaults.followupWeight : this.data.signalDefaults.retryWeight;
		const cap = this.data.signalDefaults.implicitCap;
		const weight = Math.max(-cap, Math.min(cap, rawWeight));
		this.data.feedbackSignals.push({
			id: randomUUID(),
			target: { type: "method", method },
			query,
			method,
			sentiment: weight >= 0 ? "useful" : "not_useful",
			source,
			weight,
			confidenceCap: cap,
			eventId: randomUUID(),
			timestamp: Date.now(),
		});
	}

	getMethodHints(query: string): MethodHint[] {
		const eventScores = new Map<string, { method: string; score: number; signals: number; cap: number }>();
		for (const signal of this.data.feedbackSignals) {
			if (!queryMatches(signal.query, query)) continue;
			const method = this.methodForSignal(signal);
			if (!method) continue;
			const key = `${signal.eventId}\0${method}`;
			const current = eventScores.get(key) ?? { method, score: 0, signals: 0, cap: signal.confidenceCap };
			current.score += signal.weight;
			current.signals++;
			current.cap = Math.max(current.cap, signal.confidenceCap);
			eventScores.set(key, current);
		}
		const scores = new Map<string, { score: number; signals: number }>();
		for (const eventScore of eventScores.values()) {
			const cappedScore = Math.max(-eventScore.cap, Math.min(eventScore.cap, eventScore.score));
			const current = scores.get(eventScore.method) ?? { score: 0, signals: 0 };
			current.score += cappedScore;
			current.signals += eventScore.signals;
			scores.set(eventScore.method, current);
		}
		return Array.from(scores.entries())
			.map(([method, stats]) => ({
				method,
				score: stats.score,
				confidence: Math.min(1, stats.signals / 5),
				reason: `${stats.signals} feedback signal(s) matched this query; advisory only, not a method disable rule`,
			}))
			.sort((a, b) => b.score - a.score || b.confidence - a.confidence || a.method.localeCompare(b.method));
	}

	getInsights(query: string): RetrievalInsight[] {
		return this.data.insights
			.filter((insight) => insightMatches(insight, query))
			.sort(
				(a, b) =>
					b.confidence - a.confidence ||
					b.supportingSignalCount - a.supportingSignalCount ||
					b.updatedAt - a.updatedAt ||
					a.domain.localeCompare(b.domain),
			);
	}

	// Compatibility projection for existing callers/tests while product code migrates to MethodHint wording.
	getMethodPriority(query: string): Array<{ method: string; score: number }> {
		return this.getMethodHints(query).map((hint) => ({ method: hint.method, score: hint.score }));
	}

	// Compatibility helpers: not persisted as v3 entries.
	append(entry: Omit<MemoryEntry, "id" | "timestamp">): MemoryEntry {
		const full: MemoryEntry = { id: randomUUID(), timestamp: Date.now(), ...entry };
		this.legacyEntries.set(full.id, full);
		if (entry.outcome !== "pending") {
			const sentiment = entry.outcome === "useful" ? "useful" : "not_useful";
			this.data.feedbackSignals.push({
				id: full.id,
				target: { type: "method", method: entry.method },
				query: entry.query,
				method: entry.method,
				sentiment,
				source: "explicit",
				weight:
					sentiment === "useful"
						? this.data.signalDefaults.explicitWeight
						: -this.data.signalDefaults.explicitWeight,
				confidenceCap: 1,
				eventId: randomUUID(),
				timestamp: full.timestamp,
			});
		}
		return full;
	}

	getEntries(): readonly MemoryEntry[] {
		return Array.from(this.legacyEntries.values());
	}

	registerAttempt(attempt: SearchAttempt): void {
		for (const source of attempt.sources) this.legacySourceToAttemptId.set(source, attempt.id);
	}

	recordResultFeedback(feedback: ResultFeedback[]): void {
		const bySource = new Map(feedback.map((item) => [item.source, item.useful]));
		for (const [source, useful] of bySource) {
			const attemptId = this.legacySourceToAttemptId.get(source);
			const entry = attemptId ? this.legacyEntries.get(attemptId) : undefined;
			if (!entry) continue;
			if (entry.outcome === "pending" || (entry.outcome === "not_useful" && useful)) {
				entry.outcome = useful ? "useful" : "not_useful";
				this.recordFeedback(entry.query, entry.method, useful);
			}
		}
	}

	resolvePendingEntries(query: string, method: string | null, outcome: "useful" | "not_useful"): void {
		for (const entry of this.legacyEntries.values()) {
			if (entry.outcome !== "pending") continue;
			if (entry.query !== query) continue;
			if (method !== null && entry.method !== method) continue;
			entry.outcome = outcome;
			this.recordFeedback(entry.query, entry.method, outcome === "useful");
		}
	}

	recordFeedback(query: string, methodName: string, satisfied: boolean): void {
		const sentiment: FeedbackSentiment = satisfied ? "useful" : "not_useful";
		this.data.feedbackSignals.push({
			id: randomUUID(),
			target: { type: "method", method: methodName },
			query,
			method: methodName,
			sentiment,
			source: "explicit",
			weight: satisfied ? this.data.signalDefaults.explicitWeight : -this.data.signalDefaults.explicitWeight,
			confidenceCap: 1,
			eventId: randomUUID(),
			timestamp: Date.now(),
		});
	}

	private readPersistedData(): MemorySchemaV4 {
		if (!existsSync(this.storagePath)) return emptyMemoryV4();
		try {
			const parsed: unknown = JSON.parse(readFileSync(this.storagePath, "utf-8"));
			if (isV4(parsed)) return normalizeV4(parsed);
		} catch (error) {
			if (!(error instanceof Error)) throw error;
		}
		return this.incompatibleMemory();
	}

	private incompatibleMemory(): MemorySchemaV4 {
		console.warn(RESET_WARNING);
		const data = emptyMemoryV4();
		data.warnings.push({
			code: "memory-reset",
			message: "Retrieval memory was reset because it was not v4-compatible",
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

	private mergeWithPersisted(persisted: MemorySchemaV4): MemorySchemaV4 {
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

		const baselineSignalIds = new Set(this.persistedData.feedbackSignals.map((signal) => signal.id));
		const signalKeys = new Set<string>();
		merged.feedbackSignals = merged.feedbackSignals.filter((signal) => {
			const key = feedbackSignalKey(signal);
			if (signalKeys.has(key)) return false;
			signalKeys.add(key);
			return true;
		});
		for (const signal of this.data.feedbackSignals) {
			if (baselineSignalIds.has(signal.id)) continue;
			const key = feedbackSignalKey(signal);
			if (signalKeys.has(key)) continue;
			signalKeys.add(key);
			merged.feedbackSignals.push(signal);
		}

		const baselineWarningKeys = new Set(this.persistedData.warnings.map(warningKey));
		const warningKeys = new Set(merged.warnings.map(warningKey));
		for (const warning of this.data.warnings) {
			const key = warningKey(warning);
			if (baselineWarningKeys.has(key) || warningKeys.has(key)) continue;
			warningKeys.add(key);
			merged.warnings.push(warning);
		}

		const baselinePendingKeys = new Set(
			this.persistedData.pendingInsightSignals.map((item) => feedbackSignalKey(item.signal)),
		);
		const pendingKeys = new Set(merged.pendingInsightSignals.map((item) => feedbackSignalKey(item.signal)));
		for (const item of this.data.pendingInsightSignals) {
			const key = feedbackSignalKey(item.signal);
			if (baselinePendingKeys.has(key) || pendingKeys.has(key)) continue;
			pendingKeys.add(key);
			merged.pendingInsightSignals.push(item);
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
				? Math.max(0, insight.supportingSignalCount - baseline.supportingSignalCount)
				: insight.supportingSignalCount;
			const support = existing.supportingSignalCount + supportDelta;
			merged.insights[existingIndex] = {
				...existing,
				recommendedSources: Array.from(
					new Set([...existing.recommendedSources, ...insight.recommendedSources]),
				).slice(0, 3),
				recommendedMethods: Array.from(
					new Set([...existing.recommendedMethods, ...insight.recommendedMethods]),
				).slice(0, 3),
				rationale: insight.updatedAt >= existing.updatedAt ? insight.rationale : existing.rationale,
				supportingSignalCount: support,
				confidence: Math.max(existing.confidence, insight.confidence, Math.min(1, support / 100)),
				createdAt: Math.min(existing.createdAt, insight.createdAt),
				updatedAt: Math.max(existing.updatedAt, insight.updatedAt),
			};
		}

		return merged;
	}

	private capData(): void {
		const evictedSignals =
			this.data.feedbackSignals.length > MAX_RECORDS
				? this.data.feedbackSignals.slice(0, this.data.feedbackSignals.length - MAX_RECORDS).map((signal) => ({
						signal,
						method: this.methodForSignal(signal),
						source: this.sourceForSignal(signal),
					}))
				: [];
		this.extractInsightsFromEvictedSignals(evictedSignals);

		this.data.curatedResults = this.data.curatedResults.slice(-MAX_RECORDS);
		this.data.evidenceChunks = this.data.evidenceChunks.slice(-MAX_RECORDS);
		this.data.feedbackSignals = this.data.feedbackSignals.slice(-MAX_RECORDS);
		this.data.insights = this.data.insights
			.sort(
				(a, b) =>
					b.confidence - a.confidence ||
					b.supportingSignalCount - a.supportingSignalCount ||
					b.updatedAt - a.updatedAt ||
					a.domain.localeCompare(b.domain),
			)
			.slice(0, MAX_INSIGHTS);
		this.data.warnings = this.data.warnings.slice(-MAX_WARNINGS);
	}

	private extractInsightsFromEvictedSignals(evictedSignals: readonly InsightExtractionSignal[]): void {
		const candidates = [...this.data.pendingInsightSignals, ...evictedSignals];
		const completeBatchCount = Math.floor(candidates.length / INSIGHT_BATCH_SIZE);
		this.data.pendingInsightSignals = candidates.slice(completeBatchCount * INSIGHT_BATCH_SIZE);
		if (completeBatchCount === 0) return;
		const now = Date.now();
		try {
			for (let i = 0; i < completeBatchCount; i++) {
				const batch = candidates.slice(i * INSIGHT_BATCH_SIZE, (i + 1) * INSIGHT_BATCH_SIZE);
				this.mergeInsights(this.insightExtractor(batch, now));
			}
		} catch {
			console.warn(INSIGHT_WARNING);
			this.data.warnings.push({
				code: "insight-extraction-failed",
				message: "Retrieval insight extraction failed; memory save continued without blocking capping",
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
			const support = existing.supportingSignalCount + insight.supportingSignalCount;
			this.data.insights[existingIndex] = {
				...existing,
				recommendedSources: sources,
				recommendedMethods: methods,
				rationale: insight.rationale,
				supportingSignalCount: support,
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

	private methodForSignal(signal: FeedbackSignal): string | undefined {
		if (signal.method) return signal.method;
		const target = signal.target;
		if (target.type === "method") return target.method;
		if (target.type === "evidence_chunk") {
			return this.data.evidenceChunks.find((chunk) => chunk.stableEvidenceId === target.stableEvidenceId)?.method;
		}
		if (target.type === "curated_result") {
			const result = this.data.curatedResults.find((entry) => entry.resultId === target.resultId);
			const firstEvidence = result?.evidenceIds[0];
			return firstEvidence
				? this.data.evidenceChunks.find((chunk) => chunk.stableEvidenceId === firstEvidence)?.method
				: undefined;
		}
		return undefined;
	}

	private sourceForSignal(signal: FeedbackSignal): string | undefined {
		const target = signal.target;
		if (target.type === "evidence_chunk") {
			return this.data.evidenceChunks.find((chunk) => chunk.stableEvidenceId === target.stableEvidenceId)?.source;
		}
		if (target.type === "curated_result") {
			const result = this.data.curatedResults.find((entry) => entry.resultId === target.resultId);
			const firstEvidence = result?.evidenceIds[0];
			return firstEvidence
				? this.data.evidenceChunks.find((chunk) => chunk.stableEvidenceId === firstEvidence)?.source
				: undefined;
		}
		return undefined;
	}
}

export function normalizeSessionEvidenceRef(
	input: Omit<SessionEvidenceRef, "stableEvidenceId"> & { readonly stableEvidenceId?: string },
): SessionEvidenceRef {
	if (input.stableEvidenceId && isPathOpaqueIdentifier(input.stableEvidenceId)) {
		return { ...input, stableEvidenceId: input.stableEvidenceId };
	}
	return normalizeEvidenceRef(input);
}
