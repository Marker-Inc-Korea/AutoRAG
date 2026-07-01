import { createHash, randomUUID } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
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
}

const DEFAULT_SIGNAL_DEFAULTS: SignalDefaults = {
	explicitWeight: 1,
	followupWeight: 0.25,
	retryWeight: -0.25,
	implicitCap: 0.5,
};
const MAX_RECORDS = 500;
const MAX_WARNINGS = 50;
const RESET_WARNING = "[AutoRAG] Retrieval memory is not v4-compatible; starting fresh";

function emptyMemoryV4(): MemorySchemaV4 {
	return {
		version: 4,
		curatedResults: [],
		evidenceChunks: [],
		feedbackSignals: [],
		signalDefaults: DEFAULT_SIGNAL_DEFAULTS,
		warnings: [],
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function isV4(data: unknown): data is MemorySchemaV4 {
	return (
		isRecord(data) &&
		data.version === 4 &&
		Array.isArray(data.curatedResults) &&
		Array.isArray(data.evidenceChunks) &&
		Array.isArray(data.feedbackSignals) &&
		isRecord(data.signalDefaults) &&
		typeof data.signalDefaults.explicitWeight === "number" &&
		Array.isArray(data.warnings)
	);
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

export class RetrievalMemory {
	private readonly storagePath: string;
	private data: MemorySchemaV4 = emptyMemoryV4();
	private legacyEntries = new Map<string, MemoryEntry>();
	private legacySourceToAttemptId = new Map<string, string>();

	constructor(options: RetrievalMemoryOptions) {
		this.storagePath = options.storagePath;
	}

	load(): void {
		this.legacyEntries = new Map();
		this.legacySourceToAttemptId = new Map();
		if (!existsSync(this.storagePath)) {
			this.data = emptyMemoryV4();
			return;
		}
		try {
			const parsed: unknown = JSON.parse(readFileSync(this.storagePath, "utf-8"));
			if (isV4(parsed)) {
				this.data = parsed;
				return;
			}
			this.resetIncompatible();
		} catch {
			this.resetIncompatible();
		}
	}

	save(): void {
		const dir = dirname(this.storagePath);
		if (!existsSync(dir)) {
			throw new Error(`Memory storage directory does not exist: ${dir}`);
		}
		this.capData();
		const tmpPath = `${this.storagePath}.tmp`;
		writeFileSync(tmpPath, `${JSON.stringify(this.data, null, 2)}\n`, "utf-8");
		renameSync(tmpPath, this.storagePath);
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

	private resetIncompatible(): void {
		console.warn(RESET_WARNING);
		this.data = emptyMemoryV4();
		this.data.warnings.push({
			code: "memory-reset",
			message: "Retrieval memory was reset because it was not v4-compatible",
			timestamp: Date.now(),
		});
	}

	private capData(): void {
		this.data.curatedResults = this.data.curatedResults.slice(-MAX_RECORDS);
		this.data.evidenceChunks = this.data.evidenceChunks.slice(-MAX_RECORDS);
		this.data.feedbackSignals = this.data.feedbackSignals.slice(-MAX_RECORDS);
		this.data.warnings = this.data.warnings.slice(-MAX_WARNINGS);
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
}

export function normalizeSessionEvidenceRef(
	input: Omit<SessionEvidenceRef, "stableEvidenceId"> & { readonly stableEvidenceId?: string },
): SessionEvidenceRef {
	if (input.stableEvidenceId && isPathOpaqueIdentifier(input.stableEvidenceId)) {
		return { ...input, stableEvidenceId: input.stableEvidenceId };
	}
	return normalizeEvidenceRef(input);
}
