import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export interface MemoryEntry {
	id: string;
	query: string;
	method: string;
	outcome: "pending" | "useful" | "not_useful";
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

interface MemoryDataV3 {
	version: 3;
	entries: MemoryEntry[];
}

interface MemoryDataV2 {
	version: 2;
	entries: Array<{
		id: string;
		query: string;
		method: string;
		outcome: "success" | "failure";
		timestamp: number;
		metadata?: { resultCount?: number };
	}>;
}

interface MemoryDataV1 {
	patterns: Record<string, Record<string, { success: number; fail: number }>>;
}

const MAX_ENTRIES = 500;
const MAX_PENDING_ATTEMPTS = 100;

function isV1(data: unknown): data is MemoryDataV1 {
	if (typeof data !== "object" || data === null) return false;
	const obj = data as Record<string, unknown>;
	return typeof obj.patterns === "object" && obj.patterns !== null && !("version" in obj);
}

function isV2(data: unknown): data is MemoryDataV2 {
	if (typeof data !== "object" || data === null) return false;
	const obj = data as Record<string, unknown>;
	return obj.version === 2 && Array.isArray(obj.entries);
}

function isV3(data: unknown): data is MemoryDataV3 {
	if (typeof data !== "object" || data === null) return false;
	const obj = data as Record<string, unknown>;
	return obj.version === 3 && Array.isArray(obj.entries);
}

function migrateV1(data: MemoryDataV1): MemoryDataV2 {
	const now = Date.now();
	const entries: MemoryDataV2["entries"] = [];

	for (const [patternKey, methods] of Object.entries(data.patterns)) {
		for (const [methodName, stats] of Object.entries(methods)) {
			for (let i = 0; i < stats.success; i++) {
				entries.push({
					id: randomUUID(),
					query: patternKey,
					method: methodName,
					outcome: "success",
					timestamp: now,
				});
			}
			for (let i = 0; i < stats.fail; i++) {
				entries.push({
					id: randomUUID(),
					query: patternKey,
					method: methodName,
					outcome: "failure",
					timestamp: now,
				});
			}
		}
	}

	return { version: 2, entries };
}

function migrateV2(data: MemoryDataV2): MemoryDataV3 {
	return {
		version: 3,
		entries: data.entries.map((entry) => ({
			...entry,
			outcome: (entry.outcome === "success" ? "useful" : "not_useful") as "useful" | "not_useful",
		})),
	};
}

export interface RetrievalMemoryOptions {
	storagePath: string;
}

export class RetrievalMemory {
	private readonly storagePath: string;
	private data: MemoryDataV3 = { version: 3, entries: [] };
	private readonly attempts = new Map<string, SearchAttempt>();
	private readonly sourceToAttemptId = new Map<string, string>();

	constructor(options: RetrievalMemoryOptions) {
		this.storagePath = options.storagePath;
	}

	load(): void {
		if (!existsSync(this.storagePath)) {
			this.data = { version: 3, entries: [] };
			return;
		}
		try {
			const content = readFileSync(this.storagePath, "utf-8");
			const parsed: unknown = JSON.parse(content);
			if (isV3(parsed)) {
				this.data = parsed;
			} else if (isV2(parsed)) {
				this.data = migrateV2(parsed);
				this.save();
			} else if (isV1(parsed)) {
				this.data = migrateV2(migrateV1(parsed));
				this.save();
			} else {
				console.warn("[AutoRAG] Memory file has unexpected structure, starting fresh");
				this.data = { version: 3, entries: [] };
			}
		} catch {
			console.warn(`[AutoRAG] Could not parse memory file at ${this.storagePath}, starting fresh`);
			this.data = { version: 3, entries: [] };
		}
	}

	save(): void {
		const tmpPath = `${this.storagePath}.tmp`;
		const dir = dirname(this.storagePath);
		if (!existsSync(dir)) {
			throw new Error(`Memory storage directory does not exist: ${dir}`);
		}
		if (this.data.entries.length > MAX_ENTRIES) {
			this.data.entries = this.data.entries.slice(-MAX_ENTRIES);
		}
		writeFileSync(tmpPath, JSON.stringify(this.data, null, 2), "utf-8");
		renameSync(tmpPath, this.storagePath);
	}

	append(entry: Omit<MemoryEntry, "id" | "timestamp">): MemoryEntry {
		const full: MemoryEntry = {
			id: randomUUID(),
			timestamp: Date.now(),
			...entry,
		};
		this.data.entries.push(full);
		return full;
	}

	getEntries(): readonly MemoryEntry[] {
		return this.data.entries;
	}

	registerAttempt(attempt: SearchAttempt): void {
		this.attempts.set(attempt.id, attempt);
		for (const source of attempt.sources) {
			this.sourceToAttemptId.set(source, attempt.id);
		}
		if (this.attempts.size > MAX_PENDING_ATTEMPTS) {
			const oldest = this.attempts.keys().next().value as string;
			this.clearAttempt(oldest);
		}
	}

	recordResultFeedback(feedback: ResultFeedback[]): void {
		const resolvedAttemptIds = new Set<string>();

		for (const fb of feedback) {
			const attemptId = this.sourceToAttemptId.get(fb.source);
			if (!attemptId) continue;

			resolvedAttemptIds.add(attemptId);
			const entry = this.data.entries.find((e) => e.id === attemptId);
			if (!entry) continue;
			if (entry.outcome !== "pending" && entry.outcome !== "not_useful") continue;

			if (fb.useful) {
				entry.outcome = "useful";
			} else if (entry.outcome === "pending") {
				entry.outcome = "not_useful";
			}
		}

		for (const attemptId of resolvedAttemptIds) {
			this.clearAttempt(attemptId);
		}
	}

	resolvePendingEntries(query: string, method: string | null, outcome: "useful" | "not_useful"): void {
		for (const entry of this.data.entries) {
			if (entry.outcome !== "pending") continue;
			if (entry.query !== query) continue;
			if (method !== null && entry.method !== method) continue;
			entry.outcome = outcome;
		}
		for (const [attemptId, attempt] of this.attempts) {
			if (attempt.query === query && (method === null || attempt.method === method)) {
				this.clearAttempt(attemptId);
			}
		}
	}

	recordFeedback(query: string, methodName: string, satisfied: boolean): void {
		this.append({
			query,
			method: methodName,
			outcome: satisfied ? "useful" : "not_useful",
		});
	}

	getMethodPriority(query: string): Array<{ method: string; score: number }> {
		if (this.data.entries.length === 0) return [];

		const queryLower = query.toLowerCase();
		const methodScores: Record<string, { useful: number; not_useful: number }> = {};

		for (const entry of this.data.entries) {
			if (entry.outcome === "pending") continue;

			const entryLower = entry.query.toLowerCase();
			if (entryLower !== queryLower && !entryLower.includes(queryLower) && !queryLower.includes(entryLower)) {
				continue;
			}

			if (!methodScores[entry.method]) {
				methodScores[entry.method] = { useful: 0, not_useful: 0 };
			}
			methodScores[entry.method][entry.outcome]++;
		}

		return Object.entries(methodScores)
			.map(([method, stats]) => {
				const total = stats.useful + stats.not_useful;
				return { method, score: total > 0 ? stats.useful / total : 0 };
			})
			.sort((a, b) => b.score - a.score);
	}

	private clearAttempt(attemptId: string): void {
		const attempt = this.attempts.get(attemptId);
		if (attempt) {
			for (const source of attempt.sources) {
				if (this.sourceToAttemptId.get(source) === attemptId) {
					this.sourceToAttemptId.delete(source);
				}
			}
			this.attempts.delete(attemptId);
		}
	}
}
