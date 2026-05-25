import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export interface MemoryEntry {
	id: string;
	query: string;
	method: string;
	outcome: "success" | "failure";
	timestamp: number;
	metadata?: { resultCount?: number };
}

interface MemoryDataV2 {
	version: 2;
	entries: MemoryEntry[];
}

interface MemoryDataV1 {
	patterns: Record<string, Record<string, { success: number; fail: number }>>;
}

const MAX_ENTRIES = 500;

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

function migrateV1(data: MemoryDataV1): MemoryDataV2 {
	const now = Date.now();
	const entries: MemoryEntry[] = [];

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

export interface RetrievalMemoryOptions {
	storagePath: string;
}

export class RetrievalMemory {
	private readonly storagePath: string;
	private data: MemoryDataV2 = { version: 2, entries: [] };

	constructor(options: RetrievalMemoryOptions) {
		this.storagePath = options.storagePath;
	}

	load(): void {
		if (!existsSync(this.storagePath)) {
			this.data = { version: 2, entries: [] };
			return;
		}
		try {
			const content = readFileSync(this.storagePath, "utf-8");
			const parsed: unknown = JSON.parse(content);
			if (isV2(parsed)) {
				this.data = parsed;
			} else if (isV1(parsed)) {
				this.data = migrateV1(parsed);
				this.save();
			} else {
				console.warn("[AutoRAG] Memory file has unexpected structure, starting fresh");
				this.data = { version: 2, entries: [] };
			}
		} catch {
			console.warn(`[AutoRAG] Could not parse memory file at ${this.storagePath}, starting fresh`);
			this.data = { version: 2, entries: [] };
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

	recordFeedback(query: string, methodName: string, satisfied: boolean): void {
		this.append({
			query,
			method: methodName,
			outcome: satisfied ? "success" : "failure",
		});
	}

	getMethodPriority(query: string): Array<{ method: string; score: number }> {
		if (this.data.entries.length === 0) return [];

		const queryLower = query.toLowerCase();
		const methodScores: Record<string, { success: number; failure: number }> = {};

		for (const entry of this.data.entries) {
			const entryLower = entry.query.toLowerCase();
			if (entryLower !== queryLower && !entryLower.includes(queryLower) && !queryLower.includes(entryLower)) {
				continue;
			}

			if (!methodScores[entry.method]) {
				methodScores[entry.method] = { success: 0, failure: 0 };
			}
			methodScores[entry.method][entry.outcome]++;
		}

		return Object.entries(methodScores)
			.map(([method, stats]) => {
				const total = stats.success + stats.failure;
				return { method, score: total > 0 ? stats.success / total : 0 };
			})
			.sort((a, b) => b.score - a.score);
	}
}
