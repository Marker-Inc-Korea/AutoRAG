import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

interface MethodStats {
	success: number;
	fail: number;
}

interface MemoryData {
	patterns: Record<string, Record<string, MethodStats>>;
}

const STOP_WORDS = new Set(["a", "an", "the", "is", "in", "on", "at", "to", "for", "of", "and", "or", "with"]);

function extractKeywords(query: string): string[] {
	return query
		.toLowerCase()
		.split(/\s+/)
		.filter((w) => w.length > 2 && !STOP_WORDS.has(w));
}

function keywordOverlap(a: string[], b: string[]): number {
	const setB = new Set(b);
	return a.filter((w) => setB.has(w)).length;
}

export interface RetrievalMemoryOptions {
	storagePath: string;
}

export class RetrievalMemory {
	private readonly storagePath: string;
	private data: MemoryData = { patterns: {} };

	constructor(options: RetrievalMemoryOptions) {
		this.storagePath = options.storagePath;
	}

	load(): void {
		if (!existsSync(this.storagePath)) {
			this.data = { patterns: {} };
			return;
		}
		try {
			const content = readFileSync(this.storagePath, "utf-8");
			const parsed = JSON.parse(content) as MemoryData;
			if (parsed && typeof parsed.patterns === "object") {
				this.data = parsed;
			} else {
				console.warn("[AutoRAG] Memory file has unexpected structure, starting fresh");
				this.data = { patterns: {} };
			}
		} catch {
			console.warn(`[AutoRAG] Could not parse memory file at ${this.storagePath}, starting fresh`);
			this.data = { patterns: {} };
		}
	}

	save(): void {
		const tmpPath = `${this.storagePath}.tmp`;
		const dir = dirname(this.storagePath);
		if (!existsSync(dir)) {
			throw new Error(`Memory storage directory does not exist: ${dir}`);
		}
		writeFileSync(tmpPath, JSON.stringify(this.data, null, 2), "utf-8");
		renameSync(tmpPath, this.storagePath);
	}

	recordFeedback(query: string, methodName: string, satisfied: boolean): void {
		const keywords = extractKeywords(query);
		const patternKey = keywords.sort().join(" ");
		if (!patternKey) return;

		if (!this.data.patterns[patternKey]) {
			this.data.patterns[patternKey] = {};
		}
		const pattern = this.data.patterns[patternKey];
		if (!pattern[methodName]) {
			pattern[methodName] = { success: 0, fail: 0 };
		}
		if (satisfied) {
			pattern[methodName].success++;
		} else {
			pattern[methodName].fail++;
		}
	}

	getMethodPriority(query: string): Array<{ method: string; score: number }> {
		const keywords = extractKeywords(query);
		if (keywords.length === 0) return [];

		const methodScores: Record<string, { success: number; fail: number }> = {};

		for (const [patternKey, methods] of Object.entries(this.data.patterns)) {
			const patternKeywords = patternKey.split(" ");
			const overlap = keywordOverlap(keywords, patternKeywords);
			if (overlap === 0) continue;

			for (const [methodName, stats] of Object.entries(methods)) {
				if (!methodScores[methodName]) {
					methodScores[methodName] = { success: 0, fail: 0 };
				}
				methodScores[methodName].success += stats.success * overlap;
				methodScores[methodName].fail += stats.fail * overlap;
			}
		}

		return Object.entries(methodScores)
			.map(([method, stats]) => ({
				method,
				score: stats.success + stats.fail > 0 ? stats.success / (stats.success + stats.fail) : 0,
			}))
			.sort((a, b) => b.score - a.score);
	}
}
