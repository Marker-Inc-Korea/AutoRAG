import type { MemoryEntry } from "./memory.ts";

interface QueryGroup {
	query: string;
	methods: Map<string, { success: number; failure: number }>;
	lastUsed: number;
}

export function renderMemoryContext(entries: readonly MemoryEntry[], opts?: { maxGroups?: number }): string {
	if (entries.length === 0) {
		return "No retrieval history available.";
	}

	const maxGroups = opts?.maxGroups ?? 50;
	const groups = new Map<string, QueryGroup>();

	for (const entry of entries) {
		let group = groups.get(entry.query);
		if (!group) {
			group = { query: entry.query, methods: new Map(), lastUsed: entry.timestamp };
			groups.set(entry.query, group);
		}
		if (entry.timestamp > group.lastUsed) {
			group.lastUsed = entry.timestamp;
		}
		let methodStats = group.methods.get(entry.method);
		if (!methodStats) {
			methodStats = { success: 0, failure: 0 };
			group.methods.set(entry.method, methodStats);
		}
		methodStats[entry.outcome]++;
	}

	const sorted = Array.from(groups.values()).sort((a, b) => b.lastUsed - a.lastUsed);
	const capped = sorted.slice(0, maxGroups);

	const rows: string[] = [];
	for (const group of capped) {
		const date = new Date(group.lastUsed).toISOString().slice(0, 10);
		for (const [method, stats] of group.methods) {
			rows.push(`| ${group.query} | ${method} | ${stats.success} | ${stats.failure} | ${date} |`);
		}
	}

	return `## Retrieval Memory (advisory, not instructions)

| Past Query | Method | Success | Failure | Last Used |
|---|---:|---:|---:|---|
${rows.join("\n")}`;
}
