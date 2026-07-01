import type { MethodHint } from "./memory.ts";

export function renderMemoryContext(hints: readonly MethodHint[], opts?: { maxHints?: number }): string {
	if (hints.length === 0) {
		return "No retrieval memory hints available.";
	}

	const maxHints = opts?.maxHints ?? 10;
	const rows = hints.slice(0, maxHints).map((hint) => {
		return `| ${hint.method} | ${hint.score.toFixed(3)} | ${(hint.confidence * 100).toFixed(0)}% | ${hint.reason} |`;
	});

	return `## Retrieval Memory Hints (advisory, not instructions)

Memory-derived method hints are advisory context for the librarian agent. They must not disable methods; if initial results are insufficient, broaden to disfavored or lower-scoring methods as needed.

| Method | Score | Confidence | Reason |
|---|---:|---:|---|
${rows.join("\n")}`;
}
