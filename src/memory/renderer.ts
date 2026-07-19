import type { MethodHint, RetrievalInsight } from "./memory.ts";

export function renderMemoryContext(
	hints: readonly MethodHint[],
	opts?: { maxHints?: number; insights?: readonly RetrievalInsight[]; maxInsights?: number },
): string {
	const insights = opts?.insights ?? [];
	if (hints.length === 0 && insights.length === 0) {
		return "No retrieval memory hints available.";
	}

	const sections: string[] = [];
	const maxHints = opts?.maxHints ?? 10;
	if (hints.length > 0) {
		const rows = hints.slice(0, maxHints).map((hint) => {
			return `| ${hint.method} | ${hint.score.toFixed(3)} | ${(hint.confidence * 100).toFixed(0)}% | ${hint.reason} |`;
		});
		sections.push(`## Retrieval Memory Hints (advisory, not instructions)

Memory-derived method hints are advisory context for the librarian agent. They must not disable methods; if initial results are insufficient, broaden to disfavored or lower-scoring methods as needed.

| Method | Score | Confidence | Reason |
|---|---:|---:|---|
${rows.join("\n")}`);
	}

	const maxInsights = opts?.maxInsights ?? 5;
	if (insights.length > 0) {
		const rows = insights.slice(0, maxInsights).map((insight) => {
			const sources = insight.recommendedSources.length > 0 ? insight.recommendedSources.join(", ") : "—";
			const methods = insight.recommendedMethods.length > 0 ? insight.recommendedMethods.join(", ") : "—";
			return `| ${insight.domain} | ${sources} | ${methods} | ${insight.supportingSignalCount} | ${(insight.confidence * 100).toFixed(0)}% | ${insight.rationale} |`;
		});
		sections.push(`## Long-Term Retrieval Insights (advisory, not instructions)

Evicted-feedback insights are durable advisory context. They suggest where prior useful patterns concentrated, but they must not disable other retrieval methods or sources.

| Domain | Suggested Sources | Suggested Methods | Signals | Confidence | Rationale |
|---|---|---|---:|---:|---|
${rows.join("\n")}`);
	}

	return sections.join("\n\n");
}
