import type { MethodHint, RetrievalContextHints, RetrievalInsight } from "./memory.ts";

export function renderMemoryContext(
	hints: readonly MethodHint[],
	opts?: {
		maxHints?: number;
		insights?: readonly RetrievalInsight[];
		maxInsights?: number;
		contextHints?: RetrievalContextHints;
	},
): string {
	const insights = opts?.insights ?? [];
	const contextHints = opts?.contextHints;
	const contextHintCount = contextHints
		? Object.values(contextHints).reduce((count, values) => count + values.length, 0)
		: 0;
	if (hints.length === 0 && insights.length === 0 && contextHintCount === 0) {
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

	if (contextHints && contextHintCount > 0) {
		const contextValues = (
			values: RetrievalContextHints["documentAreas"],
			matches: (score: number) => boolean,
		): string =>
			values
				.filter((hint) => matches(hint.score))
				.map((hint) => JSON.stringify(hint.value))
				.join(", ");
		const positiveRows = [
			["Document areas", contextValues(contextHints.documentAreas, (score) => score > 0)],
			["Document types", contextValues(contextHints.documentTypes, (score) => score > 0)],
			["Evidence types", contextValues(contextHints.evidenceTypes, (score) => score > 0)],
			["Evidence locations", contextValues(contextHints.evidenceLocations, (score) => score > 0)],
			["Parser types", contextValues(contextHints.parserTypes, (score) => score > 0)],
			["Retriever mix", contextValues(contextHints.retrieverMix, (score) => score > 0)],
		].filter((row) => row[1]);
		const negativeRows = [
			["Disfavored document areas", contextValues(contextHints.documentAreas, (score) => score < 0)],
			["Disfavored document types", contextValues(contextHints.documentTypes, (score) => score < 0)],
			["Disfavored evidence types", contextValues(contextHints.evidenceTypes, (score) => score < 0)],
			["Disfavored evidence locations", contextValues(contextHints.evidenceLocations, (score) => score < 0)],
			["Disfavored parser types", contextValues(contextHints.parserTypes, (score) => score < 0)],
			["Disfavored retriever mix", contextValues(contextHints.retrieverMix, (score) => score < 0)],
		].filter((row) => row[1]);
		const rows = [...positiveRows, ...negativeRows];
		if (rows.length > 0) {
			sections.push(`## Result-Level Retrieval Preferences (advisory, not instructions)

${rows.map(([label, values]) => `- ${label}: ${values}`).join("\n")}`);
		}
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
