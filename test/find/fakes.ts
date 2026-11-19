import type { FindJudge, FindRuntime, FindSearchAnswer } from "../../src/find/types.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

export interface FakeRuntimeOptions {
	readonly datasources?: readonly string[];
	readonly searchSelected?: (
		query: string,
		selection?: {
			readonly datasourceIds?: readonly string[];
			readonly local?: boolean;
			readonly methods?: readonly string[];
		},
		options?: { readonly topK?: number; readonly scope?: string },
	) => Promise<FindSearchAnswer>;
	readonly searchFsearch?: FindRuntime["searchFsearch"];
	readonly searchEverything?: FindRuntime["searchEverything"];
}

export function fakeRuntime(options: FakeRuntimeOptions = {}): FindRuntime {
	return {
		listDatasources: () => (options.datasources ?? []).map((datasourceId) => ({ datasourceId })),
		searchSelected: async (query, selection, opts) => {
			if (options.searchSelected !== undefined) return options.searchSelected(query, selection, opts);
			return { results: [], diagnostics: [], unsearched: [] };
		},
		searchFsearch: options.searchFsearch ?? (async () => ({ ok: true, backend: "fsearch-cli", results: [] })),
		searchEverything: options.searchEverything ?? (async () => ({ ok: true, results: [] })),
	};
}

export function result(overrides: Partial<RetrievalResult> = {}): RetrievalResult {
	return {
		id: overrides.id ?? "id",
		content: overrides.content ?? "content",
		source: overrides.source ?? "/tmp/doc.md",
		score: overrides.score ?? 1,
		metadata: overrides.metadata ?? {},
	};
}

export function answer(overrides: Partial<FindSearchAnswer> = {}): FindSearchAnswer {
	return {
		results: overrides.results ?? [],
		diagnostics: overrides.diagnostics ?? [],
		unsearched: overrides.unsearched ?? [],
	};
}

/**
 * A judge whose verdict per question is looked up by index in the supplied
 * probability list (null = no verdict for that question).
 */
export function judgeFrom(probabilities: readonly (number | null)[]): FindJudge {
	return async (_state, questions) => ({
		verdicts: questions.map((question, index) => ({
			id: question.id ?? `q${index}`,
			type: "noul" as const,
			answer: probabilities[index] ?? null,
			confidence: 1,
			escalate: false,
		})),
	});
}
