import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";
import { completeSimple } from "@earendil-works/pi-ai/compat";

/** Hard cap on search queries one question may decompose into. */
export const MAX_DECOMPOSED_QUERIES = 5;

/** Prompt in, raw completion text out. The seam the decomposer calls. */
export type DecompositionCompleter = (prompt: string) => Promise<string>;

/** A pi-ai model plus the credential used only for decomposition calls. */
export interface DecompositionModel {
	readonly model: Model<Api>;
	readonly apiKey?: string;
}

export function buildDecompositionPrompt(query: string): string {
	return (
		`Split the user question below into search queries for a retrieval system.\n\n` +
		`Rules:\n` +
		`- Return at most ${MAX_DECOMPOSED_QUERIES} queries; use fewer when fewer cover the question.\n` +
		`- Each query is short, self-contained, and targets one fact or sub-question. Resolve pronouns and keep names, dates, and identifiers.\n` +
		`- Together the queries must cover everything needed to answer the question.\n` +
		`- Keep the question's language.\n` +
		`- Reply with JSON only, no prose: {"queries": ["...", "..."]}\n\n` +
		`User question: ${query}`
	);
}

function stringsFrom(value: unknown): string[] | undefined {
	const list = Array.isArray(value)
		? value
		: typeof value === "object" && value !== null && "queries" in value
			? value.queries
			: undefined;
	return Array.isArray(list) ? list.filter((item): item is string => typeof item === "string") : undefined;
}

function parseJsonStrings(text: string): string[] | undefined {
	for (const [open, close] of [
		["{", "}"],
		["[", "]"],
	] as const) {
		const start = text.indexOf(open);
		const end = text.lastIndexOf(close);
		if (start === -1 || end <= start) continue;
		try {
			const parsed = stringsFrom(JSON.parse(text.slice(start, end + 1)));
			if (parsed !== undefined) return parsed;
		} catch {
			// Not JSON in this bracket pair; try the next shape.
		}
	}
	return undefined;
}

/**
 * Parse the model's reply into at most {@link MAX_DECOMPOSED_QUERIES} unique,
 * non-empty queries. Accepts `{"queries": [...]}`, a bare array (optionally
 * fenced or wrapped in prose), or one query per line as a last resort.
 */
export function parseDecomposedQueries(text: string): string[] {
	const candidates =
		parseJsonStrings(text) ??
		text
			.split(/\r?\n/u)
			.map((line) => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/u, ""))
			.filter((line) => !line.trimStart().startsWith("```"));
	const seen = new Set<string>();
	const queries: string[] = [];
	for (const candidate of candidates) {
		const query = candidate.trim();
		const key = query.toLowerCase();
		if (query.length === 0 || seen.has(key)) continue;
		seen.add(key);
		queries.push(query);
		if (queries.length === MAX_DECOMPOSED_QUERIES) break;
	}
	return queries;
}

/**
 * Decompose one question into up to {@link MAX_DECOMPOSED_QUERIES} search
 * queries. A reply with no usable query falls back to the original question,
 * so the caller always has something to search.
 */
export async function decomposeQuery(complete: DecompositionCompleter, query: string): Promise<string[]> {
	const queries = parseDecomposedQueries(await complete(buildDecompositionPrompt(query)));
	return queries.length > 0 ? queries : [query];
}

/** Completer over a pi-ai model: one user turn, thinking off, short output. */
export function createModelDecompositionCompleter(
	target: DecompositionModel,
	signal?: AbortSignal,
): DecompositionCompleter {
	return async (prompt) => {
		const result: AssistantMessage = await completeSimple(
			target.model,
			{ messages: [{ role: "user", content: prompt, timestamp: Date.now() }] },
			{
				...(target.apiKey !== undefined ? { apiKey: target.apiKey } : {}),
				...(signal !== undefined ? { signal } : {}),
				maxTokens: 1024,
			},
		);
		if (result.stopReason === "error" || result.stopReason === "aborted") {
			throw new Error(result.errorMessage ?? `query decomposition ${result.stopReason}`);
		}
		return result.content
			.filter((part): part is Extract<(typeof result.content)[number], { type: "text" }> => part.type === "text")
			.map((part) => part.text)
			.join("");
	};
}
