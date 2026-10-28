import { collectNameCandidates, MAX_NAME_CANDIDATES } from "./name-candidates.ts";
import type { DecompositionCompleter } from "./query-decomposition.ts";

export function buildNameProposalPrompt(query: string): string {
	return (
		`The user wants a file or folder located by its name. List the names it may have, for a file-name index.\n\n` +
		`Rules:\n` +
		`- Return at most ${MAX_NAME_CANDIDATES} names, most likely first; use fewer when fewer cover it.\n` +
		`- Each name is a short plain fragment that would appear in a file name (a word, an abbreviation, a spelling, a likely extension such as pdf). No search operators, no wildcards, no quotes.\n` +
		`- Cover different spellings and languages the file could be named in: the user's own words, a translation, a common abbreviation, a romanization.\n` +
		`- Keep names, dates, and identifiers the user gave.\n` +
		`- Reply with JSON only, no prose: {"names": ["...", "..."]}\n\n` +
		`User question: ${query}`
	);
}

function stringsFrom(value: unknown): string[] | undefined {
	const list = Array.isArray(value)
		? value
		: typeof value === "object" && value !== null && "names" in value
			? value.names
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
 * Ask the model which names the file the user describes may have. Accepts
 * `{"names": [...]}`, a bare array (optionally fenced), or one name per line.
 * Returns up to {@link MAX_NAME_CANDIDATES} distinct names; an unusable reply
 * returns none, so the caller falls back to the question itself.
 */
export async function proposeFileNames(complete: DecompositionCompleter, query: string): Promise<string[]> {
	const text = await complete(buildNameProposalPrompt(query));
	const candidates =
		parseJsonStrings(text) ??
		text
			.split(/\r?\n/u)
			.map((line) => line.replace(/^\s*(?:[-*•]|\d+[.)])\s*/u, ""))
			.filter((line) => !line.trimStart().startsWith("```"));
	const [first = "", ...rest] = candidates;
	return [...collectNameCandidates(first, rest).candidates];
}
