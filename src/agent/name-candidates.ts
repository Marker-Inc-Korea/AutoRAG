/**
 * Parallel file-name search over several candidate names.
 *
 * File-name search is cheap (an index lookup), but the right name is rarely
 * known: the user's word, its translation, an abbreviation, a likely
 * extension. Instead of one call per guess, every candidate is searched at the
 * same time and the hits are merged, with each hit labelled by the candidates
 * that found it.
 */

/** Candidates searched in one call; the rest are reported as dropped, never silently ignored. */
export const MAX_NAME_CANDIDATES = 8;

export interface NameCandidates {
	readonly candidates: readonly string[];
	/** Distinct candidates beyond {@link MAX_NAME_CANDIDATES} that were not searched. */
	readonly dropped: number;
}

/**
 * The query first, then the alternatives, trimmed and without repeats. Case
 * variants count as one candidate unless the search is case-sensitive.
 */
export function collectNameCandidates(
	query: string,
	alternatives: readonly string[] = [],
	options: { readonly matchCase?: boolean } = {},
): NameCandidates {
	const seen = new Set<string>();
	const distinct: string[] = [];
	for (const raw of [query, ...alternatives]) {
		const candidate = raw.trim();
		if (candidate.length === 0) continue;
		const key = options.matchCase === true ? candidate : candidate.toLowerCase();
		if (seen.has(key)) continue;
		seen.add(key);
		distinct.push(candidate);
	}
	return {
		candidates: distinct.slice(0, MAX_NAME_CANDIDATES),
		dropped: Math.max(0, distinct.length - MAX_NAME_CANDIDATES),
	};
}

/** What one file-name search returned: hits, or a failure that names why. Backends' own results satisfy it. */
export type NameSearchResult =
	| { readonly ok: true; readonly results: readonly { readonly path: string }[] }
	| { readonly ok: false; readonly reason: string; readonly message: string };

/** A search that threw, reported as the failure of its own candidate. */
export interface NameSearchFailure {
	readonly ok: false;
	readonly reason: string;
	readonly message: string;
}

/** `Result` is the backend's own result type, so callers keep fields such as `backend` and `total`. */
export interface NameSearchOutcome<Result extends NameSearchResult = NameSearchResult> {
	readonly candidate: string;
	readonly result: Result | NameSearchFailure;
}

/**
 * Search every candidate at once. A search that throws fails only its own
 * candidate; the others still return.
 */
export async function searchNameCandidates<Result extends NameSearchResult>(
	candidates: readonly string[],
	search: (candidate: string) => Promise<Result>,
): Promise<NameSearchOutcome<Result>[]> {
	return Promise.all(
		candidates.map(async (candidate): Promise<NameSearchOutcome<Result>> => {
			try {
				return { candidate, result: await search(candidate) };
			} catch (error) {
				return {
					candidate,
					result: {
						ok: false,
						reason: "search-failed",
						message: error instanceof Error ? error.message : String(error),
					},
				};
			}
		}),
	);
}

export interface MergedNameHit<Entry extends { readonly path: string }> {
	readonly entry: Entry;
	/** Every candidate that found this path, in candidate order. */
	readonly candidates: readonly string[];
}

/**
 * One list of hits, each path once. The name indexes (Everything, FSearch) match
 * names exactly and return their hits in a sort order, not by relevance, so no
 * hit is "better" than another within one candidate. What does carry signal is
 * how many of the searched names found a path: those come first. Within the
 * same count, the candidates' lists are interleaved position by position, so the
 * cut at `limit` is shared among the candidates instead of one filling it.
 */
export function mergeNameHits<Entry extends { readonly path: string }>(
	outcomes: readonly {
		readonly candidate: string;
		readonly result: { readonly ok: true; readonly results: readonly Entry[] } | NameSearchFailure;
	}[],
	limit: number,
): MergedNameHit<Entry>[] {
	const lists = outcomes.flatMap((outcome) =>
		outcome.result.ok ? [{ candidate: outcome.candidate, entries: outcome.result.results }] : [],
	);
	const order = new Map(outcomes.map((outcome, index) => [outcome.candidate, index]));
	const byPath = new Map<string, { entry: Entry; candidates: string[] }>();
	const longest = Math.max(0, ...lists.map((list) => list.entries.length));
	for (let position = 0; position < longest; position++) {
		for (const list of lists) {
			const entry = list.entries[position];
			if (entry === undefined) continue;
			const existing = byPath.get(entry.path);
			if (existing === undefined) byPath.set(entry.path, { entry, candidates: [list.candidate] });
			else if (!existing.candidates.includes(list.candidate)) existing.candidates.push(list.candidate);
		}
	}
	// Map insertion order is the interleaved order; the sort is stable, so it only lifts multi-name matches.
	return [...byPath.values()]
		.sort((a, b) => b.candidates.length - a.candidates.length)
		.slice(0, limit)
		.map((hit) => ({
			entry: hit.entry,
			candidates: [...hit.candidates].sort((x, y) => (order.get(x) ?? 0) - (order.get(y) ?? 0)),
		}));
}

/** One hit as a numbered line; `foundBy` is shown only when several names were searched. */
export function renderNameHitLine(
	index: number,
	entry: {
		readonly path: string;
		readonly type: string;
		readonly size?: number | undefined;
		readonly dateModified?: string | undefined;
	},
	foundBy?: readonly string[],
): string {
	const parts = [`[${index + 1}] ${entry.type} ${entry.path}`];
	if (entry.size !== undefined) parts.push(`size=${entry.size}`);
	if (entry.dateModified !== undefined) parts.push(`modified=${entry.dateModified}`);
	if (foundBy !== undefined) parts.push(`(found by: ${foundBy.join(", ")})`);
	return parts.join(" ");
}

/** Lines that say which candidates failed and which were never searched; empty when everything ran. */
export function renderNameSearchProblems(
	outcomes: readonly { readonly candidate: string; readonly result: { readonly ok: true } | NameSearchFailure }[],
	dropped: number,
): string[] {
	const lines: string[] = [];
	for (const outcome of outcomes) {
		if (!outcome.result.ok) {
			lines.push(`${JSON.stringify(outcome.candidate)} failed: ${outcome.result.reason}: ${outcome.result.message}`);
		}
	}
	if (dropped > 0) {
		lines.push(
			`${dropped} more names were not searched (at most ${MAX_NAME_CANDIDATES} per call); call again with those names.`,
		);
	}
	return lines;
}
