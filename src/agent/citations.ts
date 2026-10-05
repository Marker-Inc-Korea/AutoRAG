/**
 * Bracketed answer citations (`[n]`) must resolve to a `results[].number` of
 * the same response (issue #1788). A markdown image target `(<...>)` is
 * matched first and kept verbatim so a bracketed number inside a real file
 * path is never mistaken for a citation; `[n](...)` is a markdown link, not a
 * citation.
 */
const CITATION_PATTERN = /\(<[^>]*>\)|[ \t]*\[(\d+)\](?!\()/gu;

/** Sorted, de-duplicated citation numbers in `answer` with no matching result. */
export function unresolvedCitations(answer: string, results: readonly { readonly number: number }[]): number[] {
	const known = new Set(results.map((result) => result.number));
	const unresolved = new Set<number>();
	for (const match of answer.matchAll(CITATION_PATTERN)) {
		if (match[1] === undefined) continue;
		const number = Number(match[1]);
		if (!known.has(number)) unresolved.add(number);
	}
	return [...unresolved].sort((a, b) => a - b);
}

/** Remove citation markers whose number has no matching result. */
export function stripUnresolvedCitations(
	answer: string,
	results: readonly { readonly number: number }[],
): { readonly answer: string; readonly unresolved: readonly number[] } {
	const unresolved = unresolvedCitations(answer, results);
	if (unresolved.length === 0) return { answer, unresolved };
	const drop = new Set(unresolved);
	const stripped = answer.replace(CITATION_PATTERN, (match, digits: string | undefined) =>
		digits !== undefined && drop.has(Number(digits)) ? "" : match,
	);
	return { answer: stripped, unresolved };
}

export function formatCitationList(numbers: readonly number[]): string {
	return numbers.length === 0 ? "none" : numbers.map((number) => `[${number}]`).join(", ");
}

/**
 * Throw a corrective error when `answer` cites numbers absent from `results`.
 * Emit tools surface the message to the model as a tool error so it re-emits
 * with one consistent numbering.
 */
export function assertCitationsResolve(
	label: string,
	answer: string,
	results: readonly { readonly number: number }[],
): void {
	const unresolved = unresolvedCitations(answer, results);
	if (unresolved.length === 0) return;
	const emitted = [...new Set(results.map((result) => result.number))].sort((a, b) => a - b);
	throw new Error(
		`${label}: answer cites ${formatCitationList(unresolved)} but results only contain ${formatCitationList(emitted)}. ` +
			"Every bracketed citation in answer must be the number of a result in results of this same call; " +
			"candidate numbers from retrieval context or an earlier answer are not citation numbers. Re-emit with consistent numbering.",
	);
}
