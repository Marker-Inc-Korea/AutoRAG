/**
 * Bracketed answer citations (`[n]`) must resolve to a `results[].number` of
 * the same response (issue #1788). A markdown image target `(<...>)` is
 * skipped verbatim so a bracketed number inside a real file path is never
 * mistaken for a citation; `[n](...)` is a markdown link, not a citation.
 *
 * `answer` is model- or caller-supplied, so this is a single linear scan rather
 * than a regex: alternations like `\(<[^>]*>\)` / `[ \t]*\[` backtrack
 * quadratically on inputs such as repeated `(<` or long whitespace runs.
 */
interface CitationMarker {
	/** Start of the marker, including the spaces/tabs directly before `[`. */
	readonly start: number;
	/** Index just past the closing `]`. */
	readonly end: number;
	readonly number: number;
}

function isAsciiDigit(code: number): boolean {
	return code >= 48 && code <= 57;
}

function citationMarkers(answer: string): CitationMarker[] {
	const markers: CitationMarker[] = [];
	// End of the last consumed token; leading whitespace never reaches back past it.
	let floor = 0;
	// First `>` at or after the current image-target body; cached so repeated `(<` stays linear.
	let closeIndex = -1;
	let index = 0;
	while (index < answer.length) {
		const char = answer[index];
		if (char === "(" && answer[index + 1] === "<") {
			if (closeIndex < index + 2) {
				const found = answer.indexOf(">", index + 2);
				closeIndex = found === -1 ? answer.length : found;
			}
			if (answer[closeIndex + 1] === ")") {
				index = closeIndex + 2;
				floor = index;
				continue;
			}
			index += 1;
			continue;
		}
		if (char === "[") {
			let digitsEnd = index + 1;
			while (digitsEnd < answer.length && isAsciiDigit(answer.charCodeAt(digitsEnd))) digitsEnd += 1;
			if (digitsEnd > index + 1 && answer[digitsEnd] === "]" && answer[digitsEnd + 1] !== "(") {
				let start = index;
				while (start > floor && (answer[start - 1] === " " || answer[start - 1] === "\t")) start -= 1;
				markers.push({ start, end: digitsEnd + 1, number: Number(answer.slice(index + 1, digitsEnd)) });
				index = digitsEnd + 1;
				floor = index;
				continue;
			}
		}
		index += 1;
	}
	return markers;
}

/** Sorted, de-duplicated citation numbers in `answer` with no matching result. */
export function unresolvedCitations(answer: string, results: readonly { readonly number: number }[]): number[] {
	const known = new Set(results.map((result) => result.number));
	const unresolved = new Set<number>();
	for (const marker of citationMarkers(answer)) {
		if (!known.has(marker.number)) unresolved.add(marker.number);
	}
	return [...unresolved].sort((a, b) => a - b);
}

/** Remove citation markers whose number has no matching result. */
export function stripUnresolvedCitations(
	answer: string,
	results: readonly { readonly number: number }[],
): { readonly answer: string; readonly unresolved: readonly number[] } {
	const known = new Set(results.map((result) => result.number));
	const unresolved = new Set<number>();
	let stripped = "";
	let cursor = 0;
	for (const marker of citationMarkers(answer)) {
		if (known.has(marker.number)) continue;
		unresolved.add(marker.number);
		stripped += answer.slice(cursor, marker.start);
		cursor = marker.end;
	}
	if (unresolved.size === 0) return { answer, unresolved: [] };
	return { answer: stripped + answer.slice(cursor), unresolved: [...unresolved].sort((a, b) => a - b) };
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
