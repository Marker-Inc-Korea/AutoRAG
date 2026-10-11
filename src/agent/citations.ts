/**
 * Bracketed citations in an answer. Two grammars share one scanner:
 *
 * - **Final form** `[n]`: a number that must resolve to a `results[].number`
 *   of the same response (issue #1788).
 * - **Model form** `[e3]`, `[e3, e7]`, `[file:/abs/path]`: the evidence ids the
 *   retrieval tools print next to each result, or a local file the model opened
 *   itself. The harness resolves them against the run's evidence ledger and
 *   rewrites them into the final form.
 *
 * A markdown image target `(<...>)` is skipped verbatim so a bracketed token
 * inside a real file path is never mistaken for a citation; `[n](...)` is a
 * markdown link, not a citation.
 *
 * `answer` is model- or caller-supplied, so this is a single linear scan rather
 * than a regex: alternations like `\(<[^>]*>\)` / `[ \t]*\[` backtrack
 * quadratically on inputs such as repeated `(<` or long whitespace runs.
 */
export type AnswerMarker =
	| { readonly kind: "number"; readonly start: number; readonly end: number; readonly number: number }
	| { readonly kind: "evidence"; readonly start: number; readonly end: number; readonly ids: readonly string[] }
	| { readonly kind: "file"; readonly start: number; readonly end: number; readonly path: string };

/** How many `]` a `[file:...]` marker may skip to find the end of a path that itself contains brackets. */
const MAX_FILE_MARKER_CLOSERS = 8;

const FILE_MARKER_PREFIX = "[file:";

function isAsciiDigit(code: number): boolean {
	return code >= 48 && code <= 57;
}

function digitsEndAt(text: string, from: number): number {
	let end = from;
	while (end < text.length && isAsciiDigit(text.charCodeAt(end))) end += 1;
	return end;
}

/**
 * `[2011] SGHC 222`, `[1999] 2 SLR 392`: a bracketed number that the sentence
 * keeps reading through (a space, then a letter or digit) is part of a law-report
 * or year reference, not a citation. A citation sits at the end of a clause:
 * punctuation, a line break, another marker, or the end of the text follows it.
 */
function continuesAsProse(answer: string, from: number): boolean {
	if (answer[from] !== " " && answer[from] !== "\u00a0") return false;
	const next = answer.codePointAt(from + 1);
	return next !== undefined && /[\p{L}\p{N}]/u.test(String.fromCodePoint(next));
}

/**
 * `[e3]` or `[e3, e7]` starting at `index`: the ids and the index just past the
 * closing `]`, or undefined when the text there is not an evidence marker.
 */
function evidenceMarkerAt(answer: string, index: number): { readonly ids: string[]; readonly end: number } | undefined {
	const ids: string[] = [];
	let cursor = index + 1;
	for (;;) {
		if (answer[cursor] !== "e") return undefined;
		const digitsEnd = digitsEndAt(answer, cursor + 1);
		if (digitsEnd === cursor + 1) return undefined;
		ids.push(answer.slice(cursor, digitsEnd));
		cursor = digitsEnd;
		if (answer[cursor] === "]") return { ids, end: cursor + 1 };
		if (answer[cursor] !== ",") return undefined;
		cursor += 1;
		while (answer[cursor] === " " || answer[cursor] === "\t") cursor += 1;
	}
}

/** The path inside `[file:<path>]`, tolerating an optional `<...>` wrapper. */
function filePathFrom(raw: string): string {
	const trimmed = raw.trim();
	return trimmed.startsWith("<") && trimmed.endsWith(">") ? trimmed.slice(1, -1).trim() : trimmed;
}

/**
 * `[file:/abs/path]` starting at `index`. A path may contain `]` (Korean file
 * names such as `[최종] 계약서.pdf` are common), so the marker ends at the first
 * `]` whose preceding text is a path `isFile` accepts.
 */
function fileMarkerAt(
	answer: string,
	index: number,
	isFile: (path: string) => boolean,
): { readonly path: string; readonly end: number } | undefined {
	if (!answer.startsWith(FILE_MARKER_PREFIX, index)) return undefined;
	const bodyStart = index + FILE_MARKER_PREFIX.length;
	let closer = answer.indexOf("]", bodyStart);
	for (let attempt = 0; closer !== -1 && attempt < MAX_FILE_MARKER_CLOSERS; attempt++) {
		const path = filePathFrom(answer.slice(bodyStart, closer));
		if (path.length > 0 && isFile(path)) return { path, end: closer + 1 };
		closer = answer.indexOf("]", closer + 1);
	}
	return undefined;
}

/** Every citation marker in `answer`, in order. `isFile` decides which `[file:...]` paths are real. */
export function answerMarkers(answer: string, isFile: (path: string) => boolean = () => false): AnswerMarker[] {
	const markers: AnswerMarker[] = [];
	// End of the last consumed token; leading whitespace never reaches back past it.
	let floor = 0;
	// First `>` at or after the current image-target body; cached so repeated `(<` stays linear.
	let closeIndex = -1;
	let index = 0;
	const leadingStart = (from: number): number => {
		let start = from;
		while (start > floor && (answer[start - 1] === " " || answer[start - 1] === "\t")) start -= 1;
		return start;
	};
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
			const digitsEnd = digitsEndAt(answer, index + 1);
			if (
				digitsEnd > index + 1 &&
				answer[digitsEnd] === "]" &&
				answer[digitsEnd + 1] !== "(" &&
				!continuesAsProse(answer, digitsEnd + 1)
			) {
				markers.push({
					kind: "number",
					start: leadingStart(index),
					end: digitsEnd + 1,
					number: Number(answer.slice(index + 1, digitsEnd)),
				});
				index = digitsEnd + 1;
				floor = index;
				continue;
			}
			const evidence = evidenceMarkerAt(answer, index);
			if (evidence !== undefined && answer[evidence.end] !== "(") {
				markers.push({ kind: "evidence", start: leadingStart(index), end: evidence.end, ids: evidence.ids });
				index = evidence.end;
				floor = index;
				continue;
			}
			const file = fileMarkerAt(answer, index, isFile);
			if (file !== undefined && answer[file.end] !== "(") {
				markers.push({ kind: "file", start: leadingStart(index), end: file.end, path: file.path });
				index = file.end;
				floor = index;
				continue;
			}
		}
		index += 1;
	}
	return markers;
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
	for (const marker of answerMarkers(answer)) {
		if (marker.kind !== "number" || known.has(marker.number)) continue;
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

/** Numbers that appear more than once, ascending, each listed once. */
function duplicateNumbers(sortedNumbers: readonly number[]): number[] {
	const duplicates: number[] = [];
	for (let index = 1; index < sortedNumbers.length; index++) {
		const number = sortedNumbers[index] as number;
		if (number === sortedNumbers[index - 1] && duplicates[duplicates.length - 1] !== number) duplicates.push(number);
	}
	return duplicates;
}

/**
 * Throw a corrective error unless `results` and `mapping` carry the same
 * numbers, exactly one entry each. Duplicates are rejected even when both
 * sides repeat them: a repeated number collapses into a single registry entry,
 * so the other entry's evidence would be lost.
 */
export function assertResultsMappingOneToOne(
	label: string,
	results: readonly { readonly number: number }[],
	mapping: readonly { readonly number: number }[],
): void {
	const resultNumbers = results.map((result) => result.number).sort((a, b) => a - b);
	const mappingNumbers = mapping.map((entry) => entry.number).sort((a, b) => a - b);
	const resultDuplicates = duplicateNumbers(resultNumbers);
	const mappingDuplicates = duplicateNumbers(mappingNumbers);
	const sameNumbers =
		resultNumbers.length === mappingNumbers.length &&
		resultNumbers.every((number, index) => number === mappingNumbers[index]);
	if (sameNumbers && resultDuplicates.length === 0 && mappingDuplicates.length === 0) return;
	const duplicateNotes = [
		resultDuplicates.length > 0 ? `results repeat ${formatCitationList(resultDuplicates)}` : undefined,
		mappingDuplicates.length > 0 ? `mapping repeats ${formatCitationList(mappingDuplicates)}` : undefined,
	].filter((note) => note !== undefined);
	throw new Error(
		`${label}: result numbers and mapping numbers must be one-to-one, but results contain ${formatCitationList(resultNumbers)} ` +
			`and mapping contains ${formatCitationList(mappingNumbers)}` +
			(duplicateNotes.length > 0 ? ` (${duplicateNotes.join("; ")})` : "") +
			". Give every result a unique number with exactly one mapping entry carrying the same number and no mapping entry without a result.",
	);
}
