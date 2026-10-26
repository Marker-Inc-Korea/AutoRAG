/** A line range a cited chunk occupies in its parsed document. */
export interface CitedLineRange {
	readonly lineStart?: number;
	readonly lineEnd?: number;
}

const NUMBER = String.raw`(\d{1,6})`;
const RANGE_JOINER = String.raw`\s*[-–~]\s*`;

/**
 * Line references as prose writes them: `lines 7-9`, `line 8`, `L8`, `7~9행`,
 * `12줄`, `8번째 줄`. The English forms need a word boundary before `line`, so
 * `baseline 12` and `outline 40` are not references.
 */
const LINE_REFERENCES: readonly RegExp[] = [
	new RegExp(String.raw`\blines?\s*${NUMBER}(?:${RANGE_JOINER}${NUMBER})?`, "giu"),
	new RegExp(String.raw`\bL${NUMBER}(?:${RANGE_JOINER}L?${NUMBER})?\b`, "gu"),
	new RegExp(String.raw`${NUMBER}(?:${RANGE_JOINER}${NUMBER})?\s*(?:번째\s*)?[줄행]`, "gu"),
];

function claimedLines(prose: string): number[] {
	const found: { index: number; numbers: number[] }[] = [];
	for (const pattern of LINE_REFERENCES) {
		for (const match of prose.matchAll(pattern)) {
			const numbers = match
				.slice(1)
				.filter((value): value is string => value !== undefined)
				.map(Number);
			found.push({ index: match.index ?? 0, numbers });
		}
	}
	return found.sort((a, b) => a.index - b.index).flatMap((entry) => entry.numbers);
}

/**
 * Line numbers `prose` states that no cited chunk covers, each once, in order of
 * appearance. The line range of a chunk is measured from the parsed document,
 * so a number outside every cited range is one the model made up or miscounted.
 *
 * Nothing can be checked when no cited chunk carries a line range, so that
 * yields no claims rather than flagging every number.
 */
export function findUnverifiedLineClaims(prose: string, cited: readonly CitedLineRange[]): number[] {
	const ranges = cited.flatMap((range) =>
		range.lineStart === undefined ? [] : [{ start: range.lineStart, end: range.lineEnd ?? range.lineStart }],
	);
	if (ranges.length === 0) return [];
	const unverified: number[] = [];
	for (const line of claimedLines(prose)) {
		if (ranges.some((range) => line >= range.start && line <= range.end)) continue;
		if (!unverified.includes(line)) unverified.push(line);
	}
	return unverified;
}
