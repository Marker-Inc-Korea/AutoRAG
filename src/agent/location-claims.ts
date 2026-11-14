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

/** A page range a cited chunk occupies in its document. */
export interface CitedPageRange {
	readonly pageStart?: number;
	readonly pageEnd?: number;
}

/**
 * Page references as prose writes them: `page 2`, `pages 2-3`, `p. 3`, `2페이지`, `2~3쪽`. The English forms
 * need a word boundary before `page`, so `homepage 40` is not a reference; `장` is not a page word here
 * because it is also a chapter (`제5장`).
 */
const PAGE_REFERENCES: readonly RegExp[] = [
	new RegExp(String.raw`\bpages?\s*${NUMBER}(?:${RANGE_JOINER}${NUMBER})?`, "giu"),
	new RegExp(String.raw`\bpp?\.\s*${NUMBER}(?:${RANGE_JOINER}${NUMBER})?`, "giu"),
	new RegExp(String.raw`${NUMBER}(?:${RANGE_JOINER}${NUMBER})?\s*(?:페이지|쪽)`, "gu"),
];

function claimedPages(prose: string): number[] {
	const found: { index: number; numbers: number[] }[] = [];
	for (const pattern of PAGE_REFERENCES) {
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
 * Page numbers `prose` states that no cited chunk covers, each once, in order of appearance. A chunk's page
 * range is measured by the parser, so a number outside every cited range is one the model made up. Nothing
 * can be checked when no cited chunk carries a page, so that yields no claims.
 */
export function findUnverifiedPageClaims(prose: string, cited: readonly CitedPageRange[]): number[] {
	const ranges = cited.flatMap((range) =>
		range.pageStart === undefined ? [] : [{ start: range.pageStart, end: range.pageEnd ?? range.pageStart }],
	);
	if (ranges.length === 0) return [];
	const unverified: number[] = [];
	for (const page of claimedPages(prose)) {
		if (ranges.some((range) => page >= range.start && page <= range.end)) continue;
		if (!unverified.includes(page)) unverified.push(page);
	}
	return unverified;
}
