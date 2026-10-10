/**
 * Remembers which search query and retrieval method surfaced each retrieval
 * result during a run, so a final answer's evidence refs (which carry only a
 * source, method, and text) can be mapped back to the query that found them.
 * Retrieval memory needs both to store an evidence record; the refs alone do
 * not know their query.
 */

/** The query, retrieval method, and source of a remembered result. */
export interface EvidenceOrigin {
	readonly query: string;
	readonly method: string;
	readonly source: string;
}

/** One retrieval result the agent saw, together with the query that produced it. */
export interface EvidenceOriginCandidate extends EvidenceOrigin {
	readonly content: string;
}

/** The parts of a cited evidence ref that identify its origin. */
export interface EvidenceRefLike {
	readonly source: string;
	readonly method: string;
	readonly excerpt?: string;
	readonly content?: string;
}

/** Oldest candidates beyond this are dropped, bounding memory per run. */
const MAX_CANDIDATES = 2000;

/** Stored content beyond this is truncated; matching only needs its head. */
const MAX_CONTENT_LENGTH = 4000;

/** A ref text shorter than this never matches by containment or prefix. */
const MIN_MATCH_LENGTH = 12;

interface StoredCandidate {
	readonly query: string;
	readonly method: string;
	readonly source: string;
	readonly content: string;
}

/** Case-, whitespace-, and Unicode-normalization-insensitive form for comparison. */
function normalize(text: string): string {
	return text.normalize("NFC").toLowerCase().replace(/\s+/g, " ").trim();
}

/** Length of the shared leading run of two strings. */
function commonPrefixLength(a: string, b: string): number {
	const limit = Math.min(a.length, b.length);
	let length = 0;
	while (length < limit && a[length] === b[length]) {
		length += 1;
	}
	return length;
}

/** The matched candidate as a resolved origin. */
function toOrigin(candidate: StoredCandidate): EvidenceOrigin {
	return { query: candidate.query, method: candidate.method, source: candidate.source };
}

/**
 * Insertion-ordered index of retrieval results keyed by their source. Matching
 * is deliberately text-based: a final answer's ref text is the same excerpt the
 * retrieval returned, so it shares the candidate's content even when the ref
 * carries a different method.
 */
export class EvidenceOriginIndex {
	private readonly candidates: StoredCandidate[] = [];
	/** Sources already remembered, to the normalized contents seen for each. */
	private readonly seen = new Map<string, Set<string>>();

	/**
	 * Remember one retrieval result and the query/method that produced it.
	 * First write wins for an identical (source, normalized content) pair.
	 */
	add(candidate: EvidenceOriginCandidate): void {
		const content = normalize(candidate.content).slice(0, MAX_CONTENT_LENGTH);
		const seenContents = this.seen.get(candidate.source);
		if (seenContents?.has(content)) {
			return;
		}
		if (seenContents) {
			seenContents.add(content);
		} else {
			this.seen.set(candidate.source, new Set([content]));
		}
		this.candidates.push({
			query: candidate.query,
			method: candidate.method,
			source: candidate.source,
			content,
		});
		if (this.candidates.length > MAX_CANDIDATES) {
			const dropped = this.candidates.shift();
			if (dropped) {
				this.seen.get(dropped.source)?.delete(dropped.content);
			}
		}
	}

	/** Where `ref` came from, or undefined when no remembered result matches. */
	resolve(ref: EvidenceRefLike): EvidenceOrigin | undefined {
		const rawText = ref.excerpt ?? ref.content ?? "";

		// Source-less refs (fast answers) match across every source by text alone.
		if (ref.source === "") {
			return this.resolveSourceless(rawText);
		}

		const matches = this.candidates.filter((candidate) => candidate.source === ref.source);
		if (matches.length === 0) {
			return undefined;
		}

		// 1. A non-empty chunk the ref text equals, contains, or is contained by.
		const refText = normalize(rawText);
		for (const candidate of matches) {
			if (candidate.content.length === 0) {
				continue;
			}
			if (candidate.content === refText) {
				return toOrigin(candidate);
			}
			if (
				refText.length >= MIN_MATCH_LENGTH &&
				(candidate.content.includes(refText) || refText.includes(candidate.content))
			) {
				return toOrigin(candidate);
			}
		}

		// 2. A source seen only once can only be that candidate.
		if (matches.length === 1) {
			return toOrigin(matches[0]!);
		}

		// 3. Several chunks: the longest shared prefix wins, else the earliest.
		let best = matches[0]!;
		let bestPrefix = -1;
		for (const candidate of matches) {
			const prefix = commonPrefixLength(candidate.content, refText);
			if (prefix >= MIN_MATCH_LENGTH && prefix > bestPrefix) {
				best = candidate;
				bestPrefix = prefix;
			}
		}
		return toOrigin(best);
	}

	/**
	 * Match a ref that carries no source (its text is the evidence text the
	 * answering model cited, possibly several excerpts joined by newlines)
	 * against every remembered candidate by text alone.
	 */
	private resolveSourceless(rawText: string): EvidenceOrigin | undefined {
		const refText = normalize(rawText);
		const needles =
			refText.length >= MIN_MATCH_LENGTH
				? [
						refText,
						...rawText
							.split("\n")
							.map(normalize)
							.filter((line) => line.length >= MIN_MATCH_LENGTH),
					]
				: [];
		if (needles.length === 0) {
			return undefined;
		}
		for (const candidate of this.candidates) {
			// Both sides must be long enough; a short equality is not a match.
			if (candidate.content.length < MIN_MATCH_LENGTH) {
				continue;
			}
			for (const needle of needles) {
				if (
					candidate.content === needle ||
					candidate.content.includes(needle) ||
					needle.includes(candidate.content)
				) {
					return toOrigin(candidate);
				}
			}
		}
		return undefined;
	}

	clear(): void {
		this.candidates.length = 0;
		this.seen.clear();
	}
}
