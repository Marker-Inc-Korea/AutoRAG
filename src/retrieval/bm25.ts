/**
 * Lightweight BM25-style lexical scorer shared by retrieval surfaces.
 *
 * Terms match document tokens exactly or as a prefix (min 2 chars), so
 * agglutinative suffixes — Korean particles/endings (인증서 → 인증서를) and
 * English inflections (index → indexing) — still rank. Prefix hits are
 * slightly discounted against exact hits (0.75 weight).
 */

/**
 * Lowercase `text` and split on anything that is not a letter, digit, or
 * Hangul syllable, discarding empty tokens.
 */
export function tokenize(text: string): string[] {
	return (
		text
			.toLowerCase()
			// Split on anything that is not a letter, digit, or Hangul syllable.
			.split(/[^\p{L}\p{N}]+/u)
			.filter((token) => token.length > 0)
	);
}

/**
 * Score each entry of `documents` against `query` with a BM25-style formula.
 * Returns one score per document, in the same order; unmatched documents
 * score 0. Empty query terms or zero documents yield an array of zeros.
 */
export function bm25Scores(query: string, documents: readonly string[]): number[] {
	const docCount = documents.length;
	if (docCount === 0) return [];

	const queryTerms = [...new Set(tokenize(query))];
	if (queryTerms.length === 0) return documents.map(() => 0);

	const tokenized = documents.map((document) => tokenize(document));
	const avgLength = tokenized.reduce((sum, tokens) => sum + tokens.length, 0) / docCount || 1;

	// Weighted term frequency per (term, doc): exact = 1, prefix = 0.75.
	const termFrequency = (term: string, tokens: readonly string[]): number => {
		let tf = 0;
		for (const token of tokens) {
			if (token === term) tf += 1;
			else if (term.length >= 2 && token.startsWith(term)) tf += 0.75;
		}
		return tf;
	};
	const frequencies = queryTerms.map((term) => tokenized.map((tokens) => termFrequency(term, tokens)));
	const documentFrequency = frequencies.map((perDoc) => perDoc.reduce((count, tf) => count + (tf > 0 ? 1 : 0), 0));

	const k1 = 1.2;
	const b = 0.75;
	return documents.map((_document, index) => {
		const tokens = tokenized[index] ?? [];
		let score = 0;
		for (const [termIndex] of queryTerms.entries()) {
			const df = documentFrequency[termIndex] ?? 0;
			if (df === 0) continue;
			const tf = frequencies[termIndex]?.[index] ?? 0;
			if (tf === 0) continue;
			const idf = Math.log(1 + (docCount - df + 0.5) / (df + 0.5));
			score += (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * tokens.length) / avgLength));
		}
		return score;
	});
}
