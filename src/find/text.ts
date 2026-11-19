/**
 * Text helpers for the whole-search groups: the relevance gate's deterministic
 * lexical pass, the one-line snippet, and the platform file-name regexes.
 *
 * The lexical pass is ported from the Finder's `matchesQueryLexicallyOrApprox`
 * with the approximate edit-distance step dropped, so it stays dependency-free.
 */

/** Passed text a hit carries, capped at this many characters. */
export const SEARCH_HIT_CONTENT_LIMIT = 2000;
const SNIPPET_LENGTH = 160;

const WORD_TOKEN = /[\p{L}\p{N}]+/gu;
const ASCII_ONLY = /^[\x00-\x7f]*$/u;
const ASCII_WORD = /^[a-z0-9]+$/u;
const ASCII_ALNUM_CHAR = /[a-z0-9]/u;

// Built with Object.fromEntries, not an object literal: a literal `then` key would
// make the table a thenable (biome `noThenProperty`).
const ENGLISH_STOPWORDS: Readonly<Record<string, true>> = Object.fromEntries(
	(
		"a an and are as at be by do for from had has have he her his how if in into is it its me my no not of on onto or our she so " +
		"that the their them then there these they this to up us was we were what when where which who why with you your"
	)
		.split(" ")
		.map((word): [string, true] => [word, true]),
);

const KOREAN_QUERY_STOPWORDS: Readonly<Record<string, true>> = Object.fromEntries(
	"찾아줘 찾아 알려줘 알려 보여줘 보여 검색 검색해줘 관련 대한 대해 어디 무엇 언제 어떻게"
		.split(" ")
		.map((word): [string, true] => [word, true]),
);

/** Common Korean trailing postpositions (조사) stripped from multi-syllable Hangul nouns. */
const KOREAN_POSTPOSITIONS = /(?:에서|으로|부터|까지|이랑|하고|은|는|이|가|을|를|의|에|로|과|와|도|만)$/u;

/** The passage as one line, cut to {@link SNIPPET_LENGTH} around the first query term that appears in it. */
export function contentSnippet(text: string, query: string): string {
	const flat = text.replace(/\s+/gu, " ").trim();
	if (flat.length <= SNIPPET_LENGTH) return flat;
	const lower = flat.toLowerCase();
	const terms = query
		.toLowerCase()
		.split(/\s+/u)
		.filter((term) => term.length > 0);
	const found = terms.map((term) => lower.indexOf(term)).filter((index) => index >= 0);
	const at = found.length === 0 ? 0 : Math.min(...found);
	const start = Math.max(0, Math.min(at - Math.floor(SNIPPET_LENGTH / 3), flat.length - SNIPPET_LENGTH));
	const end = start + SNIPPET_LENGTH;
	return `${start > 0 ? "…" : ""}${flat.slice(start, end).trim()}${end < flat.length ? "…" : ""}`;
}

/** The passage flattened to one line and capped at {@link SEARCH_HIT_CONTENT_LIMIT} characters. */
export function capContent(text: string): string {
	const flat = text.replace(/\s+/gu, " ").trim();
	return flat.length <= SEARCH_HIT_CONTENT_LIMIT ? flat : `${flat.slice(0, SEARCH_HIT_CONTENT_LIMIT - 1)}…`;
}

function toSearchKey(text: string): string {
	const normalized = text.normalize("NFKC").toLowerCase();
	let out = "";
	for (const match of normalized.matchAll(WORD_TOKEN)) out += match[0];
	return out;
}

function containsAsciiToken(haystack: string, token: string): boolean {
	let index = haystack.indexOf(token);
	while (index !== -1) {
		const before = index > 0 ? haystack[index - 1]! : "";
		const afterIndex = index + token.length;
		const after = afterIndex < haystack.length ? haystack[afterIndex]! : "";
		if (!ASCII_ALNUM_CHAR.test(before) && !ASCII_ALNUM_CHAR.test(after)) return true;
		index = haystack.indexOf(token, index + 1);
	}
	return false;
}

function koreanStem(token: string): string | null {
	if (ASCII_ONLY.test(token) || token.length < 3) return null;
	const stripped = token.replace(KOREAN_POSTPOSITIONS, "");
	return stripped.length >= 2 && stripped !== token ? stripped : null;
}

/**
 * Fast deterministic pass-through before Jev: exact query word match (NFKC
 * case-folded, ASCII boundary-aware so `_`/`-`/Hangul neighbors work while
 * partial stems like `art` in `start` do not, plus Korean postposition
 * stripping like `비행기를` → `비행기`). The approximate edit-distance step the
 * Finder had is intentionally dropped: this is dependency-free.
 */
export function matchesQueryLexically(query: string, candidateText: string): boolean {
	const normQuery = query.normalize("NFKC").toLowerCase().replace(/\s+/gu, " ").trim();
	const normCandidate = candidateText.normalize("NFKC").toLowerCase().replace(/\s+/gu, " ").trim();
	if (normQuery.length === 0 || normCandidate.length === 0) return false;
	if (normQuery.length >= 2 && normCandidate.includes(normQuery)) return true;

	const rawTokens = [...normQuery.matchAll(WORD_TOKEN)].map((match) => match[0]);
	const meaningful = rawTokens.filter((token) => {
		if (ENGLISH_STOPWORDS[token] === true || KOREAN_QUERY_STOPWORDS[token] === true) return false;
		return token.length >= 2;
	});
	const tokens = meaningful.length > 0 ? meaningful : rawTokens.filter((token) => token.length >= 2);

	for (const token of tokens) {
		if (ASCII_WORD.test(token)) {
			if (containsAsciiToken(normCandidate, token)) return true;
		} else {
			if (normCandidate.includes(token)) return true;
			const stem = koreanStem(token);
			if (stem !== null && normCandidate.includes(stem)) return true;
		}
	}

	const qKey = tokens.length > 0 ? tokens.join("") : toSearchKey(normQuery);
	const cKey = toSearchKey(normCandidate);
	return qKey.length >= 2 && cKey.includes(qKey);
}

/** Escape a literal query so an index's regex mode matches it as a substring. */
export function escapeSearchRegex(text: string): string {
	return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Regex for the fsearch index. macOS stores Hangul file names decomposed (NFD)
 * while typed queries are composed (NFC), and the index compares bytes, so a
 * name matches only when the query uses the same form. Sending both forms as an
 * alternation finds either spelling; a query that has one form (ASCII) stays the
 * plain escaped substring.
 */
export function fsearchRegexQuery(text: string): string {
	const trimmed = text.trim();
	const forms = [...new Set([trimmed.normalize("NFC"), trimmed.normalize("NFD")])];
	if (forms.length === 1) return escapeSearchRegex(trimmed);
	return `(?:${forms.map(escapeSearchRegex).join("|")})`;
}

/**
 * Everything's `regex:` search function is the working regex surface on the
 * bundled ES build. A bare space ends the function's argument, so spaces become
 * `\x20` to keep a literal substring.
 */
export function everythingRegexQuery(text: string): string {
	return `regex:${escapeSearchRegex(text.trim()).replace(/ /g, "\\x20")}`;
}
