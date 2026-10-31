import { createHash } from "node:crypto";
import { countTokens } from "gpt-tokenizer/encoding/o200k_base";

/**
 * Token estimate for the context guard.
 *
 * pi's `estimateTokens` guesses chars/4, which is right for English prose and
 * wrong by 2-4x for the text parsers produce from binaries and chat exports:
 * punctuation, control characters, emoji, CJK and encoded blobs run 0.5-1 token
 * per character. This counts with the `o200k_base` BPE instead. It is not every
 * provider's tokenizer (measured against hosted models it lands within ~20% of
 * DeepSeek and GPT counts and under Claude's on prose, emoji and Korean), so the
 * guard keeps its own headroom and the overflow ceiling in `cloud/context-overflow.ts`
 * for a provider that still counts more.
 */

/** Special-token text in a parsed file is plain text to the model; the default would throw on it. */
const PLAIN_TEXT = { disallowedSpecial: new Set<string>() };
/** Below this, counting is cheaper than hashing for the cache. */
const CACHE_MIN_CHARS = 2_000;
const CACHE_ENTRIES = 64;
/** Keyed by length plus SHA-1 so the cache never holds a megabyte string alive. */
const cache = new Map<string, number>();

/** Tokens in `text`, never fewer than the chars/4 guess so prose keeps the budget it had. */
export function estimateTextTokens(text: string): number {
	return Math.max(Math.ceil(text.length / 4), countText(text));
}

/** How many tokens the pieces add beyond their chars/4 guesses; zero for prose. */
export function tokensBeyondCharGuess(parts: readonly string[]): number {
	let excess = 0;
	for (const part of parts) excess += Math.max(0, countText(part) - Math.ceil(part.length / 4));
	return excess;
}

function countText(text: string): number {
	if (text.length === 0) return 0;
	if (text.length < CACHE_MIN_CHARS) return countTokens(text, PLAIN_TEXT);
	const key = `${text.length}:${createHash("sha1").update(text).digest("base64")}`;
	const hit = cache.get(key);
	if (hit !== undefined) {
		// Re-insert so the Map's insertion order is least-recently-used first.
		cache.delete(key);
		cache.set(key, hit);
		return hit;
	}
	const count = countTokens(text, PLAIN_TEXT);
	cache.set(key, count);
	if (cache.size > CACHE_ENTRIES) cache.delete(cache.keys().next().value as string);
	return count;
}
