/**
 * Incremental extractor for the top-level `answer` string of a streamed
 * tool-call JSON argument buffer (`emit_fast_answer` /
 * `emit_autorag_results`). Tool-call arguments arrive as raw
 * `toolcall_delta` JSON fragments, and the curated answer lives inside the
 * first key of that JSON — this recovers the answer text as it streams in,
 * without waiting for the tool call to complete.
 *
 * Returns the decoded answer text received so far, or `undefined` while the
 * answer value has not started arriving. Only complete characters are
 * returned: escape sequences or surrogate pairs split across deltas are
 * held back until they complete, so every returned prefix is safe to render.
 */

const ESCAPES: Readonly<Record<string, string>> = {
	'"': '"',
	"\\": "\\",
	"/": "/",
	b: "\b",
	f: "\f",
	n: "\n",
	r: "\r",
	t: "\t",
};

function skipWhitespace(raw: string, start: number): number {
	let index = start;
	while (index < raw.length && /\s/u.test(raw[index])) index += 1;
	return index;
}

/** Read one complete JSON string (opening quote at `start`); `undefined` while it is still streaming. */
function readCompleteString(raw: string, start: number): { value: string; end: number } | undefined {
	if (raw[start] !== '"') return undefined;
	let value = "";
	let index = start + 1;
	while (index < raw.length) {
		const char = raw[index];
		if (char === '"') return { value, end: index + 1 };
		if (char !== "\\") {
			value += char;
			index += 1;
			continue;
		}
		if (index + 1 >= raw.length) return undefined;
		const escapeChar = raw[index + 1];
		if (escapeChar === "u") {
			if (index + 6 > raw.length) return undefined;
			const hex = raw.slice(index + 2, index + 6);
			if (!/^[0-9a-fA-F]{4}$/u.test(hex)) return undefined;
			value += String.fromCharCode(Number.parseInt(hex, 16));
			index += 6;
			continue;
		}
		const mapped = ESCAPES[escapeChar];
		if (mapped === undefined) return undefined;
		value += mapped;
		index += 2;
	}
	return undefined;
}

/** Skip one complete JSON value starting at `start`; `undefined` while it is still streaming. */
function skipValue(raw: string, start: number): number | undefined {
	let index = start;
	let depth = 0;
	while (index < raw.length) {
		const char = raw[index];
		if (char === '"') {
			const string = readCompleteString(raw, index);
			if (string === undefined) return undefined;
			if (depth === 0) return string.end;
			index = string.end;
			continue;
		}
		if (char === "{" || char === "[") {
			depth += 1;
			index += 1;
			continue;
		}
		if (char === "}" || char === "]") {
			if (depth === 0) return undefined;
			depth -= 1;
			index += 1;
			if (depth === 0) return index;
			continue;
		}
		if (depth === 0 && (char === "," || char === "}")) return undefined;
		index += 1;
	}
	return undefined;
}

/** Drop a trailing high surrogate that may pair with the next delta chunk. */
function withoutDanglingSurrogate(text: string): string {
	const last = text.length - 1;
	if (last >= 0) {
		const code = text.charCodeAt(last);
		if (code >= 0xd800 && code <= 0xdbff) return text.slice(0, last);
	}
	return text;
}

/** Decode the string content starting after the opening quote at `start`, tolerating a truncated tail. */
function decodePartialString(raw: string, start: number): string {
	let value = "";
	let index = start;
	while (index < raw.length) {
		const char = raw[index];
		if (char === '"') return value;
		if (char !== "\\") {
			value += char;
			index += 1;
			continue;
		}
		if (index + 1 >= raw.length) return withoutDanglingSurrogate(value);
		const escapeChar = raw[index + 1];
		if (escapeChar === "u") {
			if (index + 6 > raw.length) return withoutDanglingSurrogate(value);
			const hex = raw.slice(index + 2, index + 6);
			if (!/^[0-9a-fA-F]{4}$/u.test(hex)) return withoutDanglingSurrogate(value);
			value += String.fromCharCode(Number.parseInt(hex, 16));
			index += 6;
			continue;
		}
		const mapped = ESCAPES[escapeChar];
		if (mapped === undefined) return withoutDanglingSurrogate(value);
		value += mapped;
		index += 2;
	}
	return withoutDanglingSurrogate(value);
}

/**
 * Extract the answer text streamed so far from a partial tool-call
 * arguments JSON buffer. The FIRST top-level `answer` key wins, so a nested
 * `"answer"` occurrence inside a later value (e.g. a result summary) can
 * never shadow it; while an earlier sibling value is still streaming, the
 * answer has not started and `undefined` is returned.
 */
export function extractPartialAnswer(raw: string): string | undefined {
	let index = skipWhitespace(raw, 0);
	if (index >= raw.length || raw[index] !== "{") return undefined;
	index = skipWhitespace(raw, index + 1);
	for (;;) {
		if (index >= raw.length) return undefined;
		if (raw[index] === "}") return undefined;
		const key = readCompleteString(raw, index);
		if (key === undefined) return undefined;
		index = skipWhitespace(raw, key.end);
		if (index >= raw.length || raw[index] !== ":") return undefined;
		index = skipWhitespace(raw, index + 1);
		if (index >= raw.length) return undefined;
		if (key.value === "answer") {
			if (raw[index] !== '"') return undefined;
			return decodePartialString(raw, index + 1);
		}
		const skipped = skipValue(raw, index);
		if (skipped === undefined) return undefined;
		index = skipWhitespace(raw, skipped);
		if (index >= raw.length) return undefined;
		if (raw[index] === ",") {
			index = skipWhitespace(raw, index + 1);
			continue;
		}
		return undefined;
	}
}
