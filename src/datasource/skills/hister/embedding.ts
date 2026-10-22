/**
 * Sharing AutoRAG's embedding backend with Hister.
 *
 * Hister keeps its own vector store and embeds with whatever `semantic_search` in its config.yml names, so the
 * only way the two can use one model is to copy the shared endpoint (the `minsync serve` daemon by default) into
 * that block. Mismatched models are not an error anywhere: the vectors simply do not line up and results degrade
 * silently. This module derives the block from {@link ResolvedEmbeddingEndpoint}, reads what Hister currently has,
 * and rewrites only the six keys involved. Hister itself is never modified; its config.yml is its documented
 * interface.
 *
 * `yaml` is a dev dependency of this package, so the file is edited line by line. Only the `semantic_search`
 * block's own keys are touched and every other byte is preserved; the tests compare the result with a real YAML
 * parser.
 */

import type { ResolvedEmbeddingEndpoint } from "../../embedding-env.ts";

/** The `semantic_search` keys that must agree with the shared embedding backend. */
export interface HisterSemanticSettings {
	readonly enable: boolean;
	readonly embedding_endpoint: string;
	readonly embedding_model: string;
	readonly dimensions: number;
	readonly query_prefix: string;
	readonly document_prefix: string;
}

export type HisterSemanticKey = keyof HisterSemanticSettings;

const KEYS: readonly HisterSemanticKey[] = [
	"enable",
	"embedding_endpoint",
	"embedding_model",
	"dimensions",
	"query_prefix",
	"document_prefix",
];

/**
 * What Hister uses when its config does not say (v0.20.0 `config.go`): semantic search off, an Ollama endpoint, and
 * a `query: ` prefix. Used only to report a missing key; the apply step always writes every key.
 */
export const HISTER_SEMANTIC_DEFAULTS: HisterSemanticSettings = {
	enable: false,
	embedding_endpoint: "http://localhost:11434/v1/embeddings",
	embedding_model: "qwen3-embedding:8b",
	dimensions: 2000,
	query_prefix: "query: ",
	document_prefix: "",
};

/**
 * Keys whose change makes Hister's stored vectors stale. Hister fingerprints the endpoint, model, dimensions and
 * document prefix (not the query prefix) and reports `Embedding settings differ ... run hister reindex`.
 */
const FINGERPRINTED: readonly HisterSemanticKey[] = [
	"enable",
	"embedding_endpoint",
	"embedding_model",
	"dimensions",
	"document_prefix",
];

export type DeriveResult =
	| { readonly ok: true; readonly settings: HisterSemanticSettings }
	| { readonly ok: false; readonly reason: string };

/**
 * The settings that make Hister embed exactly like AutoRAG does. Refuses what Hister cannot do faithfully:
 * - a TEI endpoint (Hister only calls the OpenAI `/v1/embeddings` route);
 * - a remote endpoint (browsing text would leave the machine, and Hister's config would have to hold the key).
 */
export function deriveHisterSemanticSettings(endpoint: ResolvedEmbeddingEndpoint): DeriveResult {
	if (endpoint.protocol !== "openai") {
		return { ok: false, reason: "Hister only speaks the OpenAI embeddings route; this embedding backend uses TEI" };
	}
	if (endpoint.backend !== "daemon") {
		return {
			ok: false,
			reason:
				"the shared embedding backend is a remote endpoint: browsing text would leave this machine, and Hister's config would have to hold the API key. Run `minsync serve` to share a local model",
		};
	}
	const root = endpoint.baseUrl.replace(/\/+$/u, "").replace(/\/v1$/u, "");
	return {
		ok: true,
		settings: {
			enable: true,
			embedding_endpoint: `${root}/v1/embeddings`,
			embedding_model: endpoint.model,
			dimensions: endpoint.dimension,
			query_prefix: endpoint.queryPrefix ?? "",
			document_prefix: endpoint.passagePrefix ?? "",
		},
	};
}

export function differingSemanticKeys(
	current: HisterSemanticSettings,
	wanted: HisterSemanticSettings,
): readonly HisterSemanticKey[] {
	return KEYS.filter((key) => current[key] !== wanted[key]).sort();
}

export function reindexRequired(changed: readonly HisterSemanticKey[]): boolean {
	return changed.some((key) => FINGERPRINTED.includes(key));
}

// ---------------------------------------------------------------------------------------------------------------
// config.yml reading and editing
// ---------------------------------------------------------------------------------------------------------------

const BLOCK = "semantic_search";

interface Located {
	/** Index of the `semantic_search:` line. */
	readonly header: number;
	/** First line after the block (exclusive end). */
	readonly end: number;
	/** Indentation of the block's keys. */
	readonly indent: string;
	/** Each key's line range inside the block, `[start, end)`. */
	readonly entries: ReadonlyMap<string, { readonly start: number; readonly end: number }>;
	/** True for `semantic_search: {}` / `{ }`, which has no lines of its own. */
	readonly emptyFlow: boolean;
}

type Locate =
	| { readonly ok: true; readonly found: Located | undefined }
	| { readonly ok: false; readonly reason: string };

function indentOf(line: string): number {
	return line.length - line.trimStart().length;
}

function isBlank(line: string): boolean {
	const trimmed = line.trim();
	return trimmed === "" || trimmed.startsWith("#");
}

/**
 * Whether a single- or double-quoted YAML scalar is still open at the end of `line`. Only the value part of a
 * `key: value` line, or a continuation line, can open or close one; `'` inside a plain word is just a character.
 */
function quoteStateAfter(line: string, open: "'" | '"' | undefined): "'" | '"' | undefined {
	let state = open;
	let text = line;
	if (state === undefined) {
		const value = /^\s*[A-Za-z_][\w-]*\s*:\s*(.*)$/u.exec(line)?.[1];
		if (value === undefined) return undefined;
		const quote = value[0];
		if (quote !== "'" && quote !== '"') return undefined;
		state = quote;
		text = value.slice(1);
	}
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (state === '"' && ch === "\\") {
			i += 1;
			continue;
		}
		if (ch !== state) continue;
		if (state === "'" && text[i + 1] === "'") {
			i += 1;
			continue;
		}
		return undefined;
	}
	return state;
}

function locate(lines: readonly string[]): Locate {
	const header = lines.findIndex((line) => /^semantic_search\s*:/u.test(line));
	if (header === -1) return { ok: true, found: undefined };
	const rest =
		lines[header]
			?.replace(/^semantic_search\s*:/u, "")
			.replace(/\s+#.*$/u, "")
			.trim() ?? "";
	if (rest === "{}" || rest === "{ }") {
		return { ok: true, found: { header, end: header + 1, indent: "    ", entries: new Map(), emptyFlow: true } };
	}
	if (rest !== "") {
		return {
			ok: false,
			reason: `${BLOCK} in flow style or with an inline value cannot be edited safely; rewrite it as an indented block`,
		};
	}
	// A top-level key ends the block. A column-0 line while a quoted scalar is still open is that scalar's
	// continuation, which Hister's YAML parser folds (a hand edit produced exactly this shape).
	let end = header + 1;
	let indent: string | undefined;
	let open: "'" | '"' | undefined;
	for (let i = header + 1; i < lines.length; i++) {
		const line = lines[i] ?? "";
		if (open === undefined && isBlank(line)) {
			end = i + 1;
			continue;
		}
		if (open === undefined && indentOf(line) === 0) break;
		if (open === undefined) indent ??= line.slice(0, indentOf(line));
		open = quoteStateAfter(line, open);
		end = i + 1;
	}
	while (end > header + 1 && isBlank(lines[end - 1] ?? "")) end -= 1;
	const entries = new Map<string, { start: number; end: number }>();
	const keyIndent = indent?.length ?? 4;
	let current: { key: string; start: number } | undefined;
	const close = (at: number): void => {
		if (current !== undefined) entries.set(current.key, { start: current.start, end: at });
	};
	let inQuote: "'" | '"' | undefined;
	for (let i = header + 1; i < end; i++) {
		const line = lines[i] ?? "";
		if (inQuote === undefined && isBlank(line)) continue;
		const match = inQuote === undefined ? /^(\s+)([A-Za-z_][\w-]*)\s*:/u.exec(line) : null;
		if (match !== null && (match[1]?.length ?? 0) === keyIndent) {
			close(i);
			current = { key: match[2] ?? "", start: i };
		}
		inQuote = quoteStateAfter(line, inQuote);
	}
	close(end);
	return { ok: true, found: { header, end, indent: indent ?? "    ", entries, emptyFlow: false } };
}

/** Unfold the YAML scalar that starts on `lines[0]` (after `key:`) and continues on `lines[1..]`. */
function scalarOf(first: string, continuation: readonly string[]): string {
	const value = first.replace(/^[^:]*:\s*/u, "");
	const trimmed = value.trim();
	if (/^[|>][+-]?\d*\s*(#.*)?$/u.test(trimmed)) {
		const folded = trimmed.startsWith(">");
		const chomp = /[+-]/u.exec(trimmed)?.[0];
		const body = continuation.map((line) => line.trim());
		while (body.length > 0 && body[body.length - 1] === "") body.pop();
		const text = body.join(folded ? " " : "\n");
		return chomp === "+" || chomp === undefined ? `${text}\n` : text;
	}
	const joined = [value, ...continuation.map((line) => line.trim())].join(" ").trim();
	if (joined.startsWith('"')) {
		const close = lastUnescapedQuote(joined);
		return unescapeDouble(joined.slice(1, close === -1 ? undefined : close));
	}
	if (joined.startsWith("'")) {
		const close = joined.lastIndexOf("'");
		return joined.slice(1, close <= 0 ? undefined : close).replace(/''/gu, "'");
	}
	return joined.replace(/\s+#.*$/u, "").trim();
}

function lastUnescapedQuote(text: string): number {
	for (let i = text.length - 1; i > 0; i--) {
		if (text[i] !== '"') continue;
		let slashes = 0;
		for (let j = i - 1; j >= 0 && text[j] === "\\"; j--) slashes += 1;
		if (slashes % 2 === 0) return i;
	}
	return -1;
}

function unescapeDouble(text: string): string {
	return text.replace(/\\(["\\/nrt0]|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})/gu, (_all, code: string) => {
		if (code === "n") return "\n";
		if (code === "r") return "\r";
		if (code === "t") return "\t";
		if (code === "0") return "\0";
		if (code === '"' || code === "\\" || code === "/") return code;
		return String.fromCharCode(Number.parseInt(code.slice(1), 16));
	});
}

export type ReadResult =
	| {
			readonly ok: true;
			readonly settings: HisterSemanticSettings;
			/** Keys the file sets itself; the rest are Hister's defaults. */
			readonly explicit: readonly HisterSemanticKey[];
	  }
	| { readonly ok: false; readonly reason: string };

export function readHisterSemanticSettings(text: string): ReadResult {
	const lines = text.split(/\r?\n/u);
	const located = locate(lines);
	if (!located.ok) return located;
	const settings: { -readonly [K in HisterSemanticKey]: HisterSemanticSettings[K] } = { ...HISTER_SEMANTIC_DEFAULTS };
	const explicit: HisterSemanticKey[] = [];
	if (located.found === undefined) return { ok: true, settings, explicit };
	for (const key of KEYS) {
		const entry = located.found.entries.get(key);
		if (entry === undefined) continue;
		const raw = scalarOf(lines[entry.start] ?? "", lines.slice(entry.start + 1, entry.end));
		explicit.push(key);
		if (key === "enable") settings.enable = raw === "true";
		else if (key === "dimensions") settings.dimensions = Number.parseInt(raw, 10);
		else settings[key] = raw;
	}
	return { ok: true, settings, explicit };
}

/** A YAML double-quoted scalar: always valid for any text, including a newline, which Hister's prefix needs. */
function doubleQuoted(value: string): string {
	return JSON.stringify(value);
}

function renderValue(key: HisterSemanticKey, settings: HisterSemanticSettings): string {
	if (key === "enable") return settings.enable ? "true" : "false";
	if (key === "dimensions") return String(settings.dimensions);
	return doubleQuoted(settings[key]);
}

export type ApplyResult =
	| { readonly ok: true; readonly text: string }
	| { readonly ok: false; readonly reason: string };

/** Set every {@link KEYS} entry inside `semantic_search`, creating the block when it is absent. */
export function applyHisterSemanticSettings(text: string, settings: HisterSemanticSettings): ApplyResult {
	const newline = text.includes("\r\n") ? "\r\n" : "\n";
	const lines = text.split(/\r?\n/u);
	const trailing = lines.length > 0 && lines[lines.length - 1] === "";
	if (trailing) lines.pop();
	const located = locate(lines);
	if (!located.ok) return located;
	const found = located.found;
	if (found === undefined) {
		const block = [`${BLOCK}:`, ...KEYS.map((key) => `    ${key}: ${renderValue(key, settings)}`)];
		const next = [...lines, ...block];
		return { ok: true, text: next.join(newline) + newline };
	}
	const indent = found.indent;
	if (found.emptyFlow) {
		const expanded = [`${BLOCK}:`, ...KEYS.map((key) => `${indent}${key}: ${renderValue(key, settings)}`)];
		lines.splice(found.header, 1, ...expanded);
		return { ok: true, text: lines.join(newline) + (trailing ? newline : "") };
	}
	// Work from the bottom so earlier line numbers stay valid.
	const missing: string[] = [];
	const edits: { start: number; end: number; replacement: string }[] = [];
	for (const key of KEYS) {
		const entry = found.entries.get(key);
		const line = `${indent}${key}: ${renderValue(key, settings)}`;
		if (entry === undefined) missing.push(line);
		else edits.push({ start: entry.start, end: entry.end, replacement: line });
	}
	edits.sort((a, b) => b.start - a.start);
	for (const edit of edits) lines.splice(edit.start, edit.end - edit.start, edit.replacement);
	if (missing.length > 0) {
		const removed = edits.reduce((sum, edit) => sum + (edit.end - edit.start - 1), 0);
		lines.splice(found.end - removed, 0, ...missing);
	}
	return { ok: true, text: lines.join(newline) + (trailing ? newline : "") };
}
