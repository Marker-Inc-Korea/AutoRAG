import { basename } from "node:path";
import type { JikjiRetryStep } from "./types.ts";

/** Characters that are shell syntax when unquoted. A command using them is not a plain argv line. */
const SHELL_SYNTAX_RE = /[;&|<>()$`\n\r*?{}~#!]/u;

/**
 * Split one POSIX-style command line into words. Handles single quotes (and the
 * `'\''` idiom jikji uses for embedded quotes), double quotes and backslash
 * escapes. Never executes or expands anything; returns `undefined` for an
 * unterminated quote or for shell syntax outside quotes, so the result is only
 * ever treated as data.
 */
function shellWords(command: string): string[] | undefined {
	const words: string[] = [];
	let current = "";
	let inWord = false;
	let i = 0;
	while (i < command.length) {
		const ch = command[i] as string;
		if (ch === "'") {
			const end = command.indexOf("'", i + 1);
			if (end === -1) return undefined;
			current += command.slice(i + 1, end);
			inWord = true;
			i = end + 1;
		} else if (ch === '"') {
			let j = i + 1;
			let closed = false;
			while (j < command.length) {
				const inner = command[j] as string;
				if (inner === '"') {
					closed = true;
					break;
				}
				if (inner === "$" || inner === "`") return undefined;
				if (inner === "\\" && j + 1 < command.length) {
					current += command[j + 1];
					j += 2;
					continue;
				}
				current += inner;
				j += 1;
			}
			if (!closed) return undefined;
			inWord = true;
			i = j + 1;
		} else if (ch === "\\") {
			if (i + 1 >= command.length) return undefined;
			current += command[i + 1];
			inWord = true;
			i += 2;
		} else if (ch === " " || ch === "\t") {
			if (inWord) {
				words.push(current);
				current = "";
				inWord = false;
			}
			i += 1;
		} else {
			if (SHELL_SYNTAX_RE.test(ch)) return undefined;
			current += ch;
			inWord = true;
			i += 1;
		}
	}
	if (inWord) words.push(current);
	return words;
}

/**
 * Read the `jikji discover` step out of a `next_commands` entry. The command is
 * data: AutoRAG builds its own argv from the parsed root, query, top-k and
 * proof, never runs the string. Returns `undefined` unless the line is exactly
 * a `discover <root> <query>` retry (`--after-jikji-retry` plus a proof, only
 * flags AutoRAG understands) whose proof matches `envelopeProof` when jikji
 * also reports one, so an unknown or tampered shape never reaches a process.
 */
export function parseJikjiRetryCommand(command: string, envelopeProof: string | undefined): JikjiRetryStep | undefined {
	const words = shellWords(command);
	if (words === undefined || words.length < 4) return undefined;
	const program = basename(words[0] as string).replace(/\.exe$/iu, "");
	if (program !== "jikji" || words[1] !== "discover") return undefined;
	const root = words[2] as string;
	const query = words[3] as string;
	if (root.length === 0 || query.length === 0) return undefined;

	let topK: number | undefined;
	let proof: string | undefined;
	let afterRetry = false;
	for (let i = 4; i < words.length; i += 1) {
		const flag = words[i] as string;
		if (flag === "--json") continue;
		if (flag === "--after-jikji-retry") {
			afterRetry = true;
		} else if (flag === "--top-k" || flag === "--retry-proof") {
			const value = words[i + 1];
			if (value === undefined) return undefined;
			i += 1;
			if (flag === "--retry-proof") {
				proof = value;
			} else {
				if (!/^[1-9][0-9]*$/u.test(value)) return undefined;
				topK = Number(value);
			}
		} else {
			return undefined;
		}
	}
	if (!afterRetry || proof === undefined || proof.length === 0) return undefined;
	if (envelopeProof !== undefined && envelopeProof.length > 0 && envelopeProof !== proof) return undefined;
	return { root, query, ...(topK === undefined ? {} : { topK }), proof };
}
