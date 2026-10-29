import { statSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

/** A tool call the gate is about to judge. */
export interface ProposedCall {
	readonly tool: string;
	readonly input: Readonly<Record<string, unknown>>;
	readonly cwd: string;
}

/** `raw` as an absolute path: relative to `cwd`, with `~` expanded. */
export function resolveToolPath(raw: string, cwd: string): string {
	return resolve(cwd, raw === "~" || raw.startsWith("~/") ? join(homedir(), raw.slice(1)) : raw);
}

export interface NamedPath {
	readonly path: string;
	/**
	 * True when the word is certainly a path (absolute, `~/`, `./`, `../`, a file name with an extension) or
	 * names something that exists. A plain word that exists nowhere (`fresh` in `mkdir fresh`) is only a
	 * candidate: the gate watches whether the call brings it into existence, and shows it to Jev only then.
	 */
	readonly always: boolean;
}

/** Most paths one call names; explicit ones are kept first when a command has more. */
const MAX_NAMED_PATHS = 40;

const WORD = /"([^"]*)"|'([^']*)'|([^\s;|&<>()`$="',]+)/g;
const EXPLICIT = /^(?:~\/|\.{1,2}\/|\/)/;
const WITH_EXTENSION = /^(?:[\w.@+%-]+\/)*[\w.@+%-]*\w\.[A-Za-z]\w{0,7}$/;
/** `s/old/new/g`, `y/abc/xyz/`: a sed script, not a path. */
const SED_SCRIPT = /^[sy]([/|#,]).*\1.*\1[a-z]*$/;

function exists(path: string): boolean {
	try {
		statSync(path);
		return true;
	} catch {
		return false;
	}
}

/**
 * The paths a call names, for the facts Jev is shown (does each exist, did the agent make it). This does not
 * decide anything and does not understand the command: it picks out the words that look like paths. A word it
 * misses is a fact Jev lacks; a word it wrongly takes is a harmless "does not exist" line.
 */
export function pathsNamedBy(call: ProposedCall): NamedPath[] {
	if (call.tool === "write" || call.tool === "edit") {
		return typeof call.input.path === "string"
			? [{ path: resolveToolPath(call.input.path, call.cwd), always: true }]
			: [];
	}
	if (call.tool !== "bash" || typeof call.input.command !== "string") return [];

	const ranked = new Map<string, { path: string; always: boolean; rank: number }>();
	for (const match of call.input.command.matchAll(WORD)) {
		const word = match[1] ?? match[2] ?? match[3];
		if (word === undefined || word === "" || word.startsWith("-") || word.includes("://") || SED_SCRIPT.test(word)) {
			continue;
		}
		if (!/^[\w./~@+%:-]+$/.test(word)) continue;
		const path = resolveToolPath(word, call.cwd);
		const explicit = EXPLICIT.test(word);
		const present = exists(path);
		const rank = explicit ? 0 : present ? 1 : WITH_EXTENSION.test(word) ? 2 : 3;
		if (!ranked.has(path)) ranked.set(path, { path, always: rank < 3, rank });
	}
	return [...ranked.values()]
		.sort((a, b) => a.rank - b.rank)
		.slice(0, MAX_NAMED_PATHS)
		.map(({ path, always }) => ({ path, always }));
}
