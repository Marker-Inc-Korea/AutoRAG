import { existsSync, statSync } from "node:fs";
import { homedir } from "node:os";

/** Where a cited local path starts. `~/ ` (a space after the tilde) is a malformed path the answer should not contain. */
const PATH_START = /\/Users\/|~\/ ?/gu;
const MAX_PATH_CHARS = 400;

export interface CitedPath {
	/** What the answer wrote, up to the part that exists on disk. */
	readonly written: string;
	/** Absolute path of the longest prefix that exists; undefined when none beyond the home directory does. */
	readonly resolved: string | undefined;
	readonly isFile: boolean;
	readonly missing: boolean;
	readonly malformed: boolean;
}

/**
 * Every local path in `text`, followed on disk. A path in prose is not delimited, and file names hold spaces and
 * parentheses, so the answer's text after the start is tried longest first until a prefix exists. The path is then
 * missing when the next character shows the answer named something deeper than that prefix.
 */
export function citedPaths(text: string): CitedPath[] {
	const out: CitedPath[] = [];
	for (const match of text.matchAll(PATH_START)) {
		const start = (match.index ?? 0) + match[0].length;
		const lineEnd = text.indexOf("\n", start);
		const tail = text.slice(start, Math.min(lineEnd < 0 ? text.length : lineEnd, start + MAX_PATH_CHARS));
		const base = match[0].startsWith("/Users/") ? "/Users/" : `${homedir()}/`;
		const malformed = match[0] === "~/ ";
		let resolvedTail: string | undefined;
		for (let length = tail.length; length > 0; length -= 1) {
			if (tail.endsWith(" ", length) && tail[length - 1] === " ") continue;
			if (existsSync(base + tail.slice(0, length))) {
				resolvedTail = tail.slice(0, length);
				break;
			}
		}
		if (resolvedTail === undefined) {
			out.push({ written: base + tail.split(/[`\s]/u)[0], resolved: undefined, isFile: false, missing: true, malformed });
			continue;
		}
		const absolute = base + resolvedTail;
		const isFile = statSync(absolute).isFile();
		const next = tail[resolvedTail.length];
		// A file is complete wherever it ends (a particle may follow it). A directory is incomplete when a longer name follows.
		const deeper =
			!isFile &&
			next !== undefined &&
			(resolvedTail.endsWith("/") ? /[\p{L}\p{N}_\-.~]/u.test(next) : /[A-Za-z0-9_\-~]/u.test(next));
		out.push({ written: absolute, resolved: absolute, isFile, missing: deeper, malformed });
	}
	return out;
}

