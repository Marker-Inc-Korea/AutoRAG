import type { Workspace } from "@nomadamas/agentdir";

/** A single file's grep result over the virtual tree (source paths never appear). */
export interface GrepHit {
	/** Virtual path of the matching file (e.g. `/docs/a.txt`). */
	virtualPath: string;
	/** First matching line's text. */
	line: string;
	/** 1-based line number of the first match. */
	lineNumber: number;
	/** Total regex matches across the file. */
	matchCount: number;
	/** Relevance score: matchCount dominates; shallower paths tie-break. */
	score: number;
}

export interface GrepOptions {
	/** Restrict the search to a virtual glob (defaults to all files under the tree). */
	pathGlob?: string;
	/** Case-insensitive matching. */
	ignoreCase?: boolean;
	/** Cap the number of returned hits (after scoring). */
	maxResults?: number;
}

/** Number of `/` separators in a virtual path (e.g. `/docs/a.txt` -> 2). */
function pathDepth(virtualPath: string): number {
	return (virtualPath.match(/\//g) ?? []).length;
}

function escapeRegExp(literal: string): string {
	return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function countMatches(line: string, regex: RegExp): number {
	return (line.match(regex) ?? []).length;
}

/**
 * Search file contents across the agentdir virtual tree.
 *
 * Implemented purely over the agentdir Node binding (`rglob` to enumerate, then
 * `readBytes` to read content) so the same core backs both the `grep` agent
 * tool and the posix `RetrievalMethod`. Directories returned by `rglob` are
 * skipped (their `readBytes` throws). Source filesystem paths never appear in
 * the results — only virtual paths.
 *
 * Score = matchCount + 1/(1 + depth) so a file with more matches ranks above
 * one with fewer, and shallower paths break ties.
 */
export async function agentdirGrep(ws: Workspace, pattern: string, options: GrepOptions = {}): Promise<GrepHit[]> {
	const glob = options.pathGlob ?? "/**/*";
	const flags = options.ignoreCase ? "gi" : "g";
	let regex: RegExp;
	try {
		regex = new RegExp(pattern, flags);
	} catch {
		regex = new RegExp(escapeRegExp(pattern), flags);
	}

	const candidates = await ws.rglob(glob);
	const hits: GrepHit[] = [];
	for (const virtualPath of candidates) {
		let content: string;
		try {
			content = (await ws.readBytes(virtualPath)).toString("utf8");
		} catch {
			continue; // directory or unreadable entry
		}
		const lines = content.split(/\r?\n/);
		let matchCount = 0;
		let firstLine = "";
		let firstLineNumber = 0;
		for (let i = 0; i < lines.length; i += 1) {
			const n = countMatches(lines[i], regex);
			if (n > 0) {
				matchCount += n;
				if (firstLineNumber === 0) {
					firstLine = lines[i];
					firstLineNumber = i + 1;
				}
			}
		}
		if (matchCount > 0) {
			hits.push({
				virtualPath,
				line: firstLine,
				lineNumber: firstLineNumber,
				matchCount,
				score: matchCount + 1 / (1 + pathDepth(virtualPath)),
			});
		}
	}

	hits.sort((a, b) => b.score - a.score);
	return options.maxResults ? hits.slice(0, options.maxResults) : hits;
}
