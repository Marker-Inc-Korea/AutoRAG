/**
 * Resolves which clone a running window belongs to, for the dev window title.
 *
 * Read from `.git` files rather than spawning git: the label is cosmetic, so a
 * missing or unusual repository must degrade to "path only" instead of failing
 * or slowing startup.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { DevLabel } from "../shared/app-info";

const SHORT_SHA_LENGTH = 7;

export interface DevLabelReader {
	(path: string): string | null;
}

export function resolveClonePath(appPath: string): string {
	return dirname(appPath);
}

function readFile(reader: DevLabelReader, path: string): string | null {
	try {
		return reader(path);
	} catch {
		return null;
	}
}

function commitFromPackedRefs(packed: string, ref: string): string | null {
	for (const line of packed.split("\n")) {
		const [sha, name] = line.trim().split(" ");
		if (name === ref && sha !== undefined && sha.length >= SHORT_SHA_LENGTH) return sha.slice(0, SHORT_SHA_LENGTH);
	}
	return null;
}

export function readDevLabel(clonePath: string, reader: DevLabelReader = (path) => readFileSync(path, "utf8")): DevLabel {
	const gitDir = join(clonePath, ".git");
	const head = readFile(reader, join(gitDir, "HEAD"))?.trim() ?? null;
	if (head === null) return { clonePath, branch: null, commit: null };
	if (!head.startsWith("ref:")) {
		// Detached HEAD: the file holds the commit itself.
		return { clonePath, branch: null, commit: head.slice(0, SHORT_SHA_LENGTH) || null };
	}
	const ref = head.slice("ref:".length).trim();
	const branch = ref.startsWith("refs/heads/") ? ref.slice("refs/heads/".length) : ref;
	const loose = readFile(reader, join(gitDir, ref))?.trim() ?? null;
	const commit =
		loose !== null && loose.length >= SHORT_SHA_LENGTH
			? loose.slice(0, SHORT_SHA_LENGTH)
			: (() => {
					const packed = readFile(reader, join(gitDir, "packed-refs"));
					return packed === null ? null : commitFromPackedRefs(packed, ref);
				})();
	return { clonePath, branch, commit };
}
