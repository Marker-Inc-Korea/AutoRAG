import { homedir } from "node:os";
import { join, resolve } from "node:path";

/**
 * Context used to build a conservative, data-driven public-output sanitizer.
 * The sanitizer redacts ONLY known filesystem roots and exact internal source
 * identifiers — never broad relative-path heuristics — so legitimate prose that
 * merely looks path-like (e.g. `docs/policy`, URLs, product names) is preserved.
 */
export interface PathSanitizerContext {
	readonly workspaceRoot: string;
	readonly searchPaths: readonly string[];
	readonly sources: Iterable<string>;
}

export interface SanitizeResult {
	readonly text: string;
	readonly redacted: boolean;
}

export interface PathSanitizer {
	sanitize(text: string): SanitizeResult;
}

const PLACEHOLDER = "[redacted]";

function escapeRegExp(value: string): string {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Builds a sanitizer that redacts:
 *  1. Absolute filesystem roots (workspace root, resolved search paths, the
 *     home directory, and the internal `.autorag` directory) together with any
 *     descendant path segments.
 *  2. Exact internal source identifiers (opaque root-relative ids from the
 *     result mapping/evidence) matched on segment boundaries.
 *
 * It deliberately avoids matching generic relative-path-looking text.
 */
export function createPathSanitizer(context: PathSanitizerContext): PathSanitizer {
	const absoluteRoots = new Set<string>();
	const addRoot = (candidate: string | undefined): void => {
		if (candidate && candidate.length > 0) absoluteRoots.add(resolve(candidate));
	};
	addRoot(context.workspaceRoot);
	addRoot(join(context.workspaceRoot, ".autorag"));
	for (const searchPath of context.searchPaths) {
		addRoot(resolve(searchPath));
		addRoot(resolve(context.workspaceRoot, searchPath));
	}
	addRoot(homedir());

	// Longest roots first so a nested root is redacted before its parent.
	const rootPatterns = [...absoluteRoots]
		.sort((a, b) => b.length - a.length)
		.map((root) => `${escapeRegExp(root)}[^\\s"'\`)\\]]*`);

	const sourcePatterns = [...new Set(context.sources)]
		.filter((source) => source.length > 0)
		.sort((a, b) => b.length - a.length)
		// Internal source identifiers are opaque root-relative paths. Anchor only on
		// the RIGHT (no trailing word/path char) so a longer unrelated filename is
		// not partially matched, but still scrub the id when it is glued to a
		// preceding token (e.g. "id/data/secret.txt").
		.map((source) => `${escapeRegExp(source)}(?![\\w./-])`);

	const patterns = [...rootPatterns, ...sourcePatterns];
	const regex = patterns.length > 0 ? new RegExp(patterns.join("|"), "g") : undefined;

	return {
		sanitize(text: string): SanitizeResult {
			if (regex === undefined || text.length === 0) return { text, redacted: false };
			let redacted = false;
			const sanitized = text.replace(regex, () => {
				redacted = true;
				return PLACEHOLDER;
			});
			return { text: sanitized, redacted };
		},
	};
}
