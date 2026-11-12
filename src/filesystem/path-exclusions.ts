import { matchesGlob, posix, win32 } from "node:path";

/**
 * An `excludePaths` entry with `*`, `?` or `[` is a glob over the whole absolute path (`**\/node_modules/**`,
 * `**\/Downloads/**`, `**\/*.log`); any other entry is a file or directory path. A glob is never resolved against a
 * directory: it would then stop matching anywhere else.
 */
export function isGlobExclusion(entry: string): boolean {
	return /[*?[]/u.test(entry);
}

/**
 * Whether an absolute path matches a glob entry. Separators are compared as `/`, and a leading `**\/` also matches at
 * the filesystem root, because `matchesGlob("/a.log", "**\/*.log")` differs between platforms in Node 24.
 */
export function matchesGlobExclusion(path: string, glob: string): boolean {
	const slashed = path.replaceAll("\\", "/");
	if (matchesGlob(slashed, glob)) return true;
	return glob.startsWith("**/") && matchesGlob(slashed, glob.slice(3));
}

/** Match exact paths and directory subtrees without matching sibling prefixes, and glob entries (see {@link isGlobExclusion}). */
export function createPathExclusionMatcher(
	excludedPaths: readonly string[],
	platform: NodeJS.Platform = process.platform,
): (path: string) => boolean {
	const paths = platform === "win32" ? win32 : posix;
	const normalize = (path: string) => {
		const absolute = paths.resolve(path);
		return platform === "win32" ? absolute.toLowerCase() : absolute;
	};
	const globs = excludedPaths
		.filter(isGlobExclusion)
		.map((glob) => (platform === "win32" ? glob.toLowerCase() : glob));
	const excluded = excludedPaths
		.filter((path) => !isGlobExclusion(path))
		.map((path) => {
			const absolute = normalize(path);
			return { absolute, prefix: absolute.endsWith(paths.sep) ? absolute : `${absolute}${paths.sep}` };
		});
	return (path) => {
		const absolute = normalize(path);
		return (
			excluded.some((entry) => absolute === entry.absolute || absolute.startsWith(entry.prefix)) ||
			globs.some((glob) => matchesGlobExclusion(absolute, glob))
		);
	};
}
