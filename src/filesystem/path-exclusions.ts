import { posix, win32 } from "node:path";

/** Match exact paths and directory subtrees without matching sibling prefixes. */
export function createPathExclusionMatcher(
	excludedPaths: readonly string[],
	platform: NodeJS.Platform = process.platform,
): (path: string) => boolean {
	const paths = platform === "win32" ? win32 : posix;
	const normalize = (path: string) => {
		const absolute = paths.resolve(path);
		return platform === "win32" ? absolute.toLowerCase() : absolute;
	};
	const excluded = excludedPaths.map((path) => {
		const absolute = normalize(path);
		return { absolute, prefix: absolute.endsWith(paths.sep) ? absolute : `${absolute}${paths.sep}` };
	});
	return (path) => {
		const absolute = normalize(path);
		return excluded.some((entry) => absolute === entry.absolute || absolute.startsWith(entry.prefix));
	};
}
