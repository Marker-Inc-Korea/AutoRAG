import { basename, relative, resolve, sep } from "node:path";

export interface SourceRoot {
	readonly rootPath: string;
	readonly prefix: string;
}

export function planSourceRoots(searchPaths: readonly string[]): readonly SourceRoot[] {
	const sorted = searchPaths.map((path) => resolve(path)).sort((a, b) => a.localeCompare(b));
	const used = new Set<string>();
	return sorted.map((rootPath) => {
		const base = basename(rootPath.replace(/[/\\]+$/, "")) || "root";
		let prefix = `/${base}`;
		let suffix = 2;
		while (used.has(prefix)) {
			prefix = `/${base}-${suffix}`;
			suffix += 1;
		}
		used.add(prefix);
		return { rootPath, prefix };
	});
}

export function sourceIdentifier(sourceRoot: SourceRoot, sourcePath: string): string {
	const rel = relative(sourceRoot.rootPath, resolve(sourcePath));
	const suffix = rel === "" ? "" : `/${rel.split(sep).join("/")}`;
	return `${sourceRoot.prefix}${suffix}`;
}
