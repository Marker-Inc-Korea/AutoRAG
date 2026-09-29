/**
 * Path helpers shared by the sidebar, breadcrumbs, and list rows.
 *
 * Paths are plain strings: OS-absolute ("/Users/x/Documents") from the fs
 * bridge, or prototype-style relative roots ("Documents/Finance") from the
 * fixtures. Both forms round-trip through these helpers.
 */

export function isAbsolutePath(path: string): boolean {
	return path.startsWith("/");
}

export function pathSegments(path: string): string[] {
	return path.split("/").filter((segment) => segment.length > 0);
}

export function basename(path: string): string {
	const segments = pathSegments(path);
	return segments[segments.length - 1] ?? "";
}

export function dirname(path: string): string {
	const segments = pathSegments(path);
	const absolute = isAbsolutePath(path);
	if (segments.length <= 1) {
		return absolute ? "/" : "";
	}
	return (absolute ? "/" : "") + segments.slice(0, -1).join("/");
}

export function joinPath(dir: string, name: string): string {
	if (dir === "") {
		return name;
	}
	return dir === "/" ? `/${name}` : `${dir}/${name}`;
}

/** Location rendered with the reference separator, e.g. "Documents › Finance". */
export function whereSegments(path: string): string {
	return pathSegments(path).join(" › ");
}

export interface Crumb {
	readonly label: string;
	readonly path: string;
	/** True when a "/" separator precedes this crumb. */
	readonly separator: boolean;
	readonly isLast: boolean;
}

/**
 * At most the last `maxSegments` segments, prefixed with a "…" crumb that
 * navigates one level above the visible window (handoff README §2 Toolbar).
 */
export function breadcrumbTrail(path: string, maxSegments = 2): Crumb[] {
	const segments = pathSegments(path);
	if (segments.length === 0) {
		return [];
	}
	const prefix = isAbsolutePath(path) ? "/" : "";
	const pathAt = (index: number): string => prefix + segments.slice(0, index + 1).join("/");
	const start = Math.max(0, segments.length - maxSegments);
	const crumbs: Crumb[] = [];
	if (start > 0) {
		crumbs.push({ label: "…", path: pathAt(start - 1), separator: false, isLast: false });
	}
	for (let i = start; i < segments.length; i++) {
		crumbs.push({
			label: segments[i] ?? "",
			path: pathAt(i),
			separator: i > start || start > 0,
			isLast: i === segments.length - 1,
		});
	}
	return crumbs;
}

export interface NavTarget {
	readonly name: string;
	readonly path: string;
}

/** The sidebar entry that owns `path` — the longest matching nav path. */
export function activeNavPath(path: string, targets: readonly NavTarget[]): string | null {
	let best: string | null = null;
	for (const target of targets) {
		const prefix = target.path.endsWith("/") ? target.path : `${target.path}/`;
		const owns = path === target.path || path.startsWith(prefix);
		if (owns && (best === null || target.path.length > best.length)) {
			best = target.path;
		}
	}
	return best;
}
