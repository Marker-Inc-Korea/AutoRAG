/**
 * Path helpers shared by the sidebar, breadcrumbs, and list rows.
 *
 * Paths are plain strings and every convention round-trips: OS-absolute posix
 * ("/Users/x/Documents"), Windows drive and UNC ("C:\Users\x",
 * "\\server\share\x") from the fs bridge, and prototype-style relative roots
 * ("Documents/Finance") from the fixtures.
 */

interface PathRoot {
	readonly prefix: string;
	readonly separator: "/" | "\\";
	/** Leading segments owned by the root; never rendered as crumb labels. */
	readonly skip: number;
}

const DRIVE_ROOT = /^([A-Za-z]:)[\\/]/;
const UNC_ROOT = /^(?:\\\\|\/\/)([^\\/]+)[\\/]([^\\/]+)(?:[\\/]|$)/;

function pathRoot(path: string): PathRoot | null {
	const drive = DRIVE_ROOT.exec(path);
	if (drive !== null) {
		return { prefix: `${drive[1]}\\`, separator: "\\", skip: 1 };
	}
	const unc = UNC_ROOT.exec(path);
	if (unc !== null) {
		return { prefix: `\\\\${unc[1]}\\${unc[2]}\\`, separator: "\\", skip: 2 };
	}
	if (path.startsWith("/")) {
		return { prefix: "/", separator: "/", skip: 0 };
	}
	return null;
}

function separatorOf(path: string): "/" | "\\" {
	const root = pathRoot(path);
	if (root !== null) {
		return root.separator;
	}
	return path.lastIndexOf("\\") > path.lastIndexOf("/") ? "\\" : "/";
}

export function isAbsolutePath(path: string): boolean {
	return pathRoot(path) !== null;
}

export function pathSegments(path: string): string[] {
	return path.split(/[\\/]+/).filter((segment) => segment.length > 0);
}

export function basename(path: string): string {
	const segments = pathSegments(path);
	return segments[segments.length - 1] ?? "";
}

export function dirname(path: string): string {
	const segments = pathSegments(path);
	const root = pathRoot(path);
	if (root === null) {
		if (segments.length <= 1) {
			return "";
		}
		return segments.slice(0, -1).join("/");
	}
	const rest = segments.slice(root.skip);
	if (rest.length <= 1) {
		return root.prefix;
	}
	return root.prefix + rest.slice(0, -1).join(root.separator);
}

export function joinPath(dir: string, name: string): string {
	if (dir === "") {
		return name;
	}
	if (dir.endsWith("/") || dir.endsWith("\\")) {
		return dir + name;
	}
	return `${dir}${separatorOf(dir)}${name}`;
}

/** Location rendered with the reference separator, e.g. "Documents › Finance". */
export function whereSegments(path: string): string {
	return pathSegments(path).join(" › ");
}

export interface Crumb {
	readonly label: string;
	readonly path: string;
	/** True when a separator precedes this crumb. */
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
	const root = pathRoot(path);
	const prefix = root?.prefix ?? "";
	const separator = root?.separator ?? separatorOf(path);
	const parts = segments.slice(root?.skip ?? 0);
	if (parts.length === 0) {
		return [];
	}
	const pathAt = (index: number): string => prefix + parts.slice(0, index + 1).join(separator);
	const start = Math.max(0, parts.length - maxSegments);
	const crumbs: Crumb[] = [];
	if (start > 0) {
		crumbs.push({ label: "…", path: pathAt(start - 1), separator: false, isLast: false });
	}
	for (let i = start; i < parts.length; i++) {
		crumbs.push({
			label: parts[i] ?? "",
			path: pathAt(i),
			separator: i > start || start > 0,
			isLast: i === parts.length - 1,
		});
	}
	return crumbs;
}

export interface NavTarget {
	readonly name: string;
	readonly path: string;
}

/** `path` with a trailing separator, so ownership checks never match a partial segment. */
export function parentPrefix(path: string): string {
	return path.endsWith("/") || path.endsWith("\\") ? path : path + separatorOf(path);
}

/** The sidebar entry that owns `path` — the longest matching nav path. */
export function activeNavPath(path: string, targets: readonly NavTarget[]): string | null {
	let best: string | null = null;
	for (const target of targets) {
		const owns = path === target.path || path.startsWith(parentPrefix(target.path));
		if (owns && (best === null || target.path.length > best.length)) {
			best = target.path;
		}
	}
	return best;
}
