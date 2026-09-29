export const APP_NAME = "AutoRAG Finder";
export const APP_VERSION = "0.1.0";

/**
 * Which clone (and revision) a running window belongs to. Multiple clones of
 * this repository are normal, so the label tells windows apart during
 * development: absolute clone path plus git branch and short commit.
 */
export interface DevLabel {
	readonly clonePath: string;
	readonly branch: string | null;
	readonly commit: string | null;
}

/** `feat/x@612ae5f`, or just the path when git metadata is unavailable. */
export function formatDevLabel(label: DevLabel): string {
	if (label.branch === null && label.commit === null) return label.clonePath;
	return `${label.clonePath} (${label.branch ?? "detached"}@${label.commit ?? "unknown"})`;
}

export function formatWindowTitle(appName: string, version: string, devLabel?: DevLabel | null): string {
	const normalizedVersion = version.startsWith("v") ? version.slice(1) : version;
	const base = `${appName} v${normalizedVersion}`;
	return devLabel === undefined || devLabel === null ? base : `${base} — ${formatDevLabel(devLabel)}`;
}
