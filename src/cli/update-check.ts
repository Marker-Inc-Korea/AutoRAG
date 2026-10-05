/**
 * AutoRAG's own update check. Pi's interactive host reports its own version and
 * changelog; this module answers "is a newer `@autorag/librarian` published?"
 * so the CLI and TUI can notify on AutoRAG's terms.
 *
 * The check is best-effort: any network/parse failure yields `status: "error"`
 * and never throws, so a launch is never blocked or failed by the lookup.
 */

export const AUTORAG_PACKAGE_NAME = "@autorag/librarian";
export const AUTORAG_UPDATE_COMMAND = `bun install -g ${AUTORAG_PACKAGE_NAME}`;
export const AUTORAG_LATEST_VERSION_URL = `https://registry.npmjs.org/${encodeURIComponent(AUTORAG_PACKAGE_NAME)}/latest`;
export const AUTORAG_UPDATE_CHECK_TIMEOUT_MS = 2500;

const SEMVER_PATTERN = /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/;

export type AutoRAGUpdateStatus = "available" | "up-to-date" | "skipped" | "error";

export interface AutoRAGUpdateResult {
	readonly status: AutoRAGUpdateStatus;
	readonly packageName: string;
	readonly currentVersion: string;
	readonly latestVersion?: string;
	readonly installCommand: string;
}

export interface CheckAutoRAGUpdateOptions {
	/** The running package version to compare against the registry. */
	readonly currentVersion: string;
	/** Registry endpoint override; primarily a test/QA seam. */
	readonly url?: string;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly env?: Readonly<Record<string, string | undefined>>;
}

interface ParsedVersion {
	major: number;
	minor: number;
	patch: number;
	prerelease: string | undefined;
}

function parseVersion(value: string): ParsedVersion | undefined {
	const match = SEMVER_PATTERN.exec(value.trim());
	if (match === null) return undefined;
	return {
		major: Number(match[1]),
		minor: Number(match[2]),
		patch: Number(match[3]),
		prerelease: match[4],
	};
}

/** Semver precedence by numeric core, then release-over-prerelease. `undefined` when either side is unparsable. */
export function comparePackageVersions(left: string, right: string): number | undefined {
	const a = parseVersion(left);
	const b = parseVersion(right);
	if (a === undefined || b === undefined) return undefined;
	if (a.major !== b.major) return a.major - b.major;
	if (a.minor !== b.minor) return a.minor - b.minor;
	if (a.patch !== b.patch) return a.patch - b.patch;
	if (a.prerelease === b.prerelease) return 0;
	if (a.prerelease === undefined) return 1;
	if (b.prerelease === undefined) return -1;
	// Two distinct prereleases: treat as equal so a pre-release train never
	// triggers a spurious "update available" notice.
	return 0;
}

export async function checkAutoRAGUpdate(options: CheckAutoRAGUpdateOptions): Promise<AutoRAGUpdateResult> {
	const env = options.env ?? process.env;
	const base = {
		packageName: AUTORAG_PACKAGE_NAME,
		currentVersion: options.currentVersion,
		installCommand: AUTORAG_UPDATE_COMMAND,
	} as const;
	if (
		env.AUTORAG_NO_UPDATE_CHECK !== undefined &&
		env.AUTORAG_NO_UPDATE_CHECK !== "" &&
		env.AUTORAG_NO_UPDATE_CHECK !== "0"
	) {
		return { ...base, status: "skipped" };
	}
	const url = options.url ?? env.AUTORAG_UPDATE_CHECK_URL ?? AUTORAG_LATEST_VERSION_URL;
	const fetchImpl = options.fetchImpl ?? fetch;
	try {
		const response = await fetchImpl(url, {
			headers: { accept: "application/json" },
			signal: AbortSignal.timeout(options.timeoutMs ?? AUTORAG_UPDATE_CHECK_TIMEOUT_MS),
		});
		if (!response.ok) return { ...base, status: "error" };
		const payload = (await response.json()) as { version?: unknown };
		if (typeof payload.version !== "string" || payload.version.length === 0) return { ...base, status: "error" };
		const latestVersion = payload.version;
		const comparison = comparePackageVersions(latestVersion, options.currentVersion);
		if (comparison === undefined) return { ...base, status: "error" };
		return { ...base, status: comparison > 0 ? "available" : "up-to-date", latestVersion };
	} catch {
		return { ...base, status: "error" };
	}
}

/** One-line TUI/CLI notice, or `undefined` when there is nothing to announce. */
export function renderAutoRAGUpdateNotice(result: AutoRAGUpdateResult): string | undefined {
	if (result.status !== "available" || result.latestVersion === undefined) return undefined;
	return `AutoRAG v${result.latestVersion} is available (you have v${result.currentVersion}). Update with \`${result.installCommand}\`.`;
}
