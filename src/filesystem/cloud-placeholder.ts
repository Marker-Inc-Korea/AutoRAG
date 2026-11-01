import { execFile } from "node:child_process";
import { realpathSync } from "node:fs";
import { lstat, readdir } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Darwin `SF_DATALESS` (st_flags): the file's data lives only in the cloud. `UF_DATALESS` is not a flag;
 * `0x8000` is `UF_HIDDEN`, which dotfiles carry, so it must never be read as a placeholder marker.
 */
export const SF_DATALESS = 0x4000_0000;
export const FILE_PROVIDER_XATTR = "com.apple.file-provider-domain-id";

/** Windows FILE_ATTRIBUTE_* bits carried by Cloud Files placeholders and their pin state. */
export const WINDOWS_FILE_ATTRIBUTE_PINNED = 0x0008_0000;
export const WINDOWS_FILE_ATTRIBUTE_UNPINNED = 0x0010_0000;
export const WINDOWS_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x0040_0000;

const MAC_CLOUD_PATH_MARKERS = ["/Library/CloudStorage/", "/Library/Mobile Documents/"] as const;

const WINDOWS_CLOUD_PATH_MARKERS = [
	"\\OneDrive",
	"/OneDrive",
	"\\Google Drive",
	"/Google Drive",
	"\\GoogleDriveFS",
	"/GoogleDriveFS",
	"\\My Drive",
	"/My Drive",
	"\\Shared drives",
	"/Shared drives",
] as const;

export function pathLooksLikeCloudRoot(target: string): boolean {
	const normalized = target.replaceAll("\\", "/");
	return (
		MAC_CLOUD_PATH_MARKERS.some((marker) => normalized.includes(marker)) ||
		WINDOWS_CLOUD_PATH_MARKERS.some((marker) => normalized.toLowerCase().includes(marker.toLowerCase()))
	);
}

export function homeCloudRoots(home = homedir()): readonly string[] {
	const normalizedHome = home.replaceAll("\\", "/").replace(/\/+$/, "");
	return [`${normalizedHome}/Library/CloudStorage`, `${normalizedHome}/Library/Mobile Documents`];
}

/** Whether `st_flags` (the decimal number `stat -f %f` prints) marks a data-less File Provider placeholder. */
export function datalessFromDarwinFlags(flags: number): boolean {
	return Number.isFinite(flags) && ((flags >>> 0) & SF_DATALESS) !== 0;
}

/**
 * Whether a decimal `FileAttributes` word marks a Cloud Files placeholder whose bytes live only in the cloud.
 * Measured on Windows 11 with OneDrive: an online-only placeholder carries `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`;
 * `FILE_ATTRIBUTE_OFFLINE` is NOT set on it, so an earlier check that required the offline bit missed every
 * real placeholder (#2043).
 */
export function datalessFromWindowsAttributes(attributes: number): boolean {
	return Number.isFinite(attributes) && (attributes & WINDOWS_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS) !== 0;
}

/** Which cloud backs a path. Whether AutoRAG may release a provider's files again is up to {@link CloudFileProbe.canRelease}. */
export type CloudProvider = "icloud" | "google-drive" | "onedrive" | "dropbox" | "file-provider";

export function cloudProviderFromPath(target: string): CloudProvider {
	const lower = target.toLowerCase();
	if (lower.includes("googledrive") || lower.includes("google drive") || lower.includes("drivefs"))
		return "google-drive";
	if (lower.includes("onedrive")) return "onedrive";
	if (lower.includes("dropbox")) return "dropbox";
	if (lower.includes("mobile documents") || lower.includes("clouddocs")) return "icloud";
	return "file-provider";
}

/** Provider named by a File Provider domain id such as `com.apple.CloudDocs.iCloudDriveFileProvider/<uuid>`. */
export function cloudProviderFromDomain(domain: string): CloudProvider {
	const lower = domain.toLowerCase();
	if (lower.includes("clouddocs") || lower.includes("icloud")) return "icloud";
	if (lower.includes("google")) return "google-drive";
	if (lower.includes("onedrive")) return "onedrive";
	if (lower.includes("dropbox")) return "dropbox";
	return "file-provider";
}

/** One entry of a cloud-root listing: its name and whether it is a directory. Never carries file contents. */
export interface CloudDirEntry {
	readonly name: string;
	readonly isDirectory: () => boolean;
}

function normalizeCloudPath(value: string): string {
	let normalized = value.replaceAll("\\", "/");
	if (normalized.length > 1) normalized = normalized.replace(/\/+$/u, "");
	return normalized;
}

/**
 * Whether `target` is `parent` itself or lies under it. Pure: compares normalized paths only.
 * Case-insensitive on Windows, where a drive letter and file names may differ in case.
 */
export function isPathInside(target: string, parent: string): boolean {
	const normalizedParent = normalizeCloudPath(parent);
	if (normalizedParent === "") return false;
	const left = normalizeCloudPath(target);
	const [a, b] =
		process.platform === "win32" ? [left.toLowerCase(), normalizedParent.toLowerCase()] : [left, normalizedParent];
	return a === b || a.startsWith(`${b}/`);
}

/**
 * What kind of non-physical location a machine root is:
 * - `file-provider`: a macOS File Provider directory (a `~/Library/CloudStorage` mount, iCloud Drive, or a
 *   directory carrying `com.apple.file-provider-domain-id`, such as the iCloud-synced `~/Desktop`/`~/Documents`).
 * - `cloud-files`: a Windows Cloud Files sync root (OneDrive, Dropbox, iCloud for Windows, …).
 * - `network`: a network share or virtual filesystem — an SMB/NFS/AFP/WebDAV/FUSE mount, a mapped network
 *   drive, a UNC path, or a Windows network/FUSE volume.
 */
export type CloudRootKind = "file-provider" | "cloud-files" | "network";

/** One detected non-physical root on this machine: its absolute path and what it is. */
export interface CloudRoot {
	readonly path: string;
	readonly kind: CloudRootKind;
}

/**
 * Best-effort canonical absolute path: symlinks are resolved when the path exists, and the input is returned
 * unchanged otherwise (a path that does not exist, or one whose lookup fails). macOS reports a mount point at
 * its real path — `mount_webdav … /var/…` shows up as `/private/var/…` — so both sides of a root comparison
 * are canonicalized. Resolving a path reads directory metadata only, never file contents.
 */
export function canonicalPath(value: string): string {
	try {
		return realpathSync(value);
	} catch {
		return value;
	}
}

/**
 * Rewrites a walked path into the same canonical space as {@link CloudRoot.path}: the search root's own
 * symlinks are resolved once, so a file reached through `/var/…` compares against a mount root that `mount`
 * reports at `/private/var/…`. The result is a comparison key, not a path to open: a path under a rewritten
 * search root comes back in the comparable (`/`-separated) form, and a path under no such search root is
 * returned unchanged. Pure apart from one `realpath` per search root.
 */
export function canonicalPathMapper(searchRoots: readonly string[]): (path: string) => string {
	const forms = searchRoots
		.map((root) => ({ raw: normalizeCloudPath(root), canonical: normalizeCloudPath(canonicalPath(root)) }))
		.filter(({ raw, canonical }) => raw !== "" && raw !== canonical)
		.sort((left, right) => right.raw.length - left.raw.length);
	if (forms.length === 0) return (path) => path;
	return (path) => {
		const normalized = normalizeCloudPath(path);
		for (const { raw, canonical } of forms) {
			if (isPathInside(normalized, raw)) return `${canonical}${normalized.slice(raw.length)}`;
		}
		return path;
	};
}

/** What a search root's relationship to this machine's non-physical roots is. */
export interface CloudScope {
	/** The search root is a cloud or network folder (or lies inside one): readers that open contents skip it. */
	readonly insideCloudRoot: boolean;
	/** Cloud or network folders nested under the search root: readers that must not read them exclude these. */
	readonly nestedCloudRoots: readonly CloudRoot[];
}

/**
 * Classify a search root against the machine's non-physical roots. Jikji `prepare`, the dupey exact-duplicate
 * scan, and the cloud fingerprints all use this: a root inside a cloud or network folder is skipped entirely
 * (reading a cloud placeholder downloads it; a network file is re-read over the wire), and folders nested
 * inside a root are excluded. Both sides are canonicalized, so a search root or a mount reported at its real
 * path (`/private/var/…`) still matches; paths that do not exist compare as given.
 */
export function cloudScope(searchRoot: string, cloudRoots: readonly CloudRoot[]): CloudScope {
	const canonicalSearchRoot = canonicalPath(searchRoot);
	const roots = cloudRoots.map((root) => ({ path: canonicalPath(root.path), kind: root.kind }));
	return {
		insideCloudRoot: roots.some((root) => isPathInside(canonicalSearchRoot, root.path)),
		nestedCloudRoots: roots.filter(
			(root) => isPathInside(root.path, canonicalSearchRoot) && !isPathInside(canonicalSearchRoot, root.path),
		),
	};
}

export type CloudEvictResult = { readonly ok: true } | { readonly ok: false; readonly message: string };

/**
 * What a refresh needs to know about cloud files, and the one action it may take on them. Every method
 * is read-only except {@link CloudFileProbe.evict}, which only drops the local copy of a file.
 */
export interface CloudFileProbe {
	/**
	 * Absolute roots of the non-physical folders on this machine (each File Provider mount and File
	 * Provider–managed directory on macOS; the registered Cloud Files sync roots, network/FUSE volumes, and the
	 * search roots that are UNC paths on Windows). `searchRoots` are the roots a refresh is about to walk, so a
	 * folder provider that is not under a known mount (iCloud Desktop/Documents) can be found through the
	 * `com.apple.file-provider-domain-id` xattr. Listing reads directory metadata only, never file contents.
	 * Readers that open originals — Jikji's prepare pass, the dupey exact-duplicate scan, and the cloud
	 * fingerprints — use this to skip or exclude them.
	 */
	cloudRoots(searchRoots: readonly string[]): Promise<readonly CloudRoot[]>;
	/** The subset of `paths` whose data is not on disk. Reading one of these downloads it. */
	datalessAmong(paths: readonly string[]): Promise<ReadonlySet<string>>;
	/** Provider behind a placeholder, or `undefined` when the path is not under a File Provider domain. */
	providerOf(path: string): Promise<CloudProvider | undefined>;
	/**
	 * Whether {@link CloudFileProbe.evict} is verified to release this provider's files on this platform. A refresh
	 * downloads a placeholder only when this is true; any other placeholder is skipped unread.
	 */
	canRelease(provider: CloudProvider | undefined): boolean;
	/** Release the local copy of a downloaded file; the file stays in the cloud and keeps its metadata. */
	evict(path: string): Promise<CloudEvictResult>;
}

/** Paths per `stat` invocation: keeps argv well under the OS limit for long cloud paths. */
const STAT_BATCH = 200;

export interface CloudExecOptions {
	readonly timeoutMs?: number;
	readonly maxBuffer?: number;
}

/** Runs a helper process. Injected by tests so they never need a real File Provider mount. */
export type CloudExec = (
	file: string,
	args: readonly string[],
	options?: CloudExecOptions,
) => Promise<{ readonly stdout: string; readonly stderr: string }>;

const defaultCloudExec: CloudExec = (file, args, options) =>
	execFileAsync(file, [...args], {
		timeout: options?.timeoutMs ?? 30_000,
		...(options?.maxBuffer === undefined ? {} : { maxBuffer: options.maxBuffer }),
	});

/**
 * JXA escaped into `osascript -l JavaScript`. It calls `evictUbiquitousItemAtURLError` — the one public
 * Foundation release call that works for every File Provider measured here (iCloud, Google Drive, OneDrive, Dropbox),
 * with no helper binary and no entitlement.
 *
 * The failed eviction's NSError out-param must never be read: on a refusal, dereferencing it segfaults
 * osascript (measured exit 139). The program passes a Ref and ignores it; a useful refusal reason comes from
 * the file's resource keys instead, which are safe to read. The target path is the helper's last argument.
 */
const JXA_RELEASE_PROGRAM = `ObjC.import("Foundation");
function run(argv) {
	var url = $.NSURL.fileURLWithPath(argv[0]);
	var errorRef = Ref();
	var released = $.NSFileManager.defaultManager.evictUbiquitousItemAtURLError(url, errorRef);
	if (released) return JSON.stringify({ ok: true });
	function resourceValue(key) {
		var ref = Ref();
		if (!url.getResourceValueForKeyError(ref, key, null)) return null;
		var value = ref[0];
		return value === undefined || value === null ? null : ObjC.unwrap(value);
	}
	return JSON.stringify({
		ok: false,
		isUploaded: resourceValue($.NSURLUbiquitousItemIsUploadedKey),
		isUploading: resourceValue($.NSURLUbiquitousItemIsUploadingKey),
		downloadingStatus: resourceValue($.NSURLUbiquitousItemDownloadingStatusKey)
	});
}`;

/** Providers measured to release through `evictUbiquitousItem` on macOS; the probe's release allow-list. */
const RELEASABLE_DARWIN_PROVIDERS: Readonly<Partial<Record<CloudProvider, true>>> = {
	icloud: true,
	"google-drive": true,
	onedrive: true,
	dropbox: true,
};

interface ReleaseOutcome {
	readonly ok: boolean;
	readonly isUploaded?: unknown;
	readonly isUploading?: unknown;
	readonly downloadingStatus?: unknown;
}

function parseReleaseOutcome(stdout: string): ReleaseOutcome | undefined {
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout.trim());
	} catch {
		return undefined;
	}
	if (typeof parsed !== "object" || parsed === null || !("ok" in parsed) || typeof parsed.ok !== "boolean") {
		return undefined;
	}
	return {
		ok: parsed.ok,
		...("isUploaded" in parsed ? { isUploaded: parsed.isUploaded } : {}),
		...("isUploading" in parsed ? { isUploading: parsed.isUploading } : {}),
		...("downloadingStatus" in parsed ? { downloadingStatus: parsed.downloadingStatus } : {}),
	};
}

function describeReleaseRefusal(outcome: ReleaseOutcome): string {
	if (outcome.isUploading === true) return "The file is still uploading to the cloud, so it cannot be released yet.";
	if (outcome.isUploaded === false)
		return "The file has not finished uploading to the cloud, so it cannot be released yet.";
	const status = typeof outcome.downloadingStatus === "string" ? outcome.downloadingStatus : "unknown";
	return `The cloud provider declined to release the local copy (download status ${status}).`;
}

function describeReleaseHelperFailure(error: unknown): string {
	// Node's execFile rejection carries signal/killed/code/stderr off-type; read only the fields formatted here.
	const failure = error as { signal?: unknown; killed?: unknown; code?: unknown; stderr?: unknown };
	const parts: string[] = [];
	if (failure.killed === true || typeof failure.signal === "string") {
		parts.push(`the helper was killed${typeof failure.signal === "string" ? ` by ${failure.signal}` : ""}`);
	} else if (typeof failure.code === "number") {
		parts.push(`the helper exited with code ${failure.code}`);
	}
	const stderr = typeof failure.stderr === "string" ? failure.stderr.trim() : "";
	if (stderr !== "") parts.push(stderr);
	if (parts.length === 0) parts.push(error instanceof Error ? error.message : String(error));
	return `osascript could not release the local copy: ${parts.join("; ")}`;
}

async function defaultDarwinReadDir(path: string): Promise<readonly CloudDirEntry[]> {
	const entries = await readdir(path, { withFileTypes: true });
	return entries.map((entry) => ({ name: entry.name, isDirectory: () => entry.isDirectory() }));
}

/** Paths per `xattr` invocation: keeps argv well under the OS limit while batching whole trees. */
export const XATTR_BATCH = 200;

/** Filesystem types that are never storage, even though they also lack the `local` mount option. */
const DARWIN_NON_STORAGE_FILE_SYSTEMS: Readonly<Record<string, true>> = { autofs: true };

/** The mount option every local volume carries (`apfs`, `hfs`, and `devfs` all do; network and FUSE mounts do not). */
const DARWIN_LOCAL_MOUNT_OPTION = "local";

/** One line of `/sbin/mount`: `<device> on <mountpoint> (<fstype>, <option>, …)`. */
export interface DarwinMount {
	readonly device: string;
	readonly mountPoint: string;
	readonly fileSystem: string;
	readonly options: readonly string[];
}

/**
 * Parses `/sbin/mount` output. The mount point is everything between the first ` on ` and the final ` (`, so a
 * path or device name containing spaces survives; the parenthesised tail is `<fstype>, <option>, …`.
 */
export function parseDarwinMounts(stdout: string): DarwinMount[] {
	const mounts: DarwinMount[] = [];
	for (const line of stdout.split("\n")) {
		if (line.trim() === "") continue;
		const open = line.lastIndexOf(" (");
		if (open <= 0 || !line.endsWith(")")) continue;
		const head = line.slice(0, open);
		const on = head.indexOf(" on ");
		if (on <= 0) continue;
		const mountPoint = head.slice(on + " on ".length);
		if (mountPoint === "") continue;
		const fields = line
			.slice(open + 2, -1)
			.split(",")
			.map((field) => field.trim())
			.filter((field) => field !== "");
		mounts.push({
			device: head.slice(0, on),
			mountPoint,
			fileSystem: fields[0] ?? "",
			options: fields.slice(1),
		});
	}
	return mounts;
}

/**
 * Mount points of the volumes that are not local storage: a share (SMB/NFS/AFP/WebDAV), a FUSE mount, or any
 * other filesystem that does not carry the `local` option. `autofs` (the `map auto_home` automounter) also
 * lacks `local` but is not a share, so it is skipped. Mount points are canonicalized by the caller.
 */
export function networkMountPoints(mounts: readonly DarwinMount[]): readonly string[] {
	const points: string[] = [];
	const seen = new Set<string>();
	for (const mount of mounts) {
		if (DARWIN_NON_STORAGE_FILE_SYSTEMS[mount.fileSystem] === true) continue;
		if (mount.options.includes(DARWIN_LOCAL_MOUNT_OPTION)) continue;
		if (seen.has(mount.mountPoint)) continue;
		seen.add(mount.mountPoint);
		points.push(mount.mountPoint);
	}
	return points;
}

/**
 * Parses `xattr -v -p <attr> <path>…` stdout. Each path that carries the attribute is printed as
 * `<path as given>: <value>`; paths without it print nothing to stdout (the `No such xattr` note goes to
 * stderr and the process exits nonzero). A line is matched to its candidate by prefix, longest candidate
 * first, so a path that contains `: ` is not mis-split.
 */
export function parseXattrValues(stdout: string, candidates: readonly string[]): ReadonlyMap<string, string> {
	const values = new Map<string, string>();
	if (candidates.length === 0) return values;
	const ordered = [...candidates].sort((left, right) => right.length - left.length);
	for (const line of stdout.split("\n")) {
		if (line === "") continue;
		for (const candidate of ordered) {
			if (line.startsWith(`${candidate}: `)) {
				values.set(candidate, line.slice(candidate.length + 2));
				break;
			}
		}
	}
	return values;
}

export interface DarwinCloudProbeOptions {
	/** Process runner for `stat`, the release helper, `/sbin/mount`, and `xattr`. Injected by tests. */
	readonly exec?: CloudExec;
	/** Directory listing for {@link DarwinCloudProbe.cloudRoots}; injected by tests so they need no real mounts. */
	readonly readDir?: (path: string) => Promise<readonly CloudDirEntry[]>;
	/** Home directory holding `Library/CloudStorage` and `Library/Mobile Documents`; defaults to the user's home. */
	readonly home?: string;
}

/**
 * The macOS probe. `stat -f %f` reads `st_flags`, which Node's `fs.stat` does not expose; a placeholder is
 * the file whose `SF_DATALESS` bit is set. Released with {@link DarwinCloudProbe.evict}: a Foundation
 * `evictUbiquitousItem` call through `osascript -l JavaScript`, which drops the local copy of iCloud,
 * Google Drive, OneDrive, and Dropbox placeholders alike (measured; `brctl evict` works only for iCloud, so
 * not used).
 */
export class DarwinCloudProbe implements CloudFileProbe {
	private readonly exec: CloudExec;
	private readonly readDir: (path: string) => Promise<readonly CloudDirEntry[]>;
	private readonly home: string;

	constructor(options: DarwinCloudProbeOptions = {}) {
		this.exec = options.exec ?? defaultCloudExec;
		this.readDir = options.readDir ?? defaultDarwinReadDir;
		this.home = options.home ?? homedir();
	}

	/**
	 * The machine's non-physical roots, without ever reading file contents:
	 * - each File Provider mount is its own directory directly under `~/Library/CloudStorage` (for example
	 *   `GoogleDrive-user`, `OneDrive-Work`, `Dropbox`), and iCloud Drive's own root is `~/Library/Mobile
	 *   Documents`;
	 * - every mount from `/sbin/mount` that lacks the `local` option is a network share, WebDAV, or FUSE
	 *   mount (`autofs` is not), reported by `mount` at its real path;
	 * - a directory managed by a File Provider that is not under a known mount — the iCloud-synced `~/Desktop`
	 *   and `~/Documents` carry `com.apple.file-provider-domain-id` — is found on each search root and its
	 *   ancestors (`inside`) and on each direct child directory (`nested`), batched through `xattr`.
	 *
	 * A helper that fails (no `mount`, no `xattr`, a hung share) contributes no roots rather than failing the
	 * refresh: the roots found by the other sources stand.
	 */
	async cloudRoots(searchRoots: readonly string[] = []): Promise<readonly CloudRoot[]> {
		const roots: CloudRoot[] = [];
		const seen = new Set<string>();
		const add = (path: string, kind: CloudRootKind): void => {
			const canonical = canonicalPath(path);
			const key = normalizeCloudPath(canonical);
			if (key === "" || seen.has(key)) return;
			seen.add(key);
			roots.push({ path: canonical, kind });
		};
		const [cloudStorage, mobileDocuments] = homeCloudRoots(this.home);
		for (const entry of (await this.tryReadDir(cloudStorage)) ?? []) {
			if (entry.isDirectory()) add(join(cloudStorage, entry.name), "file-provider");
		}
		if ((await this.tryReadDir(mobileDocuments)) !== undefined) add(mobileDocuments, "file-provider");
		for (const mountPoint of await this.networkMountPoints()) add(mountPoint, "network");
		for (const directory of await this.fileProviderDirectories(searchRoots)) add(directory, "file-provider");
		return roots;
	}

	/** Mount points of the volumes `/sbin/mount` reports without the `local` option; `[]` when `mount` fails. */
	private async networkMountPoints(): Promise<readonly string[]> {
		try {
			const { stdout } = await this.exec("/sbin/mount", [], { timeoutMs: 10_000, maxBuffer: 4 * 1024 * 1024 });
			return networkMountPoints(parseDarwinMounts(stdout));
		} catch {
			return [];
		}
	}

	/**
	 * Directories carrying `com.apple.file-provider-domain-id` among each search root's ancestors and direct
	 * child directories. One directory listing per search root and as few `xattr` processes as the path count
	 * allows; a failure of either contributes no directories.
	 */
	private async fileProviderDirectories(searchRoots: readonly string[]): Promise<readonly string[]> {
		if (searchRoots.length === 0) return [];
		const candidates = new Set<string>();
		for (const searchRoot of searchRoots) {
			const canonical = canonicalPath(searchRoot);
			for (let cursor = canonical; ; cursor = dirname(cursor)) {
				candidates.add(cursor);
				const parent = dirname(cursor);
				if (parent === cursor) break;
			}
			for (const entry of (await this.tryReadDir(canonical)) ?? []) {
				if (entry.isDirectory()) candidates.add(join(canonical, entry.name));
			}
		}
		const paths = [...candidates];
		const managed: string[] = [];
		for (let start = 0; start < paths.length; start += XATTR_BATCH) {
			for (const path of (await this.xattrValues(paths.slice(start, start + XATTR_BATCH))).keys())
				managed.push(path);
		}
		return managed;
	}

	/**
	 * The paths carrying {@link FILE_PROVIDER_XATTR}. `xattr` exits nonzero as soon as one path lacks the
	 * attribute, so the values it printed before that are read off the rejected helper's stdout; a helper that
	 * produced nothing usable contributes no paths.
	 */
	private async xattrValues(paths: readonly string[]): Promise<ReadonlyMap<string, string>> {
		try {
			const { stdout } = await this.exec("/usr/bin/xattr", ["-v", "-p", FILE_PROVIDER_XATTR, ...paths], {
				timeoutMs: 15_000,
				maxBuffer: 8 * 1024 * 1024,
			});
			return parseXattrValues(stdout, paths);
		} catch (error) {
			// Node's execFile rejection carries the partial stdout off-type; only the one field read here is touched.
			const failure = error as { stdout?: unknown };
			return typeof failure.stdout === "string" ? parseXattrValues(failure.stdout, paths) : new Map();
		}
	}

	private async tryReadDir(path: string): Promise<readonly CloudDirEntry[] | undefined> {
		try {
			return await this.readDir(path);
		} catch {
			return undefined;
		}
	}

	async datalessAmong(paths: readonly string[]): Promise<ReadonlySet<string>> {
		const dataless = new Set<string>();
		for (let start = 0; start < paths.length; start += STAT_BATCH) {
			const batch = paths.slice(start, start + STAT_BATCH);
			const { stdout } = await this.exec("/usr/bin/stat", ["-f", "%f", ...batch], {
				timeoutMs: 30_000,
				maxBuffer: 4 * 1024 * 1024,
			});
			const flags = stdout.split("\n").filter(Boolean);
			for (const [index, path] of batch.entries()) {
				if (datalessFromDarwinFlags(Number.parseInt(flags[index] ?? "", 10))) dataless.add(path);
			}
		}
		return dataless;
	}

	async providerOf(path: string): Promise<CloudProvider | undefined> {
		const domain = await readFileProviderDomainUpwards(path);
		return domain === undefined ? undefined : cloudProviderFromDomain(domain);
	}

	canRelease(provider: CloudProvider | undefined): boolean {
		return provider !== undefined && RELEASABLE_DARWIN_PROVIDERS[provider] === true;
	}

	async evict(path: string): Promise<CloudEvictResult> {
		let stdout: string;
		try {
			stdout = (
				await this.exec("/usr/bin/osascript", ["-l", "JavaScript", "-e", JXA_RELEASE_PROGRAM, path], {
					timeoutMs: 30_000,
				})
			).stdout;
		} catch (error) {
			return { ok: false, message: describeReleaseHelperFailure(error) };
		}
		const outcome = parseReleaseOutcome(stdout);
		if (outcome === undefined) {
			const detail = stdout.trim();
			return {
				ok: false,
				message: `osascript returned no release result${detail === "" ? "" : `: ${detail}`}`,
			};
		}
		return outcome.ok ? { ok: true } : { ok: false, message: describeReleaseRefusal(outcome) };
	}
}

/** Paths per PowerShell process: long cloud paths still keep the command line under the Windows limit. */
export const WINDOWS_ATTRIBUTE_BATCH = 100;

export interface PowerShellResult {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
}

/** Runs one PowerShell process. The default spawns `powershell.exe -NoProfile -NonInteractive -Command <script>`. */
export type PowerShellRunner = (script: string, timeoutMs: number) => Promise<PowerShellResult>;

/** One local sync root registered by a Cloud Files provider, with the name Explorer shows for it. */
export interface WindowsSyncRoot {
	readonly path: string;
	readonly label: string;
}

/**
 * Reads `FileAttributes` for a batch of paths in one process: one decimal attribute word per line, in batch
 * order, and `ERR` for a path that could not be read. The list is passed as a PowerShell array literal.
 */
export function buildWindowsAttributeScript(paths: readonly string[]): string {
	return (
		"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8\n" +
		"$ErrorActionPreference='SilentlyContinue'\n" +
		`foreach($p in @(${paths.map(quotePowerShell).join(",")})){` +
		"$i=Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue;" +
		"if($null -eq $i){'ERR'}else{$i.Attributes.value__}}\n"
	);
}

/** Dumps every Cloud Files sync root and its Explorer display name, one `path<TAB>label` line each. */
export function buildWindowsSyncRootsScript(): string {
	return (
		"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8\n" +
		"$k='HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\SyncRootManager'\n" +
		"if(Test-Path $k){Get-ChildItem $k|ForEach-Object{" +
		"$u=Join-Path $_.PSPath 'UserSyncRoots';" +
		"if(Test-Path $u){$p=Get-ItemProperty -LiteralPath $u -ErrorAction SilentlyContinue;" +
		"$p.PSObject.Properties|Where-Object{$_.Name -notlike 'PS*'}|ForEach-Object{" +
		'"$($_.Name)$([char]9)$($_.Value)"}}}}\n'
	);
}

/**
 * Opens one file with a Cloud Files oplock and dehydrates it synchronously with `CfDehydratePlaceholder`.
 * Prints `OPEN=` and `DEHYDRATE=` as hex HRESULTs and, on failure, `MSG=` with the OS message. Dehydration
 * is provider-agnostic and does not change the file's pin state; it refuses a file the provider has not
 * finished uploading, which is exactly the refusal a refresh wants to surface.
 */
export function buildWindowsDehydrateScript(path: string): string {
	return (
		"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8\n" +
		"$ErrorActionPreference='Stop'\n" +
		"Add-Type -Namespace Autorag -Name CfApi -MemberDefinition @'\n" +
		'[DllImport("cldapi.dll", CharSet=CharSet.Unicode, SetLastError=false)] public static extern int CfOpenFileWithOplock(string FilePath, uint Flags, out System.IntPtr ProtectedHandle);\n' +
		'[DllImport("cldapi.dll", CharSet=CharSet.Unicode, SetLastError=false)] public static extern int CfDehydratePlaceholder(System.IntPtr FileHandle, long StartingOffset, long Length, uint DehydrateFlags, System.IntPtr Overlapped);\n' +
		'[DllImport("cldapi.dll", SetLastError=false)] public static extern void CfCloseHandle(System.IntPtr FileHandle);\n' +
		"'@\n" +
		"$h=[IntPtr]::Zero\n" +
		`$open=[Autorag.CfApi]::CfOpenFileWithOplock(${quotePowerShell(path)},0x1,[ref]$h)\n` +
		"'OPEN={0:X8}' -f ($open -band 0xFFFFFFFF)\n" +
		"if($open -ne 0){if($h -ne [IntPtr]::Zero){[Autorag.CfApi]::CfCloseHandle($h)};" +
		'$m=([System.Runtime.InteropServices.Marshal]::GetExceptionForHR($open)).Message;"MSG=$m";exit 0}\n' +
		"$dry=[Autorag.CfApi]::CfDehydratePlaceholder($h,0,-1,0,[IntPtr]::Zero)\n" +
		"[Autorag.CfApi]::CfCloseHandle($h)\n" +
		"'DEHYDRATE={0:X8}' -f ($dry -band 0xFFFFFFFF)\n" +
		'if($dry -ne 0){$m=([System.Runtime.InteropServices.Marshal]::GetExceptionForHR($dry)).Message;"MSG=$m"}\n'
	);
}

/** Maps one attribute word per output line back to the batch paths, in order; unparseable lines are dropped. */
export function parseWindowsAttributeOutput(stdout: string, paths: readonly string[]): Map<string, number> {
	const lines = stdout.split(/\r?\n/);
	const attributes = new Map<string, number>();
	for (const [index, path] of paths.entries()) {
		const value = Number.parseInt((lines[index] ?? "").trim(), 10);
		if (Number.isFinite(value)) attributes.set(path, value);
	}
	return attributes;
}

/** Parses the `path<TAB>label` lines of {@link buildWindowsSyncRootsScript} into sync roots. */
export function parseWindowsSyncRoots(stdout: string): WindowsSyncRoot[] {
	const roots: WindowsSyncRoot[] = [];
	for (const line of stdout.split(/\r?\n/)) {
		const tab = line.indexOf("\t");
		if (tab <= 0) continue;
		const path = line.slice(0, tab).trim();
		if (path !== "") roots.push({ path, label: line.slice(tab + 1).trim() });
	}
	return roots;
}

/** Provider a sync root's Explorer display name belongs to, from the names the providers register. */
export function providerFromWindowsSyncRoot(label: string): CloudProvider {
	const lower = label.toLowerCase();
	if (lower.includes("onedrive") || lower.includes("one drive")) return "onedrive";
	if (lower.includes("google")) return "google-drive";
	return "file-provider";
}

/** `Win32_LogicalDisk.DriveType` for a mapped network drive. */
export const WINDOWS_DRIVE_TYPE_NETWORK = 4;

/** One logical disk as `Win32_LogicalDisk` reports it. */
export interface WindowsLogicalDisk {
	/** The drive letter, for example `C:`. */
	readonly deviceId: string;
	readonly driveType: number;
	readonly fileSystem: string;
	readonly volumeName: string;
}

/** Dumps every logical disk, one `deviceId<TAB>driveType<TAB>fileSystem<TAB>volumeName` line each. */
export function buildWindowsDiskScript(): string {
	return (
		"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8\n" +
		"Get-CimInstance -ClassName Win32_LogicalDisk | ForEach-Object {" +
		"$_.DeviceID + [char]9 + $_.DriveType + [char]9 + $_.FileSystem + [char]9 + $_.VolumeName}\n"
	);
}

/** Parses the tab-separated lines of {@link buildWindowsDiskScript}. */
export function parseWindowsLogicalDisks(stdout: string): WindowsLogicalDisk[] {
	const disks: WindowsLogicalDisk[] = [];
	for (const line of stdout.split(/\r?\n/)) {
		if (line.trim() === "") continue;
		const [deviceId, driveType, fileSystem, ...rest] = line.split("\t");
		const type = Number.parseInt((driveType ?? "").trim(), 10);
		if (!Number.isFinite(type)) continue;
		disks.push({
			deviceId: (deviceId ?? "").trim(),
			driveType: type,
			fileSystem: (fileSystem ?? "").trim(),
			volumeName: rest.join("\t").trim(),
		});
	}
	return disks;
}

/** Filesystem names a FUSE-compatible mount (WinFsp, rclone's `FUSE`, Dokan) reports on Windows. */
const WINDOWS_FUSE_FILE_SYSTEM_PREFIXES = ["fuse", "winfsp", "dokan"] as const;

/**
 * The non-physical roots among a disk list:
 * - `DriveType 4` is a mapped network drive (`X:\`), always `network`;
 * - a FUSE-compatible volume (WinFsp, Dokan, rclone) is a virtual filesystem, treated as `network`;
 * - Google Drive for desktop mounts a drive letter the Cloud Files registry does not list, recognized by its
 *   `GoogleDriveFS` filesystem or `Google Drive` volume name — **unverified**: no Windows host with Google
 *   Drive for desktop installed has been measured.
 */
export function windowsRootsFromDisks(disks: readonly WindowsLogicalDisk[]): readonly CloudRoot[] {
	const roots: CloudRoot[] = [];
	for (const disk of disks) {
		if (disk.deviceId === "") continue;
		const path = `${disk.deviceId}\\`;
		const fileSystem = disk.fileSystem.toLowerCase();
		if (disk.driveType === WINDOWS_DRIVE_TYPE_NETWORK) {
			roots.push({ path, kind: "network" });
			continue;
		}
		if (WINDOWS_FUSE_FILE_SYSTEM_PREFIXES.some((prefix) => fileSystem.startsWith(prefix))) {
			roots.push({ path, kind: "network" });
			continue;
		}
		if (fileSystem.startsWith("googledrivefs") || disk.volumeName.toLowerCase().includes("google drive")) {
			roots.push({ path, kind: "cloud-files" });
		}
	}
	return roots;
}

/**
 * The share root of a UNC path — `\\server\share` for `\\server\share\folder`. Returns `undefined` for a local
 * path and for a device path (`\\?\…`, `\\.\…`), which is not a share.
 */
export function uncShareRoot(target: string): string | undefined {
	const normalized = target.replaceAll("/", "\\");
	if (!normalized.startsWith("\\\\") || normalized.startsWith("\\\\?\\") || normalized.startsWith("\\\\.\\"))
		return undefined;
	const rest = normalized.slice(2);
	const separator = rest.indexOf("\\");
	if (separator <= 0) return undefined;
	const server = rest.slice(0, separator);
	const share = rest.slice(separator + 1).split("\\")[0] ?? "";
	if (server === "" || share === "") return undefined;
	return `\\\\${server}\\${share}`;
}

/**
 * Maps the `OPEN`/`DEHYDRATE`/`MSG` lines of {@link buildWindowsDehydrateScript} to an eviction result.
 * `HRESULT 0` is success; anything else (including a process that produced no lines) is a failure whose
 * message is the OS text when the script printed one, otherwise the process stderr.
 */
export function parseWindowsDehydrateOutput(result: PowerShellResult): CloudEvictResult {
	const open = /^OPEN=(?:0x)?([0-9A-Fa-f]{8})$/mu.exec(result.stdout);
	if (open === null)
		return { ok: false, message: windowsFailureMessage(result, "the release command produced no result") };
	if (Number.parseInt(open[1] as string, 16) !== 0) {
		return {
			ok: false,
			message: windowsFailureMessage(result, `could not open the file for release (0x${open[1]})`),
		};
	}
	const dehydrate = /^DEHYDRATE=(?:0x)?([0-9A-Fa-f]{8})$/mu.exec(result.stdout);
	if (dehydrate === null) {
		return { ok: false, message: windowsFailureMessage(result, "the release command produced no result") };
	}
	const hresult = dehydrate[1] as string;
	if (Number.parseInt(hresult, 16) === 0) return { ok: true };
	const refusal = WINDOWS_DEHYDRATE_REFUSALS[hresult.toUpperCase()];
	return {
		ok: false,
		message: refusal ?? windowsFailureMessage(result, `the provider refused to dehydrate the file (0x${hresult})`),
	};
}

/** `CfDehydratePlaceholder` refusals the platform enforces, measured against a Cloud Files sync root. */
const WINDOWS_DEHYDRATE_REFUSALS: Readonly<Record<string, string>> = {
	"80070179": "The file has changes the cloud provider has not synced yet, so it cannot be released.",
	"80070188": "The file is pinned to stay on this device, so it is not released.",
};

function windowsFailureMessage(result: PowerShellResult, fallback: string): string {
	const printed = /^MSG=(.*)$/mu.exec(result.stdout)?.[1]?.trim();
	if (printed !== undefined && printed !== "") return printed;
	const stderr = result.stderr.trim();
	return stderr !== "" ? stderr : fallback;
}

export interface WindowsCloudProbeOptions {
	/** Runs PowerShell; tests inject one that returns canned output instead of spawning a process. */
	readonly run?: PowerShellRunner;
}

/**
 * The Windows probe. A Cloud Files placeholder carries `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`; the provider is the
 * registered sync root that contains the path (else the path's own marker); the local copy is released with
 * `CfDehydratePlaceholder`. Release is allowed for every Cloud Files sync root because the platform itself refuses
 * to dehydrate a file with unsynced changes or a pinned file (measured: 0x80070179 and 0x80070188). Verified against
 * a synthetic cfapi provider only; OneDrive, Dropbox and other commercial providers were not measured.
 */
export class WindowsCloudProbe implements CloudFileProbe {
	private readonly run: PowerShellRunner;
	private syncRoots: Promise<readonly WindowsSyncRoot[]> | undefined;
	private disks: Promise<readonly CloudRoot[]> | undefined;

	constructor(options: WindowsCloudProbeOptions = {}) {
		this.run = options.run ?? defaultPowerShellRunner;
	}

	/**
	 * The machine's non-physical roots, none of which reads a file's contents: the registered Cloud Files sync
	 * roots (registry), the mapped network and FUSE-compatible volume letters (`Win32_LogicalDisk`, one
	 * PowerShell process), and the share root of any search root that is a UNC path (`\\server\share\…`).
	 */
	async cloudRoots(searchRoots: readonly string[] = []): Promise<readonly CloudRoot[]> {
		const roots: CloudRoot[] = [];
		const seen = new Set<string>();
		const add = (path: string, kind: CloudRootKind): void => {
			const canonical = canonicalPath(path);
			const key = normalizeCloudPath(canonical).toLowerCase();
			if (key === "" || seen.has(key)) return;
			seen.add(key);
			roots.push({ path: canonical, kind });
		};
		for (const root of await this.loadSyncRoots()) add(root.path, "cloud-files");
		for (const root of await this.loadDiskRoots()) add(root.path, root.kind);
		for (const searchRoot of searchRoots) {
			const share = uncShareRoot(searchRoot);
			if (share !== undefined) add(share, "network");
		}
		return roots;
	}

	async datalessAmong(paths: readonly string[]): Promise<ReadonlySet<string>> {
		const dataless = new Set<string>();
		for (let start = 0; start < paths.length; start += WINDOWS_ATTRIBUTE_BATCH) {
			const batch = paths.slice(start, start + WINDOWS_ATTRIBUTE_BATCH);
			const result = await this.run(buildWindowsAttributeScript(batch), 30_000);
			if (!result.ok) {
				throw new Error(
					`Could not read Windows cloud file attributes: ${result.stderr.trim() || "powershell.exe failed"}`,
				);
			}
			for (const [path, attributes] of parseWindowsAttributeOutput(result.stdout, batch)) {
				if (datalessFromWindowsAttributes(attributes)) dataless.add(path);
			}
		}
		return dataless;
	}

	async providerOf(path: string): Promise<CloudProvider | undefined> {
		const root = longestSyncRoot(await this.loadSyncRoots(), path);
		return root === undefined ? cloudProviderFromPath(path) : providerFromWindowsSyncRoot(root.label);
	}

	canRelease(provider: CloudProvider | undefined): boolean {
		return provider !== undefined;
	}

	async evict(path: string): Promise<CloudEvictResult> {
		return parseWindowsDehydrateOutput(await this.run(buildWindowsDehydrateScript(path), 60_000));
	}

	/** The sync roots are read once per probe; a failure falls back to the path markers, never aborts a refresh. */
	private loadSyncRoots(): Promise<readonly WindowsSyncRoot[]> {
		this.syncRoots ??= this.run(buildWindowsSyncRootsScript(), 15_000).then(
			(result) => (result.ok ? parseWindowsSyncRoots(result.stdout) : []),
			() => [],
		);
		return this.syncRoots;
	}

	/** The disk list is read once per probe in one PowerShell process; a failure contributes no roots. */
	private loadDiskRoots(): Promise<readonly CloudRoot[]> {
		this.disks ??= this.run(buildWindowsDiskScript(), 30_000).then(
			(result) => (result.ok ? windowsRootsFromDisks(parseWindowsLogicalDisks(result.stdout)) : []),
			() => [],
		);
		return this.disks;
	}
}

/** Longest registered sync root containing `path`, so a nested root wins over its parent. */
function longestSyncRoot(roots: readonly WindowsSyncRoot[], target: string): WindowsSyncRoot | undefined {
	const normalized = target.replaceAll("/", "\\").toLowerCase();
	let match: WindowsSyncRoot | undefined;
	let matchLength = -1;
	for (const root of roots) {
		const prefix = root.path.replaceAll("/", "\\").replace(/\\+$/u, "").toLowerCase();
		if (prefix === "" || prefix.length <= matchLength) continue;
		if (normalized === prefix || normalized.startsWith(`${prefix}\\`)) {
			match = root;
			matchLength = prefix.length;
		}
	}
	return match;
}

async function defaultPowerShellRunner(script: string, timeoutMs: number): Promise<PowerShellResult> {
	try {
		const { stdout, stderr } = await execFileAsync(
			"powershell.exe",
			["-NoProfile", "-NonInteractive", "-Command", script],
			{ timeout: timeoutMs, maxBuffer: 16 * 1024 * 1024, windowsHide: true },
		);
		return { ok: true, stdout, stderr };
	} catch (error) {
		const record = typeof error === "object" && error !== null ? (error as Record<string, unknown>) : {};
		return {
			ok: false,
			stdout: typeof record.stdout === "string" ? record.stdout : "",
			stderr: typeof record.stderr === "string" ? record.stderr : String(error),
		};
	}
}

/** The probe for this platform, or `undefined` where AutoRAG has no cloud-aware handling yet. */
export function defaultCloudProbe(): CloudFileProbe | undefined {
	if (process.platform === "darwin") return new DarwinCloudProbe();
	if (process.platform === "win32") return new WindowsCloudProbe();
	return undefined;
}

async function readFileProviderDomain(target: string): Promise<string | undefined> {
	try {
		await lstat(target);
		const { stdout } = await execFileAsync("/usr/bin/xattr", ["-p", FILE_PROVIDER_XATTR, target], {
			timeout: 2_000,
		});
		const value = stdout.trim();
		return value.length > 0 ? value : undefined;
	} catch {
		return undefined;
	}
}

async function readFileProviderDomainUpwards(target: string): Promise<string | undefined> {
	let cursor = target;
	for (let depth = 0; depth < 12; depth += 1) {
		const domain = await readFileProviderDomain(cursor);
		if (domain !== undefined) return domain;
		const parent = dirname(cursor);
		if (parent === cursor) return undefined;
		cursor = parent;
	}
	return undefined;
}

function quotePowerShell(value: string): string {
	return `'${value.replaceAll("'", "''")}'`;
}
