import { execFile } from "node:child_process";
import { lstat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, relative, sep } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/**
 * Darwin `SF_DATALESS` (st_flags): the file's data lives only in the cloud. `UF_DATALESS` is not a flag;
 * `0x8000` is `UF_HIDDEN`, which dotfiles carry, so it must never be read as a placeholder marker.
 */
export const SF_DATALESS = 0x4000_0000;
export const FILE_PROVIDER_XATTR = "com.apple.file-provider-domain-id";

/** Windows FILE_ATTRIBUTE_* bits commonly used by Cloud Files placeholders. */
export const WINDOWS_FILE_ATTRIBUTE_OFFLINE = 0x0000_1000;
export const WINDOWS_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x0004_0000;
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

export type FilesystemKind = "local" | "file-provider";

export interface FilesystemClassification {
	readonly kind: FilesystemKind;
	readonly provider?: string;
	readonly reason: "path-marker" | "file-provider-xattr" | "cloud-volume" | "placeholder-attribute" | "local";
}

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

export async function classifyFilesystemRoot(target: string): Promise<FilesystemClassification> {
	if (pathLooksLikeCloudRoot(target)) {
		return { kind: "file-provider", reason: "path-marker", provider: cloudProviderFromPath(target) };
	}
	if (process.platform === "darwin") return classifyDarwinRoot(target);
	if (process.platform === "win32") return classifyWindowsRoot(target);
	return { kind: "local", reason: "local" };
}

/** Whether `st_flags` (the decimal number `stat -f %f` prints) marks a data-less File Provider placeholder. */
export function datalessFromDarwinFlags(flags: number): boolean {
	return Number.isFinite(flags) && ((flags >>> 0) & SF_DATALESS) !== 0;
}

export async function isDatalessPlaceholder(target: string): Promise<boolean> {
	if (process.platform === "darwin") {
		try {
			const { stdout } = await execFileAsync("/usr/bin/stat", ["-f", "%f", target], { timeout: 2_000 });
			return datalessFromDarwinFlags(Number.parseInt(stdout.trim(), 10));
		} catch {
			return false;
		}
	}
	if (process.platform === "win32") {
		const attributes = await windowsFileAttributes(target);
		return (
			attributes !== undefined &&
			(attributes & WINDOWS_FILE_ATTRIBUTE_OFFLINE) !== 0 &&
			(attributes & (WINDOWS_FILE_ATTRIBUTE_RECALL_ON_OPEN | WINDOWS_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)) !== 0
		);
	}
	return false;
}

export interface MaterializedWalk {
	readonly materialized: readonly string[];
	readonly skippedDataless: number;
}

export async function listMaterializedFiles(
	root: string,
	options: { readonly limit?: number; readonly timeoutMs?: number } = {},
): Promise<MaterializedWalk> {
	const limit = options.limit ?? 20_000;
	if (process.platform === "darwin") {
		try {
			const { stdout } = await execFileAsync("/usr/bin/find", [root, "-type", "f", "!", "-flags", "+dataless"], {
				timeout: options.timeoutMs ?? 15_000,
				maxBuffer: 16 * 1024 * 1024,
			});
			return {
				materialized: stdout.split("\n").filter(Boolean).slice(0, limit),
				skippedDataless: await countDatalessMacFiles(root, options.timeoutMs),
			};
		} catch {
			return { materialized: [], skippedDataless: 0 };
		}
	}
	if (process.platform === "win32") {
		try {
			const script =
				`Get-ChildItem -LiteralPath ${quotePowerShell(root)} -File -Recurse -Force | ` +
				"Where-Object { -not ($_.Attributes.ToString() -match 'Offline') } | " +
				"Select-Object -ExpandProperty FullName";
			const { stdout } = await execFileAsync(
				"powershell.exe",
				["-NoProfile", "-NonInteractive", "-Command", script],
				{
					timeout: options.timeoutMs ?? 15_000,
					maxBuffer: 16 * 1024 * 1024,
				},
			);
			return { materialized: stdout.split(/\r?\n/).filter(Boolean).slice(0, limit), skippedDataless: 0 };
		} catch {
			return { materialized: [], skippedDataless: 0 };
		}
	}
	return { materialized: [], skippedDataless: 0 };
}

export function relativeCloudHint(root: string, filePath: string): string {
	const rel = relative(root, filePath);
	return rel === "" ? filePath : rel.split(sep).join("/");
}

async function classifyDarwinRoot(target: string): Promise<FilesystemClassification> {
	let cursor = target;
	for (let depth = 0; depth < 8; depth += 1) {
		const provider = await readFileProviderDomain(cursor);
		if (provider !== undefined) return { kind: "file-provider", reason: "file-provider-xattr", provider };
		const parent = dirname(cursor);
		if (parent === cursor) break;
		cursor = parent;
	}
	return { kind: "local", reason: "local" };
}

async function classifyWindowsRoot(target: string): Promise<FilesystemClassification> {
	const attributes = await windowsFileAttributes(target);
	if (
		attributes !== undefined &&
		(attributes & WINDOWS_FILE_ATTRIBUTE_OFFLINE) !== 0 &&
		(attributes & (WINDOWS_FILE_ATTRIBUTE_RECALL_ON_OPEN | WINDOWS_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS)) !== 0
	) {
		return { kind: "file-provider", reason: "placeholder-attribute", provider: "windows-cloud-files" };
	}
	if (await windowsCloudVolume(target)) {
		return { kind: "file-provider", reason: "cloud-volume", provider: "windows-cloud-volume" };
	}
	return { kind: "local", reason: "local" };
}

async function windowsFileAttributes(target: string): Promise<number | undefined> {
	try {
		const script = `(Get-Item -LiteralPath ${quotePowerShell(target)} -Force).Attributes.value__`;
		const { stdout } = await execFileAsync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
			timeout: 2_000,
		});
		const value = Number.parseInt(stdout.trim(), 10);
		return Number.isFinite(value) ? value : undefined;
	} catch {
		return undefined;
	}
}

async function windowsCloudVolume(target: string): Promise<boolean> {
	try {
		const root = /^[A-Za-z]:/.test(target) ? `${target.slice(0, 2)}\\` : target;
		const script =
			`Get-Volume -DriveLetter '${root[0]}' -ErrorAction SilentlyContinue | ` +
			"Select-Object -Property FileSystem,FileSystemLabel | ConvertTo-Json -Compress";
		const { stdout } = await execFileAsync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
			timeout: 2_000,
		});
		return /drivefs|google drive|onedrive|cloud/i.test(stdout);
	} catch {
		return false;
	}
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

async function countDatalessMacFiles(root: string, timeoutMs = 15_000): Promise<number> {
	try {
		const { stdout } = await execFileAsync("/usr/bin/find", [root, "-flags", "+dataless"], {
			timeout: timeoutMs,
			maxBuffer: 16 * 1024 * 1024,
		});
		return stdout.split("\n").filter(Boolean).length;
	} catch {
		return 0;
	}
}

/** Which cloud backs a path. Whether AutoRAG may release a provider's files again is up to {@link CloudFileProbe.canRelease}. */
export type CloudProvider = "icloud" | "google-drive" | "onedrive" | "file-provider";

export function cloudProviderFromPath(target: string): CloudProvider {
	const lower = target.toLowerCase();
	if (lower.includes("googledrive") || lower.includes("google drive") || lower.includes("drivefs"))
		return "google-drive";
	if (lower.includes("onedrive")) return "onedrive";
	if (lower.includes("mobile documents") || lower.includes("clouddocs")) return "icloud";
	return "file-provider";
}

/** Provider named by a File Provider domain id such as `com.apple.CloudDocs.iCloudDriveFileProvider/<uuid>`. */
export function cloudProviderFromDomain(domain: string): CloudProvider {
	const lower = domain.toLowerCase();
	if (lower.includes("clouddocs") || lower.includes("icloud")) return "icloud";
	if (lower.includes("google")) return "google-drive";
	if (lower.includes("onedrive")) return "onedrive";
	return "file-provider";
}

export type CloudEvictResult = { readonly ok: true } | { readonly ok: false; readonly message: string };

/**
 * What a refresh needs to know about cloud files, and the one action it may take on them. Every method
 * is read-only except {@link CloudFileProbe.evict}, which only drops the local copy of a file.
 */
export interface CloudFileProbe {
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

/**
 * The macOS probe. `stat -f %f` reads `st_flags`, which Node's `fs.stat` does not expose; a placeholder is
 * the file whose `SF_DATALESS` bit is set. `brctl evict` is iCloud-only: for other providers it refuses
 * with "outside of any CloudDocs app library", so this probe never calls it for them.
 */
export class DarwinCloudProbe implements CloudFileProbe {
	async datalessAmong(paths: readonly string[]): Promise<ReadonlySet<string>> {
		const dataless = new Set<string>();
		for (let start = 0; start < paths.length; start += STAT_BATCH) {
			const batch = paths.slice(start, start + STAT_BATCH);
			const { stdout } = await execFileAsync("/usr/bin/stat", ["-f", "%f", ...batch], {
				timeout: 30_000,
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
		return provider === "icloud";
	}

	async evict(path: string): Promise<CloudEvictResult> {
		try {
			await execFileAsync("/usr/bin/brctl", ["evict", path], { timeout: 30_000 });
			return { ok: true };
		} catch (error) {
			const stderr = typeof error === "object" && error !== null && "stderr" in error ? error.stderr : undefined;
			return {
				ok: false,
				message: typeof stderr === "string" && stderr.trim() !== "" ? stderr.trim() : String(error),
			};
		}
	}
}

/** The probe for this platform, or `undefined` where AutoRAG has no cloud-aware handling yet. */
export function defaultCloudProbe(): CloudFileProbe | undefined {
	return process.platform === "darwin" ? new DarwinCloudProbe() : undefined;
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
