import { execFile } from "node:child_process";
import { lstat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, relative, sep } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** Darwin st_flags bit used by iCloud/File Provider data-less placeholders. */
export const UF_DATALESS = 0x0000_8000;
export const FILE_PROVIDER_XATTR = "com.apple.file-provider-domain-id";

/** Windows FILE_ATTRIBUTE_* bits commonly used by Cloud Files placeholders. */
export const WINDOWS_FILE_ATTRIBUTE_OFFLINE = 0x0000_1000;
export const WINDOWS_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x0004_0000;
export const WINDOWS_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x0040_0000;

const MAC_CLOUD_PATH_MARKERS = [
	`${sep}Library${sep}CloudStorage${sep}`,
	`${sep}Library${sep}Mobile Documents${sep}`,
] as const;

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
	const normalized = target.replaceAll("\\", sep);
	return (
		MAC_CLOUD_PATH_MARKERS.some((marker) => normalized.includes(marker)) ||
		WINDOWS_CLOUD_PATH_MARKERS.some((marker) => normalized.toLowerCase().includes(marker.toLowerCase()))
	);
}

export function homeCloudRoots(home = homedir()): readonly string[] {
	return [join(home, "Library", "CloudStorage"), join(home, "Library", "Mobile Documents")];
}

export async function classifyFilesystemRoot(target: string): Promise<FilesystemClassification> {
	if (pathLooksLikeCloudRoot(target)) {
		return { kind: "file-provider", reason: "path-marker", provider: cloudProviderFromPath(target) };
	}
	if (process.platform === "darwin") return classifyDarwinRoot(target);
	if (process.platform === "win32") return classifyWindowsRoot(target);
	return { kind: "local", reason: "local" };
}

export async function isDatalessPlaceholder(target: string): Promise<boolean> {
	if (process.platform === "darwin") {
		try {
			const { stdout } = await execFileAsync("/usr/bin/stat", ["-f", "%f", target], { timeout: 2_000 });
			const flags = Number.parseInt(stdout.trim(), 16);
			return Number.isFinite(flags) && (flags & UF_DATALESS) !== 0;
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

function cloudProviderFromPath(target: string): string {
	const lower = target.toLowerCase();
	if (lower.includes("googledrive") || lower.includes("google drive") || lower.includes("drivefs"))
		return "google-drive";
	if (lower.includes("onedrive")) return "onedrive";
	if (lower.includes("mobile documents") || lower.includes("clouddocs")) return "icloud";
	return "file-provider";
}

function quotePowerShell(value: string): string {
	return `'${value.replaceAll("'", "''")}'`;
}
