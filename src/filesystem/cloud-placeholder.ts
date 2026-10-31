import { execFile } from "node:child_process";
import { lstat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname } from "node:path";
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

/**
 * Maps the `OPEN`/`DEHYDRATE`/`MSG` lines of {@link buildWindowsDehydrateScript} to an eviction result.
 * `HRESULT 0` is success; anything else (including a process that produced no lines) is a failure whose
 * message is the OS text when the script printed one, otherwise the process stderr.
 */
export function parseWindowsDehydrateOutput(result: PowerShellResult): CloudEvictResult {
	const open = /^OPEN=0x([0-9A-Fa-f]{8})$/mu.exec(result.stdout);
	if (open === null)
		return { ok: false, message: windowsFailureMessage(result, "the release command produced no result") };
	if (Number.parseInt(open[1] as string, 16) !== 0) {
		return {
			ok: false,
			message: windowsFailureMessage(result, `could not open the file for release (0x${open[1]})`),
		};
	}
	const dehydrate = /^DEHYDRATE=0x([0-9A-Fa-f]{8})$/mu.exec(result.stdout);
	if (dehydrate === null) {
		return { ok: false, message: windowsFailureMessage(result, "the release command produced no result") };
	}
	if (Number.parseInt(dehydrate[1] as string, 16) === 0) return { ok: true };
	return {
		ok: false,
		message: windowsFailureMessage(result, `the provider refused to dehydrate the file (0x${dehydrate[1]})`),
	};
}

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
 * The Windows probe. Cloud Files placeholders carry `FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS`; the provider is
 * the registered sync root that contains the path (falling back to the path's own marker), and the local
 * copy is released with `CfDehydratePlaceholder`. Only OneDrive is enabled by {@link canRelease} because it
 * is the only provider whose release was verified live.
 */
export class WindowsCloudProbe implements CloudFileProbe {
	private readonly run: PowerShellRunner;
	private syncRoots: Promise<readonly WindowsSyncRoot[]> | undefined;

	constructor(options: WindowsCloudProbeOptions = {}) {
		this.run = options.run ?? defaultPowerShellRunner;
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
		return provider !== undefined && WINDOWS_RELEASABLE.has(provider);
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
}

/** Providers whose release path was verified live on Windows. Extend only after a measured run. */
const WINDOWS_RELEASABLE: ReadonlySet<CloudProvider> = new Set<CloudProvider>(["onedrive"]);

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
