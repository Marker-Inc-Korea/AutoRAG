import { execFile } from "node:child_process";
import { lstat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, relative, sep } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

/** Darwin `st_flags` bit. Present on iCloud/File Provider placeholders. */
export const UF_DATALESS = 0x0000_8000;
export const FILE_PROVIDER_XATTR = "com.apple.file-provider-domain-id";

const CLOUD_PATH_MARKERS = [
	`${sep}Library${sep}CloudStorage${sep}`,
	`${sep}Library${sep}Mobile Documents${sep}`,
] as const;

export type FilesystemKind = "local" | "file-provider";

export interface FilesystemClassification {
	readonly kind: FilesystemKind;
	readonly provider?: string;
	readonly reason: "path-marker" | "file-provider-xattr" | "local";
}

export function pathLooksLikeCloudRoot(target: string): boolean {
	return CLOUD_PATH_MARKERS.some((marker) => target.includes(marker));
}

export function homeCloudRoots(home = homedir()): readonly string[] {
	return [
		join(home, "Library", "CloudStorage"),
		join(home, "Library", "Mobile Documents"),
	];
}

export async function classifyFilesystemRoot(target: string): Promise<FilesystemClassification> {
	if (pathLooksLikeCloudRoot(target)) {
		return { kind: "file-provider", reason: "path-marker", provider: "cloud-storage-path" };
	}
	if (process.platform !== "darwin") {
		return { kind: "local", reason: "local" };
	}
	let cursor = target;
	for (let depth = 0; depth < 8; depth += 1) {
		const provider = await readFileProviderDomain(cursor);
		if (provider !== undefined) {
			return { kind: "file-provider", reason: "file-provider-xattr", provider };
		}
		const parent = dirname(cursor);
		if (parent === cursor) break;
		cursor = parent;
	}
	return { kind: "local", reason: "local" };
}

export async function isDatalessPlaceholder(target: string): Promise<boolean> {
	if (process.platform !== "darwin") return false;
	try {
		const { stdout } = await execFileAsync("/usr/bin/stat", ["-f", "%f", target], { timeout: 2_000 });
		const flags = Number.parseInt(stdout.trim(), 16);
		return Number.isFinite(flags) && (flags & UF_DATALESS) !== 0;
	} catch {
		return false;
	}
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
	if (process.platform !== "darwin") {
		return { materialized: [], skippedDataless: 0 };
	}
	try {
		const { stdout } = await execFileAsync("/usr/bin/find", [root, "-type", "f", "!", "-flags", "+dataless"], {
			timeout: options.timeoutMs ?? 15_000,
			maxBuffer: 16 * 1024 * 1024,
		});
		const materialized = stdout.split("\n").filter((line) => line.length > 0).slice(0, limit);
		let skippedDataless = 0;
		try {
			const skipped = await execFileAsync("/usr/bin/find", [root, "-flags", "+dataless"], {
				timeout: options.timeoutMs ?? 15_000,
				maxBuffer: 16 * 1024 * 1024,
			});
			skippedDataless = skipped.stdout.split("\n").filter((line) => line.length > 0).length;
		} catch {
			skippedDataless = 0;
		}
		return { materialized, skippedDataless };
	} catch {
		return { materialized: [], skippedDataless: 0 };
	}
}

export function relativeCloudHint(root: string, filePath: string): string {
	const rel = relative(root, filePath);
	return rel === "" ? filePath : rel.split(sep).join("/");
}

async function readFileProviderDomain(target: string): Promise<string | undefined> {
	try {
		await lstat(target);
	} catch {
		return undefined;
	}
	try {
		const { stdout } = await execFileAsync("/usr/bin/xattr", ["-p", FILE_PROVIDER_XATTR, target], {
			timeout: 2_000,
		});
		const value = stdout.trim();
		return value.length > 0 ? value : undefined;
	} catch {
		return undefined;
	}
}
