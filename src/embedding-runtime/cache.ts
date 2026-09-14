import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { chmod, copyFile, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, normalize, relative, resolve } from "node:path";
import { promisify } from "node:util";
import JSZip from "jszip";
import { resolveAutoRAGHome } from "../config/home.ts";
import { CacheError } from "./types.ts";

export { CacheError } from "./types.ts";

export interface CacheAsset {
	readonly id: string;
	readonly url: string;
	readonly filename: string;
	readonly sha256: string;
	readonly kind?: "model" | "runtime";
	readonly archiveMembers?: readonly string[];
}
export function cacheDirectory(kind: "models" | "runtime", cacheRoot = resolveAutoRAGHome()): string {
	return join(cacheRoot, kind);
}

export interface CacheOptions {
	readonly cacheRoot?: string;
	readonly offline?: boolean;
	readonly fetch?: typeof globalThis.fetch;
}

export async function verifyCacheEntry(path: string, expectedSha256: string): Promise<string> {
	validateHash(expectedSha256);
	try {
		const hash = await hashFile(path);
		if (hash !== expectedSha256) throw new CacheError("hash-mismatch", `SHA-256 mismatch for ${path}.`, { path });
		return hash;
	} catch (error) {
		if (error instanceof CacheError) throw error;
		throw new CacheError("io", `Unable to verify cache entry ${path}.`, { path, cause: error });
	}
}

export async function downloadAsset(asset: CacheAsset, options: CacheOptions = {}): Promise<string> {
	validateHash(asset.sha256);
	const directory = assetDirectory(asset, options.cacheRoot);
	const destination = join(directory, basename(asset.filename));
	await mkdir(directory, { recursive: true });
	try {
		if (await verifyIfPresent(destination, asset.sha256)) return finalizeAsset(destination, asset);
	} catch {
		await rm(destination, { force: true });
	}
	if (options.offline)
		throw new CacheError("offline-missing", `Asset ${asset.id} is missing or corrupt in offline mode.`);
	const part = `${destination}.part`;
	try {
		const response = await (options.fetch ?? fetch)(asset.url, { signal: AbortSignal.timeout(1_000) });
		if (!response.ok || !response.body)
			throw new CacheError("download", `Download failed for ${asset.url}: HTTP ${response.status}.`);
		const writer = createWriteStream(part);
		try {
			for await (const chunk of response.body) {
				if (!writer.write(chunk)) await new Promise<void>((resolve) => writer.once("drain", resolve));
			}
		} finally {
			await new Promise<void>((resolve, reject) => {
				writer.end(() => resolve());
				writer.once("error", reject);
			});
		}
		await verifyCacheEntry(part, asset.sha256);
		await rename(part, destination);
		return await finalizeAsset(destination, asset);
	} catch (error) {
		await rm(part, { force: true });
		if (error instanceof CacheError) throw error;
		throw new CacheError("download", `Download failed for ${asset.url}.`, { cause: error });
	}
}

export async function importAsset(sourcePath: string, asset: CacheAsset, options: CacheOptions = {}): Promise<string> {
	validateHash(asset.sha256);
	await verifyCacheEntry(sourcePath, asset.sha256);
	const directory = assetDirectory(asset, options.cacheRoot);
	const destination = join(directory, basename(asset.filename));
	await mkdir(directory, { recursive: true });
	const part = `${destination}.part`;
	try {
		await copyFile(sourcePath, part);
		await verifyCacheEntry(part, asset.sha256);
		await rename(part, destination);
		return await finalizeAsset(destination, asset);
	} catch (error) {
		await rm(part, { force: true });
		throw error;
	}
}

function assetDirectory(asset: CacheAsset, cacheRoot?: string): string {
	return cacheDirectory(asset.kind === "runtime" ? "runtime" : "models", cacheRoot);
}

async function finalizeAsset(archivePath: string, asset: CacheAsset): Promise<string> {
	if (asset.kind !== "runtime") return archivePath;
	if (!asset.archiveMembers?.length)
		throw new CacheError("missing-member", `Runtime asset ${asset.id} has no expected archive members.`);
	const extracted = `${archivePath}.extracted`;
	if (await membersPresent(extracted, asset.archiveMembers)) return serverPath(extracted, asset.archiveMembers);
	const partDirectory = `${extracted}.part-dir`;
	await rm(partDirectory, { recursive: true, force: true });
	try {
		await mkdir(partDirectory, { recursive: true });
		if (asset.filename.endsWith(".tar.gz") || asset.filename.endsWith(".tgz")) {
			await extractTarGz(archivePath, partDirectory);
		} else if (asset.filename.endsWith(".zip")) {
			await extractZip(archivePath, partDirectory);
		} else {
			throw new CacheError("extraction", `Unsupported runtime archive format: ${asset.filename}.`);
		}
		await validateMembers(partDirectory, asset.archiveMembers);
		const executable = serverPath(partDirectory, asset.archiveMembers);
		await chmod(executable, 0o755);
		await rm(extracted, { recursive: true, force: true });
		await rename(partDirectory, extracted);
		return serverPath(extracted, asset.archiveMembers);
	} catch (error) {
		await rm(partDirectory, { recursive: true, force: true });
		if (error instanceof CacheError) throw error;
		throw new CacheError("extraction", `Unable to extract runtime asset ${asset.id}.`, {
			path: archivePath,
			cause: error,
		});
	}
}

async function extractTarGz(archivePath: string, destination: string): Promise<void> {
	const run = promisify(execFile);
	const { stdout } = await run("tar", ["-tzf", archivePath], { encoding: "utf8" });
	for (const entry of stdout.split("\n").filter(Boolean)) validateArchiveEntry(entry);
	await run("tar", ["-xzf", archivePath, "-C", destination]);
}

async function writeBinary(path: string, bytes: Uint8Array): Promise<void> {
	await writeFile(path, bytes);
}

async function extractZip(archivePath: string, destination: string): Promise<void> {
	const zip = await JSZip.loadAsync(await readFile(archivePath));
	for (const [name, entry] of Object.entries(zip.files)) {
		validateArchiveEntry(name);
		const target = resolve(destination, normalize(name));
		if (entry.dir) {
			await mkdir(target, { recursive: true });
			continue;
		}
		await mkdir(dirname(target), { recursive: true });
		await writeBinary(target, await entry.async("uint8array"));
	}
}

function validateArchiveEntry(entry: string): void {
	const normalized = normalize(entry);
	if (
		normalized === ".." ||
		normalized.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) ||
		normalized.startsWith("/")
	)
		throw new CacheError("extraction", `Unsafe runtime archive member: ${entry}.`);
}

async function validateMembers(root: string, members: readonly string[]): Promise<void> {
	for (const member of members) {
		const path = resolve(root, member);
		if (relative(root, path).startsWith(".."))
			throw new CacheError("missing-member", `Invalid expected runtime member: ${member}.`);
		try {
			if (!(await stat(path)).isFile()) throw new Error("not a file");
		} catch (error) {
			throw new CacheError("missing-member", `Runtime archive is missing expected member ${member}.`, {
				path,
				cause: error,
			});
		}
	}
}

async function membersPresent(root: string, members: readonly string[]): Promise<boolean> {
	try {
		await validateMembers(root, members);
		return true;
	} catch {
		return false;
	}
}

function serverPath(root: string, members: readonly string[]): string {
	const server = members.find(
		(member) => basename(member) === "llama-server" || basename(member) === "llama-server.exe",
	);
	if (!server) throw new CacheError("missing-member", "Runtime manifest does not identify a llama-server executable.");
	return join(root, server);
}

async function verifyIfPresent(path: string, hash: string): Promise<boolean> {
	try {
		await stat(path);
		await verifyCacheEntry(path, hash);
		return true;
	} catch {
		return false;
	}
}
async function hashFile(path: string): Promise<string> {
	const hash = createHash("sha256");
	for await (const chunk of createReadStream(path)) hash.update(chunk);
	return hash.digest("hex");
}
function validateHash(hash: string): void {
	if (!/^[a-f0-9]{64}$/i.test(hash)) throw new CacheError("invalid-hash", "Expected a 64-character SHA-256 hash.");
}
