import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { copyFile, mkdir, rename, rm, stat } from "node:fs/promises";
import { basename, join } from "node:path";
import { resolveAutoRAGHome } from "../config/home.ts";
import { CacheError } from "./types.ts";

export { CacheError } from "./types.ts";

export interface CacheAsset {
	readonly id: string;
	readonly url: string;
	readonly filename: string;
	readonly sha256: string;
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
	const models = cacheDirectory("models", options.cacheRoot);
	const destination = join(models, basename(asset.filename));
	await mkdir(models, { recursive: true });
	try {
		if (await verifyIfPresent(destination, asset.sha256)) return destination;
	} catch {
		await rm(destination, { force: true });
	}
	if (options.offline)
		throw new CacheError("offline-missing", `Asset ${asset.id} is missing or corrupt in offline mode.`);
	const part = `${destination}.part`;
	try {
		const response = await (options.fetch ?? fetch)(asset.url);
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
		return destination;
	} catch (error) {
		await rm(part, { force: true });
		if (error instanceof CacheError) throw error;
		throw new CacheError("download", `Download failed for ${asset.url}.`, { cause: error });
	}
}

export async function importAsset(sourcePath: string, asset: CacheAsset, options: CacheOptions = {}): Promise<string> {
	validateHash(asset.sha256);
	await verifyCacheEntry(sourcePath, asset.sha256);
	const destination = join(cacheDirectory("models", options.cacheRoot), basename(asset.filename));
	await mkdir(cacheDirectory("models", options.cacheRoot), { recursive: true });
	const part = `${destination}.part`;
	try {
		await copyFile(sourcePath, part);
		await verifyCacheEntry(part, asset.sha256);
		await rename(part, destination);
		return destination;
	} catch (error) {
		await rm(part, { force: true });
		throw error;
	}
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
