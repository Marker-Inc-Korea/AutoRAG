import { createHash } from "node:crypto";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { CacheError, downloadAsset, importAsset, verifyCacheEntry } from "../../src/embedding-runtime/cache.ts";

const bytes = Buffer.from("verified model bytes");
const sha256 = createHash("sha256").update(bytes).digest("hex");
const asset = { id: "fixture", url: "https://fixture.invalid/model.gguf", filename: "model.gguf", sha256 };

async function root() {
	return mkdtemp(join(tmpdir(), "autorag-embedding-cache-"));
}

describe("embedding runtime cache", () => {
	it("cleans .part after hash mismatch and leaves no final file", async () => {
		const cacheRoot = await root();
		await expect(
			downloadAsset({ ...asset, sha256: "0".repeat(64) }, { cacheRoot, fetch: async () => new Response(bytes) }),
		).rejects.toBeInstanceOf(CacheError);
		await expect(stat(join(cacheRoot, "models", asset.filename))).rejects.toThrow();
		await expect(stat(join(cacheRoot, "models", `${asset.filename}.part`))).rejects.toThrow();
	});

	it("renames atomically and does not expose partial files", async () => {
		const cacheRoot = await root();
		const result = await downloadAsset(asset, { cacheRoot, fetch: async () => new Response(bytes) });
		expect(result).toBe(join(cacheRoot, "models", asset.filename));
		expect(await readFile(result)).toEqual(bytes);
	});

	it("rejects offline missing assets without attempting network", async () => {
		const cacheRoot = await root();
		let called = false;
		await expect(
			downloadAsset(asset, {
				cacheRoot,
				offline: true,
				fetch: async () => {
					called = true;
					return new Response(bytes);
				},
			}),
		).rejects.toThrow(/offline/i);
		expect(called).toBe(false);
	});

	it("imports and verifies local files, rejecting mismatches", async () => {
		const cacheRoot = await root();
		const source = join(cacheRoot, "source.gguf");
		await writeFile(source, bytes);
		expect(await importAsset(source, asset, { cacheRoot })).toContain("model.gguf");
		await writeFile(source, "wrong");
		await expect(importAsset(source, asset, { cacheRoot })).rejects.toBeInstanceOf(CacheError);
	});

	it("detects corrupted cache entries", async () => {
		const cacheRoot = await root();
		const path = await downloadAsset(asset, { cacheRoot, fetch: async () => new Response(bytes) });
		await writeFile(path, "corrupt");
		await expect(verifyCacheEntry(path, sha256)).rejects.toBeInstanceOf(CacheError);
	});
});
