import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
	CacheError,
	downloadAsset,
	hasVerifiedAsset,
	importAsset,
	PROVISIONED_RUNTIME_MARKER,
	verifyCacheEntry,
} from "../../src/embedding-runtime/cache.ts";

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

	it("GREEN — slow download (3s, within 10s budget) succeeds with configurable timeout", async () => {
		const cacheRoot = await root();
		const fetch: typeof globalThis.fetch = (_url, init) =>
			new Promise((resolve, reject) => {
				const signal = (init as RequestInit | undefined)?.signal;
				const timer = setTimeout(async () => {
					resolve(new Response(bytes));
				}, 3_000);
				signal?.addEventListener("abort", () => {
					clearTimeout(timer);
					reject(signal.reason);
				});
			});
		const result = await downloadAsset(asset, {
			cacheRoot,
			fetch,
			downloadTimeoutMs: 10_000, // 10s budget covers the 3s server delay
		});
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

	it("routes model and runtime assets separately and extracts a tar.gz server matching real llama.cpp layout", async () => {
		const cacheRoot = await root();
		const fixture = await mkdtemp(join(cacheRoot, "fixture-"));
		await mkdir(join(fixture, "llama-b10951"), { recursive: true });
		await writeFile(join(fixture, "llama-b10951", "llama-server"), "#!/bin/sh\n");
		const archive = join(cacheRoot, "runtime.tar.gz");
		execFileSync("tar", ["-czf", archive, "-C", fixture, "llama-b10951/llama-server"]);
		const archiveBytes = await readFile(archive);
		const runtimeAsset = {
			id: "runtime-fixture",
			url: "https://fixture.invalid/runtime.tar.gz",
			filename: "runtime.tar.gz",
			sha256: createHash("sha256").update(archiveBytes).digest("hex"),
			kind: "runtime" as const,
			archiveMembers: ["llama-b10951/llama-server"],
		};
		const runtimePath = await downloadAsset(runtimeAsset, {
			cacheRoot,
			fetch: async () => new Response(archiveBytes),
		});
		expect(runtimePath).toBe(join(cacheRoot, "runtime", "runtime.tar.gz.extracted", "llama-b10951", "llama-server"));
		if (process.platform !== "win32") expect((await stat(runtimePath)).mode & 0o111).toBeGreaterThan(0);
		expect(await downloadAsset(asset, { cacheRoot, fetch: async () => new Response(bytes) })).toBe(
			join(cacheRoot, "models", "model.gguf"),
		);
	});

	it("rejects a runtime archive with missing expected members and cleans extraction state", async () => {
		const cacheRoot = await root();
		const fixture = await mkdtemp(join(cacheRoot, "fixture-"));
		await mkdir(join(fixture, "llama-b10951"), { recursive: true });
		await writeFile(join(fixture, "llama-b10951", "not-server"), "nope");
		const archive = join(cacheRoot, "runtime.tar.gz");
		execFileSync("tar", ["-czf", archive, "-C", fixture, "llama-b10951/not-server"]);
		const archiveBytes = await readFile(archive);
		const runtimeAsset = {
			id: "bad-runtime",
			url: "https://fixture.invalid/runtime.tar.gz",
			filename: "bad-runtime.tar.gz",
			sha256: createHash("sha256").update(archiveBytes).digest("hex"),
			kind: "runtime" as const,
			archiveMembers: ["llama-b10951/llama-server"],
		};
		await expect(
			downloadAsset(runtimeAsset, { cacheRoot, fetch: async () => new Response(archiveBytes) }),
		).rejects.toMatchObject({
			code: "missing-member",
		});
		await expect(stat(join(cacheRoot, "runtime", "bad-runtime.tar.gz.extracted.part-dir"))).rejects.toThrow();
	});

	describe("a runtime shipped already extracted (signed desktop builds)", () => {
		const runtimeAsset = {
			id: "runtime-fixture",
			url: "https://fixture.invalid/runtime.tar.gz",
			filename: "runtime.tar.gz",
			sha256: "a".repeat(64),
			kind: "runtime" as const,
			archiveMembers: ["llama-b10951/llama-server"],
		};

		async function provision(cacheRoot: string, marker: string) {
			const extracted = join(cacheRoot, "runtime", "runtime.tar.gz.extracted");
			await mkdir(join(extracted, "llama-b10951"), { recursive: true });
			await writeFile(join(extracted, "llama-b10951", "llama-server"), "#!/bin/sh\n");
			await writeFile(join(extracted, PROVISIONED_RUNTIME_MARKER), `${marker}\n`);
			return join(extracted, "llama-b10951", "llama-server");
		}

		it("is used offline without the archive when its marker names the pinned archive", async () => {
			const cacheRoot = await root();
			const server = await provision(cacheRoot, runtimeAsset.sha256);
			let fetched = false;
			const fetch: typeof globalThis.fetch = async () => {
				fetched = true;
				return new Response("unused");
			};
			expect(await hasVerifiedAsset(runtimeAsset, cacheRoot)).toBe(true);
			expect(await downloadAsset(runtimeAsset, { cacheRoot, offline: true, fetch })).toBe(server);
			expect(fetched).toBe(false);
		});

		it("is ignored when its marker names another archive, so the pinned one is still required", async () => {
			const cacheRoot = await root();
			await provision(cacheRoot, "b".repeat(64));
			expect(await hasVerifiedAsset(runtimeAsset, cacheRoot)).toBe(false);
			await expect(downloadAsset(runtimeAsset, { cacheRoot, offline: true })).rejects.toMatchObject({
				code: "offline-missing",
			});
		});

		it("is ignored when an expected member is missing", async () => {
			const cacheRoot = await root();
			const server = await provision(cacheRoot, runtimeAsset.sha256);
			await rm(server);
			expect(await hasVerifiedAsset(runtimeAsset, cacheRoot)).toBe(false);
		});
	});
});
