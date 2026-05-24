import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { PosixRetrieval } from "../../src/retrieval/posix.ts";

const FIXTURE_DIR = resolve("test/fixtures/sample-project");

describe("PosixRetrieval", () => {
	it("describe() returns correct descriptor", () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const desc = posix.describe();
		expect(desc.name).toBe("posix");
		expect(desc.type).toBe("posix");
		expect(desc.status).toBe("active");
		expect(desc.capabilities).toContain("grep");
		expect(desc.capabilities).toContain("glob");
	});

	it("finds content matches with grep", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("function", {});
		expect(results.length).toBeGreaterThan(0);
		const sources = results.map((r) => r.source);
		const hasMain = sources.some((s) => s.includes("main.ts"));
		const hasUtils = sources.some((s) => s.includes("utils.ts"));
		expect(hasMain || hasUtils).toBe(true);
	});

	it("returns empty array for no matches", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("absolutely_nonexistent_string_xyz_12345", {});
		expect(results).toEqual([]);
	});

	it("respects topK limit", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("function", { topK: 1 });
		expect(results.length).toBeLessThanOrEqual(1);
	});

	it("respects scope option", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("function", { scope: resolve(FIXTURE_DIR, "src") });
		for (const r of results) {
			expect(r.source).toContain("src");
		}
	});

	it("result shape matches RetrievalResult interface", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("function", {});
		if (results.length > 0) {
			const r = results[0];
			expect(typeof r.id).toBe("string");
			expect(typeof r.content).toBe("string");
			expect(typeof r.source).toBe("string");
			expect(typeof r.score).toBe("number");
			expect(typeof r.metadata).toBe("object");
		}
	});

	it("finds TypeScript files with glob pattern", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("**/*.ts", { topK: 20 });
		expect(results.length).toBeGreaterThan(0);
		for (const r of results) {
			expect(r.source).toMatch(/\.ts$/);
		}
	});

	it("finds markdown files with glob pattern", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("**/*.md", { topK: 20 });
		expect(results.length).toBeGreaterThan(0);
		for (const r of results) {
			expect(r.source).toMatch(/\.md$/);
		}
	});

	it("handles AbortSignal cancellation", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const controller = new AbortController();
		controller.abort();
		await expect(posix.retrieve("function", { signal: controller.signal })).rejects.toThrow();
	});

	it("does not include hidden files by default", async () => {
		const posix = new PosixRetrieval({ defaultScope: FIXTURE_DIR });
		const results = await posix.retrieve("secret config", {});
		const hiddenResults = results.filter((r) => r.source.includes(".hidden-file"));
		expect(hiddenResults.length).toBe(0);
	});
});
