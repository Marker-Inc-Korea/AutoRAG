import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { countFiles, MAX_TIMER_MS, scaledPrepareTimeoutMs } from "../../src/jikji/prepare-budget.ts";

describe("scaledPrepareTimeoutMs", () => {
	it("adds the per-file budget to the base", () => {
		expect(scaledPrepareTimeoutMs(60_000, 10, 1_000)).toBe(70_000);
	});

	it("has no policy ceiling: 10M files still exceed the old 30 minute wall", () => {
		expect(scaledPrepareTimeoutMs(60_000, 5_000, 10_000_000)).toBeGreaterThan(1_800_000);
	});

	it("clamps only to the timer limit so setTimeout never fires early", () => {
		expect(scaledPrepareTimeoutMs(60_000, 5_000, 10_000_000_000)).toBe(MAX_TIMER_MS);
	});
});

describe("countFiles", () => {
	let root: string;

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-jikji-count-"));
		mkdirSync(join(root, "a", "b"), { recursive: true });
		mkdirSync(join(root, ".hidden"), { recursive: true });
		writeFileSync(join(root, "1.txt"), "x");
		writeFileSync(join(root, "a", "2.txt"), "x");
		writeFileSync(join(root, "a", "b", "3.txt"), "x");
		writeFileSync(join(root, ".dot.txt"), "x");
		writeFileSync(join(root, ".hidden", "4.txt"), "x");
		symlinkSync(join(root, "1.txt"), join(root, "link.txt"));
	});

	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	it("counts regular files recursively, skipping hidden entries and symlinks", async () => {
		expect(await countFiles(root, false)).toBe(3);
	});

	it("includes hidden entries when asked", async () => {
		expect(await countFiles(root, true)).toBe(5);
	});

	it("returns 0 for a missing root", async () => {
		expect(await countFiles(join(root, "nope"), false)).toBe(0);
	});

	it("stops walking once the signal aborts", async () => {
		const controller = new AbortController();
		controller.abort();
		expect(await countFiles(root, false, controller.signal)).toBe(0);
	});
});
