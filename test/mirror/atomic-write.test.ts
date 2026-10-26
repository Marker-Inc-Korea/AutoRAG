import type * as Fs from "node:fs";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renameWithRetry, writeFileAtomic } from "../../src/mirror/atomic-write.ts";

const fsMock = vi.hoisted(() => ({
	renameHook: undefined as ((from: string, to: string) => void) | undefined,
}));

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof Fs>("node:fs");
	return {
		...actual,
		renameSync: (from: Fs.PathLike, to: Fs.PathLike) => {
			fsMock.renameHook?.(String(from), String(to));
			return actual.renameSync(from, to);
		},
	};
});

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-atomic-write-"));
});

afterEach(() => {
	fsMock.renameHook = undefined;
	rmSync(root, { recursive: true, force: true });
});

function errno(code: string): NodeJS.ErrnoException {
	const error: NodeJS.ErrnoException = new Error(`${code}: injected failure`);
	error.code = code;
	return error;
}

/** Fail the first `failures` renames with `code`, then let the real rename through. */
function failRenames(code: string, failures: number): { readonly attempts: () => number } {
	let attempts = 0;
	fsMock.renameHook = () => {
		attempts += 1;
		if (attempts <= failures) throw errno(code);
	};
	return { attempts: () => attempts };
}

describe("renameWithRetry", () => {
	it.each(["EPERM", "EACCES", "EBUSY"])("retries a transient %s rename until it succeeds", (code) => {
		const from = join(root, "from.tmp");
		const to = join(root, "to.json");
		writeFileSync(from, "new");
		writeFileSync(to, "old");
		const counter = failRenames(code, 3);

		renameWithRetry(from, to, { backoffMs: 0 });

		expect(counter.attempts()).toBe(4);
		expect(readFileSync(to, "utf8")).toBe("new");
	});

	it("throws a non-retryable rename error on the first attempt", () => {
		const from = join(root, "from.tmp");
		writeFileSync(from, "new");
		const counter = failRenames("ENOSPC", 10);

		expect(() => renameWithRetry(from, join(root, "to.json"), { backoffMs: 0 })).toThrow(/ENOSPC/);
		expect(counter.attempts()).toBe(1);
	});

	it("rethrows the original error once the retry budget is spent", () => {
		const from = join(root, "from.tmp");
		writeFileSync(from, "new");
		const counter = failRenames("EPERM", 100);

		expect(() => renameWithRetry(from, join(root, "to.json"), { retries: 2, backoffMs: 0 })).toThrow(/EPERM/);
		expect(counter.attempts()).toBe(3);
	});
});

describe("writeFileAtomic", () => {
	it("replaces the destination through a rename when nothing interferes", () => {
		const path = join(root, "out.md");
		writeFileSync(path, "old");

		writeFileAtomic(path, "new");

		expect(readFileSync(path, "utf8")).toBe("new");
		expect(readdirSync(root)).toEqual(["out.md"]);
	});

	it("survives a transient EPERM on the rename (held destination)", () => {
		const path = join(root, "out.md");
		writeFileSync(path, "old");
		const counter = failRenames("EPERM", 2);

		writeFileAtomic(path, "new");

		expect(counter.attempts()).toBe(3);
		expect(readFileSync(path, "utf8")).toBe("new");
		expect(readdirSync(root)).toEqual(["out.md"]);
	});

	it("falls back to an in-place overwrite when the rename never succeeds, leaving no temp file", () => {
		const path = join(root, "out.md");
		writeFileSync(path, "old");
		failRenames("EPERM", 100);

		writeFileAtomic(path, "new");

		expect(readFileSync(path, "utf8")).toBe("new");
		expect(readdirSync(root)).toEqual(["out.md"]);
	});

	it("does not mask a non-retryable rename failure and cleans its temp file", () => {
		const path = join(root, "out.md");
		writeFileSync(path, "old");
		failRenames("ENOSPC", 100);

		expect(() => writeFileAtomic(path, "new")).toThrow(/ENOSPC/);

		expect(readFileSync(path, "utf8")).toBe("old");
		expect(readdirSync(root)).toEqual(["out.md"]);
	});
});
