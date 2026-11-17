import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { acquireRefreshLock, refreshLockPath } from "../../src/mirror/refresh-lock.ts";

let roots: string[] = [];

afterEach(() => {
	for (const root of roots) rmSync(root, { recursive: true, force: true });
	roots = [];
});

describe("refresh lock", () => {
	it("refuses a second owner until the first owner releases", () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-refresh-lock-"));
		roots.push(root);

		const first = acquireRefreshLock(root);
		const second = acquireRefreshLock(root);

		expect(first).toBeDefined();
		expect(second).toBeUndefined();
		expect(existsSync(refreshLockPath(root))).toBe(true);
		first?.release();
		const afterRelease = acquireRefreshLock(root);
		expect(afterRelease).toBeDefined();
		afterRelease?.release();
	});
});
