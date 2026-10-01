import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { selectExactDuplicateExclusions } from "../../src/dupey/index.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("exact duplicate filter", () => {
	it("keeps only the newest file for each canonical content hash", async () => {
		const root = mkdtempSync(join(tmpdir(), "dupey-filter-"));
		roots.push(root);
		mkdirSync(join(root, "docs"));
		const oldPath = join(root, "docs", "old.txt");
		const newPath = join(root, "docs", "new.txt");
		writeFileSync(oldPath, "same");
		writeFileSync(newPath, "same");
		utimesSync(oldPath, 1, 1);
		utimesSync(newPath, 2, 2);
		const result = await selectExactDuplicateExclusions(root, {
			dir: root,
			files: [
				{ path: "docs/old.txt", content_hash: "same-hash" },
				{ path: "docs/new.txt", content_hash: "same-hash" },
			],
			families: [],
			errors: [],
		});
		expect(result.keepers).toEqual(new Set([newPath]));
		expect(result.excluded).toEqual(new Set([oldPath]));
	});

	it("skips unavailable copies when choosing the keeper", async () => {
		const root = mkdtempSync(join(tmpdir(), "dupey-filter-"));
		roots.push(root);
		mkdirSync(join(root, "docs"));
		mkdirSync(join(root, "private"));
		const publicCopy = join(root, "docs", "report.txt");
		const privateCopy = join(root, "private", "report.txt");
		writeFileSync(publicCopy, "same");
		writeFileSync(privateCopy, "same");
		utimesSync(publicCopy, 1, 1);
		utimesSync(privateCopy, 2, 2);
		const unavailable = new Set([join(root, "private")]);
		const result = await selectExactDuplicateExclusions(
			root,
			{
				dir: root,
				files: [
					{ path: "docs/report.txt", content_hash: "same-hash" },
					{ path: "private/report.txt", content_hash: "same-hash" },
				],
				families: [],
				errors: [],
			},
			(path) => {
				for (const excluded of unavailable) {
					if (path === excluded || path.startsWith(`${excluded}/`)) return true;
				}
				return false;
			},
		);
		expect(result.keepers).toEqual(new Set([publicCopy]));
		expect(result.excluded).toEqual(new Set());
	});
});
