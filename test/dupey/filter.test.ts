import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { selectExactDuplicateExclusions } from "../../src/dupey/index.ts";
import { isPathExcluded } from "../../src/mirror/index.ts";

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
			(path) => isPathExcluded(path, unavailable),
		);
		expect(result.keepers).toEqual(new Set([publicCopy]));
		expect(result.excluded).toEqual(new Set());
	});

	it("uses dupey's ranking when it disagrees with filesystem mtime", async () => {
		const root = mkdtempSync(join(tmpdir(), "dupey-filter-"));
		roots.push(root);
		mkdirSync(join(root, "docs"));
		const internalNewest = join(root, "docs", "internal-newest.docx");
		const fsNewest = join(root, "docs", "fs-newest.docx");
		writeFileSync(internalNewest, "same");
		writeFileSync(fsNewest, "same");
		// Filesystem mtime says `fsNewest` is newer, but dupey read the document's
		// internal modified timestamp and ranked `internalNewest` first.
		utimesSync(internalNewest, 1, 1);
		utimesSync(fsNewest, 2, 2);
		const result = await selectExactDuplicateExclusions(root, {
			dir: root,
			files: [
				{ path: "docs/internal-newest.docx", content_hash: "same-hash" },
				{ path: "docs/fs-newest.docx", content_hash: "same-hash" },
			],
			families: [
				{
					id: 0,
					relation: "exact",
					files: [internalNewest, fsNewest],
					members: [],
					edges: [],
					pick: {
						ranked: [
							{
								path: internalNewest,
								rank: 1,
								score: 1,
								reasons: [{ name: "internal_modified", detail: "document timestamp is newest" }],
							},
							{ path: fsNewest, rank: 2, score: 0, reasons: [] },
						],
					},
				},
			],
			errors: [],
		});
		expect(result.keepers).toEqual(new Set([internalNewest]));
		expect(result.excluded).toEqual(new Set([fsNewest]));
	});

	it("skips a dupey-ranked copy that is unavailable and keeps the next one", async () => {
		const root = mkdtempSync(join(tmpdir(), "dupey-filter-"));
		roots.push(root);
		mkdirSync(join(root, "docs"));
		mkdirSync(join(root, "private"));
		const preferred = join(root, "private", "report.docx");
		const fallback = join(root, "docs", "report.docx");
		writeFileSync(preferred, "same");
		writeFileSync(fallback, "same");
		utimesSync(preferred, 2, 2);
		utimesSync(fallback, 1, 1);
		const unavailable = new Set([join(root, "private")]);
		const result = await selectExactDuplicateExclusions(
			root,
			{
				dir: root,
				files: [
					{ path: "private/report.docx", content_hash: "same-hash" },
					{ path: "docs/report.docx", content_hash: "same-hash" },
				],
				families: [
					{
						id: 0,
						relation: "exact",
						files: [preferred, fallback],
						members: [],
						edges: [],
						pick: {
							ranked: [
								{
									path: preferred,
									rank: 1,
									score: 1,
									reasons: [{ name: "internal_modified", detail: "newest" }],
								},
								{ path: fallback, rank: 2, score: 0, reasons: [] },
							],
						},
					},
				],
				errors: [],
			},
			(path) => isPathExcluded(path, unavailable),
		);
		expect(result.keepers).toEqual(new Set([fallback]));
		expect(result.excluded).toEqual(new Set());
	});
});
