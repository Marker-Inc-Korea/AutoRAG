import { mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	buildVersionDiffSystemPrompt,
	buildVersionDiffUserPrompt,
	buildVersionHistory,
	cleanVersionDiffSummary,
	compareDocumentVersions,
	diffTextLines,
	formatCompactDiffForLlm,
	summarizeVersionDiff,
	summarizeVersionDiffWithLlm,
} from "../../src/dupey/index.ts";

describe("diffTextLines", () => {
	it("returns identical status when texts match after normalizing CRLF", () => {
		const diff = diffTextLines("alpha\r\nbeta\r\n", "alpha\nbeta\n");
		expect(diff.identical).toBe(true);
		expect(diff.addedCount).toBe(0);
		expect(diff.removedCount).toBe(0);
		expect(diff.hunks).toEqual([]);
	});

	it("tracks added and removed lines with surrounding context", () => {
		const prev = ["Line 1", "Line 2", "Budget: $10,000", "Line 4", "Line 5"].join("\n");
		const curr = ["Line 1", "Line 2", "Budget: $12,500", "Approved by CFO", "Line 4", "Line 5"].join("\n");

		const diff = diffTextLines(prev, curr, { contextLines: 1 });
		expect(diff.identical).toBe(false);
		expect(diff.addedCount).toBe(2);
		expect(diff.removedCount).toBe(1);
		expect(diff.hunks).toHaveLength(1);
		expect(diff.hunks[0]?.lines).toEqual([
			{ kind: "context", text: "Line 2", previousLine: 2, currentLine: 2 },
			{ kind: "removed", text: "Budget: $10,000", previousLine: 3, currentLine: null },
			{ kind: "added", text: "Budget: $12,500", previousLine: null, currentLine: 3 },
			{ kind: "added", text: "Approved by CFO", previousLine: null, currentLine: 4 },
			{ kind: "context", text: "Line 4", previousLine: 4, currentLine: 5 },
		]);
	});

	it("splits distant edits into separate hunks", () => {
		const base = Array.from({ length: 12 }, (_, i) => `row-${i + 1}`);
		const next = [...base];
		next[0] = "row-1-updated";
		next[11] = "row-12-updated";

		const diff = diffTextLines(base.join("\n"), next.join("\n"), { contextLines: 1 });
		expect(diff.hunks).toHaveLength(2);
		expect(diff.addedCount).toBe(2);
		expect(diff.removedCount).toBe(2);
	});

	it("handles completely different large texts without blowing up", () => {
		const prev = Array.from({ length: 600 }, (_, i) => `old line ${i}`).join("\n");
		const curr = Array.from({ length: 600 }, (_, i) => `new line ${i}`).join("\n");

		const diff = diffTextLines(prev, curr);
		expect(diff.identical).toBe(false);
		expect(diff.addedCount).toBe(600);
		expect(diff.removedCount).toBe(600);
	});
});

describe("summarizeVersionDiff", () => {
	it("extracts concise non-empty added and removed previews without markdown noise", () => {
		const diff = diffTextLines(
			["# Q3 Plan", "", "- Old budget: 100M", "- Old deadline: Oct 1"].join("\n"),
			["# Q3 Plan", "", "- New budget: 120M", "- Added risk review", "- New deadline: Oct 15"].join("\n"),
		);
		const summary = summarizeVersionDiff(diff, { maxHighlights: 2, maxLineLength: 60 });
		expect(summary.addedCount).toBe(3);
		expect(summary.removedCount).toBe(2);
		expect(summary.addedHighlights).toEqual(["New budget: 120M", "Added risk review"]);
		expect(summary.removedHighlights).toEqual(["Old budget: 100M", "Old deadline: Oct 1"]);
	});

	it("skips blank lines and separator-only markdown lines in highlights", () => {
		const diff = diffTextLines("intro\n", "intro\n\n---\n\n> Important approval note\n");
		const summary = summarizeVersionDiff(diff);
		expect(summary.addedHighlights).toEqual(["Important approval note"]);
		expect(summary.removedHighlights).toEqual([]);
	});

	it("truncates overly long lines cleanly with an ellipsis", () => {
		const longLine = "A".repeat(140);
		const diff = diffTextLines("", longLine);
		const summary = summarizeVersionDiff(diff, { maxLineLength: 40 });
		expect(summary.addedHighlights[0]?.length).toBeLessThanOrEqual(40);
		expect(summary.addedHighlights[0]?.endsWith("…")).toBe(true);
	});
});

describe("compareDocumentVersions & buildVersionHistory", () => {
	const tempDirs: string[] = [];

	afterEach(() => {
		for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
	});

	function createTempDir(): string {
		const dir = mkdtempSync(join(tmpdir(), "autorag-version-diff-"));
		tempDirs.push(dir);
		return dir;
	}

	it("orders a family chronologically by last modified date and diffs each version against its immediate predecessor", async () => {
		const dir = createTempDir();
		const v1 = join(dir, "proposal_v1.md");
		const v2 = join(dir, "proposal_v2.md");
		const v3 = join(dir, "proposal_final.md");

		writeFileSync(v1, "# Proposal\nScope: Phase 1\nBudget: 50M\n", "utf8");
		writeFileSync(v2, "# Proposal\nScope: Phase 1\nBudget: 65M\nOwner: Mina\n", "utf8");
		writeFileSync(v3, "# Proposal\nScope: Phase 1 + Phase 2\nBudget: 65M\nOwner: Mina\nStatus: Approved\n", "utf8");

		utimesSync(v1, new Date("2026-10-01T09:00:00Z"), new Date("2026-10-01T09:00:00Z"));
		utimesSync(v2, new Date("2026-10-05T12:00:00Z"), new Date("2026-10-05T12:00:00Z"));
		utimesSync(v3, new Date("2026-10-10T15:30:00Z"), new Date("2026-10-10T15:30:00Z"));

		const history = await buildVersionHistory([
			{ path: v2, relation: "near" },
			{ path: v3, relation: "near" },
			{ path: v1, relation: "contains" },
		]);

		// Latest version first (by modified date), oldest last
		expect(history.versions.map((v) => v.path)).toEqual([v3, v2, v1]);
		expect(history.versions[0]?.isLatest).toBe(true);
		expect(history.versions[1]?.isLatest).toBe(false);
		expect(history.versions[2]?.isLatest).toBe(false);

		// Latest (v3) compared against immediate predecessor (v2)
		const latestEntry = history.versions[0]!;
		expect(latestEntry.previousPath).toBe(v2);
		expect(latestEntry.diff?.status).toBe("changed");
		expect(latestEntry.diff?.summary.addedHighlights).toContain("Scope: Phase 1 + Phase 2");
		expect(latestEntry.diff?.summary.addedHighlights).toContain("Status: Approved");
		expect(latestEntry.diff?.summary.removedHighlights).toContain("Scope: Phase 1");

		// Middle (v2) compared against v1
		const middleEntry = history.versions[1]!;
		expect(middleEntry.previousPath).toBe(v1);
		expect(middleEntry.diff?.status).toBe("changed");
		expect(middleEntry.diff?.summary.addedHighlights).toContain("Budget: 65M");
		expect(middleEntry.diff?.summary.addedHighlights).toContain("Owner: Mina");
		expect(middleEntry.diff?.summary.removedHighlights).toContain("Budget: 50M");

		// Oldest (v1) is the base version — no predecessor
		const baseEntry = history.versions[2]!;
		expect(baseEntry.previousPath).toBeNull();
		expect(baseEntry.diff).toBeNull();
	});

	it("reports identical status when two versions have exact same text", async () => {
		const dir = createTempDir();
		const a = join(dir, "copy1.txt");
		const b = join(dir, "copy2.txt");
		writeFileSync(a, "Identical content\nLine 2\n", "utf8");
		writeFileSync(b, "Identical content\nLine 2\n", "utf8");

		const diff = await compareDocumentVersions(a, b);
		expect(diff.status).toBe("identical");
		expect(diff.summary.addedCount).toBe(0);
		expect(diff.summary.removedCount).toBe(0);
	});

	it("reports unsupported status cleanly for binary/image formats without throwing", async () => {
		const dir = createTempDir();
		const img1 = join(dir, "photo1.png");
		const img2 = join(dir, "photo2.png");
		writeFileSync(img1, Buffer.from([0x89, 0x50, 0x4e, 0x47]));
		writeFileSync(img2, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00]));

		const diff = await compareDocumentVersions(img1, img2);
		expect(diff.status).toBe("unsupported");
		expect(diff.hunks).toEqual([]);
	});

	it("honors explicit mtimeMs provided by caller when sorting versions", async () => {
		const dir = createTempDir();
		const a = join(dir, "a.txt");
		const b = join(dir, "b.txt");
		writeFileSync(a, "old\n", "utf8");
		writeFileSync(b, "old\nnew\n", "utf8");

		const history = await buildVersionHistory([
			{ path: a, mtimeMs: 1000 },
			{ path: b, mtimeMs: 2000 },
		]);

		expect(history.versions[0]?.path).toBe(b);
		expect(history.versions[0]?.isLatest).toBe(true);
		expect(history.versions[0]?.previousPath).toBe(a);
		expect(history.versions[0]?.diff?.summary.addedHighlights).toEqual(["new"]);
	});
});

describe("sLLM version diff summary", () => {
	it("builds an ultra-short system prompt with optional language hint", () => {
		const ko = buildVersionDiffSystemPrompt("ko");
		expect(ko.length).toBeLessThan(200);
		expect(ko).toContain("Korean");

		const auto = buildVersionDiffSystemPrompt();
		expect(auto.length).toBeLessThan(200);
		expect(auto).toContain("document language");
	});

	it("formats only added/removed non-empty lines within the character budget", () => {
		const diff = diffTextLines("keep\nold line\nkeep2", "keep\nnew line\nadded line\nkeep2");
		const compact = formatCompactDiffForLlm(diff.hunks, 200);
		expect(compact).toContain("- old line");
		expect(compact).toContain("+ new line");
		expect(compact).toContain("+ added line");
		expect(compact).not.toContain("keep");
	});

	it("cleans markdown bullets, quotes, and trims overlong replies", () => {
		expect(cleanVersionDiffSummary('  - "예산이 1억에서 1.2억으로 상향됨."  ')).toBe(
			"예산이 1억에서 1.2억으로 상향됨.",
		);
		expect(cleanVersionDiffSummary("   ")).toBeNull();
		const long = "가".repeat(300);
		const cleaned = cleanVersionDiffSummary(long);
		expect(cleaned?.length).toBeLessThanOrEqual(240);
		expect(cleaned?.endsWith("…")).toBe(true);
	});

	it("calls the completer with a concise prompt and returns cleaned summary text", async () => {
		const diff = diffTextLines("Budget: 100M\nStatus: Draft", "Budget: 120M\nStatus: Approved");
		const calls: Array<{ systemPrompt: string; userPrompt: string }> = [];
		const result = await summarizeVersionDiffWithLlm(
			async (input) => {
				calls.push(input);
				return "예산을 100M에서 120M으로 상향하고 상태를 승인으로 변경했습니다.";
			},
			{
				previousName: "plan_v1.md",
				currentName: "plan_v2.md",
				addedCount: diff.addedCount,
				removedCount: diff.removedCount,
				hunks: diff.hunks,
				language: "ko",
			},
		);

		expect(calls).toHaveLength(1);
		expect(calls[0]?.systemPrompt).toContain("Korean");
		expect(calls[0]?.userPrompt).toBe(
			buildVersionDiffUserPrompt({
				previousName: "plan_v1.md",
				currentName: "plan_v2.md",
				addedCount: diff.addedCount,
				removedCount: diff.removedCount,
				hunks: diff.hunks,
			}),
		);
		expect(result).toBe("예산을 100M에서 120M으로 상향하고 상태를 승인으로 변경했습니다.");
	});

	it("skips calling the completer when hunks contain no non-empty changes", async () => {
		let called = false;
		const result = await summarizeVersionDiffWithLlm(
			async () => {
				called = true;
				return "should not run";
			},
			{
				previousName: "a.md",
				currentName: "b.md",
				addedCount: 0,
				removedCount: 0,
				hunks: [],
			},
		);
		expect(called).toBe(false);
		expect(result).toBeNull();
	});
});
