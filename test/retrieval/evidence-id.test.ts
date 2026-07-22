import { describe, expect, it } from "vitest";
import {
	isPathOpaqueIdentifier,
	normalizeEvidenceRef,
	normalizeEvidenceText,
	stableEvidenceId,
} from "../../src/retrieval/evidence-id.ts";

describe("stable evidence IDs", () => {
	it("normalizes evidence text before hashing", () => {
		expect(normalizeEvidenceText("  Cafe\u0301\r\nline  ")).toBe("Café\nline");
	});

	it("uses path-opaque backend retrieval IDs with method namespace", () => {
		expect(stableEvidenceId({ method: "minsync", source: "/docs/a.md", retrievalResultId: "chunk-123" })).toBe(
			"minsync:chunk-123",
		);
		expect(
			stableEvidenceId({ method: "minsync", source: "/docs/a.md", retrievalResultId: "minsync:chunk-123" }),
		).toBe("minsync:chunk-123");
	});

	it("rejects filesystem-like retrieval IDs and hashes opaque source plus chunk identity", () => {
		const id = stableEvidenceId({
			method: "posix",
			source: "/docs/a.md",
			retrievalResultId: "/Users/me/docs/a.md:12",
			lineNumber: 12,
			excerpt: " Important fact ",
		});
		expect(id).toMatch(/^posix:[0-9a-f]{24}$/u);
		expect(id).not.toContain("/Users");
	});

	it("treats absolute and relative filesystem syntax as non-opaque", () => {
		expect(isPathOpaqueIdentifier("chunk-1")).toBe(true);
		expect(isPathOpaqueIdentifier("/tmp/file.txt")).toBe(false);
		expect(isPathOpaqueIdentifier("docs/file.txt")).toBe(false);
		expect(isPathOpaqueIdentifier("../file.txt")).toBe(false);
		expect(isPathOpaqueIdentifier("C:\\docs\\file.txt")).toBe(false);
	});

	it("throws when no path-opaque backend ID or evidence text exists", () => {
		expect(() => stableEvidenceId({ method: "grep", source: "/docs/a.md", retrievalResultId: "/docs/a.md" })).toThrow(
			/excerpt or content/,
		);
	});

	it("returns normalized evidence references", () => {
		const ref = normalizeEvidenceRef({ method: "bm25", source: "/docs/a.md", content: "text", chunkIndex: 1 });
		expect(ref.stableEvidenceId).toMatch(/^bm25:/u);
		expect(ref.chunkIndex).toBe(1);
	});
});
