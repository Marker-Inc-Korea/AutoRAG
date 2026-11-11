import { describe, expect, it } from "vitest";
import { createScanDuplicateDocumentsTool, type ScanDuplicateDocumentsDetails } from "../../src/agent/dupey-tool.ts";

describe("scan_duplicate_documents tool", () => {
	it("returns duplicate families without changing files", async () => {
		const tool = createScanDuplicateDocumentsTool({
			async scanDuplicateDocuments() {
				return {
					scans: [{ dir: "/docs", files: [], errors: [], families: [] }],
					familyCount: 0,
					exactDuplicateCount: 0,
				};
			},
		});
		const result = await tool.execute("call-1", {});
		expect(tool.name).toBe("scan_duplicate_documents");
		expect(result.details).toMatchObject({ familyCount: 0, exactDuplicateCount: 0 });
		expect(result.content[0]).toMatchObject({ type: "text" });
	});

	it("surfaces family members, relation, scores, and the selected candidate to the model", async () => {
		const details: ScanDuplicateDocumentsDetails = {
			scans: [
				{
					dir: "/docs",
					files: [],
					errors: [],
					families: [
						{
							id: 0,
							relation: "exact",
							files: ["/docs/internal-newest.txt", "/docs/fs-newest.txt"],
							members: [
								{
									path: "/docs/internal-newest.txt",
									relation: "exact",
									near_score: 1,
									jaccard: 1,
									containment: 1,
								},
								{ path: "/docs/fs-newest.txt", relation: "exact", near_score: 1, jaccard: 1, containment: 1 },
							],
							edges: [],
							pick: {
								ranked: [
									{
										path: "/docs/internal-newest.txt",
										rank: 1,
										score: 1,
										reasons: [{ name: "internal_modified", detail: "document timestamp is newest" }],
									},
									{ path: "/docs/fs-newest.txt", rank: 2, score: 0, reasons: [] },
								],
							},
						},
						{
							id: 1,
							relation: "near",
							files: ["/docs/draft-a.txt", "/docs/draft-b.txt"],
							members: [
								{ path: "/docs/draft-a.txt", relation: "near", near_score: 0.93, jaccard: 0.93 },
								{ path: "/docs/draft-b.txt", relation: "near", near_score: 0.93, jaccard: 0.93 },
							],
							edges: [],
						},
					],
				},
			],
			familyCount: 2,
			exactDuplicateCount: 2,
		};
		const tool = createScanDuplicateDocumentsTool({
			async scanDuplicateDocuments() {
				return details;
			},
		});

		const result = await tool.execute("call-1", {});
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n");

		// Exact family: both member paths, the relation, and dupey's selected candidate + reason.
		expect(text).toContain("/docs/internal-newest.txt");
		expect(text).toContain("/docs/fs-newest.txt");
		expect(text).toContain("exact");
		expect(text).toContain("selected: /docs/internal-newest.txt");
		expect(text).toContain("internal_modified");

		// Near family: paths, relation, and similarity score.
		expect(text).toContain("/docs/draft-a.txt");
		expect(text).toContain("/docs/draft-b.txt");
		expect(text).toContain("near");
		expect(text).toContain("0.93");
	});
});
