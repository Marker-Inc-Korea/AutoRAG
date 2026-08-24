import { describe, expect, it } from "vitest";
import { createScanDuplicateDocumentsTool } from "../../src/agent/dupey-tool.ts";

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
});
