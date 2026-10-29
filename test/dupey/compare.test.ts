import { describe, expect, it } from "vitest";
import { compareWithDupey, DupeyCliError } from "../../src/dupey/index.ts";

function table(rows: Record<string, string>): string {
	return Object.entries(rows)
		.map(([key, value]) => `${key}\t${value}`)
		.join("\n");
}

describe("compareWithDupey", () => {
	it("passes both files as absolute paths", async () => {
		const seen: string[][] = [];
		await compareWithDupey("a.docx", "b.hwpx", {
			run: async (args) => {
				seen.push([...args]);
				return table({ exact_equal: "true", near_score: "1.0000", jaccard: "1.0000" });
			},
		});
		expect(seen[0]?.[0]).toBe("compare");
		expect(seen[0]?.[1]).toMatch(/a\.docx$/);
		expect(seen[0]?.[2]).toMatch(/b\.hwpx$/);
	});

	it("classifies equal canonical text as exact", async () => {
		const result = await compareWithDupey("a", "b", {
			run: async () =>
				table({
					exact_equal: "true",
					near_score: "1.0000",
					jaccard: "1.0000",
					containment_a_in_b: "1.0000",
					containment_b_in_a: "1.0000",
				}),
		});
		expect(result).toMatchObject({ relation: "exact", nearScore: 1, jaccard: 1 });
	});

	it("classifies a high MinHash score as near and a low one as different", async () => {
		const near = await compareWithDupey("a", "b", {
			run: async () => table({ exact_equal: "false", near_score: "0.9922", jaccard: "0.9933" }),
		});
		expect(near).toMatchObject({ relation: "near", nearScore: 0.9922 });
		const different = await compareWithDupey("a", "b", {
			run: async () => table({ exact_equal: "false", near_score: "0.1016", jaccard: "0.1605" }),
		});
		expect(different).toMatchObject({ relation: "different", nearScore: 0.1016 });
	});

	it("reports formats dupey cannot read as unsupported, not as an error", async () => {
		const result = await compareWithDupey("a.docx", "a.html", {
			run: async () => {
				throw new DupeyCliError(
					"dupey failed with exit code 1: Error: extract a.html\n\nCaused by:\n    unsupported format: a.html",
				);
			},
		});
		expect(result).toEqual({ relation: "unsupported" });
	});

	it("rethrows other dupey failures and malformed output", async () => {
		await expect(
			compareWithDupey("a", "b", {
				run: async () => {
					throw new DupeyCliError("dupey timed out.");
				},
			}),
		).rejects.toThrow("timed out");
		await expect(compareWithDupey("a", "b", { run: async () => "garbage" })).rejects.toBeInstanceOf(DupeyCliError);
	});
});
