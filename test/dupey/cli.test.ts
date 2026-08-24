import { describe, expect, it } from "vitest";
import { DupeyCliError, scanWithDupey } from "../../src/dupey/index.ts";

describe("dupey CLI adapter", () => {
	it("runs JSON scan and normalizes the directory", async () => {
		const result = await scanWithDupey("docs", {
			run: async (args) => {
				expect(args).toEqual(["scan", expect.stringMatching(/\/docs$/), "--json"]);
				return JSON.stringify({ dir: "/docs", files: [], families: [], errors: [] });
			},
		});
		expect(result.files).toEqual([]);
	});

	it("rejects malformed output", async () => {
		await expect(scanWithDupey(".", { run: async () => "not-json" })).rejects.toBeInstanceOf(DupeyCliError);
	});
});
