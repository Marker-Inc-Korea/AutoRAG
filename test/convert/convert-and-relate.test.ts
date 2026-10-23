import { describe, expect, it } from "vitest";
import { convertAndRelate, type FileeConvertResult } from "../../src/convert/index.ts";
import { DupeyCliError } from "../../src/dupey/index.ts";

function result(files: FileeConvertResult["files"]): FileeConvertResult {
	const converted = files.filter((file) => file.state === "done").length;
	return { preset: "X", target: "x", converted, skipped: 0, failed: files.length - converted, files };
}

const compareText = (rows: Record<string, string>) =>
	Object.entries(rows)
		.map(([key, value]) => `${key}\t${value}`)
		.join("\n");

describe("convertAndRelate", () => {
	it("asks dupey whether each output would join its source's family", async () => {
		const compared: string[][] = [];
		const out = await convertAndRelate(
			{ inputs: ["/in"], to: "hwpx" },
			{
				filee: {
					run: async () => ({
						stdout: JSON.stringify(
							result([
								{ source: "/in/a.docx", state: "done", outputs: ["/in/a.hwpx"] },
								{ source: "/in/b.docx", state: "done", outputs: ["/in/b.hwpx"] },
							]),
						),
						stderr: "",
						exitCode: 0,
					}),
				},
				dupey: {
					run: async (args) => {
						compared.push([...args]);
						return args[1]?.endsWith("a.docx")
							? compareText({ exact_equal: "true", near_score: "1.0000", jaccard: "1.0000" })
							: compareText({ exact_equal: "false", near_score: "0.2000", jaccard: "0.1000" });
					},
				},
			},
		);
		expect(compared).toHaveLength(2);
		expect(out.links).toEqual([
			expect.objectContaining({
				source: "/in/a.docx",
				output: "/in/a.hwpx",
				dupey: expect.objectContaining({ relation: "exact" }),
			}),
			expect.objectContaining({
				source: "/in/b.docx",
				output: "/in/b.hwpx",
				dupey: expect.objectContaining({ relation: "different" }),
			}),
		]);
	});

	it("skips failed files and never compares a skipped one", async () => {
		let compares = 0;
		const out = await convertAndRelate(
			{ inputs: ["/in"], to: "pdf" },
			{
				filee: {
					run: async () => ({
						stdout: JSON.stringify(
							result([
								{ source: "/in/a.docx", state: "failed", outputs: [], error: "bad" },
								{ source: "/in/b.docx", state: "skipped", outputs: [] },
							]),
						),
						stderr: "",
						exitCode: 1,
					}),
				},
				dupey: {
					run: async () => {
						compares += 1;
						return "";
					},
				},
			},
		);
		expect(compares).toBe(0);
		expect(out.links).toEqual([]);
	});

	it("marks formats dupey cannot read as unsupported and keeps converting", async () => {
		const out = await convertAndRelate(
			{ inputs: ["/in/a.docx"], to: "html" },
			{
				filee: {
					run: async () => ({
						stdout: JSON.stringify(result([{ source: "/in/a.docx", state: "done", outputs: ["/in/a.html"] }])),
						stderr: "",
						exitCode: 0,
					}),
				},
				dupey: {
					run: async () => {
						throw new DupeyCliError("dupey failed with exit code 1: unsupported format: a.html");
					},
				},
			},
		);
		expect(out.links[0]?.dupey).toEqual({ relation: "unsupported" });
	});

	it("degrades to unavailable when dupey itself cannot run", async () => {
		const out = await convertAndRelate(
			{ inputs: ["/in/a.docx"], to: "pdf" },
			{
				filee: {
					run: async () => ({
						stdout: JSON.stringify(result([{ source: "/in/a.docx", state: "done", outputs: ["/in/a.pdf"] }])),
						stderr: "",
						exitCode: 0,
					}),
				},
				dupey: {
					run: async () => {
						throw new DupeyCliError("Unable to run dupey: spawn dupey ENOENT");
					},
				},
			},
		);
		expect(out.result.converted).toBe(1);
		expect(out.links[0]?.dupey).toMatchObject({ relation: "unavailable", reason: expect.stringContaining("dupey") });
	});

	it("does not call dupey at all when it is disabled", async () => {
		const out = await convertAndRelate(
			{ inputs: ["/in/a.docx"], to: "pdf" },
			{
				filee: {
					run: async () => ({
						stdout: JSON.stringify(result([{ source: "/in/a.docx", state: "done", outputs: ["/in/a.pdf"] }])),
						stderr: "",
						exitCode: 0,
					}),
				},
				dupey: false,
			},
		);
		expect(out.links).toEqual([]);
	});

	it("propagates a filee failure unchanged", async () => {
		await expect(
			convertAndRelate(
				{ inputs: ["/in"], to: "nope" },
				{
					filee: { run: async () => ({ stdout: "", stderr: "filee: unknown format 'nope'.", exitCode: 2 }) },
					dupey: false,
				},
			),
		).rejects.toThrow("unknown format 'nope'");
	});
});
