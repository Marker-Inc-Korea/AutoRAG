import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Value } from "typebox/value";
import { describe, expect, it, vi } from "vitest";
import { reportSchema } from "../../src/agent/results.ts";
import { main } from "../../src/cli/index.ts";
import * as publicApi from "../../src/index.ts";

describe("public AutoRAG-lite facade", () => {
	it("exports a loop-free factory for headless retrieval and reporting", () => {
		const factory = "createAutoRAGLite" in publicApi ? publicApi.createAutoRAGLite : undefined;
		expect(typeof factory).toBe("function");
	});
});

describe("lite report bridge", () => {
	it("rejects malformed report input without success or persistence", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-report-invalid-"));
		try {
			const configPath = join(root, "config.json");
			const memoryPath = join(root, "memory.json");
			const reportPath = join(root, "report.json");
			writeFileSync(
				configPath,
				JSON.stringify({
					searchPaths: [root],
					workspacePath: root,
					memoryPath,
					minSync: { autoInstall: false },
					jikji: false,
				}),
			);
			writeFileSync(reportPath, "{not-json");
			const error = vi.spyOn(process.stderr, "write").mockReturnValue(true);

			const code = await main([
				"lite",
				"report",
				"opaque query",
				"--input",
				reportPath,
				"--config",
				configPath,
				"--json",
			]);

			expect(code).toBe(2);
			expect(String(error.mock.calls.map(([line]) => line).join(""))).toContain('"ok":false');
			expect(existsSync(memoryPath)).toBe(false);
		} finally {
			vi.restoreAllMocks();
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("persists opaque report data for existing evidence commands", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-report-valid-"));
		try {
			const configPath = join(root, "config.json");
			const memoryPath = join(root, "memory.json");
			const reportPath = join(root, "report.json");
			writeFileSync(
				configPath,
				JSON.stringify({
					searchPaths: [root],
					workspacePath: root,
					memoryPath,
					minSync: { autoInstall: false },
					jikji: false,
				}),
			);
			writeFileSync(
				reportPath,
				JSON.stringify({
					answer: "[1] opaque result",
					results: [
						{
							number: 1,
							title: "Opaque result",
							summary: "Submitted without reading its source",
							confidence: 0.75,
							evidence: [{ excerpt: "opaque evidence" }],
						},
					],
					mapping: [
						{
							number: 1,
							source: "file:///definitely-do-not-read",
							method: "fixture",
							content: "opaque source content",
						},
					],
				}),
			);
			const output = vi.spyOn(process.stdout, "write").mockReturnValue(true);

			const reportCode = await main([
				"lite",
				"report",
				"opaque query",
				"--input",
				reportPath,
				"--config",
				configPath,
				"--json",
			]);
			expect(reportCode).toBe(0);
			const reportOutput = String(output.mock.calls.at(-1)?.[0] ?? "");
			const parsed: unknown = JSON.parse(reportOutput);
			expect(parsed).toMatchObject({ ok: true });
			expect(typeof parsed).toBe("object");
			if (typeof parsed !== "object" || parsed === null || !("sessionId" in parsed)) return;
			expect(typeof parsed.sessionId).toBe("string");
			if (typeof parsed.sessionId !== "string") return;

			const evidenceCode = await main([
				"evidence",
				parsed.sessionId,
				"--result",
				"1",
				"--config",
				configPath,
				"--json",
			]);
			expect(evidenceCode).toBe(0);
			expect(String(output.mock.calls.at(-1)?.[0] ?? "")).toContain("file:///definitely-do-not-read");
		} finally {
			vi.restoreAllMocks();
			rmSync(root, { recursive: true, force: true });
		}
	});
});

describe("reportSchema (input contract of `autorag report` / the MCP report tool)", () => {
	function validReport() {
		return {
			answer: "answer [1]",
			results: [
				{
					number: 1,
					title: "Result one",
					summary: "summary one",
					evidence: [{ excerpt: "snippet", lineNumber: 12 }],
					confidence: 0.9,
				},
			],
			mapping: [
				{
					number: 1,
					source: "/data/one.txt",
					method: "bm25",
					content: "snippet text",
					evidenceRefs: [{ method: "bm25", source: "/data/one.txt", content: "snippet text", chunkIndex: 3 }],
				},
			],
		};
	}

	it("accepts a well-formed report with an explicit number -> source mapping", () => {
		expect(Value.Check(reportSchema, validReport())).toBe(true);
	});

	it("accepts a report whose mapping entry omits the optional evidenceRefs and warnings", () => {
		const report = validReport();
		const { evidenceRefs: _evidenceRefs, ...mapping } = report.mapping[0] as Record<string, unknown>;
		expect(Value.Check(reportSchema, { ...report, mapping: [mapping] })).toBe(true);
	});

	it("requires answer, results, and mapping", () => {
		const report = validReport() as Record<string, unknown>;
		for (const key of ["answer", "results", "mapping"]) {
			const { [key]: _omitted, ...rest } = report;
			expect(Value.Check(reportSchema, rest)).toBe(false);
		}
	});

	it("rejects a confidence outside 0..1", () => {
		const report = validReport();
		expect(Value.Check(reportSchema, { ...report, results: [{ ...report.results[0], confidence: 1.5 }] })).toBe(
			false,
		);
	});

	it("rejects a non-integer result number", () => {
		const report = validReport();
		expect(Value.Check(reportSchema, { ...report, results: [{ ...report.results[0], number: 1.5 }] })).toBe(false);
	});

	it("rejects an evidence ref missing its method or source", () => {
		const report = validReport();
		expect(
			Value.Check(reportSchema, {
				...report,
				mapping: [{ ...report.mapping[0], evidenceRefs: [{ source: "/data/one.txt" }] }],
			}),
		).toBe(false);
	});
});
