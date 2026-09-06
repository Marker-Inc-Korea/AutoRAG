import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
	type AutoRAGResultsDetails,
	createEmitResultsTool,
	EMIT_AUTORAG_RESULTS_TOOL_NAME,
} from "../../src/agent/emit-results-tool.ts";
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
				JSON.stringify({ searchPaths: [root], workspacePath: root, memoryPath, minSync: false, jikji: false }),
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

	it("persists opaque report data for existing evidence and feedback commands", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-lite-report-valid-"));
		try {
			const configPath = join(root, "config.json");
			const memoryPath = join(root, "memory.json");
			const reportPath = join(root, "report.json");
			writeFileSync(
				configPath,
				JSON.stringify({ searchPaths: [root], workspacePath: root, memoryPath, minSync: false, jikji: false }),
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

			const feedbackCode = await main([
				"feedback",
				parsed.sessionId,
				"--useful",
				"1",
				"--config",
				configPath,
				"--json",
			]);
			expect(feedbackCode).toBe(0);
			expect(String(output.mock.calls.at(-1)?.[0] ?? "")).toContain('"applied":true');
		} finally {
			vi.restoreAllMocks();
			rmSync(root, { recursive: true, force: true });
		}
	});
});

describe("createEmitResultsTool", () => {
	it("returns typed details with terminate and forwards them to the capture sink", async () => {
		let captured: AutoRAGResultsDetails | undefined;
		const tool = createEmitResultsTool((details) => {
			captured = details;
		});

		expect(tool.name).toBe(EMIT_AUTORAG_RESULTS_TOOL_NAME);

		const result = await tool.execute("call-1", {
			answer: "the answer",
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
					method: "grep",
					content: "snippet",
					evidenceRefs: [
						{
							method: "grep",
							source: "/data/one.txt",
							content: "snippet",
							retrieverMix: ["bm25", "minsync"],
							parserType: "pdf",
							documentType: "manual",
							documentArea: "billing",
							evidenceType: "policy",
							evidenceLocation: "page 12",
							confidence: 0.88,
						},
					],
				},
			],
			warnings: [],
		});

		expect(result.terminate).toBe(true);
		expect(result.details.answer).toBe("the answer");
		expect(result.details.results[0].evidence[0]).toEqual({ excerpt: "snippet", lineNumber: 12 });
		expect(result.details.mapping[0]).toEqual({
			number: 1,
			source: "/data/one.txt",
			method: "grep",
			content: "snippet",
			evidenceRefs: [
				{
					method: "grep",
					source: "/data/one.txt",
					content: "snippet",
					retrieverMix: ["bm25", "minsync"],
					parserType: "pdf",
					documentType: "manual",
					documentArea: "billing",
					evidenceType: "policy",
					evidenceLocation: "page 12",
					confidence: 0.88,
				},
			],
		});
		expect(captured).toBe(result.details);
	});

	it("omits lineNumber from evidence when not provided", async () => {
		const tool = createEmitResultsTool(() => {});
		const result = await tool.execute("call-2", {
			answer: "a",
			results: [{ number: 1, title: "t", summary: "s", evidence: [{ excerpt: "e" }], confidence: 1 }],
			mapping: [{ number: 1, source: "/x", method: "grep", content: "e" }],
		});
		expect(result.details.results[0].evidence[0]).toEqual({ excerpt: "e" });
		expect(result.details.warnings).toEqual([]);
	});
});
