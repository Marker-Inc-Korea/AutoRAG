import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Value } from "typebox/value";
import { describe, expect, it, vi } from "vitest";
import {
	type AutoRAGResultsDetails,
	createEmitResultsTool,
	EMIT_AUTORAG_RESULTS_TOOL_NAME,
	emitResultsSchema,
} from "../../src/agent/emit-results-tool.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
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
	function ledgerWith(source = "/data/one.txt", content = "snippet text") {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("search_all_documents", {
			id: "bm25:one",
			source,
			content,
			score: 1,
			metadata: { method: "bm25", parserType: "pdf", chunkIndex: 3 },
		});
		return { ledger, id };
	}

	it("accepts only answer/results/warnings: the model never writes a mapping", () => {
		const tool = createEmitResultsTool(() => {}, { ledger: new EvidenceLedger(), allowLocalFiles: false });
		expect(Object.keys(tool.parameters.properties ?? {}).sort()).toEqual(["answer", "results", "warnings"]);
	});

	it("builds the mapping from harness-held evidence, terminates, and forwards details", async () => {
		const { ledger, id } = ledgerWith();
		let captured: AutoRAGResultsDetails | undefined;
		const tool = createEmitResultsTool(
			(details) => {
				captured = details;
			},
			{ ledger, allowLocalFiles: false },
		);
		expect(tool.name).toBe(EMIT_AUTORAG_RESULTS_TOOL_NAME);

		const result = await tool.execute("call-1", {
			answer: "the answer [1]",
			results: [
				{
					number: 1,
					title: "Result one",
					summary: "summary one",
					evidence: [{ excerpt: "snippet", lineNumber: 12 }],
					confidence: 0.9,
					refs: [id],
				},
			],
		});

		expect(result.terminate).toBe(true);
		expect(result.details.results[0]).toEqual({
			number: 1,
			title: "Result one",
			summary: "summary one",
			evidence: [{ excerpt: "snippet", lineNumber: 12 }],
			confidence: 0.9,
		});
		expect(result.details.mapping).toEqual([
			{
				number: 1,
				source: "/data/one.txt",
				method: "bm25",
				content: "snippet text",
				evidenceRefs: [
					{
						method: "bm25",
						source: "/data/one.txt",
						content: "snippet text",
						retrieverMix: ["bm25"],
						retrievalResultId: "bm25:one",
						chunkIndex: 3,
						parserType: "pdf",
					},
				],
			},
		]);
		expect(result.details.warnings).toEqual([]);
		expect(captured).toBe(result.details);
	});

	it("uses the recorded chunk even when the model paraphrases the excerpt", async () => {
		const { ledger, id } = ledgerWith("/data/one.txt", "the exact chunk text");
		const tool = createEmitResultsTool(() => {}, { ledger, allowLocalFiles: false });
		const result = await tool.execute("call-2", {
			answer: "a [1]",
			results: [
				{ number: 1, title: "t", summary: "s", evidence: [{ excerpt: "paraphrase" }], confidence: 1, refs: [id] },
			],
		});
		expect(result.details.mapping[0]?.content).toBe("the exact chunk text");
		expect(result.details.results[0]?.evidence[0]).toEqual({ excerpt: "paraphrase" });
	});

	it("rejects a result citing evidence no tool returned, so the model re-emits", async () => {
		const tool = createEmitResultsTool(() => {}, { ledger: new EvidenceLedger(), allowLocalFiles: false });
		await expect(
			tool.execute("call-3", {
				answer: "a [1]",
				results: [
					{ number: 1, title: "t", summary: "s", evidence: [], confidence: 1, refs: ["/docs/invented.txt"] },
				],
			}),
		).rejects.toThrow(/result 1.*invented\.txt/su);
	});

	it("accepts a real local file the model opened itself when local files are allowed", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-emit-local-"));
		try {
			const file = join(root, "opened.txt");
			writeFileSync(file, "x");
			const tool = createEmitResultsTool(() => {}, { ledger: new EvidenceLedger(), allowLocalFiles: true });
			const result = await tool.execute("call-4", {
				answer: "a [1]",
				results: [
					{ number: 1, title: "t", summary: "s", evidence: [{ excerpt: "quoted" }], confidence: 1, refs: [file] },
				],
			});
			expect(result.details.mapping[0]).toMatchObject({ source: file, method: "bash", content: "quoted" });
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("still rejects citations that point at no result and duplicate result numbers", async () => {
		const { ledger, id } = ledgerWith();
		const tool = createEmitResultsTool(() => {}, { ledger, allowLocalFiles: false });
		await expect(
			tool.execute("call-5", {
				answer: "see [2]",
				results: [{ number: 1, title: "t", summary: "s", evidence: [], confidence: 1, refs: [id] }],
			}),
		).rejects.toThrow(/\[2\]/u);
		await expect(
			tool.execute("call-6", {
				answer: "a [1]",
				results: [
					{ number: 1, title: "a", summary: "s", evidence: [], confidence: 1, refs: [id] },
					{ number: 1, title: "b", summary: "s", evidence: [], confidence: 1, refs: [id] },
				],
			}),
		).rejects.toThrow(/repeat/u);
	});

	it("requires at least one evidence ref per result in the schema", () => {
		const result = { number: 1, title: "t", summary: "s", evidence: [], confidence: 1 };
		expect(Value.Check(emitResultsSchema, { answer: "a [1]", results: [{ ...result, refs: ["e1"] }] })).toBe(true);
		expect(Value.Check(emitResultsSchema, { answer: "a [1]", results: [{ ...result, refs: [] }] })).toBe(false);
		expect(Value.Check(emitResultsSchema, { answer: "a [1]", results: [result] })).toBe(false);
	});

	it("omits lineNumber from evidence when not provided", async () => {
		const { ledger, id } = ledgerWith();
		const tool = createEmitResultsTool(() => {}, { ledger, allowLocalFiles: false });
		const result = await tool.execute("call-7", {
			answer: "a",
			results: [{ number: 1, title: "t", summary: "s", evidence: [{ excerpt: "e" }], confidence: 1, refs: [id] }],
		});
		expect(result.details.results[0].evidence[0]).toEqual({ excerpt: "e" });
		expect(result.details.warnings).toEqual([]);
	});
});
