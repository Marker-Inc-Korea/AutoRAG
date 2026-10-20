import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	CONVERT_FILES_TOOL_NAME,
	type ConvertFilesDetails,
	createConvertFilesTool,
	createListFileConversionsTool,
	LIST_FILE_CONVERSIONS_TOOL_NAME,
} from "../../src/agent/convert-tool.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;
beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-convert-agent-"));
});
afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function details(overrides: Partial<ConvertFilesDetails> = {}): ConvertFilesDetails {
	return {
		result: {
			preset: "HWPX",
			target: "hwpx",
			converted: 2,
			skipped: 0,
			failed: 1,
			files: [
				{ source: "/docs/a.docx", state: "done", outputs: ["/docs/a.hwpx"] },
				{ source: "/docs/b.md", state: "done", outputs: ["/docs/b.hwpx"], notice: "tables simplified" },
				{ source: "/docs/c.png", state: "failed", outputs: [], error: "improper image header" },
			],
		},
		links: [
			{ source: "/docs/a.docx", output: "/docs/a.hwpx", dupey: { relation: "exact", nearScore: 1, jaccard: 1 } },
			{ source: "/docs/b.md", output: "/docs/b.hwpx", dupey: { relation: "near", nearScore: 0.99, jaccard: 0.98 } },
		],
		...overrides,
	};
}

describe("convert_files tool", () => {
	it("is named for what it does and forwards every option, overwrite included", async () => {
		const seen: unknown[] = [];
		const tool = createConvertFilesTool({
			async convertFiles(request) {
				seen.push(request);
				return details();
			},
			async listFileConversions() {
				return "";
			},
		});
		expect(tool.name).toBe("convert_files");
		expect(CONVERT_FILES_TOOL_NAME).toBe("convert_files");
		await tool.execute("c1", {
			inputs: ["/docs/a.docx"],
			to: "hwpx",
			outputDir: "/out",
			recursive: true,
			quality: 70,
			overwrite: true,
			keepDates: true,
		});
		expect(seen).toEqual([
			{
				inputs: ["/docs/a.docx"],
				to: "hwpx",
				outputDir: "/out",
				recursive: true,
				quality: 70,
				overwrite: true,
				keepDates: true,
			},
		]);
	});

	it("renders per-file results with the dupey family verdict the model needs", async () => {
		const tool = createConvertFilesTool({
			async convertFiles() {
				return details();
			},
			async listFileConversions() {
				return "";
			},
		});
		const result = await tool.execute("c1", { inputs: ["/docs"], to: "hwpx" });
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n");
		expect(text).toContain("converted=2 skipped=0 failed=1");
		expect(text).toContain("/docs/a.docx -> /docs/a.hwpx");
		expect(text).toContain("dupey: exact");
		expect(text).toContain("dupey: near");
		expect(text).toContain("tables simplified");
		expect(text).toContain("/docs/c.png");
		expect(text).toContain("improper image header");
		expect(result.isError).toBeUndefined();
	});

	it("flags the result as an error only when nothing was converted", async () => {
		const failed = details({
			result: {
				preset: "PDF",
				target: "pdf",
				converted: 0,
				skipped: 0,
				failed: 1,
				files: [{ source: "/docs/c.png", state: "failed", outputs: [], error: "bad" }],
			},
			links: [],
		});
		const tool = createConvertFilesTool({
			async convertFiles() {
				return failed;
			},
			async listFileConversions() {
				return "";
			},
		});
		const result = await tool.execute("c1", { inputs: ["/docs/c.png"], to: "pdf" });
		expect(result.isError).toBe(true);
	});

	it("explains what each dupey verdict means for the corpus", async () => {
		const tool = createConvertFilesTool({
			async convertFiles() {
				return details({
					result: {
						preset: "PDF",
						target: "pdf",
						converted: 1,
						skipped: 0,
						failed: 0,
						files: [{ source: "/a.docx", state: "done", outputs: ["/a.pdf", "/a.html", "/a.odt"] }],
					},
					links: [
						{
							source: "/a.docx",
							output: "/a.pdf",
							dupey: { relation: "different", nearScore: 0.1, jaccard: 0.16 },
						},
						{ source: "/a.docx", output: "/a.html", dupey: { relation: "unsupported" } },
						{
							source: "/a.docx",
							output: "/a.odt",
							dupey: { relation: "unavailable", reason: "dupey not installed" },
						},
					],
				});
			},
			async listFileConversions() {
				return "";
			},
		});
		const result = await tool.execute("c1", { inputs: ["/a.docx"], to: "pdf" });
		const text = result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n");
		expect(text).toContain("dupey: different");
		expect(text).toContain("not grouped");
		expect(text).toContain("dupey: unsupported");
		expect(text).toContain("dupey: unavailable");
	});
});

describe("list_file_conversions tool", () => {
	it("lists formats or one format's targets", async () => {
		const asked: Array<string | undefined> = [];
		const tool = createListFileConversionsTool({
			async convertFiles() {
				return details();
			},
			async listFileConversions(format) {
				asked.push(format);
				return "HEIC converts to:\n  png, jpg";
			},
		});
		expect(tool.name).toBe("list_file_conversions");
		expect(LIST_FILE_CONVERSIONS_TOOL_NAME).toBe("list_file_conversions");
		const result = await tool.execute("c1", { from: "heic" });
		expect(asked).toEqual(["heic"]);
		expect(result.content[0]).toMatchObject({ type: "text", text: expect.stringContaining("png, jpg") });
	});
});

describe("agent wiring", () => {
	it("registers both conversion tools by default", () => {
		const agent = new AutoRAGAgent({ searchPaths: [FIXTURE_DIR], memoryPath: join(tmpDir, "memory.json") });
		const names = (
			agent as unknown as { innerAgent: { state: { tools: Array<{ name: string }> } } }
		).innerAgent.state.tools.map((tool) => tool.name);
		expect(names).toContain("convert_files");
		expect(names).toContain("list_file_conversions");
	});

	it("can disable them", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			filee: false,
		});
		const names = (
			agent as unknown as { innerAgent: { state: { tools: Array<{ name: string }> } } }
		).innerAgent.state.tools.map((tool) => tool.name);
		expect(names).not.toContain("convert_files");
		expect(names).not.toContain("list_file_conversions");
	});

	it("keeps the tool names reserved so a caller tool cannot shadow them", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [
				{
					name: "convert_files",
					label: "x",
					description: "x",
					parameters: { type: "object", properties: {} } as never,
					execute: async () => ({ content: [], details: undefined }),
				},
			],
		});
		expect((agent as unknown as { droppedCallerToolNames: string[] }).droppedCallerToolNames).toContain(
			"convert_files",
		);
	});

	it("announces the tools in the system prompt, and the dupey interplay", () => {
		const prompt = buildSystemPrompt({
			toolNames: ["bash", "convert_files", "list_file_conversions", "scan_duplicate_documents"],
			modelId: "test-model",
		});
		expect(prompt).toContain("**convert_files**");
		expect(prompt).toContain("**list_file_conversions**");
		expect(prompt).toContain("overwrite");
		const without = buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" });
		expect(without).not.toContain("convert_files");
	});
});
