import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createCheckMemoryTool } from "../../src/memory/check-memory-tool.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";

let tmpDir: string;
let memoryPath: string;

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-checkmem-test-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("createCheckMemoryTool", () => {
	it("returns summary with recommendation for matching query", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "posix", true);

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "find typescript files" });

		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("Retrieval Memory");
		expect(text).toContain("posix");
		expect(text).toContain("advisory");
	});

	it("returns 'No retrieval history' for cold start", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "anything" });

		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("No retrieval memory hints available.");
	});

	it("does not modify memory state", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("test", "posix", true);
		const countBefore = memory.getSignalCount();

		const tool = createCheckMemoryTool(memory);
		await tool.execute("test-call", { query: "test" });

		expect(memory.getSignalCount()).toBe(countBefore);
	});

	it("details contains signalCount and topMethod", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("query", "posix", true);

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "query" });

		expect(result.details).toBeDefined();
		expect(result.details!.signalCount).toBe(1);
		expect(result.details!.topMethod).toBe("posix");
	});

	it("returns durable insights with details", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 600; i++) memory.recordFeedback("photo archive lookup", "posix", true);
		memory.save();

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "photo archive lookup" });

		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("Long-Term Retrieval Insights");
		expect(text).toContain("photo archive lookup");
		expect(result.details!.insightCount).toBe(1);
	});

	it("renders result-level document and evidence preferences", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordCuratedResultsSession({
			sessionId: "context-session",
			query: "refund policy",
			results: [
				{
					number: 1,
					title: "Refunds",
					summary: "Policy",
					content: "Refund policy",
					method: "bm25",
					source: "/docs/refunds.md",
					confidence: 0.9,
					evidenceRefs: [
						normalizeSessionEvidenceRef({
							method: "bm25",
							source: "/docs/refunds.md",
							content: "Refund policy",
							documentArea: "billing",
							documentType: "policy",
							evidenceType: "rule",
							evidenceLocation: "section 4",
							parserType: "markdown",
							retrieverMix: ["bm25", "minsync"],
							confidence: 0.8,
						}),
					],
				},
			],
		});
		memory.recordFeedbackByIds([{ feedbackId: "context-session:1", useful: true }]);

		const result = await createCheckMemoryTool(memory).execute("context-call", { query: "refund policy" });
		const text = (result.content[0] as { type: "text"; text: string }).text;

		expect(text).toContain('Document areas: "billing"');
		expect(text).toContain('Document types: "policy"');
		expect(text).toContain('Evidence types: "rule"');
		expect(text).toContain('Evidence locations: "section 4"');
		expect(text).toContain('Parser types: "markdown"');
		expect(text).toContain('Retriever mix: "bm25", "minsync"');
		expect(result.details!.contextHintCount).toBe(7);
	});

	it("renders negative result-context preferences as disfavored data", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordCuratedResultsSession({
			sessionId: "negative-context",
			query: "refund policy",
			results: [
				{
					number: 1,
					title: "Refunds",
					summary: "Policy",
					content: "Refund policy",
					method: "bm25",
					source: "opaque:refunds",
					evidenceRefs: [
						normalizeSessionEvidenceRef({
							method: "bm25",
							source: "opaque:refunds",
							content: "Refund policy",
							documentArea: "billing",
							evidenceType: "rule",
						}),
					],
				},
			],
		});
		memory.recordFeedbackByIds([{ feedbackId: "negative-context:1", useful: false }]);

		const result = await createCheckMemoryTool(memory).execute("negative-call", { query: "refund policy" });
		const text = (result.content[0] as { type: "text"; text: string }).text;

		expect(text).toContain('Disfavored document areas: "billing"');
		expect(text).toContain('Disfavored evidence types: "rule"');
	});
});
