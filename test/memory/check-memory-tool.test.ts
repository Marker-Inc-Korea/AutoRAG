import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type CheckMemoryDetails, createCheckMemoryTool } from "../../src/memory/check-memory-tool.ts";
import { loadMemoryContext } from "../../src/memory/context.ts";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
import { renderMemoryContext } from "../../src/memory/renderer.ts";

let tmpDir: string;
let memoryPath: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-checkmem-"));
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

let counter = 0;

function record(overrides: Partial<JudgedEvidenceRecord> = {}): JudgedEvidenceRecord {
	counter += 1;
	const sessionId = overrides.sessionId ?? `session-${counter}`;
	return {
		id: `${sessionId}:posix:e${counter}`,
		sessionId,
		conversationId: "current",
		question: "typescript files",
		searchQuery: "typescript files",
		method: "posix",
		source: `/src/${counter}.ts`,
		stableEvidenceId: `posix:e${counter}`,
		resultNumber: 1,
		title: `Title ${counter}`,
		excerpt: `Excerpt ${counter}`,
		probability: 0.9,
		createdAt: counter,
		...overrides,
	};
}

function textOf(result: AgentToolResult<CheckMemoryDetails>): string {
	const [part] = result.content;
	if (part === undefined || part.type !== "text") throw new Error("expected text content");
	return part.text;
}

describe("createCheckMemoryTool", () => {
	it("is named check_memory and reports section counts in details", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence([
			record({ conversationId: "current", question: "typescript files", title: "Current" }),
			record({ conversationId: "past", question: "typescript files", title: "Past" }),
		]);

		const tool = createCheckMemoryTool(memory, () => ({ conversationId: "current" }));
		expect(tool.name).toBe("check_memory");
		expect(tool.label).toBe("Check Memory");

		const result = await tool.execute("call-1", { query: "typescript files" });
		expect(result.details).toEqual({ currentCount: 1, similarCount: 1, insightCount: 0 });
	});

	it("returns exactly the renderMemoryContext text for the same sections", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence([
			record({ conversationId: "current", question: "typescript files", title: "Current" }),
			record({ conversationId: "past", question: "typescript files", searchQuery: "ts files", title: "Past" }),
		]);
		const query = "typescript files";

		const tool = createCheckMemoryTool(memory, () => ({ conversationId: "current" }));
		const result = await tool.execute("call-2", { query });
		const context = await loadMemoryContext(memory, query, { conversationId: "current" });
		const expected = renderMemoryContext({
			current: context.current,
			similar: context.similar,
			insights: memory.getInsights(query),
		});

		expect(textOf(result)).toBe(expected);
	});

	it("forwards the visibility gate so hidden evidence is never shown", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence([
			record({ conversationId: "current", question: "typescript files", title: "Visible" }),
			record({ conversationId: "current", question: "typescript files", title: "Hidden" }),
		]);

		const tool = createCheckMemoryTool(memory, () => ({
			conversationId: "current",
			isVisible: (candidate) => candidate.title !== "Hidden",
		}));
		const result = await tool.execute("call-4", { query: "typescript files" });

		expect(result.details).toEqual({ currentCount: 1, similarCount: 0, insightCount: 0 });
		expect(textOf(result)).toContain("Visible");
		expect(textOf(result)).not.toContain("Hidden");
	});

	it("calls the options provider on every execute", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence([
			record({ conversationId: "current", question: "typescript files", title: "Current" }),
			record({ conversationId: "other", question: "typescript files", searchQuery: "ts files", title: "Other" }),
		]);
		let conversationId = "current";
		let calls = 0;
		const tool = createCheckMemoryTool(memory, () => {
			calls += 1;
			return { conversationId };
		});

		await tool.execute("call-5", { query: "typescript files" });
		expect(calls).toBe(1);

		conversationId = "other";
		const second = await tool.execute("call-6", { query: "typescript files" });
		expect(calls).toBe(2);
		expect(second.details.currentCount).toBe(1);
		const currentSection = textOf(second).split("## Similar Past Questions")[0] ?? "";
		expect(currentSection).toContain('"Other"');
		expect(currentSection).not.toContain('"Current"');
	});

	it("refuses nothing and returns the sentinel on a cold start", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const result = await createCheckMemoryTool(memory, () => ({ conversationId: "current" })).execute("call-3", {
			query: "anything",
		});

		expect(textOf(result)).toBe("No retrieval memory available.");
		expect(result.details).toEqual({ currentCount: 0, similarCount: 0, insightCount: 0 });
	});
});
