import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-search-documents-test-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("AutoRAGAgent searchDocuments", () => {
	it("returns structured curated results when retrieval finds documents", async () => {
		// Given: a document collection with meeting notes and isolated memory.
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
		});

		// When: a parent agent calls the structured search API.
		const response = await agent.searchDocuments("Meeting", { topK: 1 });

		// Then: the response exposes numbered citable units without source filesystem paths.
		expect(response.query).toBe("Meeting");
		expect(response.sessionId).toMatch(/[0-9a-f-]{36}/);
		expect(response.results).toHaveLength(1);
		const [result] = response.results;
		expect(result).toEqual({
			number: 1,
			title: "Meeting notes from 2024-01-15",
			summary: "Meeting notes from 2024-01-15",
			evidence: [{ excerpt: "Meeting notes from 2024-01-15", lineNumber: 1 }],
			confidence: 1,
			feedbackId: `${response.sessionId}:1`,
		});
		expect(response.answer).toContain("[1] Meeting notes from 2024-01-15");
		expect(response.searched).toBe(1);
		expect(response.warnings).toEqual([]);
		expect(JSON.stringify(response)).not.toContain("/Users/");
		expect(JSON.stringify(response)).not.toContain("test/fixtures/sample-project");
	});

	it("returns an empty structured response when query is blank", async () => {
		// Given: a blank query and a missing source path that would fail if retrieval ran.
		const agent = new AutoRAGAgent({
			searchPaths: [join(tmpDir, "missing-source")],
			memoryPath: join(tmpDir, "memory.json"),
			workspacePath: tmpDir,
		});

		// When: the parent agent calls the structured search API.
		const response = await agent.searchDocuments("   ", { topK: 1 });

		// Then: no retrieval occurs and the caller receives a typed warning.
		expect(response.query).toBe("");
		expect(response.results).toEqual([]);
		expect(response.answer).toBe("");
		expect(response.searched).toBe(0);
		expect(response.warnings).toEqual(["empty-query"]);
	});

	it("registers search result numbers for feedback resolution", async () => {
		// Given: a structured search result backed by pending memory.
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
			workspacePath: tmpDir,
		});
		const response = await agent.searchDocuments("Meeting", { topK: 1 });

		// When: the caller marks the numbered result useful.
		agent.recordFeedbackByNumbers(response.sessionId, [1]);

		// Then: the stored feedback resolves the pending search attempt.
		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries().map((entry) => entry.outcome)).toEqual(["useful"]);
	});

	it("does not apply blank-query feedback to the previous search", async () => {
		// Given: a completed search followed by a blank-query structured response.
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
			workspacePath: tmpDir,
		});
		await agent.searchDocuments("Meeting", { topK: 1 });
		const blank = await agent.searchDocuments("   ", { topK: 1 });

		// When: the caller submits feedback for the blank-query session.
		agent.submitFeedback(blank.sessionId, true);

		// Then: the prior search remains pending instead of being resolved by the blank session.
		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries().map((entry) => entry.outcome)).toEqual(["pending"]);
	});

	it("makes a blank-query response the current session for default feedback calls", async () => {
		// Given: a completed search followed by a blank-query structured response.
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
			workspacePath: tmpDir,
		});
		await agent.searchDocuments("Meeting", { topK: 1 });
		await agent.searchDocuments("   ", { topK: 1 });

		// When: the caller uses default-session feedback and registry helpers.
		agent.submitFeedback(undefined, true);

		// Then: the blank session is current and prior search memory remains pending.
		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(agent.getResultRegistry().size).toBe(0);
		expect(memory.getEntries().map((entry) => entry.outcome)).toEqual(["pending"]);
	});

	it("keeps feedback state independent for each returned feedback id", async () => {
		// Given: a search that returns multiple independently citable units.
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
			workspacePath: tmpDir,
		});
		const response = await agent.searchDocuments("function|Meeting", { topK: 3 });

		// When: the caller marks the first and second numbered units in separate calls.
		agent.recordFeedbackByNumbers(response.sessionId, [1]);
		agent.recordFeedbackByNumbers(response.sessionId, [2]);

		// Then: both feedback IDs resolve independently.
		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		expect(memory.getEntries().map((entry) => entry.outcome)).toEqual(["useful", "useful", "pending"]);
	});
});
