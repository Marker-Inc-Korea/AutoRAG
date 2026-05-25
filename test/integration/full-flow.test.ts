import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-integration-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("Full flow integration", () => {
	it("agent creates with manifests and includes them in system prompt", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			manifestDir: "test/fixtures/manifests",
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("codebase-vectors");
		expect(prompt).toContain("documentation-index");
	});

	it("feedback → memory → priority flow works end-to-end", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});

		agent["lastQuery"] = "search typescript code";
		agent.submitFeedback(true);
		agent.submitFeedback(true);
		agent.submitFeedback(true);

		expect(existsSync(memPath)).toBe(true);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const priority = memory.getMethodPriority("search typescript code");
		expect(priority.length).toBeGreaterThan(0);
		expect(priority[0].method).toBe("posix");
	});

	it("agent has 5 retrieval methods registered", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const methods = agent.getRegistry().list();
		expect(methods.length).toBe(5);
	});

	it("submitFeedback falls back to posix when no afterToolCall recorded", () => {
		const memPath = join(tmpDir, "memory.json");
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: memPath,
		});
		agent["lastQuery"] = "cold start query";
		agent.submitFeedback(true);

		const memory = new RetrievalMemory({ storagePath: memPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0].method).toBe("posix");
		expect(entries[0].outcome).toBe("success");
	});

	it("agent system prompt includes check_memory in tools", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("check_memory");
		expect(prompt).toContain("Memory & Strategy");
	});
});
