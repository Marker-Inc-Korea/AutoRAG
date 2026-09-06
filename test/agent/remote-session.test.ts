import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string | undefined;

interface AgentInternals {
	innerAgent: {
		state: {
			tools: Array<{ name: string }>;
		};
	};
}

afterEach(() => {
	if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
	tmpDir = undefined;
});

describe("AutoRAGAgent remote-session tool surface", () => {
	it("excludes local-only tools while retaining retrieval and emit tools", () => {
		tmpDir = mkdtempSync(join(tmpdir(), "autorag-remote-session-"));
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			remoteSession: true,
		});
		const names = (agent as unknown as AgentInternals).innerAgent.state.tools.map((tool) => tool.name);

		for (const name of ["bash", "jikji_find", "check_memory"]) {
			expect(names, name).not.toContain(name);
		}
		for (const name of [
			"lexical_search_local_docs",
			"semantic_search_local_docs",
			"search_all_documents",
			"search_datasource_documents",
			"emit_autorag_results",
		]) {
			expect(names, name).toContain(name);
		}

		const prompt = agent.getSystemPrompt();
		for (const name of ["bash", "jikji_find", "check_memory"]) {
			expect(prompt).not.toContain(`- **${name}**:`);
		}
	});
});
