import { describe, expect, it } from "vitest";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

function prompt(
	toolNames = [
		"bash",
		"search_all_documents",
		"search_bm25_documents",
		"search_minsync_documents",
		"check_memory",
		"emit_autorag_results",
	],
) {
	return buildSystemPrompt({ toolNames, manifests: [], jikjiIndexingEnabled: true, modelId: "test-model" });
}

describe("buildSystemPrompt single-agent contract", () => {
	it("assigns retrieval, reading, judgment, and curation to one agent", () => {
		const text = prompt();
		expect(text).toContain("retrieve candidates");
		expect(text).toContain("read the relevant source material directly");
		expect(text).toContain("judge the evidence");
		expect(text).toContain("emit_autorag_results");
		expect(text).not.toMatch(/subagent|explorer|delegat|Assignment V1|pi-subagents/i);
	});

	it("keeps retrieval, memory, datasource trust, and Jikji guidance", () => {
		const text = prompt();
		expect(text).toContain("search_all_documents");
		expect(text).toContain("search_bm25_documents");
		expect(text).toContain("search_minsync_documents");
		expect(text).toContain("check_memory");
		expect(text).toContain("default-deny");
		expect(text).toContain("Jikji Local Discovery");
	});

	it("fails closed when no tools are provided", () => {
		expect(prompt([])).toMatch(/blocked\/degraded state/i);
	});
});
