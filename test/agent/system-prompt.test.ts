import { describe, expect, it } from "vitest";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

function buildPrompt(options: { jikjiIndexingEnabled?: boolean; toolNames?: string[] } = {}): string {
	return buildSystemPrompt({
		toolNames: options.toolNames ?? [
			"bash",
			"jikji_find",
			"search_all_documents",
			"search_bm25_documents",
			"search_minsync_documents",
			"search_datasource_documents",
			"check_memory",
		],
		manifests: [],
		jikjiIndexingEnabled: options.jikjiIndexingEnabled,
	});
}

describe("buildSystemPrompt subagent orchestration contract", () => {
	it("makes the sol orchestrator and luna explorers distinct and mandatory", () => {
		const prompt = buildPrompt();

		expect(prompt).toContain("## Subagent Orchestration");
		expect(prompt).toContain("gpt-5.6-sol");
		expect(prompt).toContain("gpt-5.6-luna");
		expect(prompt).toContain("pi-subagents");
		expect(prompt).toMatch(/no single-agent fallback/i);
		expect(prompt).toMatch(/missing.*capability.*fatal/i);
		expect(prompt).toContain("sole orchestrator");
		expect(prompt).toMatch(/judgment|sufficiency|conflict|freshness/i);
		expect(prompt).toMatch(/follow-up|follow up/i);
		expect(prompt).toMatch(/final curation|curation/i);
		expect(prompt).toContain("bounded seed packs for an explorer");
		expect(prompt).toContain("Each candidate handoff MUST include");
		expect(prompt).not.toContain("Your job: find relevant files, read their contents");
	});

	it("specifies the explorer request and evidence handoff contract", () => {
		const prompt = buildPrompt();

		expect(prompt).toMatch(/original query/i);
		expect(prompt).toMatch(/selected retrieval method/i);
		expect(prompt).toMatch(/query variants/i);
		expect(prompt).toMatch(/weak(?:ly)? relevant/i);
		expect(prompt).toMatch(/search and read many documents/i);
		expect(prompt).toMatch(/evidence/i);
		expect(prompt).toContain("retrievedAt");
		expect(prompt).toMatch(/asOf|temporal metadata|explicit unknown/i);
		expect(prompt).toMatch(/creation\/modification timing/i);
	});

	it("keeps Jikji, datasource trust, and structured termination rules explicit", () => {
		const prompt = buildPrompt();

		expect(prompt).toContain("jikji_find");
		expect(prompt).toContain("agent_should_not_rerank");
		expect(prompt).toContain("raw_fallback_after_retry");
		expect(prompt).toContain("default-deny");
		expect(prompt).toContain("allowedTags");
		expect(prompt).toContain("allowedScopes");
		expect(prompt).toContain("emit_autorag_results");
		expect(prompt).toMatch(/exactly once.*final action|final action.*exactly once/i);
	});

	it("preserves the exact Jikji handoff and tool-call policy when enabled", () => {
		const prompt = buildPrompt({ jikjiIndexingEnabled: true });

		expect(prompt).toContain("## Jikji Local Discovery (Seed Policy)");
		expect(prompt).toContain("orchestrator calls `jikji_find` FIRST");
		expect(prompt).toContain("delegates answer_paths to the read-only explorer");
		expect(prompt).toContain("handoff_action");
		expect(prompt).toContain("tool_call_policy");
		for (const value of [
			"direct_use",
			"jikji_retry",
			"raw_fallback_after_retry",
			"stop_after_find",
			"forbidden_tools",
			"allowed_followups",
		]) {
			expect(prompt).toContain(value);
		}
	});

	it("fails closed in the no-search-tools branch", () => {
		const prompt = buildPrompt({ toolNames: [] });

		expect(prompt).toContain("No search tools were provided");
		expect(prompt).toContain("blocked/degraded state");
		expect(prompt).toContain("do not claim a completed search");
	});

	it("orders orchestrator ownership before explorer assignment and workflow", () => {
		const prompt = buildPrompt();

		const ownership = prompt.indexOf("### Exclusive orchestrator responsibilities");
		const assignment = prompt.indexOf("### Explorer assignment contract");
		const workflow = prompt.indexOf("## Workflow");

		expect(ownership).toBeGreaterThan(-1);
		expect(assignment).toBeGreaterThan(ownership);
		expect(workflow).toBeGreaterThan(assignment);
	});

	it("scopes retrieval instructions to explorer sections", () => {
		const prompt = buildPrompt({ jikjiIndexingEnabled: true });
		const toolsStart = prompt.indexOf("## Explorer Tools");
		const strategyStart = prompt.indexOf("## Search Strategy");
		const toolsSection = prompt.slice(toolsStart, strategyStart);
		const jikjiStart = prompt.indexOf("## Jikji Local Discovery (Seed Policy)");
		const outputStart = prompt.indexOf("## Output Format");
		const jikjiSection = prompt.slice(jikjiStart, outputStart);

		expect(toolsSection).toContain("process-bound seed retrieval");
		expect(toolsSection).toContain("must delegate the underlying document reading");
		expect(jikjiSection).toContain("Seed Policy");
		expect(jikjiSection).toContain("read-only explorer");
	});
});
