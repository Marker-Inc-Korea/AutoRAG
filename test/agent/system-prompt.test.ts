import { describe, expect, it } from "vitest";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

function buildPrompt(
	options: {
		jikjiIndexingEnabled?: boolean;
		toolNames?: string[];
		orchestratorModelId?: string;
		explorerModelId?: string;
	} = {},
): string {
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
		orchestratorModelId: options.orchestratorModelId,
		explorerModelId: options.explorerModelId,
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
		expect(prompt).toMatch(/top-level.*subagent.*agentScope.*user/i);
		expect(prompt).toMatch(/nested.*omit.*agentScope/i);
		expect(prompt).toMatch(/top-level.*subagent.*artifacts.*false/i);
		expect(prompt).toMatch(/single.*tasks.*chain.*parallel/i);
		expect(prompt).toMatch(/nested.*autorag-explorer.*task.*omit.*artifacts/i);
		expect(prompt).not.toMatch(/every.*autorag-explorer.*task.*artifacts.*false/i);
		expect(prompt).not.toContain("Your job: find relevant files, read their contents");
	});

	it("uses configured role model ids without retaining default identity claims", () => {
		const prompt = buildPrompt({
			orchestratorModelId: "custom-orchestrator",
			explorerModelId: "custom-explorer",
		});

		expect(prompt).toContain("custom-orchestrator");
		expect(prompt).toContain("custom-explorer");
		expect(prompt).not.toContain("gpt-5.6-sol");
		expect(prompt).not.toContain("gpt-5.6-luna");
	});

	it("specifies the explorer request and evidence handoff contract", () => {
		const prompt = buildPrompt();

		expect(prompt).toMatch(/original query/i);
		expect(prompt).toMatch(/selected retrieval or discovery path/i);
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

	it("separates parent seed retrieval from contained explorer tools", () => {
		const prompt = buildPrompt({ jikjiIndexingEnabled: true });
		const retrievalStart = prompt.indexOf("## Parent-Owned Seed Retrieval");
		const toolsStart = prompt.indexOf("## Explorer Tools");
		const strategyStart = prompt.indexOf("## Search Strategy");
		const quickReferenceStart = prompt.indexOf("## Tool Ownership Quick Reference");
		const retrievalSection = prompt.slice(retrievalStart, toolsStart);
		const toolsSection = prompt.slice(toolsStart, strategyStart);
		const jikjiStart = prompt.indexOf("## Jikji Local Discovery (Seed Policy)");
		const outputStart = prompt.indexOf("## Output Format");
		const jikjiSection = prompt.slice(jikjiStart, outputStart);

		expect(retrievalStart).toBeGreaterThan(-1);
		expect(toolsStart).toBeGreaterThan(retrievalStart);
		expect(strategyStart).toBeGreaterThan(toolsStart);
		expect(quickReferenceStart).toBeGreaterThan(strategyStart);
		expect(retrievalSection).toContain("parent orchestrator owns");
		expect(retrievalSection).toContain("parent-owned process-bound seed retrieval");
		expect(retrievalSection).toMatch(/delegate/);
		expect(toolsSection).toContain("read-only `read`/`grep`/`find`/`ls`");
		expect(toolsSection).toContain("exactly one normalized assigned search root");
		expect(toolsSection).toContain("preserve every inherited scope restriction");
		expect(jikjiSection).toContain("Seed Policy");
		expect(jikjiSection).toContain("read-only explorer");
		expect(prompt.slice(quickReferenceStart)).toContain("Parent-owned process-bound seed retrieval");
		expect(prompt.slice(quickReferenceStart)).not.toContain("Explorer-only");
	});

	it("does not grant forbidden shell or scope escape capabilities", () => {
		const prompt = buildPrompt({ jikjiIndexingEnabled: true });

		for (const forbidden of [
			/\bbash\b/i,
			/\bshell\b/i,
			/search parent directories/i,
			/remove (?:the )?scope restriction/i,
			/explorer-only/i,
		]) {
			expect(prompt).not.toMatch(forbidden);
		}
	});
});
