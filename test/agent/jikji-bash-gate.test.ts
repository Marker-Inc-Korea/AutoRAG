import { chmodSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { JIKJI_FIND_TOOL_NAME, type MergedJikjiPolicy } from "../../src/agent/jikji-find-tool.ts";

let root: string;
let docs: string;
let binaryPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-gate-"));
	docs = join(root, "docs");
	binaryPath = join(root, "fake-jikji.mjs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "target.txt"), "content here\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/**
 * Build a fake jikji binary that prints a hand-authored answer-pack for `find`
 * and a trivial prepare summary otherwise. The answer-pack's policy fields are
 * fully controlled so we can test every gate branch.
 */
function writeFakeJikji(
	handoff: string,
	opts?: {
		stopAfterFind?: boolean;
		forbiddenBash?: boolean;
		agentShouldNotRerank?: boolean;
		allowedFollowups?: readonly string[];
	},
): void {
	const stopAfterFind = opts?.stopAfterFind ?? false;
	const forbiddenBash = opts?.forbiddenBash ?? false;
	const noRerank = opts?.agentShouldNotRerank ?? false;
	const allowedFollowups = opts?.allowedFollowups ?? [];
	const pack = {
		answer_paths: ["target.txt"],
		paths: ["target.txt"],
		candidates: [{ path: "target.txt", next_read: "original" }],
		evidence_pack: [{ path: "target.txt", next_read: "original" }],
		handoff_action: handoff,
		tool_call_policy: {
			stop_after_find: stopAfterFind,
			forbidden_tools: forbiddenBash ? ["bash"] : [],
			allowed_followups: allowedFollowups,
		},
		agent_should_not_rerank: noRerank,
	};
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
if (process.argv.slice(2)[0] === "find") {
	console.log(JSON.stringify(${JSON.stringify(pack)}));
} else {
	console.log(JSON.stringify({ prepared: true }));
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

/** Execute the bash tool directly and return its result details. */
async function executeBash(
	agent: AutoRAGAgent,
	command = "echo hello",
): Promise<{ content: string; details: Record<string, unknown> }> {
	const innerAgent = (agent as unknown as { innerAgent: { state: { tools: AgentTool[] } } }).innerAgent;
	const bashTool = innerAgent.state.tools.find((tool) => tool.name === "bash");
	if (bashTool === undefined) throw new Error("bash tool not found");
	const result = await bashTool.execute("test-call-id", { command }, undefined);
	return {
		content: result.content[0]?.type === "text" ? result.content[0].text : "",
		details: result.details as Record<string, unknown>,
	};
}

/** Execute the jikji_find tool directly and return its result details. */
async function executeJikjiFind(
	agent: AutoRAGAgent,
	query: string,
): Promise<{ content: string; details: Record<string, unknown> }> {
	const innerAgent = (agent as unknown as { innerAgent: { state: { tools: AgentTool[] } } }).innerAgent;
	const tool = innerAgent.state.tools.find((t) => t.name === JIKJI_FIND_TOOL_NAME);
	if (tool === undefined) throw new Error("jikji_find tool not found");
	const result = await tool.execute("test-call-id", { query }, undefined);
	return {
		content: result.content[0]?.type === "text" ? result.content[0].text : "",
		details: result.details as Record<string, unknown>,
	};
}

/** Get the current active jikji policy from the agent internals. */
function getActivePolicy(agent: AutoRAGAgent): MergedJikjiPolicy | undefined {
	return (agent as unknown as { activeJikjiPolicy: MergedJikjiPolicy | undefined }).activeJikjiPolicy;
}

/**
 * Simulate an active searchDocuments run so findJikji persists run-scoped state
 * (activeJikjiPolicy / jikjiFindCallCount). Per blocker 4, direct out-of-run
 * calls do not mutate run state.
 */
function beginRun(agent: AutoRAGAgent): void {
	(agent as unknown as { activeRun: boolean }).activeRun = true;
}

function endRun(agent: AutoRAGAgent): void {
	(agent as unknown as { activeRun: boolean }).activeRun = false;
}

describe("Jikji bash gate enforcement", () => {
	it("allows bash when no jikji policy is active and jikji is not configured (regression)", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).not.toBe(true);
		expect(result.content).toContain("hello");
	});

	it("denies bash pre-find when jikji is configured (find-first)", async () => {
		writeFakeJikji("raw_fallback_after_retry");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		// No jikji_find called yet this run → find-first blocks bash.
		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
		expect(result.content).toContain("jikji_find");
		expect(result.content).not.toContain("hello");
	});

	it("denies bash when forbiddenTools includes bash", async () => {
		writeFakeJikji("direct_use", { forbiddenBash: true, stopAfterFind: true, agentShouldNotRerank: true });
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target");
		endRun(agent);
		expect(getActivePolicy(agent)?.forbiddenTools).toContain("bash");

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
		expect(result.content).not.toContain("hello");
	});

	it("denies bash when stopAfterFind is true", async () => {
		writeFakeJikji("direct_use", { stopAfterFind: true, agentShouldNotRerank: true });
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target");
		endRun(agent);
		expect(getActivePolicy(agent)?.stopAfterFind).toBe(true);

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
	});

	it("denies bash when handoffAction is direct_use", async () => {
		writeFakeJikji("direct_use", { agentShouldNotRerank: true });
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target");
		endRun(agent);
		expect(getActivePolicy(agent)?.handoffAction).toBe("direct_use");

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
	});

	it("denies bash when handoffAction is jikji_retry", async () => {
		writeFakeJikji("jikji_retry");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target");
		endRun(agent);
		expect(getActivePolicy(agent)?.handoffAction).toBe("jikji_retry");

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
	});

	it("denies bash on first raw_fallback_after_retry find (rawFallbackAllowed false)", async () => {
		writeFakeJikji("raw_fallback_after_retry");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target");
		endRun(agent);
		expect(getActivePolicy(agent)?.handoffAction).toBe("raw_fallback_after_retry");
		expect(getActivePolicy(agent)?.rawFallbackAllowed).toBe(false);

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
	});

	it("allows bash on second raw_fallback_after_retry find (rawFallbackAllowed true)", async () => {
		writeFakeJikji("raw_fallback_after_retry");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target"); // first find: rawFallbackAllowed=false
		expect(getActivePolicy(agent)?.rawFallbackAllowed).toBe(false);

		await agent.findJikji("target"); // second find: rawFallbackAllowed=true
		expect(getActivePolicy(agent)?.rawFallbackAllowed).toBe(true);
		endRun(agent);

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).not.toBe(true);
		expect(result.content).toContain("hello");
	});

	it("NEGATIVE mixed-root: direct_use+stop_after_find root overrides raw_fallback_after_retry root → bash blocked", async () => {
		// Two search roots with different fake binaries.
		const docs2 = join(root, "docs2");
		mkdirSync(docs2, { recursive: true });
		writeFileSync(join(docs2, "other.txt"), "other content\n");

		// Root 1: direct_use + stop_after_find + forbidden bash (very restrictive)
		const pack1 = {
			answer_paths: ["target.txt"],
			paths: ["target.txt"],
			candidates: [{ path: "target.txt", next_read: "original" }],
			evidence_pack: [{ path: "target.txt", next_read: "original" }],
			handoff_action: "direct_use",
			tool_call_policy: {
				stop_after_find: true,
				forbidden_tools: ["bash"],
				allowed_followups: [],
			},
			agent_should_not_rerank: true,
		};
		// Root 2: raw_fallback_after_retry (permissive)
		const pack2 = {
			answer_paths: ["other.txt"],
			paths: ["other.txt"],
			candidates: [{ path: "other.txt", next_read: "original" }],
			evidence_pack: [{ path: "other.txt", next_read: "original" }],
			handoff_action: "raw_fallback_after_retry",
			tool_call_policy: {
				stop_after_find: false,
				forbidden_tools: [],
				allowed_followups: [],
			},
			agent_should_not_rerank: false,
		};

		// Single binary that dispatches based on the root argument.
		const dispatchBinary = join(root, "fake-jikji-dispatch.mjs");
		writeFileSync(
			dispatchBinary,
			`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "find") {
	const rootArg = args[1];
		if (rootArg === ${JSON.stringify(realpathSync(docs))}) {
		console.log(JSON.stringify(${JSON.stringify(pack1)}));
	} else {
		console.log(JSON.stringify(${JSON.stringify(pack2)}));
	}
} else {
	console.log(JSON.stringify({ prepared: true }));
}
`,
		);
		chmodSync(dispatchBinary, 0o755);

		const agent = new AutoRAGAgent({
			searchPaths: [docs, docs2],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath: dispatchBinary },
		});

		beginRun(agent);
		await agent.findJikji("target");
		endRun(agent);
		const policy = getActivePolicy(agent);
		// Restrictive-wins: direct_use overrides raw_fallback_after_retry
		expect(policy?.handoffAction).toBe("direct_use");
		expect(policy?.stopAfterFind).toBe(true);
		expect(policy?.forbiddenTools).toContain("bash");

		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
		expect(result.content).not.toContain("hello");
	});

	it("clears activeJikjiPolicy between searchDocuments runs (find-first re-enforced)", async () => {
		writeFakeJikji("direct_use", { stopAfterFind: true, agentShouldNotRerank: true });
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		await agent.findJikji("target");
		expect(getActivePolicy(agent)).toBeDefined();
		endRun(agent);

		// Simulate the end-of-run cleanup that searchDocuments does in finally.
		(agent as unknown as { activeJikjiPolicy: MergedJikjiPolicy | undefined }).activeJikjiPolicy = undefined;
		(agent as unknown as { jikjiFindCallCount: number }).jikjiFindCallCount = 0;

		// After cleanup count===0 → find-first gate blocks bash until the next
		// jikji_find (find-first is enforced per run).
		const result = await executeBash(agent, "echo hello");
		expect(result.details.blockedByJikjiPolicy).toBe(true);
		expect(result.content).toContain("jikji_find");
	});

	it("releases find-first after a failed jikji_find (all roots unavailable → bash fallback)", async () => {
		const missingBinary = join(root, "does-not-exist-jikji");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath: missingBinary },
		});

		// Pre-find: bash blocked.
		const pre = await executeBash(agent, "echo hello");
		expect(pre.details.blockedByJikjiPolicy).toBe(true);

		// A failed find increments the count under an active run, releasing the
		// find-first gate. Policy stays undefined → bash allowed (fallback).
		beginRun(agent);
		const result = await agent.findJikji("target");
		expect(result.answerPack).toBeUndefined();
		expect(result.policy).toBeUndefined();
		endRun(agent);

		const post = await executeBash(agent, "echo hello");
		expect(post.details.blockedByJikjiPolicy).not.toBe(true);
		expect(post.content).toContain("hello");
	});

	it("root-provenance: a relative path only valid under root B does NOT resolve against root A", async () => {
		const docs2 = join(root, "docs2");
		mkdirSync(docs2, { recursive: true });
		writeFileSync(join(docs2, "only-in-b.txt"), "b-only content\n");
		writeFileSync(join(docs2, "marker.txt"), "marker content\n");
		// docs (root A) has target.txt but NOT only-in-b.txt or marker.txt.

		const dispatchBinary = join(root, "fake-jikji-prov.mjs");
		// Root A returns "only-in-b.txt" — a path that exists ONLY under root B.
		// With cross-root normalization (old behavior) this would resolve to
		// docs2/only-in-b.txt. With origin-only normalization (new behavior) it
		// cannot resolve under root A and is skipped.
		const packA = {
			answer_paths: ["only-in-b.txt"],
			paths: ["only-in-b.txt"],
			candidates: [{ path: "only-in-b.txt", next_read: "original" }],
			evidence_pack: [{ path: "only-in-b.txt", next_read: "original" }],
			handoff_action: "raw_fallback_after_retry",
			tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
			agent_should_not_rerank: false,
		};
		// Root B returns "marker.txt" — distinct from root A's path so the only
		// way "only-in-b.txt" appears is via cross-root resolution from root A.
		const packB = {
			answer_paths: ["marker.txt"],
			paths: ["marker.txt"],
			candidates: [{ path: "marker.txt", next_read: "original" }],
			evidence_pack: [{ path: "marker.txt", next_read: "original" }],
			handoff_action: "raw_fallback_after_retry",
			tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
			agent_should_not_rerank: false,
		};
		writeFileSync(
			dispatchBinary,
			`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "find") {
	const rootArg = args[1];
		if (rootArg === ${JSON.stringify(realpathSync(docs))}) {
		console.log(JSON.stringify(${JSON.stringify(packA)}));
	} else {
		console.log(JSON.stringify(${JSON.stringify(packB)}));
	}
} else {
	console.log(JSON.stringify({ prepared: true }));
}
`,
		);
		chmodSync(dispatchBinary, 0o755);

		const agent = new AutoRAGAgent({
			searchPaths: [docs, docs2],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath: dispatchBinary },
		});

		beginRun(agent);
		const result = await agent.findJikji("only-in-b");
		endRun(agent);

		expect(result.answerPack).toBeDefined();
		const answerPaths = result.answerPack?.answerPaths ?? [];
		// Root B's "marker.txt" resolves against its own origin root.
		expect(answerPaths.some((p) => p.includes("marker.txt"))).toBe(true);
		// Root A's "only-in-b.txt" must NOT cross-resolve against root B.
		expect(answerPaths.some((p) => p.includes("only-in-b.txt"))).toBe(false);
	});

	it("jikji_find details include candidates, evidencePack, forbiddenTools, allowedFollowups, and perRoot", async () => {
		writeFakeJikji("direct_use", {
			forbiddenBash: true,
			stopAfterFind: true,
			agentShouldNotRerank: true,
			allowedFollowups: ["jikji_find"],
		});
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		beginRun(agent);
		const { details } = await executeJikjiFind(agent, "target");
		endRun(agent);

		expect(details.method).toBe("jikji_find");
		expect(details.answerCount).toBe(1);
		// candidates: path + nextRead
		const candidates = details.candidates as readonly { path: string; nextRead: string }[];
		expect(candidates).toHaveLength(1);
		expect(candidates[0]?.path).toContain("target.txt");
		expect(candidates[0]?.nextRead).toBe("original");
		// evidencePack: path + nextRead
		const evidence = details.evidencePack as readonly { path: string; nextRead: string }[];
		expect(evidence).toHaveLength(1);
		expect(evidence[0]?.path).toContain("target.txt");
		expect(evidence[0]?.nextRead).toBe("original");
		// forbiddenTools
		expect(details.forbiddenTools).toContain("bash");
		// allowedFollowups (from policy)
		expect(details.allowedFollowups).toContain("jikji_find");
		// perRoot: one entry per ok root
		const perRoot = details.perRoot as readonly {
			root: string;
			handoffAction: string;
			stopAfterFind: boolean;
			forbiddenTools: readonly string[];
			allowedFollowups: readonly string[];
			agentShouldNotRerank: boolean;
		}[];
		expect(perRoot).toHaveLength(1);
		expect(perRoot[0]?.root).toBe(realpathSync(docs));
		expect(perRoot[0]?.handoffAction).toBe("direct_use");
		expect(perRoot[0]?.stopAfterFind).toBe(true);
		expect(perRoot[0]?.forbiddenTools).toContain("bash");
		expect(perRoot[0]?.allowedFollowups).toContain("jikji_find");
		expect(perRoot[0]?.agentShouldNotRerank).toBe(true);
	});
});
