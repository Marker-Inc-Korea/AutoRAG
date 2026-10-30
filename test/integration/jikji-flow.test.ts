import {
	chmodSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { JIKJI_FIND_TOOL_NAME } from "../../src/agent/jikji-find-tool.ts";

let root: string;
let docs: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-flow-"));
	docs = join(root, "docs");
	binaryPath = join(root, "fake-jikji.mjs");
	logPath = join(root, "jikji-calls.jsonl");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "q3-report.txt"), "Q3 document exists for Jikji lookup.\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

/**
 * Fake jikji binary that branches on argv[0] (the subcommand):
 * - `find`: prints a hand-authored answer-pack JSON
 * - `prepare`: prints a prepare summary {root,files,folders,deleted,agent_map}
 */
function writeFakeJikji(answerPack?: Record<string, unknown>): void {
	const pack = answerPack ?? {
		answer_paths: ["q3-report.txt"],
		paths: ["q3-report.txt"],
		candidates: [{ path: "q3-report.txt", next_read: "original", label: "Q3 report" }],
		evidence_pack: [{ path: "q3-report.txt", next_read: "original" }],
		handoff_action: "direct_use",
		tool_call_policy: {
			stop_after_find: true,
			forbidden_tools: ["bash"],
			allowed_followups: [],
		},
		agent_should_not_rerank: true,
	};
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
if (args[0] === "find") {
	console.log(JSON.stringify(${JSON.stringify(pack)}));
} else {
	console.log(JSON.stringify({ prepared: true, root: args[1], files: 1, folders: 0, deleted: 0, agent_map: false }));
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

function methodNames(agent: AutoRAGAgent): string[] {
	return agent
		.getMethodRegistry()
		.list()
		.map((method) => method.describe().name);
}

function toolNames(agent: AutoRAGAgent): string[] {
	return (agent as unknown as { innerAgent: { state: { tools: AgentTool[] } } }).innerAgent.state.tools.map(
		(tool) => tool.name,
	);
}

function loggedArgs(): readonly string[][] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => (JSON.parse(line) as { readonly args: string[] }).args);
}

describe("AutoRAGAgent Jikji indexing integration", () => {
	it("registers jikji_find by default", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			everything: false,
			fsearch: false,
		});

		expect(methodNames(agent)).not.toContain("jikji");
		expect(toolNames(agent)).toContain(JIKJI_FIND_TOOL_NAME);
	});

	it("keeps Jikji out of the retrieval registry when configured", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		expect(methodNames(agent)).not.toContain("jikji");
		expect(agent.getMethodRegistry().get("jikji")).toBeUndefined();
	});

	it("can configure MinSync and Jikji indexing independently", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: join(root, "missing-minsync"), workspacePath: join(root, ".autorag", "minsync") },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		expect(methodNames(agent)).toEqual(["hybrid"]);
	});

	it("registers jikji_find tool when jikji is configured", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		expect(toolNames(agent)).toContain(JIKJI_FIND_TOOL_NAME);
	});

	it("does not register jikji_find when jikji is explicitly disabled", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
		});

		expect(toolNames(agent)).not.toContain(JIKJI_FIND_TOOL_NAME);
	});

	it("runs prepare --json during refresh and uses the new local-discovery prompt section", async () => {
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		await agent.refresh(true);

		const prepareCalls = loggedArgs().filter((args) => args[0] === "prepare");
		expect(prepareCalls.length).toBeGreaterThanOrEqual(1);
		for (const call of prepareCalls) {
			expect(call).toContain("--json");
			// --no-agent-rules is emitted by default (writeAgentRules !== true)
			expect(call).toContain("--no-agent-rules");
		}

		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("## Jikji Local Discovery");
	});

	it("adds prompt guidance for Jikji local discovery section", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		const prompt = agent.getSystemPrompt();

		expect(prompt).toContain("## Jikji Local Discovery");
		expect(prompt).toContain("jikji_find");
	});

	it("emits ['find', root, query, '--json'] when findJikji is called and returns answer_paths", async () => {
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		await agent.refresh(true);
		// Clear logs from prepare so we can isolate the find call
		writeFileSync(logPath, "");

		const result = await agent.findJikji("Q3 report", { topK: 5 });

		const calls = loggedArgs();
		expect(calls.length).toBeGreaterThanOrEqual(1);
		const findCall = calls.find((args) => args[0] === "find");
		expect(findCall).toBeDefined();
		expect(findCall?.[1]).toBe(realpathSync(docs));
		expect(findCall?.[2]).toBe("Q3 report");
		expect(findCall).toContain("--json");

		expect(result.answerPack).toBeDefined();
		expect(result.answerPack?.answerPaths.length).toBeGreaterThan(0);
		expect(result.policy).toBeDefined();
		expect(result.policy?.handoffAction).toBe("direct_use");
		expect(result.policy?.stopAfterFind).toBe(true);
	});

	it("constructs and retrieves without Jikji configured and without any Python runtime", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { autoInstall: false },
		});

		await expect(agent.prepareJikji()).resolves.toBeUndefined();
		const results = await agent.retrieve("document", { topK: 1 });
		expect(results).toEqual([]);
	});

	it("surfaces a path-free degraded diagnostic when the configured Jikji binary is missing", async () => {
		const missingBinary = join(root, "does-not-exist-jikji");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath: missingBinary },
			everything: false,
			fsearch: false,
		});

		const results = await agent.prepareJikji();
		const first = results?.[0];
		expect(first).toMatchObject({ ok: false, reason: "spawn-error" });
		expect(JSON.stringify(first)).not.toContain(missingBinary);
		expect(JSON.stringify(first)).not.toContain(root);
		await agent.refresh(true);
		const diag = (await agent.getRefreshStatus()).diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-unavailable");
		expect(diag?.message).not.toContain(missingBinary);
		expect(diag?.message).not.toContain(root);
	});

	it("returns jikji-unavailable result and leaves policy undefined when binary is missing", async () => {
		const missingBinary = join(root, "does-not-exist-jikji");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath: missingBinary },
			everything: false,
			fsearch: false,
		});

		const result = await agent.findJikji("anything");
		expect(result.answerPack).toBeUndefined();
		expect(result.policy).toBeUndefined();
		expect(result.diagnostics.length).toBeGreaterThan(0);
		expect(result.diagnostics[0]?.code).toBe("jikji-unavailable");
	});

	it("serves find from the existing index and never runs prepare on a query", async () => {
		// Index builds belong to refresh; a query only reads what is there.
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		const result = await agent.findJikji("Q3 report");

		expect(result.answerPack?.answerPaths).toContain(realpathSync(join(docs, "q3-report.txt")));
		expect(loggedArgs().map((args) => args[0])).toEqual(["find"]);
	});

	// Upstream Jikji (main) rejects `prepare` on a root that overlaps an already
	// indexed one, and two prepares of the same root contend for its lock.
	it("prepares and searches only the outermost of nested or duplicated search roots", async () => {
		const nested = join(docs, "sub");
		mkdirSync(nested, { recursive: true });
		writeFileSync(join(nested, "deep.txt"), "Nested document.\n");
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [nested, docs, docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			everything: false,
			fsearch: false,
		});

		const prepared = await agent.prepareJikji();
		await agent.findJikji("Q3 report");

		const calls = loggedArgs();
		expect(prepared).toHaveLength(1);
		expect(calls.filter((args) => args[0] === "prepare").map((args) => args[1])).toEqual([realpathSync(docs)]);
		expect(calls.filter((args) => args[0] === "find").map((args) => args[1])).toEqual([realpathSync(docs)]);
	});

	it("drops user-excluded paths from the merged Jikji answer pack", async () => {
		writeFileSync(join(docs, "keep-notes.txt"), "keep\n");
		writeFileSync(join(docs, "secret-plan.txt"), "secret\n");
		writeFakeJikji({
			answer_paths: ["keep-notes.txt", "secret-plan.txt"],
			paths: ["keep-notes.txt", "secret-plan.txt"],
			candidates: [
				{ path: "keep-notes.txt", next_read: "original", label: "keep" },
				{ path: "secret-plan.txt", next_read: "original", label: "secret" },
			],
			evidence_pack: [
				{ path: "keep-notes.txt", next_read: "original" },
				{ path: "secret-plan.txt", next_read: "original" },
			],
			handoff_action: "direct_use",
			tool_call_policy: { stop_after_find: true, forbidden_tools: ["bash"], allowed_followups: [] },
			agent_should_not_rerank: true,
		});
		const excluded = join(docs, "secret-plan.txt");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath },
			excludePaths: [excluded],
			everything: false,
			fsearch: false,
		});

		await agent.refresh(true);
		const result = await agent.findJikji("plans");

		const keep = realpathSync(join(docs, "keep-notes.txt"));
		expect(result.answerPack?.answerPaths).toEqual([keep]);
		expect(result.answerPack?.paths).toEqual([keep]);
		expect(result.answerPack?.candidates.map((c) => c.path)).toEqual([keep]);
		expect(result.answerPack?.evidencePack.map((e) => e.path)).toEqual([keep]);
	});
});

describe("AutoRAGAgent Jikji prepare budget and lifecycle (#1894)", () => {
	function writeBehaviorJikji(body: string): void {
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, at: Date.now() }) + "\\n");
${body}
`,
		);
		chmodSync(binaryPath, 0o755);
	}

	function agentFor(searchPaths: string[], jikji: Record<string, unknown> = {}): AutoRAGAgent {
		return new AutoRAGAgent({
			searchPaths,
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { binaryPath, ...jikji },
			everything: false,
			fsearch: false,
		});
	}

	it("reports a timed-out root as a diagnostic and still finishes the refresh", { timeout: 30_000 }, async () => {
		writeBehaviorJikji("await new Promise(() => setInterval(() => undefined, 1000));");
		const agent = agentFor([docs], { timeoutMs: 300 });

		await expect(agent.refresh(true)).resolves.toBeDefined();

		const status = await agent.getRefreshStatus();
		const diag = status.diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-prepare-failed");
		expect(diag?.message).toMatch(/timed out/);
		expect(diag?.message).not.toContain(root);
	});

	// `autorag refresh` and `autorag status` are separate processes, so the only place an operator sees a
	// degraded Jikji is the refresh result itself. It used to carry every other index's diagnostics
	// (MinSync, Everything, FSearch) but silently dropped Jikji's: `refresh --json` printed no jikji entry
	// and a later `status` reported `jikji=configured`.
	it("returns the jikji diagnostic in the refresh result itself, not only in later status", async () => {
		writeBehaviorJikji("process.exit(3);");
		const agent = agentFor([docs]);

		const refreshed = await agent.refresh(true, { methods: ["jikji"] });

		const diag = refreshed.diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-prepare-failed");
		expect(diag?.message).not.toContain(root);
	});

	it("returns the verbatim unsupported-version diagnostic in the refresh result", async () => {
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
if (process.argv[2] === "--version") { console.log("jikji 0.1.1"); process.exit(0); }
process.exit(0);
`,
		);
		chmodSync(binaryPath, 0o755);
		// PATH is only `stale`, so no real jikji/cargo on the machine can be picked up; node is linked in
		// because the stand-in is a `#!/usr/bin/env node` script.
		const stale = join(root, "stale-bin");
		mkdirSync(stale, { recursive: true });
		const link = join(stale, "jikji");
		symlinkSync(binaryPath, link);
		symlinkSync(process.execPath, join(stale, "node"));
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false },
			jikji: { autoInstall: false, env: { PATH: stale } },
			everything: false,
			fsearch: false,
		});

		const refreshed = await agent.refresh(true, { methods: ["jikji"] });

		const diag = refreshed.diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-unsupported-version");
		expect(diag?.message).toContain("0.1.1");
		expect(diag?.message).toContain(link);
	});

	it("forwards refresh jikjiPrepare.parseTimeout to the CLI for every root", async () => {
		const second = join(root, "docs2");
		mkdirSync(second, { recursive: true });
		writeFakeJikji();
		const agent = agentFor([docs, second]);

		await agent.refresh(true, { jikjiPrepare: { parseTimeout: 90 } });

		const prepareCalls = loggedArgs().filter((args) => args[0] === "prepare");
		expect(prepareCalls).toHaveLength(2);
		for (const call of prepareCalls) {
			const flag = call.indexOf("--parse-timeout");
			expect(flag).toBeGreaterThan(-1);
			expect(call[flag + 1]).toBe("90");
		}
	});

	it("forwards a configured jikji.parseTimeout without a per-call option", async () => {
		writeFakeJikji();
		const agent = agentFor([docs], { parseTimeout: 45 });

		await agent.prepareJikji();

		const call = loggedArgs().find((args) => args[0] === "prepare") ?? [];
		expect(call).toEqual(expect.arrayContaining(["--parse-timeout", "45"]));
	});

	it("aborts the in-flight root and the queued one through the host signal", { timeout: 30_000 }, async () => {
		writeBehaviorJikji("await new Promise(() => setInterval(() => undefined, 1000));");
		const second = join(root, "docs2");
		mkdirSync(second, { recursive: true });
		const agent = agentFor([docs, second], { timeoutMs: 25_000 });
		const controller = new AbortController();

		const pending = agent.prepareJikji({ signal: controller.signal });
		const deadline = Date.now() + 10_000;
		while (Date.now() < deadline) {
			try {
				if (loggedArgs().length >= 1) break;
			} catch {
				// log not created yet
			}
			await new Promise((resolve) => setTimeout(resolve, 20));
		}
		controller.abort();

		const results = await pending;
		expect(results?.map((result) => (result.ok ? "ok" : result.reason))).toEqual(["aborted", "aborted"]);
	});
});
