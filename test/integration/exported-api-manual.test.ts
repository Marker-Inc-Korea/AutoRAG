// Manual-QA-equivalent suite exercising ONLY the public exported API surface
// via the package root (../../src/index.ts). Each MVP story appends a section
// here. Assertions focus on the real exported contract: transparency,
// structured diagnostics, refresh status, and watch behavior.

import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import {
	type FauxProviderRegistration,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/index.ts";
import { parsedOutputPath } from "../../src/mirror/paths.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-exported-api-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	registrations = [];
	writeFileSync(join(docs, "report.txt"), "Quarterly report content for retrieval.\n");
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxEmitModel(args: Record<string, unknown>) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[fauxToolCall("subagent", { agent: "autorag-explorer", model: "faux/gpt-5.6-luna", task: "explore" })],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, args)], { stopReason: "toolUse" }),
	]);
	registrations.push(reg);
	return reg.getModel();
}

function fauxSessionFactory(): NonNullable<ConstructorParameters<typeof AutoRAGAgent>[0]["sessionFactory"]> {
	return async (options) => {
		const subagentTool: AgentTool = {
			name: "subagent",
			label: "Subagent",
			description: "Test explorer",
			parameters: Type.Object({ agent: Type.String(), model: Type.String(), task: Type.String() }),
			execute: async () => ({ content: [{ type: "text", text: "explored" }], details: {} }),
		};
		const agent = new Agent({
			initialState: {
				systemPrompt: options.systemPrompt,
				model: options.model,
				tools: [subagentTool, ...options.tools],
			},
			convertToLlm: (messages) =>
				messages.filter(
					(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
				),
		});
		return {
			agent,
			prompt: async (prompt) => agent.prompt(prompt),
			abort: async () => agent.abort(),
			dispose: () => {},
		};
	};
}

describe("exported API — #19 Jikji optional non-retrieval boundary", () => {
	it("constructs and retrieves through the root export with no Jikji and no Python", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		const results = await agent.retrieve("report", { topK: 1 });
		// No retrieval method configured (posix removed) → empty results.
		expect(results).toEqual([]);
		await expect(agent.prepareJikji()).resolves.toBeUndefined();
	});

	it("exposes a path-free jikji-unavailable diagnostic for a missing binary via the root export", async () => {
		const missingBinary = join(root, "nope-jikji");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath: missingBinary },
		});

		const prepared = await agent.prepareJikji();
		expect(prepared?.[0]).toMatchObject({ ok: false, reason: "spawn-error" });
		expect(JSON.stringify(prepared)).not.toContain(missingBinary);
		expect(JSON.stringify(prepared)).not.toContain(root);
		await agent.refresh(true);
		const diag = (await agent.getRefreshStatus()).diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-unavailable");
		expect(diag?.source).toBe("jikji");
		expect(diag?.message).not.toContain(missingBinary);
		expect(diag?.message).not.toContain(root);
	});
});

describe("exported API — #6 curated output path opacity", () => {
	it("preserves real paths in the root-exported searchDocuments response and keeps the registry intact", async () => {
		const model = fauxEmitModel({
			answer: `Found in ${root}/docs/report.txt (source /docs/report.txt)`,
			results: [
				{
					number: 1,
					title: `Report at ${root}/docs`,
					summary: `Cached under ${join(root, ".autorag")}/parsed`,
					evidence: [{ excerpt: `hit in ${root}/docs/report.txt` }],
					confidence: 1,
				},
			],
			mapping: [{ number: 1, source: "/docs/report.txt", method: "grep", content: "report" }],
		});
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			sessionFactory: fauxSessionFactory(),
		});

		const response = await agent.searchDocuments("report");
		const blob = JSON.stringify(response);

		// Real paths flow through verbatim — path opacity is removed.
		expect(blob).toContain(root);
		expect(blob).toContain("/docs/report.txt");
		expect(agent.getResultRegistry(response.sessionId).get(1)?.source).toBe("/docs/report.txt");
	});
});

describe("exported API — #21 structured degraded-mode diagnostics", () => {
	it("returns a minsync-unavailable diagnostic via the root-exported retrieveWithDiagnostics", async () => {
		const missing = join(root, "missing-minsync");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: missing, workspacePath: join(root, ".autorag", "minsync") },
		});

		const { results, diagnostics } = await agent.retrieveWithDiagnostics("report", { topK: 5 });

		// No retrieval method configured (posix removed) → empty results.
		expect(results).toEqual([]);
		const minsync = diagnostics.find((d) => d.source === "minsync");
		expect(minsync?.code).toBe("minsync-unavailable");
	});

	it("surfaces bm25-degraded-fallback and unknown-warning via root-exported searchDocuments", async () => {
		const model = fauxEmitModel({
			answer: "answer",
			results: [{ number: 1, title: "A", summary: "a", evidence: [{ excerpt: "a" }], confidence: 1 }],
			mapping: [{ number: 1, source: "/docs/report.txt", method: "grep", content: "a" }],
			warnings: ["empty-query", "weird-warning"],
		});
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			bm25: { forceEngine: "typescript-fallback" },
			sessionFactory: fauxSessionFactory(),
		});
		await agent.refresh(true);

		const response = await agent.searchDocuments("report");
		const codes = (response.diagnostics ?? []).map((d) => d.code);
		expect(codes).toContain("bm25-degraded-fallback");
		expect(codes).toContain("unknown-warning");
		expect(response.warnings).toEqual(["empty-query"]);
		expect(JSON.stringify(response.diagnostics)).not.toContain(root);
	});
});

describe("exported API — #22 observable refresh status and watch", () => {
	it("exposes path-opaque getRefreshStatus transitions via the root export", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		const before = await agent.getRefreshStatus();
		expect(before.state).toBe("idle");
		expect(before.stale).toBe(true);

		await agent.refresh(true);
		const after = await agent.getRefreshStatus();
		expect(after.state).toBe("success");
		expect(after.stale).toBe(false);
		expect(after.counts?.scanned).toBeGreaterThanOrEqual(1);
		const blob = JSON.stringify(after);
		expect(blob).not.toContain(root);
		expect(blob).not.toContain("indexPath");
	});

	it(
		"keeps parsed mirrors current via the root-exported startWatchRefresh and stops cleanly",
		{ retry: 3, timeout: 30000 },
		async () => {
			const agent = new AutoRAGAgent({
				searchPaths: [docs],
				memoryPath: join(root, "memory.json"),
				workspacePath: root,
			});
			await agent.refresh(true);
			const handle = agent.startWatchRefresh({ debounceMs: 30 });

			writeFileSync(join(docs, "watched.txt"), "Watched content about ledgers.\n");
			const mirror = parsedOutputPath(root, "/docs/watched.txt");
			const start = Date.now();
			let updated = false;
			while (Date.now() - start < 10000) {
				// The parsed-mirror file appearing is the latching signal that the
				// watcher re-indexed the new source (the per-refresh `written` counter
				// resets to 0 on the next no-op refresh, so it is not a stable latch).
				if (existsSync(mirror)) {
					updated = true;
					break;
				}
				await new Promise((resolve) => setTimeout(resolve, 40));
			}
			handle.stop();
			expect(updated).toBe(true);
		},
	);
});
