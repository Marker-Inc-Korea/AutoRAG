// Manual-QA-equivalent suite exercising ONLY the public exported API surface
// via the package root (../../src/index.ts). Each MVP story appends a section
// here. Assertions focus on the real exported contract: transparency,
// structured diagnostics, refresh status, and watch behavior.

import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider, streamSimple } from "@earendil-works/pi-ai/compat";
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
	writeFileSync(
		join(docs, "report.txt"),
		"Q2 2026 revenue increased 23% year over year to $4.2M, driven by enterprise contracts.\n",
	);
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function explorerAssignment(originalQuery: string): string {
	return [
		`Original query: ${originalQuery}`,
		"Selected retrieval method: POSIX",
		`Query variants: ${originalQuery}; ${originalQuery} evidence`,
		"Required handoff fields: Retrieved at and Temporal metadata.",
	].join("\n");
}

function fauxEmitModel(originalQuery: string, args: Record<string, unknown>) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall("subagent", {
					agent: "autorag-explorer",
					agentScope: "user",
					model: "faux/gpt-5.6-luna",
					task: explorerAssignment(originalQuery),
					cwd: docs,
					artifacts: false,
				}),
			],
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
			parameters: Type.Object({
				agent: Type.String(),
				agentScope: Type.Literal("user"),
				model: Type.String(),
				task: Type.String(),
				cwd: Type.Optional(Type.String()),
				artifacts: Type.Optional(Type.Boolean()),
			}),
			execute: async () => ({
				content: [{ type: "text", text: "Explorer run completed successfully." }],
				details: {
					mode: "single",
					runId: "run-exported-api",
					results: [
						{
							agent: "autorag-explorer",
							task: explorerAssignment("report"),
							exitCode: 0,
							usage: { input: 120, output: 90, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
							model: "faux/gpt-5.6-luna",
							structuredOutput: {
								assignment: {
									originalQuery: "report",
									method: "posix",
									queryVariant: "quarterly report revenue",
									queryVariants: ["test", "quarterly report revenue", "financial report"],
								},
								evidenceCandidates: [
									{
										source: join(docs, "report.txt"),
										method: "posix",
										evidence:
											"Q2 2026 revenue increased 23% year over year to $4.2M, driven by enterprise contracts.",
										retrievedAt: "2026-07-14T05:30:00.000Z",
										sourceTemporal: { status: "asOf", asOf: "2026-06-30T00:00:00.000Z" },
										locator: "report.txt:1",
									},
								],
								summary:
									"The quarterly report attributes year-over-year revenue growth to enterprise contracts.",
							},
						},
					],
				},
			}),
		};
		const agent = new Agent({
			initialState: {
				systemPrompt: options.systemPrompt,
				model: options.model,
				tools: [subagentTool, ...options.tools],
			},
			streamFn: streamSimple,
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
		const model = fauxEmitModel("report", {
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
		// Real paths flow through verbatim — path opacity is removed.
		expect(response.answer).toContain(root);
		expect(response.results[0]?.title).toContain(root);
		expect(response.results[0]?.summary).toContain(join(root, ".autorag"));
		expect(response.results[0]?.evidence[0]?.excerpt).toContain(root);
		expect(response.answer).toContain("/docs/report.txt");
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
		const model = fauxEmitModel("report", {
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

	it("keeps parsed mirrors current via the root-exported startWatchRefresh and stops cleanly", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		await agent.refresh(true);
		let emitChange: ((filename: string | null) => void) | undefined;
		let resolveRefreshCompleted: (() => void) | undefined;
		const refreshCompleted = new Promise<void>((resolve) => {
			resolveRefreshCompleted = resolve;
		});
		const originalRefresh = agent.refresh.bind(agent);
		agent.refresh = async (...args) => {
			const result = await originalRefresh(...args);
			resolveRefreshCompleted?.();
			return result;
		};
		const handle = agent.startWatchRefresh({
			debounceMs: 0,
			watcherFactory: (_dir, onChange) => {
				emitChange = onChange;
				return { close: () => {} };
			},
		});

		try {
			writeFileSync(join(docs, "watched.txt"), "Watched content about ledgers.\n");
			emitChange?.("watched.txt");
			let timeout: NodeJS.Timeout | undefined;
			try {
				await Promise.race([
					refreshCompleted,
					new Promise<never>((_resolve, reject) => {
						timeout = setTimeout(() => reject(new Error("Timed out waiting for watch refresh")), 10_000);
					}),
				]);
			} finally {
				if (timeout !== undefined) clearTimeout(timeout);
			}
			expect(existsSync(parsedOutputPath(root, "/docs/watched.txt"))).toBe(true);
		} finally {
			handle.stop();
		}
	});
});
