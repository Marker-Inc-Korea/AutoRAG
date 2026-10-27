/**
 * Live E2E: a real model, asked for a file whose name it can only guess, passes
 * several candidate names to `fsearch_search` in one call, the names are searched
 * at the same time, and the answer names the file.
 *
 * Gated on AUTORAG_NAME_ALT_LIVE=1 plus OPENROUTER_API_KEY and fsearch-cli on
 * PATH; the offline suite never calls a model. The agent model id comes from
 * AUTORAG_NAME_ALT_MODEL (an OpenRouter id); pick one released in 2026 or later.
 *
 *   AUTORAG_NAME_ALT_LIVE=1 AUTORAG_NAME_ALT_MODEL=anthropic/claude-haiku-5.5 \
 *     bun run vitest run test/live-e2e/name-alternatives.test.ts
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveModel } from "../../src/cli/config.ts";

const MODEL_ID = process.env.AUTORAG_NAME_ALT_MODEL ?? "";
const HAS_FSEARCH = spawnSync("fsearch-cli", ["--version"]).status === 0;
const LIVE =
	process.env.AUTORAG_NAME_ALT_LIVE === "1" &&
	MODEL_ID.length > 0 &&
	(process.env.OPENROUTER_API_KEY ?? "").length > 0 &&
	HAS_FSEARCH;

describe.skipIf(!LIVE)("live: fsearch_search takes several candidate names in one call", () => {
	let root: string;
	let agent: AutoRAGAgent;
	const calls: { query: string; alternatives?: string[] }[] = [];

	beforeAll(async () => {
		root = mkdtempSync(join(tmpdir(), "autorag-name-alt-live-"));
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "환불규정-2026.hwp"), "x");
		writeFileSync(join(docs, "meeting-notes.md"), "Weekly sync.\n");
		writeFileSync(join(docs, "lunch-menu.txt"), "Bibimbap.\n");
		const model = await resolveModel({
			searchPaths: [docs],
			model: { provider: "openrouter", id: MODEL_ID },
		} as never);
		agent = new AutoRAGAgent({
			model,
			apiKey: process.env.OPENROUTER_API_KEY,
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			piAgentDir: join(root, "pi-agent"),
			jev: { backend: "openrouter" },
			minSync: { autoInstall: false },
			everything: false,
			jikji: false,
			webSearch: false,
		});
		await agent.refresh(false, { methods: ["fsearch"] });
		// Record every call the model makes to the tool, so one call carrying several names is observable.
		const internals = agent as unknown as {
			innerAgent: { state: { tools: Array<{ name: string; execute: (...args: unknown[]) => Promise<unknown> }> } };
		};
		const tool = internals.innerAgent.state.tools.find((entry) => entry.name === "fsearch_search");
		if (tool === undefined) throw new Error("fsearch_search was not registered");
		const original = tool.execute.bind(tool);
		tool.execute = (toolCallId: unknown, params: unknown, ...rest: unknown[]) => {
			calls.push(params as { query: string; alternatives?: string[] });
			return original(toolCallId, params, ...rest);
		};
	});

	afterAll(async () => {
		await agent.stopFsearch();
		rmSync(root, { recursive: true, force: true });
	});

	it("tries several spellings in one call and finds the file", async () => {
		const response = await agent.searchDocuments(
			"Use fsearch_search to find the file about the refund policy. I do not remember its exact name: it may be in Korean (환불, 규정) or English (refund, policy). Tell me its file name.",
		);

		// One call named several candidates: the parallel path, not repeated single-name calls.
		const multi = calls.find(
			(call) => (call.alternatives ?? []).filter((name) => name.trim().length > 0).length >= 1,
		);
		expect(multi, JSON.stringify(calls)).toBeDefined();
		expect(response.answer).toContain("환불규정-2026");
	}, 300_000);
});
