/**
 * Live E2E: real Jev routing plus a real model edit a temp config through the
 * `config` branch, which now carries the packaged `autorag-setup` skill as a
 * `<skill …>` invocation and leaves the session's tools untouched while the
 * planQuery path re-reads and validates the file.
 *
 * Gated on AUTORAG_SELF_CONFIG_LIVE=1 plus OPENROUTER_API_KEY. The agent model
 * id comes from AUTORAG_SELF_CONFIG_MODEL (an OpenRouter id); pick one released
 * in 2026 or later. Jev routes through its default OpenRouter backend.
 *
 *   AUTORAG_SELF_CONFIG_LIVE=1 AUTORAG_SELF_CONFIG_MODEL=<id> OPENROUTER_API_KEY=... \
 *     bun run vitest run test/live-e2e/self-config-route.test.ts
 */
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-all-tool.ts";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-minsync-tool.ts";
import { WEB_FETCH_TOOL_NAME } from "../../src/agent/web-fetch-tool.ts";
import { WEB_SEARCH_TOOL_NAME } from "../../src/agent/web-search-tool.ts";
import { resolveModel } from "../../src/cli/config.ts";

/** Tools a configuration turn must never call. */
const RETRIEVAL_TOOL_NAMES = [
	SEARCH_ALL_DOCUMENTS_TOOL_NAME,
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
	WEB_SEARCH_TOOL_NAME,
	WEB_FETCH_TOOL_NAME,
] as const;

/** The model id the temp config starts with, so the turn has something to change. */
const PLACEHOLDER_MODEL_ID = "old/placeholder-model";

const MODEL_ID = process.env.AUTORAG_SELF_CONFIG_MODEL ?? "";
const LIVE =
	process.env.AUTORAG_SELF_CONFIG_LIVE === "1" &&
	MODEL_ID.length > 0 &&
	(process.env.OPENROUTER_API_KEY ?? "").length > 0;

function routeOf(response: SearchDocumentsResponse): string | undefined {
	const routed = response.diagnostics?.find((diagnostic) => diagnostic.code === "query-routed");
	return /routed the question to (\w+)/u.exec(routed?.message ?? "")?.[1];
}

describe.skipIf(!LIVE)("live: Jev configures AutoRAG with the setup skill and no retrieval", () => {
	let root: string;
	let agent: AutoRAGAgent;
	let configPath: string;

	beforeAll(async () => {
		root = mkdtempSync(join(tmpdir(), "autorag-self-config-live-"));
		const docs = join(root, "docs");
		const workspace = join(root, "workspace");
		mkdirSync(docs, { recursive: true });
		mkdirSync(workspace, { recursive: true });
		configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			`${JSON.stringify(
				{
					model: { provider: "openrouter", id: PLACEHOLDER_MODEL_ID },
					searchPaths: [docs],
					workspacePath: workspace,
					memoryPath: join(root, "memory.json"),
				},
				null,
				2,
			)}\n`,
		);
		const model = await resolveModel({
			searchPaths: [docs],
			model: { provider: "openrouter", id: MODEL_ID },
		} as never);
		agent = new AutoRAGAgent({
			model,
			apiKey: process.env.OPENROUTER_API_KEY,
			searchPaths: [docs],
			workspacePath: workspace,
			memoryPath: join(root, "memory.json"),
			piAgentDir: join(root, "pi-agent"),
			jev: { backend: "openrouter" },
			everything: false,
			fsearch: false,
			jikji: false,
			webSearch: false,
			// The host's validator is the CLI's; this test only checks that the turn
			// writes the file and skips retrieval, so no rollback hook is needed.
			selfConfig: { configPath },
		} as never);
	});
	afterAll(() => rmSync(root, { recursive: true, force: true }));

	it("switches the temp config's model id with the setup skill and no search", async () => {
		const called: string[] = [];
		const unsubscribe = agent.subscribe((event) => {
			if (event.type === "tool_execution_start") called.push(event.toolName);
		});
		let response: SearchDocumentsResponse;
		try {
			response = await agent.searchDocuments(
				`${configPath} 파일에서 이 에이전트가 쓸 모델을 openrouter/${MODEL_ID} 로 바꿔줘. 다른 설정은 그대로 둬.`,
			);
		} finally {
			unsubscribe();
		}

		expect(routeOf(response)).toBe("config");
		expect(response.results).toEqual([]);
		// The turn edited the file with its file tools, and never searched.
		expect(called.length, "the config turn called no tool at all").toBeGreaterThan(0);
		expect(called, "the config turn used no file tool").toContain("bash");
		const offending = called.filter(
			(name) => (RETRIEVAL_TOOL_NAMES as readonly string[]).includes(name) || name.startsWith("search_datasource_"),
		);
		expect(offending, `retrieval or web tools ran on this turn: ${offending.join(", ")}`).toEqual([]);

		const written = JSON.parse(readFileSync(configPath, "utf8")) as { model?: { provider?: string; id?: string } };
		const wantedTail = MODEL_ID.split("/").at(-1) ?? MODEL_ID;
		expect(written.model?.id, `config was not pointed at ${MODEL_ID}`).toContain(wantedTail);
		expect(response.answer.length).toBeGreaterThan(0);
	}, 360_000);
});
