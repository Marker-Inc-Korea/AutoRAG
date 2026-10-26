/**
 * Live E2E: a real model drives `convert_files` and `list_file_conversions`
 * through AutoRAGAgent, with the real `filee` and `dupey` binaries.
 *
 * Gated on AUTORAG_FILEE_AGENT_LIVE=1 plus OPENROUTER_API_KEY. The model id
 * comes from AUTORAG_FILEE_AGENT_MODEL (an OpenRouter id); pick one released in 2026 or later.
 *
 *   AUTORAG_FILEE_AGENT_LIVE=1 AUTORAG_FILEE_AGENT_MODEL=<id> OPENROUTER_API_KEY=... \
 *     bun run vitest run test/live-e2e/filee-agent.test.ts
 */
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveModel } from "../../src/cli/config.ts";

const MODEL_ID = process.env.AUTORAG_FILEE_AGENT_MODEL ?? "";
const LIVE =
	process.env.AUTORAG_FILEE_AGENT_LIVE === "1" &&
	MODEL_ID.length > 0 &&
	(process.env.OPENROUTER_API_KEY ?? "").length > 0;

describe.skipIf(!LIVE)("live: a real model converts files through the agent", () => {
	let root: string;
	beforeAll(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-filee-agent-"));
	});
	afterAll(() => rmSync(root, { recursive: true, force: true }));

	it("calls convert_files with overwrite when asked, then reports dupey's verdict", async () => {
		const docs = join(root, "임의 폴더");
		const outside = join(root, "elsewhere");
		await import("node:fs").then((fs) => {
			fs.mkdirSync(docs, { recursive: true });
			fs.mkdirSync(outside, { recursive: true });
		});
		const source = join(outside, "회의록.md");
		writeFileSync(
			source,
			"# 회의록\n\n오늘 회의에서 예산 승인과 일정 조정을 논의했습니다. 다음 주까지 초안을 공유합니다.\n",
		);
		const existing = join(outside, "회의록.docx");
		writeFileSync(existing, "OLD CONTENT");

		const model = await resolveModel({
			searchPaths: [docs],
			model: { provider: "openrouter", id: MODEL_ID },
		} as never);
		const agent = new AutoRAGAgent({
			model,
			apiKey: process.env.OPENROUTER_API_KEY,
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			everything: false,
			fsearch: false,
			jikji: false,
		} as never);

		// The tools call these provider methods, so wrapping them records what the model actually sent.
		const calls: Array<{ name: string; args: unknown }> = [];
		const provider = agent as unknown as {
			convertFiles: (request: unknown, signal?: AbortSignal) => Promise<unknown>;
			listFileConversions: (from?: string, signal?: AbortSignal) => Promise<string>;
		};
		const realConvert = provider.convertFiles.bind(agent);
		const realList = provider.listFileConversions.bind(agent);
		provider.convertFiles = async (request, signal) => {
			calls.push({ name: "convert_files", args: request });
			return realConvert(request, signal);
		};
		provider.listFileConversions = async (from, signal) => {
			calls.push({ name: "list_file_conversions", args: { from } });
			return realList(from, signal);
		};

		const response = await agent.searchDocuments(
			`Convert ${source} to a DOCX file and replace the existing ${existing} with the result (overwrite it). Then tell me the dupey verdict for the converted copy.`,
			{ topK: 3 },
		);
		expect(response.answer.length).toBeGreaterThan(0);

		const convert = calls.find((call) => call.name === "convert_files");
		expect(convert, JSON.stringify(calls)).toBeDefined();
		expect(convert?.args).toMatchObject({ to: "docx", overwrite: true });
		expect(readFileSync(existing).subarray(0, 2).toString()).toBe("PK");
		expect(existsSync(join(docs, "회의록.docx"))).toBe(false);
	});
});
