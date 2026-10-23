/**
 * Live E2E: a real model plans a conversion that has no direct route
 * (JSON -> PNG goes through PDF) from the conversion map in its system prompt,
 * on a machine whose PATH has no filee-cli, so the agent also installs the
 * pinned filee-cli itself on the first conversion.
 *
 * Gated on AUTORAG_FILEE_CHAIN_LIVE=1 plus OPENROUTER_API_KEY; the model id
 * comes from AUTORAG_FILEE_AGENT_MODEL (an OpenRouter id released in 2026 or later).
 *
 *   AUTORAG_FILEE_CHAIN_LIVE=1 AUTORAG_FILEE_AGENT_MODEL=anthropic/claude-haiku-5.5 \
 *     OPENROUTER_API_KEY=... bun run vitest run test/live-e2e/filee-chain-agent.test.ts
 *
 * AUTORAG_HOME is a temp folder, so the real ~/.autorag is never read or written.
 */
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveModel } from "../../src/cli/config.ts";

const MODEL_ID = process.env.AUTORAG_FILEE_AGENT_MODEL ?? "";
const LIVE =
	process.env.AUTORAG_FILEE_CHAIN_LIVE === "1" &&
	MODEL_ID.length > 0 &&
	(process.env.OPENROUTER_API_KEY ?? "").length > 0;

describe.skipIf(!LIVE)("live: a real model chains conversions and the agent installs filee-cli", () => {
	let root: string;
	const saved = { PATH: process.env.PATH, AUTORAG_HOME: process.env.AUTORAG_HOME, NODE_ENV: process.env.NODE_ENV };
	beforeAll(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-filee-chain-"));
		process.env.AUTORAG_HOME = join(root, "home");
		// Drop only the PATH entries that hold a filee/filee-cli, so the managed install is what answers.
		process.env.PATH = (saved.PATH ?? "")
			.split(delimiter)
			.filter((dir) => !existsSync(join(dir, "filee-cli")) && !existsSync(join(dir, "filee")))
			.join(delimiter);
		// The installer refuses the network under NODE_ENV=test; this run must really download.
		process.env.NODE_ENV = "production";
	});
	afterAll(() => {
		for (const [key, value] of Object.entries(saved)) {
			if (value === undefined) delete process.env[key];
			else process.env[key] = value;
		}
		rmSync(root, { recursive: true, force: true });
	});

	it("turns a JSON file into a PNG by way of PDF", async () => {
		const docs = join(root, "docs");
		const elsewhere = join(root, "elsewhere");
		mkdirSync(docs, { recursive: true });
		mkdirSync(elsewhere, { recursive: true });
		const source = join(elsewhere, "분기실적.json");
		writeFileSync(source, JSON.stringify({ 분기: "3분기", 매출증가율: "12%", 신규고객: 320 }, null, 2));

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
			dupey: false,
			everything: false,
			fsearch: false,
			jikji: false,
		} as never);

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

		const response = await agent.searchDocuments(`${source} 파일을 PNG 이미지로 바꿔줘. 결과 PNG 경로를 알려줘.`, {
			topK: 3,
		});
		expect(response.answer.length).toBeGreaterThan(0);

		const converts = calls.filter((call) => call.name === "convert_files");
		const targets = converts.map((call) => String((call.args as { to?: string }).to));
		expect(targets, JSON.stringify(calls)).toContain("png");
		// JSON has no direct route to PNG, so a PDF (or another bridge) must come first.
		expect(converts.length, JSON.stringify(calls)).toBeGreaterThanOrEqual(2);
		expect(targets.indexOf("png")).toBeGreaterThan(0);
		const png = join(elsewhere, "분기실적.png");
		const found = [png, ...(response.answer.match(/\/[^\s"'`]+\.png/g) ?? [])].find((path) => existsSync(path));
		expect(found, response.answer).toBeDefined();
		expect(
			readFileSync(found ?? "")
				.subarray(1, 4)
				.toString(),
		).toBe("PNG");
		// The managed copy was installed by the agent loop itself.
		expect(existsSync(join(root, "home", "tools", "filee-cli"))).toBe(true);
		console.info(`calls: ${JSON.stringify(calls)}`);
	}, 900_000);
});
