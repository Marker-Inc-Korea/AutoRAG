/**
 * Live E2E: real `jikji` binary proof that candidate evidence and confidence survive from `jikji find --json`
 * into the answer pack and the baseline prefetch (#2060).
 *
 * Gated so the offline suite never needs the jikji binary. Run explicitly with `jikji` on PATH:
 *
 *   AUTORAG_JIKJI_LIVE=1 bunx vitest run test/live-e2e/jikji-evidence.test.ts
 *
 * Isolation: jikji's central index goes to a temp `JIKJI_DATA_DIR` and the agent's state to a temp workspace, so
 * the maintainer's real `~/Library/Application Support/jikji` and `~/.autorag` are never read or written.
 * The reranker is a local stub endpoint; no corpus text leaves the machine.
 */
import { spawnSync } from "node:child_process";
import { once } from "node:events";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const LIVE = process.env.AUTORAG_JIKJI_LIVE === "1" && spawnSync("jikji", ["--version"]).status === 0;

const PAY = "2026년 3분기 보험금 지급 기준 안내.\n지급 청구는 접수 후 14일 이내 처리됩니다.\n";
const INVOICE = "협력사 송장 지급 일정표.\n지급 지연 시 연체료가 부과됩니다.\n사무용품 구매 내역.\n";
const GARDEN = "식물 재배 일지.\n토마토 모종 심기와 물주기 주기.\n";

type PrefetchInternals = {
	prefetchInitialRetrievalContext: (
		query: string,
		searchQueries: readonly string[],
		options: Record<string, never>,
	) => Promise<string>;
};

const squash = (text: string): string => text.replace(/\s+/gu, " ").trim();

describe.skipIf(!LIVE)("jikji candidate evidence over the real jikji binary", () => {
	let root: string;
	let docs: string;
	let agent: AutoRAGAgent;
	let rerankRequests: { query: string; documents: string[] }[];
	let closeServer: () => void;

	beforeAll(async () => {
		root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-jikji-live-")));
		docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "pay.md"), PAY);
		writeFileSync(join(docs, "invoice.md"), INVOICE);
		writeFileSync(join(docs, "garden.md"), GARDEN);

		rerankRequests = [];
		const server = createServer((request, response) => {
			let body = "";
			request.on("data", (chunk) => {
				body += chunk;
			});
			request.on("end", () => {
				const parsed = JSON.parse(body) as { query: string; documents: string[] };
				rerankRequests.push(parsed);
				const words = parsed.query.split(/\s+/u);
				const results = parsed.documents
					.map((document, index) => ({
						document: { text: document },
						index,
						relevance_score: words.filter((word) => document.includes(word)).length / words.length,
					}))
					.sort((a, b) => b.relevance_score - a.relevance_score);
				response.writeHead(200, { "content-type": "application/json" });
				response.end(JSON.stringify({ model: "test-rerank", results }));
			});
		});
		server.listen(0, "127.0.0.1");
		await once(server, "listening");
		const address = server.address();
		const port = typeof address === "object" && address !== null ? address.port : 0;
		closeServer = () => {
			server.closeAllConnections();
			server.close();
		};

		agent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: { autoInstall: false, env: { JIKJI_DATA_DIR: join(root, "jikji-data"), HOME: root } },
			rerank: {
				provider: "openrouter",
				model: "test-rerank",
				apiKey: "test-key",
				baseUrl: `http://127.0.0.1:${port}/`,
			},
			everything: false,
			fsearch: false,
		});
		const prepared = await agent.prepareJikji();
		expect(prepared?.every((result) => result.ok)).toBe(true);
	}, 120_000);

	afterAll(() => {
		closeServer?.();
		if (root !== undefined) rmSync(root, { recursive: true, force: true });
	});

	it("returns evidence text that is really in each candidate's file, plus response confidence", async () => {
		const { answerPack } = await agent.findJikji("보험금 지급 기준");

		expect(answerPack).toBeDefined();
		expect(answerPack?.candidates.length).toBeGreaterThan(0);
		for (const candidate of answerPack?.candidates ?? []) {
			expect(candidate.evidence, candidate.path).toBeTruthy();
			const fileText = squash(readFileSync(candidate.path, "utf8"));
			for (const segment of (candidate.evidence ?? "").split(" | ")) {
				expect(fileText, `${candidate.path} evidence segment`).toContain(squash(segment).replace(/[.]$/u, ""));
			}
		}
		expect(answerPack?.candidates[0]?.path).toBe(join(docs, "pay.md"));
		expect(typeof answerPack?.confidence?.level).toBe("string");
		expect(typeof answerPack?.confidence?.score).toBe("number");
		expect(typeof answerPack?.confidence?.answerability).toBe("string");
	}, 60_000);

	it("sends jikji evidence to the reranker and still reranks whatever jikji's confidence says", async () => {
		const before = rerankRequests.length;
		const question = "분기 보험금 지급 청구 처리 기간";
		const context = await (agent as unknown as PrefetchInternals).prefetchInitialRetrievalContext(
			question,
			[question],
			{},
		);

		expect(rerankRequests).toHaveLength(before + 1);
		const documents = rerankRequests[before]?.documents ?? [];
		expect(documents.length).toBeGreaterThan(0);
		// Every jikji candidate reaches the reranker as evidence text, never as a bare absolute path.
		expect(documents.some((document) => document.startsWith("/"))).toBe(false);
		expect(documents.some((document) => document.includes("보험금 지급"))).toBe(true);
		expect(context).toContain(join(docs, "pay.md"));
		expect(context).toContain("14일 이내 처리됩니다");
	}, 60_000);
});
