/**
 * Live E2E: real `jikji` binary proof that Jikji's files reach the verification prompt as starting points and
 * stay out of the fast answer's context and the reranker pool (#2060).
 *
 * Gated so the offline suite never needs the jikji binary. Run explicitly with `jikji` on PATH:
 *
 *   AUTORAG_JIKJI_LIVE=1 bunx vitest run test/live-e2e/jikji-starting-points.test.ts
 *
 * Isolation: jikji's central index goes to a temp `JIKJI_DATA_DIR` and the agent's state to a temp workspace, so
 * the maintainer's real `~/Library/Application Support/jikji` and `~/.autorag` are never read or written.
 * MinSync is an injected method and the reranker a local stub endpoint; no corpus text leaves the machine.
 */
import { spawnSync } from "node:child_process";
import { once } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const LIVE = process.env.AUTORAG_JIKJI_LIVE === "1" && spawnSync("jikji", ["--version"]).status === 0;

const DOCS: Record<string, string> = {
	"0417.md": "# 인사규정 제27조\n퇴직금은 퇴직일로부터 14일 이내에 지급한다.\n",
	"0233.md": "# 취업규칙 별표\n성과급은 매년 3월 말일에 지급하며 지급일 이전 퇴사자는 지급 대상에서 제외한다.\n",
	"0720.md": "# 출장비 정산\n출장비는 정산서 제출 후 익월 10일에 지급한다.\n",
};

type Internals = {
	minSyncMethod: unknown;
	jikjiStartingPoints: readonly string[];
	prefetchInitialRetrievalContext: (
		query: string,
		searchQueries: readonly string[],
		options: Record<string, never>,
	) => Promise<string>;
};

describe.skipIf(!LIVE)("jikji files as verification starting points over the real jikji binary", () => {
	let root: string;
	let docs: string;
	let agent: AutoRAGAgent;
	let internals: Internals;
	let rerankRequests: { query: string; documents: string[] }[];
	let closeServer: () => void;

	beforeAll(async () => {
		root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-jikji-live-")));
		docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		for (const [name, text] of Object.entries(DOCS)) writeFileSync(join(docs, name), text);

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
		internals = agent as unknown as Internals;
		internals.minSyncMethod = {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async () => [
				{
					id: "chunk-0",
					source: join(docs, "0233.md"),
					content: DOCS["0233.md"],
					score: 0.9,
					metadata: {},
				},
				{
					id: "chunk-1",
					source: join(docs, "0720.md"),
					content: DOCS["0720.md"],
					score: 0.8,
					metadata: {},
				},
			],
		};
		const prepared = await agent.prepareJikji();
		expect(prepared?.every((result) => result.ok)).toBe(true);
	}, 120_000);

	afterAll(() => {
		closeServer?.();
		if (root !== undefined) rmSync(root, { recursive: true, force: true });
	});

	it("reranks without Jikji's paths and keeps the real files for the verification prompt", async () => {
		const before = rerankRequests.length;
		const question = "지급일 이전에 퇴사하면 지급 대상에서 제외되나요";
		const context = await internals.prefetchInitialRetrievalContext(question, [question], {});

		expect(rerankRequests).toHaveLength(before + 1);
		expect(rerankRequests[before]?.documents).toEqual([DOCS["0233.md"], DOCS["0720.md"]]);
		// The two injected MinSync chunks may show their own source paths; a file only Jikji found must not appear.
		const jikjiOnly = internals.jikjiStartingPoints.filter(
			(path) => path !== join(docs, "0233.md") && path !== join(docs, "0720.md"),
		);
		expect(jikjiOnly.length).toBeGreaterThan(0);
		for (const path of jikjiOnly) expect(context).not.toContain(path);

		expect(internals.jikjiStartingPoints.length).toBeGreaterThan(0);
		for (const path of internals.jikjiStartingPoints) expect(existsSync(path), path).toBe(true);
		expect(internals.jikjiStartingPoints[0]).toBe(join(docs, "0233.md"));

		const prompt = agent.buildRefinementPrompt(question, {}, "first answer", true, [], internals.jikjiStartingPoints);
		expect(prompt).toContain(`start exploring from them.\n- ${join(docs, "0233.md")}`);
	}, 60_000);

	it("reranks even when jikji reports high confidence and says not to rerank", async () => {
		const found = await agent.findJikji("지급");
		expect(found.answerPack?.candidates.length).toBeGreaterThan(0);
		// Upstream's own verdict on this one-word query; it must not decide whether we rerank.
		const verdict = found.answerPack?.agentShouldNotRerank;

		const before = rerankRequests.length;
		await internals.prefetchInitialRetrievalContext("지급", ["지급"], {});

		expect(rerankRequests, `jikji agentShouldNotRerank=${String(verdict)}`).toHaveLength(before + 1);
	}, 60_000);
});
