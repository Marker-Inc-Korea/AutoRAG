import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions, type AutoRAGRetrievalLimits } from "../../src/agent/agent.ts";
import type { JikjiAnswerPack } from "../../src/jikji/index.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

/** Bench-only view of the agent's private baseline builder, Jikji provider and MinSync method. */
type AgentInternals = {
	minSyncMethod: unknown;
	jikjiClient: unknown;
	findJikji: (query: string) => Promise<unknown>;
	jikjiStartingPoints: readonly string[];
	prefetchInitialRetrievalContext: (
		query: string,
		searchQueries: readonly string[],
		options: Record<string, never>,
	) => Promise<string>;
	buildRefinementPrompt: AutoRAGAgent["buildRefinementPrompt"];
};

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-start-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function pack(paths: readonly string[], overrides: Partial<JikjiAnswerPack> = {}): JikjiAnswerPack {
	return {
		answerPaths: paths,
		paths,
		candidates: paths.map((path) => ({ path, nextRead: "original" })),
		evidencePack: [],
		handoffAction: "direct_use",
		toolCallPolicy: { stopAfterFind: false, forbiddenTools: [], allowedFollowups: [] },
		agentShouldNotRerank: false,
		...overrides,
	};
}

function agentWith(
	findJikji: (query: string) => Promise<unknown>,
	extra: Partial<AutoRAGAgentOptions> = {},
	limits?: AutoRAGRetrievalLimits,
): AgentInternals {
	const agent = new AutoRAGAgent({
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		...(limits === undefined ? {} : { limits }),
		...extra,
	});
	const internals = agent as unknown as AgentInternals;
	internals.jikjiClient = {};
	internals.findJikji = findJikji;
	return internals;
}

function injectMinSync(internals: AgentInternals, contents: readonly string[]): void {
	internals.minSyncMethod = {
		isReady: () => true,
		isBinaryMissing: () => false,
		retrieve: async (): Promise<RetrievalResult[]> =>
			contents.map((content, index) => ({
				id: `row-${index}`,
				source: `/docs/chunk-${index}.md`,
				content,
				score: 1 - index / 10,
				metadata: {},
			})),
	};
}

/** Rerank endpoint that records every request and scores a document by the query words it contains. */
async function startRerankServer() {
	const requests: { query: string; documents: string[] }[] = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk) => {
			body += chunk;
		});
		request.on("end", () => {
			const parsed = JSON.parse(body) as { query: string; documents: string[] };
			requests.push(parsed);
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
	return {
		requests,
		rerank: {
			provider: "openrouter",
			model: "test-rerank",
			apiKey: "test-key",
			baseUrl: `http://127.0.0.1:${port}/`,
		} as const,
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

const options: RetrievalOptions = {};

describe("Jikji files as verification starting points (#2060)", () => {
	it("keeps Jikji's files out of the fast answer's context and the reranker pool", async () => {
		const server = await startRerankServer();
		try {
			const internals = agentWith(async () => ({ answerPack: pack(["/docs/0417.md", "/docs/0233.md"]) }), {
				rerank: server.rerank,
			});
			injectMinSync(internals, ["퇴직금은 14일 이내에 지급한다", "성과급은 3월에 지급한다"]);

			const context = await internals.prefetchInitialRetrievalContext("퇴직금 지급", ["퇴직금 지급"], {});

			expect(server.requests).toHaveLength(1);
			expect(server.requests[0]?.documents).toEqual(["퇴직금은 14일 이내에 지급한다", "성과급은 3월에 지급한다"]);
			expect(context).not.toContain("0417.md");
			expect(context).not.toContain("0233.md");
			expect(context).toContain("퇴직금은 14일 이내에 지급한다");
		} finally {
			server.close();
		}
	});

	it("keeps them out of the unranked baseline too, and numbers only what it shows", async () => {
		const internals = agentWith(async () => ({ answerPack: pack(["/docs/0417.md"]) }));
		injectMinSync(internals, ["chunk one", "chunk two"]);

		const context = await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(context).not.toContain("0417.md");
		expect(context).not.toContain("Jikji initial candidates");
		expect(context.match(/^\[e\d+\]/gmu)).toEqual(["[e1]", "[e2]"]);
	});

	it("still reranks when Jikji is confident and says not to rerank", async () => {
		const server = await startRerankServer();
		try {
			const internals = agentWith(
				async () => ({
					answerPack: pack(["/docs/wrong.pdf"], {
						agentShouldNotRerank: true,
						confidence: { level: "high", score: 0.9, answerability: "answerable_from_payload" },
					}),
				}),
				{ rerank: server.rerank },
			);
			injectMinSync(internals, ["unrelated vendor invoice", "refund exceptions require director approval"]);

			const context = await internals.prefetchInitialRetrievalContext(
				"refund exceptions director approval",
				["refund exceptions director approval"],
				{},
			);

			expect(server.requests).toHaveLength(1);
			expect(context.indexOf("refund exceptions require director approval")).toBeLessThan(
				context.indexOf("unrelated vendor invoice"),
			);
		} finally {
			server.close();
		}
	});

	it("collects every query's files once, in Jikji's order, up to prefetch.jikjiPathLimit", async () => {
		const perQuery: Record<string, string[]> = {
			first: ["/docs/a.md", "/docs/b.md", "/docs/c.md"],
			second: ["/docs/b.md", "/docs/d.md"],
		};
		const internals = agentWith(
			async (query) => ({ answerPack: pack(perQuery[query] ?? []) }),
			{},
			{
				prefetch: { jikjiPathLimit: 4 },
			},
		);
		injectMinSync(internals, []);

		await internals.prefetchInitialRetrievalContext("question", ["first", "second"], {});

		expect(internals.jikjiStartingPoints).toEqual(["/docs/a.md", "/docs/b.md", "/docs/d.md", "/docs/c.md"]);
	});

	it("has no starting points when Jikji found nothing or is unavailable", async () => {
		const internals = agentWith(async () => ({ answerPack: undefined }));
		internals.jikjiStartingPoints = ["/stale/from/last/run.md"];
		injectMinSync(internals, []);

		await internals.prefetchInitialRetrievalContext("q", ["q"], {});

		expect(internals.jikjiStartingPoints).toEqual([]);
	});

	it("tells the verification prompt to start exploring from the files, and says nothing without them", () => {
		const internals = agentWith(async () => ({ answerPack: undefined }));

		const withFiles = internals.buildRefinementPrompt(
			"q",
			options,
			"first answer",
			true,
			[],
			["/docs/0417.md", "/docs/0233.md"],
		);
		const without = internals.buildRefinementPrompt("q", options, "first answer", true, [], []);

		expect(withFiles).toContain(
			"start exploring from them.\n- /docs/0417.md\n- /docs/0233.md\n\nNow verify it rigorously.",
		);
		expect(without).not.toContain("Jikji's file discovery");
	});
});
