import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import type { JikjiAnswerPack, JikjiCandidate } from "../../src/jikji/index.ts";

/** Bench-only view of the agent's private baseline builder and Jikji provider. */
type AgentInternals = {
	jikjiClient: unknown;
	findJikji: (query: string) => Promise<unknown>;
	prefetchInitialRetrievalContext: (
		query: string,
		searchQueries: readonly string[],
		options: Record<string, never>,
	) => Promise<string>;
};

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-evidence-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function candidate(path: string, evidence?: string): JikjiCandidate {
	return { path, nextRead: "original", ...(evidence === undefined ? {} : { evidence }) };
}

function pack(candidates: readonly JikjiCandidate[], overrides: Partial<JikjiAnswerPack> = {}): JikjiAnswerPack {
	return {
		answerPaths: candidates.map((entry) => entry.path),
		paths: candidates.map((entry) => entry.path),
		candidates,
		evidencePack: [],
		handoffAction: "direct_use",
		toolCallPolicy: { stopAfterFind: false, forbiddenTools: [], allowedFollowups: [] },
		agentShouldNotRerank: false,
		...overrides,
	};
}

function agentWith(answerPack: JikjiAnswerPack, extra: Partial<AutoRAGAgentOptions> = {}): AgentInternals {
	const agent = new AutoRAGAgent({
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		...extra,
	});
	const internals = agent as unknown as AgentInternals;
	internals.jikjiClient = {};
	internals.findJikji = async () => ({ answerPack });
	return internals;
}

/** Rerank endpoint that records each request and scores a document by how many query words it contains. */
async function startRerankServer() {
	const requests: { query: string; documents: string[] }[] = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk) => {
			body += chunk;
		});
		request.on("end", () => {
			const parsed = JSON.parse(body) as { query: string; documents: string[] };
			requests.push({ query: parsed.query, documents: parsed.documents });
			const words = parsed.query.toLowerCase().split(/\s+/u);
			const results = parsed.documents
				.map((document, index) => ({
					document: { text: document },
					index,
					relevance_score: words.filter((word) => document.toLowerCase().includes(word)).length / words.length,
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

describe("Jikji candidate evidence in the baseline prefetch (#2060)", () => {
	it("shows each candidate's evidence text under its path in the unranked baseline", async () => {
		const internals = agentWith(
			pack([candidate("/docs/pay.md", "보험금 지급 청구는 14일 이내 처리됩니다"), candidate("/docs/bare.md")]),
		);

		const context = await internals.prefetchInitialRetrievalContext("보험금 지급", ["보험금 지급"], {});

		expect(context).toContain("/docs/pay.md\n보험금 지급 청구는 14일 이내 처리됩니다");
		expect(context).toMatch(/\[e\d+\] \/docs\/bare\.md$/mu);
	});

	it("sends the evidence text, not the path, to the reranker as the candidate content", async () => {
		const server = await startRerankServer();
		try {
			const internals = agentWith(
				pack([candidate("/docs/pay.md", "claims are paid within 14 days"), candidate("/docs/bare.md")]),
				{ rerank: server.rerank },
			);

			const context = await internals.prefetchInitialRetrievalContext("paid claims", ["paid claims"], {});

			expect(server.requests).toHaveLength(1);
			expect(server.requests[0]?.documents).toEqual(["claims are paid within 14 days", "/docs/bare.md"]);
			expect(context).toContain("/docs/pay.md\nclaims are paid within 14 days");
		} finally {
			server.close();
		}
	});

	it("still runs the reranker when jikji is confident and says not to rerank, and a lower candidate can win", async () => {
		const server = await startRerankServer();
		try {
			const internals = agentWith(
				pack(
					[
						candidate("/docs/wrong.pdf", "지급 일정 안내 (unrelated vendor invoice)"),
						candidate("/docs/right.md", "refund exceptions require director approval"),
					],
					{
						agentShouldNotRerank: true,
						confidence: {
							level: "high",
							score: 0.9,
							answerability: "answerable_from_payload",
							requiresLlmRerank: false,
						},
					},
				),
				{ rerank: server.rerank },
			);

			const context = await internals.prefetchInitialRetrievalContext(
				"refund exceptions director approval",
				["refund exceptions director approval"],
				{},
			);

			expect(server.requests).toHaveLength(1);
			expect(context.indexOf("/docs/right.md")).toBeGreaterThan(-1);
			expect(context.indexOf("/docs/right.md")).toBeLessThan(context.indexOf("/docs/wrong.pdf"));
		} finally {
			server.close();
		}
	});
});
