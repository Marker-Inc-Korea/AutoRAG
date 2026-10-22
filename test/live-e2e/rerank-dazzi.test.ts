/**
 * Live E2E: real hosted Dazzi Models rerank proof.
 *
 * Gated on AUTORAG_DAZZI_RERANK_LIVE=1 so the offline suite and CI (no stored
 * `/login` credential) never touch the network. Run explicitly after `/login`:
 *
 *   AUTORAG_DAZZI_RERANK_LIVE=1 [AUTORAG_DAZZI_AGENT_DIR=~/.pi/agent] bunx vitest run test/live-e2e/rerank-dazzi.test.ts
 *
 * Proves that a candidate pool containing empty and whitespace-only documents
 * reranks successfully (the server answers HTTP 400 to an empty document).
 */
import { homedir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { readDazziLoginToken } from "../../src/cloud/credential.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { OpenRouterReranker } from "../../src/retrieval/rerank.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

const LIVE = process.env.AUTORAG_DAZZI_RERANK_LIVE === "1";

// The vitest setup isolates PI_CODING_AGENT_DIR, so read the real login explicitly.
const AGENT_DIR = process.env.AUTORAG_DAZZI_AGENT_DIR ?? join(homedir(), ".pi", "agent");

const QUERY = "How do interest rates affect bond prices?";

function doc(id: string, content: string): RetrievalResult {
	return { id, content, source: `/${id}.md`, score: 0.5, metadata: {} };
}

const DOCS: readonly RetrievalResult[] = [
	doc("empty", ""),
	doc("cats", "Cats are small domesticated felines that purr when content."),
	doc("blank", " \n\t "),
	doc("bonds", "A bond's price falls when market interest rates rise, and rises when rates fall."),
	doc("bread", "Sourdough bread rises through wild yeast fermentation and long proofing."),
];

const stubMethod = (results: readonly RetrievalResult[]): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name: "posix",
		type: "posix",
		description: "stub posix",
		status: "active",
		capabilities: [],
	}),
	retrieve: async () => [...results],
});

describe.skipIf(!LIVE)("Dazzi rerank live e2e (real API)", () => {
	it("requires a stored /login credential", () => {
		expect(readDazziLoginToken(AGENT_DIR), "run `autorag tui` and /login first").toBeTruthy();
	});

	it("reranks a pool with blank documents instead of failing with HTTP 400", async () => {
		const reranker = new OpenRouterReranker({ provider: "dazzi", agentDir: AGENT_DIR });
		expect(reranker.describe().available).toBe(true);

		const out = await reranker.rerank(QUERY, DOCS);

		expect(out).toHaveLength(DOCS.length);
		expect(new Set(out.map((entry) => entry.id))).toEqual(new Set(DOCS.map((entry) => entry.id)));
		expect(out[0]?.id).toBe("bonds");
		expect(out[0]?.metadata.rerankProvider).toBe("dazzi");
		// Blank documents are never scored: they trail the ranked ones in input order.
		expect(out.slice(-2).map((entry) => entry.id)).toEqual(["empty", "blank"]);
		expect(out.at(-1)?.metadata.rerankScore).toBeUndefined();
	}, 60_000);

	it("drops blank documents when topN is set", async () => {
		const out = await new OpenRouterReranker({ provider: "dazzi", agentDir: AGENT_DIR }).rerank(QUERY, DOCS, {
			topN: 2,
		});
		expect(out).toHaveLength(2);
		expect(out[0]?.id).toBe("bonds");
		expect(out.map((entry) => entry.id)).not.toContain("empty");
		expect(out.map((entry) => entry.id)).not.toContain("blank");
	}, 60_000);

	it("reranks the engine's merged evidence without a rerank-failed diagnostic", async () => {
		const engine = new RetrievalEngine({
			reranker: new OpenRouterReranker({ provider: "dazzi", agentDir: AGENT_DIR }),
		});
		engine.register(stubMethod(DOCS));
		const { results, diagnostics } = await engine.retrieve(QUERY, { topK: 10 });
		expect(diagnostics.some((entry) => entry.code === "rerank-failed")).toBe(false);
		expect(results[0]?.id).toBe("bonds");
	}, 60_000);
});
