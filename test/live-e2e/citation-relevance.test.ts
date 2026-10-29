/**
 * Live E2E: the whole agent over real OpenRouter Jev removes a cited chunk that has nothing to do with the
 * sentence citing it, and keeps the chunk that does.
 *
 * The answering model is a deterministic faux model so the mis-citation (a table of contents and a
 * handbook intro cited for a lease fact) is guaranteed rather than hoped for; retrieval, the evidence
 * ledger, answer derivation, the citation relevance check and the memory judgment are the real code, and
 * Jev is the real hosted judge. The offline suite never calls Jev.
 *
 *   AUTORAG_CITATION_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/citation-relevance.test.ts
 */
import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

const LIVE = process.env.AUTORAG_CITATION_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);

let root: string;
let registration: FauxProviderRegistration | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-citation-live-"));
	mkdirSync(join(root, "docs"), { recursive: true });
});

afterEach(() => {
	registration?.unregister();
	rmSync(root, { recursive: true, force: true });
});

const CHUNKS: readonly { readonly id: string; readonly source: string; readonly content: string }[] = [
	{
		id: "lease",
		source: "/docs/lease.md",
		content: "The Seoul office lease renews on 2027-03-31. Monthly rent is 4,800,000 KRW.",
	},
	{
		id: "toc",
		source: "/docs/handbook.md",
		content: "목차\n1. 개요 ........ 1\n2. 임대 조건 ........ 3\n3. 계약 해지 ........ 7\n4. 부록 ........ 12",
	},
	{
		id: "intro",
		source: "/docs/onboarding.md",
		content:
			"Introduction\nThis handbook introduces the facilities team. Chapter 1 describes onboarding, Chapter 2 covers badges and parking, and Chapter 3 lists contacts.",
	},
];

describe.skipIf(!LIVE)("citation relevance, live Jev", () => {
	it("drops a cited table of contents and a cited introduction, keeps the lease chunk, and renumbers", async () => {
		registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
		// The model cites the TOC and the intro (e2, e3) next to the real chunk (e1).
		registration.setResponses([
			fauxAssistantMessage("Not sure.", { stopReason: "stop" }),
			fauxAssistantMessage(
				"- 서울 사무실 임대차 계약은 2027-03-31에 갱신됩니다 [e1][e2].\n- 월세는 4,800,000원입니다 [e1][e3].",
				{ stopReason: "stop" },
			),
		]);
		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: [join(root, "docs")],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			memoryEmbedder: false,
			webSearch: false,
			jev: { backend: "openrouter" },
		});
		const retrieve = async (): Promise<RetrievalResult[]> =>
			CHUNKS.map((chunk, index) => ({
				id: chunk.id,
				source: chunk.source,
				content: chunk.content,
				score: 1 - index * 0.1,
				metadata: { method: "semantic_search_local_docs" },
			}));
		// `minSyncMethod` is a private field and the established test seam (see jev-pipeline.test.ts): no public setter exists.
		const seam = agent as unknown as { minSyncMethod: unknown };
		seam.minSyncMethod = { isReady: () => true, isBinaryMissing: () => false, retrieve };

		const response = await agent.searchDocuments("서울 사무실 임대차 계약은 언제 갱신되고, 월세는 얼마야?");
		console.log(
			`--- answer ---\n${response.answer}\n--- results: ${response.results.map((r) => r.source).join(", ")}`,
		);
		console.log(
			response.diagnostics
				?.filter((d) => d.code.startsWith("citation-") || d.code.startsWith("evidence-judg"))
				.map((d) => `${d.code}: ${d.message}`)
				.join("\n"),
		);

		const sources = response.results.map((result) => result.source);
		expect(sources, "the lease chunk stays cited").toContain("/docs/lease.md");
		expect(sources, "the table of contents is no longer cited").not.toContain("/docs/handbook.md");
		expect(sources, "the introduction is no longer cited").not.toContain("/docs/onboarding.md");
		expect(response.answer).toMatch(/2027-03-31에 갱신됩니다 \[1\]\./u);
		expect(response.answer).not.toMatch(/\[[2-9]\]/u);
		expect(response.diagnostics?.some((d) => d.code === "citation-irrelevant")).toBe(true);
	}, 120_000);
});
