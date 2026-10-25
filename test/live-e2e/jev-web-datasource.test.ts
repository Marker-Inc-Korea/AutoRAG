/**
 * Live E2E: real Jev plus a real model decide whether the public web is one of
 * the sources a question needs, prefetch its hits, and cite them.
 *
 * Gated on AUTORAG_JEV_WEB_LIVE=1 plus OPENROUTER_API_KEY, so the offline suite
 * never calls Jev, a model, or the internet. The agent model id comes from
 * AUTORAG_JEV_WEB_MODEL (an OpenRouter id); pick one released in 2026 or later.
 *
 *   AUTORAG_JEV_WEB_LIVE=1 AUTORAG_JEV_WEB_MODEL=anthropic/claude-haiku-5.5 \
 *     OPENROUTER_API_KEY=... bun run vitest run test/live-e2e/jev-web-datasource.test.ts
 *
 * The run has no configured datasource: the gate must still run, for the web
 * candidate alone.
 */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import type { SearchDocumentDiagnostic, SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { resolveModel } from "../../src/cli/config.ts";

const MODEL_ID = process.env.AUTORAG_JEV_WEB_MODEL ?? "";
const LIVE =
	process.env.AUTORAG_JEV_WEB_LIVE === "1" && MODEL_ID.length > 0 && (process.env.OPENROUTER_API_KEY ?? "").length > 0;

function diagnostic(response: SearchDocumentsResponse, code: SearchDocumentDiagnostic["code"]): string | undefined {
	return response.diagnostics?.find((entry) => entry.code === code)?.message;
}

/** The `datasources-selected` diagnostic, split into the searched and the skipped candidate lists. */
function datasourceChoice(response: SearchDocumentsResponse): { selected: string; skipped: string } {
	const [selected = "", skipped = ""] = (diagnostic(response, "datasources-selected") ?? "").split("; skipped: ");
	return { selected, skipped };
}

function routeOf(response: SearchDocumentsResponse): string | undefined {
	return /routed the question to (\w+)/u.exec(diagnostic(response, "query-routed") ?? "")?.[1];
}

/** Hits the prefetched web search returned, read from the run's progress steps; undefined when it never ran. */
function prefetchedWebHits(agent: AutoRAGAgent): number | undefined {
	const internals = agent as unknown as {
		searchProgress: { snapshot(): { steps: readonly { id: string; status: string; detail?: string }[] } };
	};
	const step = internals.searchProgress.snapshot().steps.find((entry) => entry.id === "web");
	const hits = step?.detail === undefined ? undefined : /(\d+) hits?/u.exec(step.detail)?.[1];
	return hits === undefined ? undefined : Number(hits);
}

/** Ledger entries for web evidence: one per hit whose source is a URL. */
function webLedgerUrls(agent: AutoRAGAgent): string[] {
	const internals = agent as unknown as { evidenceLedger: EvidenceLedger };
	return internals.evidenceLedger
		.takeUnsaved()
		.filter((entry) => entry.method === "web_search" && /^https?:\/\//u.test(entry.source))
		.map((entry) => entry.source);
}

describe.skipIf(!LIVE)("live: Jev offers the public web as a datasource and prefetches it", () => {
	let root: string;
	let docs: string;
	let agent: AutoRAGAgent;

	beforeAll(async () => {
		root = mkdtempSync(join(tmpdir(), "autorag-jev-web-live-"));
		docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(
			join(docs, "release-plan.txt"),
			"Q3 release plan: the launch is scheduled for 2026-11-17, owned by Mina Park.\n",
		);
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
			fsearch: false,
			jikji: false,
		});
	});
	afterAll(() => rmSync(root, { recursive: true, force: true }));

	it("selects the public web for a recent public fact, prefetches it, and cites a URL", async () => {
		const response = await agent.searchDocuments("What is the latest stable Node.js LTS version right now?");

		// Jev routed it locally and picked the web candidate out of the gate.
		expect(routeOf(response)).toBe("local");
		expect(datasourceChoice(response).selected).toMatch(/\bweb \(p=/u);
		// The prefetched web search really ran and returned hits.
		expect(prefetchedWebHits(agent) ?? 0).toBeGreaterThan(0);
		expect(diagnostic(response, "web-search-failed")).toBeUndefined();
		expect(webLedgerUrls(agent).length).toBeGreaterThan(0);
		// The answer cites a URL, and the run's results carry it as a source.
		expect(response.answer).toMatch(/\[\d+\]/u);
		expect(response.results.some((result) => /^https?:\/\//u.test(result.source ?? ""))).toBe(true);
	}, 300_000);

	it("does not select the public web for a question about the user's own records", async () => {
		const response = await agent.searchDocuments("What does my release plan note say about the Q3 launch?");

		expect(routeOf(response)).toBe("local");
		const choice = datasourceChoice(response);
		expect(choice.selected).not.toMatch(/\bweb \(p=/u);
		expect(choice.skipped).toMatch(/\bweb \(p=/u);
		expect(prefetchedWebHits(agent)).toBeUndefined();
		expect(response.answer.length).toBeGreaterThan(0);
	}, 300_000);
});
