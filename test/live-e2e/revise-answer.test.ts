/**
 * Live E2E: a real model corrects the first answer it already gave, through `revise_answer`, in the real two-phase
 * search (fast answer, then verification) with the real librarian and pi session.
 *
 * Gated so the offline suite never calls a model or spends credits:
 *
 *   AUTORAG_REVISE_LIVE=1 [AUTORAG_REVISE_LIVE_MODEL=anthropic/claude-haiku-5.5] OPENROUTER_API_KEY=... \
 *     bunx vitest run test/live-e2e/revise-answer.test.ts
 *
 * The corpus holds an outdated policy that matches the question's words and a newer memo, in other words, that
 * overrides it. The first answer is built from what the baseline retrieval finds first. Whether the model gets it
 * wrong is its own: the test does not force it. What is asserted is the contract around a correction:
 *
 *   - the run emits `preliminary` before any `revision`, and `revision` before `complete`;
 *   - every revision names a claim of the first answer, and its `original` is text the caller actually read;
 *   - every `[n]` a revision cites is a result of the final response;
 *   - the final answer applies the revisions: a retracted or replaced claim is gone from it.
 *
 * Each model run is a sample, not a proof: the test retries up to AUTORAG_REVISE_LIVE_ATTEMPTS (default 4) runs for
 * one that produced a correction, and fails when none did, since then the contract was never exercised.
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { realMinSync } from "../helpers/real-minsync.ts";

const LIVE = process.env.AUTORAG_REVISE_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_REVISE_LIVE_MODEL ?? "anthropic/claude-haiku-5.5";
const ATTEMPTS = Number(process.env.AUTORAG_REVISE_LIVE_ATTEMPTS ?? 4);

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-revise-live-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function writeCorpus(docs: string): void {
	mkdirSync(docs, { recursive: true });
	writeFileSync(
		join(docs, "vendor-onboarding-checklist.md"),
		[
			"# Vendor onboarding checklist (v1, 2025-03)",
			"",
			"New vendors must be approved by the procurement lead before the first purchase order.",
			"The standard payment term is net 30 days.",
			"The vendor onboarding checklist is owned by Procurement Operations.",
		].join("\n"),
	);
	writeFileSync(
		join(docs, "hanbit-supply-contract-amendment.md"),
		[
			"# Hanbit Supply - contract amendment No. 3 (2026-09-12)",
			"",
			"For Hanbit Supply Co., the payment term is net 60 days, replacing the net 30 days in the standard checklist.",
			"Purchase orders above 20,000,000 KRW additionally need CFO sign-off.",
		].join("\n"),
	);
}

async function liveAgent(): Promise<AutoRAGAgent> {
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	process.env.PI_CODING_AGENT_DIR = agentDir;
	const docs = join(root, "docs");
	writeCorpus(docs);
	writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
	const config = resolveConfig({
		flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
		env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
		cwd: root,
		readOnly: true,
	});
	const resolved = await resolveAgentModel(config);
	const agent = new AutoRAGAgent({
		model: resolved.model,
		...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
		...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
		searchPaths: [docs],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		// Sparse-only: no embedding daemon to start, and corpus text never leaves the machine.
		minSync: { autoInstall: false, binaryPath: realMinSync(), embedder: { id: "none" } },
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		filee: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		jev: { backend: "openrouter" },
	});
	await agent.refresh(true);
	return agent;
}

async function collect(agent: AutoRAGAgent, query: string): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of agent.searchDocumentsStream(query)) events.push(event);
	return events;
}

const QUESTION = "한빛서플라이(Hanbit Supply) 발주 시 결제 조건은 며칠이고, 승인은 누가 해야 해?";

describe.skipIf(!LIVE)("revise_answer with a live model", () => {
	it("corrects the first answer through the tool, and the final answer applies it", async () => {
		let correctedRun: SearchDocumentsStreamEvent[] | undefined;
		const seen: string[] = [];
		for (let attempt = 1; attempt <= ATTEMPTS && correctedRun === undefined; attempt++) {
			const agent = await liveAgent();
			const events = await collect(agent, QUESTION);
			const types = events.filter((event) => event.type !== "progress").map((event) => event.type);
			seen.push(types.join(">"));
			// A run that found nothing in the corpus proves nothing about corrections: the index must have answered.
			const answered = events.some((event) => event.type === "complete" && event.response.results.length > 0);
			if (!answered) continue;
			if (types.includes("revision")) correctedRun = events;
		}
		expect(correctedRun, `no run corrected anything in ${ATTEMPTS} attempts: ${seen.join(" | ")}`).toBeDefined();
		const events = (correctedRun ?? []).filter((event) => event.type !== "progress");

		const order = events.map((event) => event.type);
		const preliminaryAt = order.indexOf("preliminary");
		const firstRevisionAt = order.indexOf("revision");
		const completeAt = order.indexOf("complete");
		expect(preliminaryAt).toBe(0);
		expect(firstRevisionAt).toBeGreaterThan(preliminaryAt);
		expect(completeAt).toBe(order.length - 1);
		expect(order.slice(firstRevisionAt, completeAt).every((type) => type === "revision")).toBe(true);

		const preliminary = events[preliminaryAt];
		const complete = events[completeAt];
		if (preliminary?.type !== "preliminary" || complete?.type !== "complete") throw new Error("unreachable");
		const resultNumbers = new Set(complete.response.results.map((result) => result.number));
		const claimsSeen = new Set<string>();

		for (const event of events) {
			if (event.type !== "revision") continue;
			const { revision } = event;
			expect(revision.reason.trim().length, "a revision always carries its reason").toBeGreaterThan(0);
			if (revision.original !== undefined) {
				// The struck text is what the caller read, not the model's retelling of it.
				expect(preliminary.response.answer, `original not in the first answer: ${revision.original}`).toContain(
					revision.original,
				);
			}
			if (revision.op !== "add") {
				expect(revision.claimId, "a revision of a claim names it").toMatch(/^c\d+$/u);
				expect(claimsSeen.has(revision.claimId ?? ""), "a claim is revised once").toBe(false);
				claimsSeen.add(revision.claimId ?? "");
			}
			for (const text of [revision.replacement ?? ""]) {
				for (const marker of text.matchAll(/\[(\d+)\]/gu)) {
					expect(
						resultNumbers.has(Number(marker[1])),
						`[${marker[1]}] cites no result of the final response`,
					).toBe(true);
				}
				expect(text, "the model's [eN] ids never reach the caller").not.toMatch(/\[e\d+\]|\[file:/u);
			}
			if (revision.op === "retract" || revision.op === "replace") {
				expect(
					complete.response.answer,
					"a retracted or replaced claim is gone from the final answer",
				).not.toContain(revision.original ?? "\u0000");
			}
			if (revision.op === "replace" || revision.op === "add") {
				expect(complete.response.answer, "a replacement or addition is in the final answer").toContain(
					(revision.replacement ?? "")
						.replace(/\s*\[\d+\]/gu, "")
						.trim()
						.slice(0, 12),
				);
			}
		}
	}, 600_000);
});
