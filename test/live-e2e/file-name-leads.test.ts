/**
 * Live E2E: real Jev decides, once verification is needed, whether file names
 * would help; a real model proposes names; the name index is searched with all of
 * them at once; and the verification phase gets the files and their directories
 * as leads. The fast answer never sees file names.
 *
 * Gated on AUTORAG_FILE_LEADS_LIVE=1 plus OPENROUTER_API_KEY and fsearch-cli on
 * PATH. The agent model id comes from AUTORAG_FILE_LEADS_MODEL (an OpenRouter
 * id); pick one released in 2026 or later.
 *
 *   AUTORAG_FILE_LEADS_LIVE=1 AUTORAG_FILE_LEADS_MODEL=anthropic/claude-haiku-5.5 \
 *     OPENROUTER_API_KEY=... bun run vitest run test/live-e2e/file-name-leads.test.ts
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { SearchDocumentDiagnostic, SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { resolveModel } from "../../src/cli/config.ts";

const MODEL_ID = process.env.AUTORAG_FILE_LEADS_MODEL ?? "";
const HAS_FSEARCH = spawnSync("fsearch-cli", ["--version"]).status === 0;
const LIVE =
	process.env.AUTORAG_FILE_LEADS_LIVE === "1" &&
	MODEL_ID.length > 0 &&
	(process.env.OPENROUTER_API_KEY ?? "").length > 0 &&
	HAS_FSEARCH;

function diagnostic(response: SearchDocumentsResponse, code: SearchDocumentDiagnostic["code"]): string | undefined {
	return response.diagnostics?.find((entry) => entry.code === code)?.message;
}

function step(agent: AutoRAGAgent, id: string): { status: string; detail?: string } | undefined {
	const internals = agent as unknown as {
		searchProgress: { snapshot(): { steps: readonly { id: string; status: string; detail?: string }[] } };
	};
	return internals.searchProgress.snapshot().steps.find((entry) => entry.id === id);
}

describe.skipIf(!LIVE)("live: file-name leads reach the verification phase, never the fast answer", () => {
	let root: string;
	let docs: string;
	let agent: AutoRAGAgent;
	/** Every prompt the session model received, in order, read from the verification and fast builders. */
	const prompts: { kind: "fast" | "verify"; text: string }[] = [];

	beforeAll(async () => {
		root = mkdtempSync(join(tmpdir(), "autorag-file-leads-live-"));
		docs = join(root, "docs");
		mkdirSync(join(docs, "billing"), { recursive: true });
		writeFileSync(join(docs, "billing", "invoice-2026-03.txt"), "March 2026 invoice. Total 1,200,000 KRW.\n");
		writeFileSync(join(docs, "billing", "receipt-2026-02.txt"), "February 2026 invoice. Total 900,000 KRW.\n");
		writeFileSync(join(docs, "lunch-menu.txt"), "Monday bibimbap. Tuesday kimchi stew.\n");
		writeFileSync(join(docs, "lease-notes.md"), "Lease notes. The security deposit is 4,800,000 KRW.\n");
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
			jikji: false,
			webSearch: false,
		});
		await agent.refresh(false, { methods: ["fsearch"] });
		// Record what each phase's prompt carried, without changing it.
		const original = {
			fast: agent.buildFastAnswerPrompt.bind(agent),
			verify: agent.buildRefinementPrompt.bind(agent),
		};
		agent.buildFastAnswerPrompt = (...args: Parameters<typeof original.fast>) => {
			const text = original.fast(...args);
			prompts.push({ kind: "fast", text });
			return text;
		};
		agent.buildRefinementPrompt = (...args: Parameters<typeof original.verify>) => {
			const text = original.verify(...args);
			prompts.push({ kind: "verify", text });
			return text;
		};
	});

	afterAll(async () => {
		await agent.stopFsearch();
		rmSync(root, { recursive: true, force: true });
	});

	it("hands verification the files and their directory when a document cannot be found by content", async () => {
		const response = await agent.searchDocuments(
			"Which file is my March 2026 invoice? I only remember it was named something like invoice or receipt.",
		);

		const fast = prompts.find((entry) => entry.kind === "fast")?.text ?? "";
		const verify = prompts.find((entry) => entry.kind === "verify")?.text ?? "";
		expect(fast.length).toBeGreaterThan(0);
		expect(fast).not.toContain("invoice-2026-03.txt");
		expect(fast).not.toContain("File-name leads");
		// The run needed verification, Jev wanted a file search, and the names were searched.
		expect(verify.length).toBeGreaterThan(0);
		expect(diagnostic(response, "file-search-selected")).toBeDefined();
		expect(step(agent, "name-proposal")?.status).toBe("done");
		expect(step(agent, "file-names")?.status).toBe("done");
		expect(diagnostic(response, "name-proposal-failed")).toBeUndefined();
		// The leads reached the verification prompt: the file, which name found it, and its directory.
		expect(verify).toContain("File-name leads");
		expect(verify).toContain(join(docs, "billing", "invoice-2026-03.txt"));
		expect(verify).toMatch(/\(found by: [^)]*invoice/u);
		expect(verify).toContain(`${join(docs, "billing")} (`);
		expect(response.answer).toContain("invoice-2026-03");
	}, 300_000);

	it("also hands leads for a question about what a named document says, since finding it by name leads to the answer", async () => {
		prompts.length = 0;
		const response = await agent.searchDocuments("What does my lease note say the security deposit is?");

		expect(diagnostic(response, "file-search-selected")).toBeDefined();
		const verify = prompts.find((entry) => entry.kind === "verify")?.text ?? "";
		expect(verify).toContain("File-name leads");
		expect(verify).toContain(join(docs, "lease-notes.md"));
		const fast = prompts.find((entry) => entry.kind === "fast")?.text ?? "";
		expect(fast).not.toContain("lease-notes.md");
	}, 300_000);

	it("adds no leads when the answer would live in a chat, mail, or the web", async () => {
		prompts.length = 0;
		const response = await agent.searchDocuments("What is the latest stable Node.js LTS version right now?");

		expect(diagnostic(response, "file-search-selected")).toBeUndefined();
		expect(step(agent, "name-proposal")).toBeUndefined();
		for (const entry of prompts) expect(entry.text).not.toContain("File-name leads");
		expect(response.answer.length).toBeGreaterThan(0);
	}, 300_000);
});
