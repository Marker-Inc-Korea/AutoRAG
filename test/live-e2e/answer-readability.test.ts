/**
 * Live E2E: a real model answers through the full two-phase search and the answer follows the readability rules.
 *
 * Gated on AUTORAG_READABILITY_LIVE=1 plus an OpenRouter credential; the offline suite never calls a model.
 * The librarian, retrieval, the real MinSync index and the model are real. The corpus is English and the
 * question is Korean, so the language rule is exercised against a source/question mismatch.
 *
 *   AUTORAG_READABILITY_LIVE=1 AUTORAG_READABILITY_LIVE_MODEL=qwen/qwen3.7-flash \
 *     bun run vitest run test/live-e2e/answer-readability.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { realMinSync } from "../helpers/real-minsync.ts";

const LIVE = process.env.AUTORAG_READABILITY_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_READABILITY_LIVE_MODEL ?? "qwen/qwen3.7-flash";

let root: string;
let savedEnv: Record<string, string | undefined>;

// The suite's setup files stub Kiwi and the pi agent dir; the real MinSync would fail to load the stub.
const RESTORED_VARIABLES = ["PI_CODING_AGENT_DIR", "KIWI_LIBRARY_PATH", "KIWI_MODEL_PATH"] as const;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "arag-readability-live-"));
	savedEnv = Object.fromEntries(RESTORED_VARIABLES.map((name) => [name, process.env[name]]));
	delete process.env.KIWI_LIBRARY_PATH;
	delete process.env.KIWI_MODEL_PATH;
});

afterEach(() => {
	for (const name of RESTORED_VARIABLES) {
		const value = savedEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	rmSync(root, { recursive: true, force: true });
});

const HANGUL = /[\u3131-\u318E\uAC00-\uD7A3]/gu;

describe.skipIf(!LIVE)("answer readability, live model", () => {
	it("answers a Korean question over English sources in Korean, bold-led, flat bullets, with bold key values", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(
			join(docs, "lease.md"),
			[
				"# Seoul office lease",
				"",
				"The Seoul office lease renews on 2027-03-31.",
				"Monthly rent is 4,800,000 KRW.",
				"The landlord must be notified of non-renewal by 2026-12-31.",
				"The security deposit is 96,000,000 KRW.",
				"",
			].join("\n"),
		);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
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
			minSync: { autoInstall: false, binaryPath: realMinSync() },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
		});
		const refreshed = await agent.refresh(true, { methods: ["parsed", "minsync"] });
		expect(refreshed.minsync?.ok, JSON.stringify(refreshed.minsync)).toBe(true);

		const response = await agent.searchDocuments(
			"서울 사무실 임대차 계약은 언제 갱신되고, 월세와 해지 통보 기한은 어떻게 돼?",
		);
		const answer = response.answer ?? "";
		console.log(`--- live answer (${MODEL_ID}) ---\n${answer}\n--- end ---`);

		// Content is grounded: the facts the sources hold reach the answer.
		expect(answer).toContain("2027");
		expect(answer).toMatch(/4,?800,?000/u);
		expect(answer).toMatch(/2026-12-31|2026년\s*12월\s*31일/u);

		// Language rule: a Korean question over English sources gets a Korean answer.
		const hangul = (answer.match(HANGUL) ?? []).length;
		const letters = (answer.match(/\p{L}/gu) ?? []).length;
		expect(hangul / Math.max(letters, 1), "share of Hangul letters").toBeGreaterThan(0.4);

		const lines = answer.split("\n").filter((line) => line.trim() !== "");

		// Lead rule: the first line is a bold conclusion, not a bullet.
		const first = lines[0] ?? "";
		expect(first, "first line").toMatch(/\*\*[^*]+\*\*/u);
		expect(first, "first line is not a bullet").not.toMatch(/^\s*[-*•]\s/u);

		// Bullet rule: a flat list (no indented bullets) within the 5-bullet cap.
		const bullets = lines.filter((line) => /^\s*([-*•]|\d+\.)\s/u.test(line));
		expect(bullets.length, "bullet count").toBeLessThanOrEqual(5);
		expect(
			bullets.filter((line) => /^\s{2,}([-*•]|\d+\.)\s/u.test(line)),
			"nested bullets",
		).toEqual([]);

		// Emphasis rule: the key values a reader scans for are in bold.
		expect(answer).toMatch(/\*\*[^*]*(2027|4,?800,?000)[^*]*\*\*/u);
	}, 300_000);
});
