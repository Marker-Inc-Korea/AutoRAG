/**
 * Live E2E: a real model is forced onto the `calculator` tool for arithmetic and date questions, through the
 * real two-phase search pipeline (fast answer, then verification) with the real librarian and pi session.
 *
 * Gated so the offline suite never calls a model or spends credits:
 *
 *   AUTORAG_CALCULATOR_LIVE=1 [AUTORAG_CALCULATOR_LIVE_MODEL=anthropic/claude-haiku-5.5] OPENROUTER_API_KEY=... \
 *     bunx vitest run test/live-e2e/calculator.test.ts
 *
 * Acceptance bar: for each question the model (1) calls `calculator` with the right operation, and (2) the
 * answer carries the exact figure the tool printed. Expected values come from an independent reference
 * (Python's decimal and datetime), not from this repository's engine.
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { CALCULATOR_TOOL_NAME } from "../../src/agent/calculator-tool.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const LIVE = process.env.AUTORAG_CALCULATOR_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_CALCULATOR_LIVE_MODEL ?? "anthropic/claude-haiku-5.5";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-calculator-live-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

interface RecordedCall {
	readonly args: Record<string, unknown>;
	readonly output: string;
}

/** Wraps the registered calculator so the test sees exactly what the model sent and what the tool answered. */
interface AgentInternals {
	readonly tools: { name: string; execute: (...args: never[]) => Promise<{ content: unknown }> }[];
}

function textOf(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.map((part) =>
			typeof part === "object" && part !== null && "text" in part && typeof part.text === "string" ? part.text : "",
		)
		.join("");
}

async function liveAgent() {
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	process.env.PI_CODING_AGENT_DIR = agentDir;
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "note.txt"), "Unrelated filler so the corpus is not empty.\n");
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
		minSync: { autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		filee: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		jev: { backend: "openrouter" },
	});
	const calls: RecordedCall[] = [];
	const tool = (agent as unknown as AgentInternals).tools.find((candidate) => candidate.name === CALCULATOR_TOOL_NAME);
	if (tool === undefined) throw new Error("the agent registered no calculator tool");
	const real = tool.execute as (
		id: string,
		params: Record<string, unknown>,
		...rest: never[]
	) => Promise<{ content: unknown }>;
	tool.execute = (async (id: string, params: Record<string, unknown>, ...rest: never[]) => {
		const result = await real(id, params, ...rest);
		calls.push({ args: params, output: textOf(result.content) });
		return result;
	}) as never;
	return { agent, calls };
}

const digitsOnly = (text: string) => text.replace(/[,\s]/g, "");

describe.skipIf(!LIVE)("calculator with a live model", () => {
	it("counts days between two dates with the tool, not from memory", async () => {
		const { agent, calls } = await liveAgent();
		const response = await agent.searchDocuments("How many days are there from 2026-01-01 to 2026-11-08?");
		const call = calls.find((candidate) => candidate.args.operation === "date_diff");
		expect(call, `calculator calls: ${JSON.stringify(calls)}`).toBeDefined();
		expect(call?.output).toContain("totalDays: 311");
		expect(response.answer).toContain("311");
	}, 300_000);

	it("multiplies big integers exactly", async () => {
		const { agent, calls } = await liveAgent();
		const response = await agent.searchDocuments("What is 123456789 times 987654321? Give every digit.");
		expect(calls.length, "the model must call the calculator").toBeGreaterThanOrEqual(1);
		expect(calls.map((call) => call.output).join("\n")).toContain("121932631112635269");
		expect(digitsOnly(response.answer)).toContain("121932631112635269");
	}, 300_000);

	it("converts a time across zones with the zone named, including daylight saving", async () => {
		const { agent, calls } = await liveAgent();
		const response = await agent.searchDocuments(
			"It is 9:00 AM on 2026-07-01 in London (Europe/London). What time is it then in Seoul?",
		);
		const call = calls.find((candidate) => candidate.args.operation === "convert_timezone");
		expect(call, `calculator calls: ${JSON.stringify(calls)}`).toBeDefined();
		expect(call?.output).toContain("2026-07-01T17:00:00+09:00");
		expect(response.answer).toMatch(/17:00|5:00\s*PM|오후 5시/i);
	}, 300_000);

	it("clamps month-end when adding a month", async () => {
		const { agent, calls } = await liveAgent();
		const response = await agent.searchDocuments("What date is exactly one month after 2026-01-31?");
		const call = calls.find((candidate) => candidate.args.operation === "date_add");
		expect(call, `calculator calls: ${JSON.stringify(calls)}`).toBeDefined();
		expect(call?.output).toContain("result: 2026-02-28");
		expect(response.answer).toMatch(/2026-02-28|February 28|2월 28일/i);
	}, 300_000);

	it("counts business days with a holiday list", async () => {
		const { agent, calls } = await liveAgent();
		const response = await agent.searchDocuments(
			"How many business days (Monday to Friday) are there from 2026-01-01 up to but not including 2026-11-08, if 2026-01-01 is a holiday?",
		);
		const call = calls.find((candidate) => candidate.args.operation === "business_days_between");
		expect(call, `calculator calls: ${JSON.stringify(calls)}`).toBeDefined();
		expect(call?.output).toContain("businessDays: 221");
		expect(response.answer).toContain("221");
	}, 300_000);
});
