import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Value } from "typebox/value";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { CALCULATOR_TOOL_NAME, createCalculatorTool } from "../../src/agent/calculator-tool.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

const NOW = new Date("2026-11-08T13:10:45Z");
const tool = createCalculatorTool({ now: () => NOW, hostTimeZone: "Asia/Seoul" });

/** Text of a tool result or message content, whether it is a string or text parts. */
function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((part) =>
			typeof part === "object" && part !== null && "text" in part && typeof part.text === "string" ? part.text : "",
		)
		.join("");
}

const textOf = (result: { content: unknown }) => contentText(result.content);

describe("calculator tool", () => {
	it("is named calculator and returns the computed fields as text for the model", async () => {
		expect(tool.name).toBe("calculator");
		expect(CALCULATOR_TOOL_NAME).toBe("calculator");
		const result = await tool.execute("c1", { operation: "date_diff", from: "2026-01-01", to: "today" });
		expect(result.isError).toBeUndefined();
		expect(textOf(result)).toContain("totalDays: 311");
		expect(textOf(result)).toContain("calendar: 10 months, 7 days");
	});

	it("evaluates expressions exactly", async () => {
		const result = await tool.execute("c1", { operation: "evaluate", expression: "123456789 * 987654321" });
		expect(textOf(result)).toContain("value: 121932631112635269");
	});

	it("reports a bad request as an error result the model can read, not as an exception", async () => {
		const result = await tool.execute("c1", { operation: "date_info", date: "2026-02-30" });
		expect(result.isError).toBe(true);
		expect(textOf(result)).toMatch(/not a real date/i);
		const missing = await tool.execute("c2", { operation: "date_diff", from: "2026-01-01" });
		expect(missing.isError).toBe(true);
		expect(textOf(missing)).toContain('requires "to"');
	});

	it("does not swallow an abort or an unexpected failure", async () => {
		const broken = createCalculatorTool({
			now: () => {
				throw new Error("clock exploded");
			},
		});
		await expect(broken.execute("c1", { operation: "now" })).rejects.toThrow("clock exploded");
	});

	it("declares a schema that accepts every operation and rejects an unknown one", () => {
		for (const operation of [
			"evaluate",
			"now",
			"date_diff",
			"date_add",
			"date_info",
			"convert_timezone",
			"business_days_between",
			"add_business_days",
		]) {
			expect(Value.Check(tool.parameters, { operation })).toBe(true);
		}
		expect(Value.Check(tool.parameters, { operation: "integrate" })).toBe(false);
		expect(Value.Check(tool.parameters, { operation: "date_add", date: "2026-01-01", days: "3" })).toBe(false);
	});

	it("tells the model what it is for and to use it instead of computing in its head", () => {
		expect(tool.description).toMatch(/date/i);
		expect(tool.description).toMatch(/time zone/i);
		expect(tool.description).toMatch(/never/i);
	});

	it("tells the model an abbreviation is a fixed offset, so it does not swap in a daylight-saving region", () => {
		const schema = JSON.stringify(tool.parameters);
		expect(schema).toMatch(/PST is -08:00 all year/);
		expect(schema).toMatch(/Never replace an abbreviation with a region/);
	});
});

describe("calculator in the system prompt", () => {
	it("lists the tool and orders the model to calculate with it, only when the tool exists", () => {
		const prompt = buildSystemPrompt({ toolNames: ["bash", "calculator"], modelId: "test-model" });
		expect(prompt).toContain("**calculator**");
		expect(prompt).toMatch(/date difference|days between/i);
		expect(prompt).toMatch(/never (do|compute|calculate)/i);
		const without = buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" });
		expect(without).not.toContain("calculator");
	});

	it("states that an abbreviation is a fixed offset and that a region is a different claim", () => {
		const prompt = buildSystemPrompt({ toolNames: ["bash", "calculator"], modelId: "test-model" });
		expect(prompt).toContain("PST is UTC-8 all year");
		expect(prompt).toMatch(/Never replace an abbreviation with a region such as America\/Los_Angeles/);
		expect(prompt).toMatch(/several zones \(CST, IST, BST\) is the user's to settle: ask/);
	});

	it("does not list it twice as a caller tool", () => {
		const prompt = buildSystemPrompt({ toolNames: ["bash", "calculator"], modelId: "test-model" });
		expect(prompt).not.toContain("**calculator**: caller-provided tool");
	});
});

let root: string;
beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-calculator-")));
});
afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

const options = () => ({ searchPaths: [root], memoryPath: join(root, "memory.json"), workspacePath: root });
/** The test reaches the agent's private tool registry; there is no public accessor and no runtime shape to validate. */
interface AgentInternals {
	readonly innerAgent: { readonly state: { readonly tools: readonly { readonly name: string }[] } };
	readonly droppedCallerToolNames: readonly string[];
}
const internalsOf = (agent: AutoRAGAgent): AgentInternals => agent as unknown as AgentInternals;
const toolNamesOf = (agent: AutoRAGAgent) =>
	internalsOf(agent).innerAgent.state.tools.map((candidate) => candidate.name);

describe("calculator agent wiring", () => {
	it("registers the calculator by default", () => {
		expect(toolNamesOf(new AutoRAGAgent(options()))).toContain("calculator");
	});

	it("keeps the name reserved so a caller tool cannot shadow it", () => {
		const agent = new AutoRAGAgent({
			...options(),
			tools: [
				{
					name: "calculator",
					label: "x",
					description: "x",
					parameters: { type: "object", properties: {} } as never,
					execute: async () => ({ content: [], details: undefined }),
				},
			],
		});
		expect(internalsOf(agent).droppedCallerToolNames).toContain("calculator");
	});

	it("lets the fast phase call it, because that phase otherwise forbids every tool", () => {
		const agent = new AutoRAGAgent(options());
		const prompt = agent.buildFastAnswerPrompt("how many days until 2026-12-25?", {}, "(no baseline)");
		expect(prompt).toMatch(/calculator is allowed/i);
	});

	it("names it in the verification phase and the extra route", () => {
		const agent = new AutoRAGAgent(options());
		expect(agent.buildExtraPrompt("add 3 months to 2026-01-31")).toMatch(/calculator/);
	});
});

describe("calculator through a real agent session", () => {
	it("runs the calculator when the model calls it and hands the exact result back", async () => {
		const registration = registerFauxProvider({ api: `faux-calc-${randomUUID()}`, models: [{ id: "calc" }] });
		const seenToolResults: string[] = [];
		const responses: FauxResponseStep[] = [
			fauxAssistantMessage(
				[fauxToolCall(CALCULATOR_TOOL_NAME, { operation: "date_diff", from: "2026-01-01", to: "2026-11-08" })],
				{
					stopReason: "toolUse",
				},
			),
			(context) => {
				for (const message of context.messages) {
					if (message.role !== "toolResult") continue;
					seenToolResults.push(contentText(message.content));
				}
				return fauxAssistantMessage("**311 days**", { stopReason: "stop" });
			},
			fauxAssistantMessage("**311 days**", { stopReason: "stop" }),
		];
		registration.setResponses(responses);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		const agent = new AutoRAGAgent({
			...options(),
			model: registration.getModel(),
			apiKey: "test-key",
			minSync: { autoInstall: false },
			jikji: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
			jev: baselineJev(),
		});
		try {
			const response = await agent.searchDocuments("How many days from 2026-01-01 to 2026-11-08?");
			expect(response.answer).toContain("311");
			expect(seenToolResults.join("\n")).toContain("totalDays: 311");
		} finally {
			registration.unregister();
		}
	});
});
