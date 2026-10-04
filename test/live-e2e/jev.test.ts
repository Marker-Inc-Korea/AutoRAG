/**
 * Live E2E: real OpenRouter proof for the `jev` decision tool.
 *
 * Gated so the offline suite never touches the network or spends credits.
 * Run explicitly with the OpenRouter key in the environment:
 *
 *   AUTORAG_JEV_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/jev.test.ts
 *
 * Acceptance bar: a narrow, unambiguous state must produce a calibrated
 * probability from Jev through the OpenRouter backend, and the verdicts must
 * carry the backend that served them.
 */
import type { ExtensionAPI, ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { createJevExtension } from "../../src/agent/jev-extension.ts";

const LIVE = process.env.AUTORAG_JEV_LIVE === "1" && (process.env.OPENROUTER_API_KEY ?? "").length > 0;

function jevTool(): ToolDefinition {
	const tools: ToolDefinition[] = [];
	(createJevExtension({ backend: "openrouter", model: "jev-latest" }) as ExtensionFactory)({
		registerTool: (tool: ToolDefinition) => tools.push(tool),
	} as unknown as ExtensionAPI);
	const tool = tools[0];
	if (tool === undefined) throw new Error("expected the jev tool to register");
	return tool;
}

async function run(tool: ToolDefinition, params: unknown): Promise<Record<string, unknown>> {
	const execute = tool.execute as unknown as (
		id: string,
		params: unknown,
	) => Promise<{ details: Record<string, unknown> }>;
	return (await execute("live-jev", params)).details;
}

describe.skipIf(!LIVE)("jev tool over OpenRouter", () => {
	it("returns a calibrated noul probability for an unambiguous state", async () => {
		const details = await run(jevTool(), {
			state: "Production checkout is down and customers cannot pay.",
			questions: [{ id: "urgent", type: "noul", question: "Does this describe an urgent production incident?" }],
		});
		const verdicts = details.verdicts as { type: string; answer: number | string | null }[];
		expect(verdicts[0]?.type).toBe("noul");
		expect(verdicts[0]?.answer as number).toBeGreaterThan(0.5);
		expect(verdicts[0]?.answer as number).toBeLessThanOrEqual(1);
		expect(details.backend).toBe("openrouter");
	}, 60_000);

	it("picks one option for a choice question", async () => {
		const details = await run(jevTool(), {
			state: "The invoice PDF fails to download when I click the receipt button.",
			questions: [
				{
					id: "department",
					type: "choice",
					question: "Which team should handle this request?",
					options: { billing: "Payments, invoices, refunds", technical: "Bugs, outages, integrations" },
				},
			],
		});
		const verdicts = details.verdicts as { type: string; answer: string | null }[];
		expect(verdicts[0]?.type).toBe("choice");
		expect(["billing", "technical"]).toContain(verdicts[0]?.answer);
	}, 60_000);
});
