/**
 * Live E2E: real OpenRouter proof for the `jev` decision tool.
 *
 * Gated so the offline suite never touches the network or spends credits.
 * Run explicitly with the OpenRouter key in the environment:
 *
 *   AUTORAG_JEV_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/jev.test.ts
 *
 * Acceptance bar: a narrow, unambiguous state must produce a calibrated
 * probability from Jev through the OpenRouter decisions API, and the tool must
 * surface the provider, model, and usage it used.
 */
import { describe, expect, it } from "vitest";
import { createJevEvaluator, createJevTool } from "../../src/jev/index.ts";

const LIVE = process.env.AUTORAG_JEV_LIVE === "1" && (process.env.OPENROUTER_API_KEY ?? "").length > 0;

describe.skipIf(!LIVE)("jev tool over OpenRouter", () => {
	it("returns a calibrated noul probability for an unambiguous state", async () => {
		const tool = createJevTool(createJevEvaluator({ backend: "openrouter", model: "jev-latest" }));
		const result = await tool.execute("live-jev-noul", {
			state: { message: "Production checkout is down and customers cannot pay." },
			questions: {
				is_urgent: {
					type: "noul",
					instructions: "Does this message describe an urgent production incident?",
				},
			},
		});

		const answer = result.details.answers.is_urgent;
		expect(answer?.type).toBe("noul");
		expect(answer?.noul).toBeGreaterThan(0.5);
		expect(answer?.noul).toBeLessThanOrEqual(1);
		expect(result.details.provider).toBe("openrouter");
		expect(result.details.model.length).toBeGreaterThan(0);
		expect(result.details.usage.totalTokens).toBeGreaterThan(0);
	}, 60_000);

	it("picks one option for a choice question", async () => {
		const tool = createJevTool(createJevEvaluator({ backend: "openrouter", model: "jev-latest" }));
		const result = await tool.execute("live-jev-choice", {
			state: { message: "The invoice PDF fails to download when I click the receipt button." },
			questions: {
				department: {
					type: "choice",
					instructions: "Which team should handle this request?",
					criteria: { billing: "Payments, invoices, refunds", technical: "Bugs, outages, integrations" },
				},
			},
		});

		const answer = result.details.answers.department;
		expect(answer?.type).toBe("choice");
		expect(["billing", "technical"]).toContain(answer?.choice);
	}, 60_000);
});
