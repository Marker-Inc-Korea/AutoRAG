/**
 * Live E2E: decision models from pi's catalog and the sLLM fallback, against real providers.
 *
 * Gated so the offline suite never touches the network or spends credits:
 *
 *   AUTORAG_DECISION_PROVIDERS_LIVE=1 OPENROUTER_API_KEY=... [FIREWORKS_API_KEY=...] [OPENAI_API_KEY=...] \
 *     bunx vitest run test/live-e2e/decision-providers.test.ts
 *
 * Acceptance bar: several decision models from different vendors (TypeSafe Jev, OpenAI's GPT-6 Luna
 * Decisions, Liquid d1, Perplexity's decider) answer a routing-shaped question correctly through
 * OpenRouter; GPT-6 Luna answers natively on OpenAI when a key is present; a small chat LLM answers the
 * same typed questions as the fallback on OpenRouter and on Fireworks; and a session whose Jev model does
 * not exist moves to the fallback with Jev's reason attached. Every chat model is a 2026+ release.
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { PiDecisionBackend } from "../../src/agent/decision-backends.ts";
import { checkDecisionModel, createDecisionModels } from "../../src/agent/decision-model-check.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { SllmDecisionBackend } from "../../src/agent/sllm-decision.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const LIVE = process.env.AUTORAG_DECISION_PROVIDERS_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const FIREWORKS = LIVE && Boolean(process.env.FIREWORKS_API_KEY);
const OPENAI = LIVE && Boolean(process.env.OPENAI_API_KEY);

/** Qwen3.7 Flash (2026-07), the default question-decomposition model. */
const OPENROUTER_SLLM = { provider: "openrouter", id: "qwen/qwen3.7-flash" };
/** NVIDIA Nemotron 3.5 Lightning (2026-08) on Fireworks. */
const FIREWORKS_SLLM = { provider: "fireworks", id: "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b" };

const STATE = "User: rm -rf ~/Documents/2025-tax-return.pdf and then empty the trash.";
const QUESTIONS = [
	{ id: "deletes", type: "noul" as const, question: "Does this request delete an existing file of the user?" },
	{
		id: "kind",
		type: "choice" as const,
		question: "What does the user want done?",
		options: { delete: "remove files", read: "read or search files", chat: "small talk" },
	},
];

/** The small chat model as the CLI resolves it: pi's catalog plus the provider credential. */
async function sllm(reference: { provider: string; id: string }): Promise<SllmDecisionBackend> {
	const root = mkdtempSync(join(tmpdir(), "autorag-decision-providers-live-"));
	try {
		writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [root] }));
		const config = resolveConfig({
			flags: { config: join(root, "config.json") },
			env: process.env,
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel({ ...config, model: reference });
		const apiKey = resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider];
		return new SllmDecisionBackend({ model: resolved.model, ...(apiKey !== undefined ? { apiKey } : {}) });
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
}

describe.skipIf(!LIVE)("decision models through pi's catalog", () => {
	it.each([
		"typesafe/jev-1.13",
		"typesafe/jev-1.14",
		"openai/gpt-6-luna-decisions",
		"liquid/d1",
		"perplexity/pplx-decider-flash-v1.2",
	])(
		"OpenRouter %s judges a destructive request as a delete",
		async (model) => {
			const judgment = await createJevJudge({ backend: "openrouter", model })(STATE, QUESTIONS);
			expect(judgment.backend).toBe("openrouter");
			expect(judgment.model).toBe(model);
			expect(judgment.answers.deletes?.answer as number).toBeGreaterThan(0.5);
			expect(judgment.answers.kind?.answer).toBe("delete");
		},
		60_000,
	);

	it.skipIf(!OPENAI)(
		"OpenAI's native Decisions API answers with gpt-6-luna",
		async () => {
			const judgment = await createJevJudge({ backend: "openai" })(STATE, QUESTIONS);
			expect(judgment.backend).toBe("openai");
			expect(judgment.answers.deletes?.answer as number).toBeGreaterThan(0.5);
			expect(judgment.answers.kind?.answer).toBe("delete");
		},
		60_000,
	);

	it("fails the session check as model-not-found for a model OpenRouter's catalog lacks", async () => {
		const error = await checkDecisionModel(
			createJevJudge({ backend: new PiDecisionBackend({ provider: "openrouter", model: "typesafe/jev-0.0-nope" }) }),
		).then(
			() => undefined,
			(caught: unknown) => caught,
		);
		expect(error).toMatchObject({ code: "decision-model-unavailable", reason: "model-not-found" });
	}, 60_000);
});

describe.skipIf(!LIVE)("sLLM fallback", () => {
	it("answers the typed questions with a small chat model on OpenRouter", async () => {
		const judgment = await createJevJudge({ backend: await sllm(OPENROUTER_SLLM) })(STATE, QUESTIONS);
		expect(judgment.backend).toBe("sllm");
		expect(judgment.model).toBe("openrouter/qwen/qwen3.7-flash");
		expect(judgment.answers.deletes?.answer as number).toBeGreaterThan(0.5);
		expect(judgment.answers.kind?.answer).toBe("delete");
	}, 120_000);

	it.skipIf(!FIREWORKS)(
		"answers the typed questions with a small chat model on Fireworks",
		async () => {
			const judgment = await createJevJudge({ backend: await sllm(FIREWORKS_SLLM) })(STATE, QUESTIONS);
			expect(judgment.model).toBe(`fireworks/${FIREWORKS_SLLM.id}`);
			expect(judgment.answers.deletes?.answer as number).toBeGreaterThan(0.5);
			expect(judgment.answers.kind?.answer).toBe("delete");
		},
		120_000,
	);

	it("takes over a session whose Jev model does not exist, and says why", async () => {
		const models = createDecisionModels({
			backend: "openrouter",
			model: "typesafe/jev-0.0-nope",
			fallback: await sllm(OPENROUTER_SLLM),
		});
		const status = await models.check();
		expect(status.backend).toBe("sllm");
		expect(status.fallbackFrom).toMatchObject({ reason: "model-not-found" });
		const judgment = await models.judge(STATE, QUESTIONS);
		expect(judgment.backend).toBe("sllm");
		expect(judgment.answers.kind?.answer).toBe("delete");
	}, 120_000);
});
