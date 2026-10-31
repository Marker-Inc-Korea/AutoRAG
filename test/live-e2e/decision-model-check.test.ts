/**
 * Live E2E: the session's decision-model check against real OpenRouter Jev.
 *
 * Gated so the offline suite never touches the network or spends credits:
 *
 *   AUTORAG_DECISION_CHECK_LIVE=1 OPENROUTER_API_KEY=... bunx vitest run test/live-e2e/decision-model-check.test.ts
 *
 * Acceptance bar: the probe passes on the live default model, a rejected key fails as `unauthorized`, a model id
 * OpenRouter does not serve fails as `model-not-found`, and a first question through `AutoRAGAgent` with a rejected
 * key stops with `decision-model-unavailable` before any chat-model call.
 */
import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { DECISION_PROVIDER_DEFAULT_MODELS } from "../../src/agent/decision-backends.ts";
import { checkDecisionModel, DecisionModelUnavailableError } from "../../src/agent/decision-model-check.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";

const KEY = process.env.OPENROUTER_API_KEY ?? "";
const LIVE = process.env.AUTORAG_DECISION_CHECK_LIVE === "1" && KEY.length > 0;

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-decision-check-live-"));
	mkdirSync(join(root, "docs"), { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

async function failure(promise: Promise<unknown>): Promise<DecisionModelUnavailableError> {
	const error = await promise.then(
		() => undefined,
		(caught: unknown) => caught,
	);
	if (!(error instanceof DecisionModelUnavailableError))
		throw new Error(`expected a decision-model error, got ${error}`);
	return error;
}

describe.skipIf(!LIVE)("decision-model check over OpenRouter", () => {
	it("passes on the live default Jev model", async () => {
		const status = await checkDecisionModel(
			createJevJudge({ backend: "openrouter", env: { OPENROUTER_API_KEY: KEY } }),
		);
		expect(status.backend).toBe("openrouter");
		expect(status.model).toContain(DECISION_PROVIDER_DEFAULT_MODELS.openrouter);
	}, 60_000);

	it("fails as unauthorized with a key OpenRouter rejects", async () => {
		const error = await failure(
			checkDecisionModel(createJevJudge({ backend: "openrouter", env: { OPENROUTER_API_KEY: "sk-or-v1-invalid" } })),
		);
		expect(error.code).toBe("decision-model-unavailable");
		expect(error.reason).toBe("unauthorized");
	}, 60_000);

	it("fails as model-not-found for a model OpenRouter does not serve", async () => {
		const error = await failure(
			checkDecisionModel(
				createJevJudge({ backend: "openrouter", model: "typesafe/jev-0.0-nope", env: { OPENROUTER_API_KEY: KEY } }),
			),
		);
		expect(error.reason).toBe("model-not-found");
	}, 60_000);

	it("stops the first question of a session before any chat-model call", async () => {
		const registration = registerFauxProvider({ api: `faux-live-${randomUUID()}`, models: [{ id: "faux" }] });
		let chatCalls = 0;
		registration.setResponses([
			() => {
				chatCalls += 1;
				return fauxAssistantMessage("never", { stopReason: "stop" });
			},
		]);
		const previous = process.env.OPENROUTER_API_KEY;
		process.env.OPENROUTER_API_KEY = "sk-or-v1-invalid";
		try {
			const agent = new AutoRAGAgent({
				model: registration.getModel(),
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				minSync: { autoInstall: false },
				jikji: false,
				memoryEmbedder: false,
				jev: { backend: "openrouter" },
			});
			const error = await failure(agent.searchDocuments("where is the signed lease?"));
			expect(error.reason).toBe("unauthorized");
			expect(error.message).toMatch(/^\[decision-model-unavailable\] AutoRAG cannot be used/u);
			expect(chatCalls).toBe(0);
		} finally {
			process.env.OPENROUTER_API_KEY = previous;
			registration.unregister();
		}
	}, 60_000);
});
