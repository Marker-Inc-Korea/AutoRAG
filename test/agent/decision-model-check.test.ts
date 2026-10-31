import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { BackendError, type JevBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import {
	checkDecisionModel,
	DECISION_MODEL_PROBE_QUESTION_ID,
	DECISION_MODEL_UNAVAILABLE_CODE,
	DecisionModelUnavailableError,
} from "../../src/agent/decision-model-check.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import type { SearchProgressTracker } from "../../src/agent/search-progress.ts";
import { BaselineDecisionBackend } from "../helpers/baseline-jev.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-decision-check-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

/** A backend whose every call fails the way a provider fails. */
function failing(error: Error): JevBackend & { calls: number } {
	return {
		name: "failing",
		calls: 0,
		async judge() {
			this.calls += 1;
			throw error;
		},
	};
}

/** Counts the probe calls a backend receives, answering every question like the baseline. */
function counting(): JevBackend & { probes: number } {
	const inner = new BaselineDecisionBackend();
	return {
		name: "counting",
		probes: 0,
		async judge(request) {
			if (request.questions.some((question) => question.id === DECISION_MODEL_PROBE_QUESTION_ID)) this.probes += 1;
			return inner.judge(request);
		},
	};
}

async function unavailable(promise: Promise<unknown>): Promise<DecisionModelUnavailableError> {
	const error = await promise.then(
		() => undefined,
		(caught: unknown) => caught,
	);
	if (!(error instanceof DecisionModelUnavailableError)) {
		throw new Error(`expected DecisionModelUnavailableError, got ${String(error)}`);
	}
	return error;
}

describe("checkDecisionModel", () => {
	it("reports the backend and model that answered the probe", async () => {
		const status = await checkDecisionModel(createJevJudge({ backend: new BaselineDecisionBackend() }));
		expect(status.backend).toBe("baseline");
		expect(status.model).toBe("baseline-decision");
	});

	it("fails with the missing-credential reason when no backend credential resolves", async () => {
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend: "openrouter", env: {} })));
		expect(error.code).toBe(DECISION_MODEL_UNAVAILABLE_CODE);
		expect(error.code).toBe("decision-model-unavailable");
		expect(error.reason).toBe("missing-credential");
		expect(error.message).toContain("[decision-model-unavailable]");
		expect(error.message).toMatch(/OPENROUTER_API_KEY/u);
		expect(error.message).toMatch(/cannot be used/u);
	});

	it.each([
		[new BackendError("openrouter", "HTTP 401: No auth credentials found", 401), "unauthorized"],
		[new BackendError("openrouter", "HTTP 403: Forbidden", 403), "unauthorized"],
		[new BackendError("openrouter", "HTTP 402: Insufficient credits", 402), "payment-required"],
		[new BackendError("openrouter", "HTTP 400: Model does-not/exist does not exist", 400), "model-not-found"],
		[new BackendError("openrouter", "HTTP 404: Not Found", 404), "model-not-found"],
		[new BackendError("openrouter", "HTTP 429: Rate limit exceeded", 429), "rate-limited"],
		[new BackendError("openrouter", "HTTP 503: upstream unavailable", 503), "unreachable"],
		[new BackendError("openrouter", "request timed out after 10000ms"), "unreachable"],
		[new BackendError("openrouter", "network error: TypeError: fetch failed"), "unreachable"],
	] as const)("classifies %s as %s", async (cause, reason) => {
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend: failing(cause) })));
		expect(error.reason).toBe(reason);
		expect(error.backend).toBe("failing");
		expect(error.detail).toContain(cause.message);
	});

	it("fails with invalid-response when the backend answers without a verdict for the probe", async () => {
		const backend: JevBackend = {
			name: "silent",
			async judge() {
				return { answers: [], model: "silent-1" };
			},
		};
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend })));
		expect(error.reason).toBe("invalid-response");
		expect(error.backend).toBe("silent");
		expect(error.detail).toMatch(/returned no answer/u);
	});
});

function fauxModel(prompts: string[]) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	registration.setResponses(
		Array.from({ length: 8 }, () => (context: { messages: readonly { role: string }[] }) => {
			prompts.push(String(context.messages.length));
			return fauxAssistantMessage("Answer.", { stopReason: "stop" });
		}),
	);
	registrations.push(registration);
	return registration.getModel();
}

/** MinSync stand-in with no hits, so a run stays offline. */
const emptyMinSync = {
	isReady: () => true,
	isBinaryMissing: () => false,
	retrieve: async () => [],
};

function agentWith(options: Partial<AutoRAGAgentOptions> & Pick<AutoRAGAgentOptions, "model">): AutoRAGAgent {
	const agent = new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
		memoryEmbedder: false,
		...options,
	});
	// The private MinSync method is the established test seam.
	const internals = agent as unknown as { minSyncMethod: unknown };
	internals.minSyncMethod = emptyMinSync;
	return agent;
}

/** A promise and its resolver. tsconfig lib is ES2022, which predates Promise.withResolvers. */
function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve: () => void = () => undefined;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

describe("first question of a session", () => {
	it("fails before any model call when the decision model is unavailable", async () => {
		const prompts: string[] = [];
		const backend = failing(new BackendError("openrouter", "HTTP 401: No auth credentials found", 401));
		const agent = agentWith({ model: fauxModel(prompts), jev: { backend } });

		const error = await unavailable(agent.searchDocuments("where is the signed lease?"));

		expect(error.reason).toBe("unauthorized");
		expect(backend.calls).toBe(1);
		expect(prompts).toEqual([]);
	});

	it("checks again on the next question after a failed check, and the agent is not left busy", async () => {
		const backend = failing(new BackendError("openrouter", "request timed out after 10000ms"));
		const agent = agentWith({ model: fauxModel([]), jev: { backend } });

		await unavailable(agent.searchDocuments("first question"));
		await unavailable(agent.searchDocuments("second question"));

		expect(backend.calls).toBe(2);
	});

	it("checks the decision model once per session, not on every question", async () => {
		const backend = counting();
		const agent = agentWith({ model: fauxModel([]), jev: { backend } });

		const first = await agent.searchDocuments("first question");
		await agent.searchDocuments("second question");

		expect(backend.probes).toBe(1);
		expect(first.diagnostics?.find((diagnostic) => diagnostic.code === "decision-model-checked")?.message).toMatch(
			/counting/u,
		);
	});

	it("shows search progress while the check is in flight, then ends it when the check fails", async () => {
		const entered = deferred();
		const released = deferred();
		const backend: JevBackend = {
			name: "slow",
			async judge() {
				entered.resolve();
				await released.promise;
				throw new BackendError("slow", "HTTP 401: User not found", 401);
			},
		};
		const agent = agentWith({ model: fauxModel([]), jev: { backend } });
		// The private tracker is the test seam the TUI widget reads.
		const internals = agent as unknown as { searchProgress: SearchProgressTracker };

		const run = unavailable(agent.searchDocuments("first question"));
		await entered.promise;
		const during = internals.searchProgress.snapshot();
		expect(during.active).toBe(true);
		expect(during.steps.map((step) => [step.id, step.status])).toEqual([["decision-model", "running"]]);
		released.resolve();
		await run;
		expect(internals.searchProgress.snapshot().active).toBe(false);
	});

	it("rejects a decision model turned off with `jev: false`", () => {
		expect(() => agentWith({ model: fauxModel([]), jev: false as unknown as undefined })).toThrow(
			DecisionModelUnavailableError,
		);
	});
});
