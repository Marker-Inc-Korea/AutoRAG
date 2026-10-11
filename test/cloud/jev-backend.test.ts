import { BackendError } from "jev-use";
import { describe, expect, it } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { AUTORAG_DEFAULT_JEV_MODEL, AutoragJevBackend } from "../../src/cloud/jev-backend.ts";
import { startFakeServer } from "./helpers/fake-server.ts";

/** Narrow a parsed JSON body to a plain object for field reads in assertions. */
function jsonObject(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

describe("AutoragJevBackend", () => {
	it("posts the native dialect to <root>/v1/jev with the account bearer key", async () => {
		const server = await startFakeServer(() => ({
			body: { model: AUTORAG_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.98 } } },
		}));
		try {
			const backend = new AutoragJevBackend({ apiKey: "dz_test", baseUrl: server.url });
			await backend.judge({
				state: { task: "summarize the report" },
				questions: [{ id: "q0", type: "noul", question: "Is the summary complete?" }],
			});
			const request = server.requests[0];
			expect(request?.method).toBe("POST");
			expect(request?.url).toBe("/v1/jev");
			expect(request?.headers.authorization).toBe("Bearer dz_test");
			expect(request?.json).toEqual({
				model: AUTORAG_DEFAULT_JEV_MODEL,
				state: '{"task":"summarize the report"}',
				questions: { q0: { type: "noul", instructions: "Is the summary complete?" } },
			});
		} finally {
			await server.close();
		}
	});

	it("sends a per-call model override and an array state as a JSON string", async () => {
		const server = await startFakeServer(() => ({
			body: { model: "custom", answers: { q0: { type: "noul", noul: 0.5 } } },
		}));
		try {
			const backend = new AutoragJevBackend({ apiKey: "dz_test", baseUrl: server.url });
			await backend.judge({
				state: ["a", "b"],
				questions: [{ id: "q0", type: "noul", question: "ok?" }],
				model: "custom",
			});
			const body = jsonObject(server.requests[0]?.json);
			expect(body?.model).toBe("custom");
			expect(body?.state).toBe('["a","b"]');
		} finally {
			await server.close();
		}
	});

	it("parses noul, choice, and score answers in question order", async () => {
		const server = await startFakeServer(() => ({
			body: {
				model: AUTORAG_DEFAULT_JEV_MODEL,
				answers: {
					n: { type: "noul", noul: 0.98 },
					c: { type: "choice", choice: "b", probabilities: { a: 0.1, b: 0.9 }, confidence: 0.9 },
					s: {
						type: "score",
						score: 1.2,
						probabilities: { "0": 0.2, "1": 0.7, "2": 0.1 },
						legend: { "0": "low" },
					},
				},
				usage: { input_tokens: 7, output_tokens: 2, cost: 0 },
			},
		}));
		try {
			const backend = new AutoragJevBackend({ apiKey: "dz_test", baseUrl: server.url });
			const response = await backend.judge({
				state: "s",
				questions: [
					{ id: "n", type: "noul", question: "yes?" },
					{ id: "c", type: "choice", question: "pick", options: { a: "one", b: "two" } },
					{ id: "s", type: "score", question: "rate", levels: ["low", "mid", "high"] },
				],
			});
			expect(response.answers).toEqual([
				{ answer: 0.98, confidence: undefined },
				{ answer: "b", confidence: 0.9, distribution: { a: 0.1, b: 0.9 } },
				{
					answer: 1.2,
					confidence: undefined,
					distribution: { "0": 0.2, "1": 0.7, "2": 0.1 },
					legend: { "0": "low" },
				},
			]);
			expect(response.usage).toEqual({ inputTokens: 7, outputTokens: 2 });
		} finally {
			await server.close();
		}
	});

	it("maps a 401 to a BackendError naming the backend, status, and server message", async () => {
		const server = await startFakeServer(() => ({
			status: 401,
			body: { error: { message: "invalid API key", type: "invalid_request_error", code: "unauthorized" } },
		}));
		try {
			const backend = new AutoragJevBackend({ apiKey: "dz_bad", baseUrl: server.url });
			const failure = await backend
				.judge({ state: "s", questions: [{ id: "q0", type: "noul", question: "ok?" }] })
				.catch((error: unknown) => error);
			expect(failure).toBeInstanceOf(BackendError);
			const error = failure as BackendError;
			expect(error.backend).toBe("autorag");
			expect(error.status).toBe(401);
			expect(error.message).toContain("[autorag]");
			expect(error.message).toContain("HTTP 401");
			expect(error.message).toContain("invalid API key");
			expect(server.requests).toHaveLength(1);
		} finally {
			await server.close();
		}
	});

	it("retries a 429 after Retry-After and then succeeds", async () => {
		let calls = 0;
		const server = await startFakeServer(() => {
			calls += 1;
			if (calls === 1) {
				return {
					status: 429,
					body: { error: { message: "rate limited", type: "rate_limit_error" } },
					headers: { "retry-after": "0" },
				};
			}
			return { body: { model: AUTORAG_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.7 } } } };
		});
		try {
			const backend = new AutoragJevBackend({ apiKey: "dz_test", baseUrl: server.url });
			const response = await backend.judge({
				state: "s",
				questions: [{ id: "q0", type: "noul", question: "ok?" }],
			});
			expect(server.requests).toHaveLength(2);
			expect(response.answers[0]?.answer).toBe(0.7);
		} finally {
			await server.close();
		}
	});

	it("throws a named, non-silent error when the account key is missing", () => {
		expect(() => new AutoragJevBackend({ env: {} })).toThrow(/AUTORAG_API_KEY/);
	});
});

describe("createJevJudge with the autorag backend", () => {
	it("escalates an unreachable backend with a readable hint instead of throwing", async () => {
		const server = await startFakeServer(() => ({
			status: 401,
			body: { error: { message: "invalid API key", type: "invalid_request_error" } },
		}));
		try {
			const judge = createJevJudge({
				backend: "autorag",
				env: { AUTORAG_API_KEY: "dz_bad", AUTORAG_BASE_URL: server.url },
			});
			const judgment = await judge({ task: "x" }, [{ type: "noul", question: "Is it done?" }]);
			expect(judgment.backend).toBe("autorag");
			expect(judgment.escalated).toBe(true);
			expect(judgment.verdicts[0]?.reason).toBe("unreachable");
			expect(judgment.verdicts[0]?.hint).toContain("HTTP 401");
		} finally {
			await server.close();
		}
	});

	it("rejects on first call when the account key is missing", async () => {
		const judge = createJevJudge({ backend: "autorag", env: {} });
		await expect(judge({ task: "x" }, [{ type: "noul", question: "ok?" }])).rejects.toThrow(/AUTORAG_API_KEY/);
	});

	it("judges a noul question through the hosted endpoint", async () => {
		const server = await startFakeServer(() => ({
			body: { model: AUTORAG_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.91 } } },
		}));
		try {
			const judge = createJevJudge({
				backend: "autorag",
				env: { AUTORAG_API_KEY: "dz_test", AUTORAG_BASE_URL: server.url },
			});
			const judgment = await judge({ task: "x" }, [{ type: "noul", question: "Is it done?" }]);
			expect(judgment.verdicts[0]?.answer).toBe(0.91);
			expect(judgment.escalated).toBe(false);
		} finally {
			await server.close();
		}
	});
});
