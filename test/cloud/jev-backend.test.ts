import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BackendError } from "jev-use";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { DAZZI_LOGIN_REQUIRED_MESSAGE } from "../../src/cloud/credential.ts";
import { DAZZI_DEFAULT_JEV_MODEL, DazziJevBackend } from "../../src/cloud/jev-backend.ts";
import { startFakeServer } from "./helpers/fake-server.ts";

let agentDir: string;

beforeEach(() => {
	agentDir = mkdtempSync(join(tmpdir(), "autorag-jev-dazzi-"));
});

afterEach(() => {
	rmSync(agentDir, { recursive: true, force: true });
});

/** Store a `/login` credential the way Pi does. */
function signIn(token = "dz_login"): void {
	writeFileSync(
		join(agentDir, "auth.json"),
		JSON.stringify({ dazzi: { type: "oauth", access: token, refresh: "", expires: Number.MAX_SAFE_INTEGER } }),
	);
}

/** Narrow a parsed JSON body to a plain object for field reads in assertions. */
function jsonObject(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

describe("DazziJevBackend", () => {
	it("posts the native dialect to <root>/v1/jev with the account bearer key", async () => {
		const server = await startFakeServer(() => ({
			body: { model: DAZZI_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.98 } } },
		}));
		try {
			const backend = new DazziJevBackend({ apiKey: "dz_test", baseUrl: server.url });
			await backend.judge({
				state: { task: "summarize the report" },
				questions: [{ id: "q0", type: "noul", question: "Is the summary complete?" }],
			});
			const request = server.requests[0];
			expect(request?.method).toBe("POST");
			expect(request?.url).toBe("/v1/jev");
			expect(request?.headers.authorization).toBe("Bearer dz_test");
			expect(request?.json).toEqual({
				model: DAZZI_DEFAULT_JEV_MODEL,
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
			const backend = new DazziJevBackend({ apiKey: "dz_test", baseUrl: server.url });
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
				model: DAZZI_DEFAULT_JEV_MODEL,
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
			const backend = new DazziJevBackend({ apiKey: "dz_test", baseUrl: server.url });
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
			const backend = new DazziJevBackend({ apiKey: "dz_bad", baseUrl: server.url });
			const failure = await backend
				.judge({ state: "s", questions: [{ id: "q0", type: "noul", question: "ok?" }] })
				.catch((error: unknown) => error);
			expect(failure).toBeInstanceOf(BackendError);
			const error = failure as BackendError;
			expect(error.backend).toBe("dazzi");
			expect(error.status).toBe(401);
			expect(error.message).toContain("[dazzi]");
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
			return { body: { model: DAZZI_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.7 } } } };
		});
		try {
			const backend = new DazziJevBackend({ apiKey: "dz_test", baseUrl: server.url });
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

	it("reads the bearer token from the stored login credential", async () => {
		signIn("dz_login");
		const server = await startFakeServer(() => ({
			body: { model: DAZZI_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.5 } } },
		}));
		try {
			const backend = new DazziJevBackend({ agentDir, baseUrl: server.url });
			await backend.judge({ state: "s", questions: [{ id: "q0", type: "noul", question: "ok?" }] });
			expect(server.requests[0]?.headers.authorization).toBe("Bearer dz_login");
		} finally {
			await server.close();
		}
	});

	it("throws a named, non-silent error when not signed in, even if AUTORAG_API_KEY is set", () => {
		expect(() => new DazziJevBackend({ agentDir, env: { AUTORAG_API_KEY: "dz_env" } })).toThrow(
			DAZZI_LOGIN_REQUIRED_MESSAGE,
		);
	});
});

describe("createJevJudge with the dazzi backend", () => {
	it("escalates an unreachable backend with a readable hint instead of throwing", async () => {
		signIn("dz_bad");
		const server = await startFakeServer(() => ({
			status: 401,
			body: { error: { message: "invalid API key", type: "invalid_request_error" } },
		}));
		try {
			const judge = createJevJudge({ backend: "dazzi", agentDir, env: { AUTORAG_BASE_URL: server.url } });
			const judgment = await judge({ task: "x" }, [{ type: "noul", question: "Is it done?" }]);
			expect(judgment.backend).toBe("dazzi");
			expect(judgment.escalated).toBe(true);
			expect(judgment.verdicts[0]?.reason).toBe("unreachable");
			expect(judgment.verdicts[0]?.hint).toContain("HTTP 401");
		} finally {
			await server.close();
		}
	});

	it("rejects on first call when not signed in", async () => {
		const judge = createJevJudge({ backend: "dazzi", agentDir, env: {} });
		await expect(judge({ task: "x" }, [{ type: "noul", question: "ok?" }])).rejects.toThrow(
			/jev\.backend "dazzi" needs a sign-in/,
		);
	});

	it("picks the hosted backend when signed in and no backend is named", async () => {
		signIn("dz_test");
		const server = await startFakeServer(() => ({
			body: { model: DAZZI_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.8 } } },
		}));
		try {
			const judge = createJevJudge({ agentDir, env: { AUTORAG_BASE_URL: server.url } });
			const judgment = await judge({ task: "x" }, [{ type: "noul", question: "ok?" }]);
			expect(judgment.backend).toBe("dazzi");
			expect(judgment.verdicts[0]?.answer).toBe(0.8);
		} finally {
			await server.close();
		}
	});

	it("names the credentials to set when no backend is named, nobody is signed in, and no key is set", async () => {
		for (const name of [
			"OPENROUTER_API_KEY",
			"OPENAI_API_KEY",
			"TYPESAFE_API_KEY",
			"AI_GATEWAY_API_KEY",
			"CLOUDFLARE_API_KEY",
			"OPENCODE_API_KEY",
		]) {
			vi.stubEnv(name, "");
		}
		try {
			const judge = createJevJudge({ agentDir, env: {} });
			await expect(judge({ task: "x" }, [{ type: "noul", question: "ok?" }])).rejects.toThrow(
				/No Jev credentials found: set OPENROUTER_API_KEY.*sign in to Dazzi Models/u,
			);
		} finally {
			vi.unstubAllEnvs();
		}
	});

	it("judges a noul question through the hosted endpoint", async () => {
		signIn("dz_test");
		const server = await startFakeServer(() => ({
			body: { model: DAZZI_DEFAULT_JEV_MODEL, answers: { q0: { type: "noul", noul: 0.91 } } },
		}));
		try {
			const judge = createJevJudge({ backend: "dazzi", agentDir, env: { AUTORAG_BASE_URL: server.url } });
			const judgment = await judge({ task: "x" }, [{ type: "noul", question: "Is it done?" }]);
			expect(judgment.verdicts[0]?.answer).toBe(0.91);
			expect(judgment.escalated).toBe(false);
		} finally {
			await server.close();
		}
	});
});
