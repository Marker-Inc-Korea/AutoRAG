/**
 * Model-native and keyless web search providers (PR #1592 review comment
 * 5699346366, item 1): port oh-my-pi's credential-reusing providers —
 * Gemini grounding, Anthropic/OpenAI/xAI native web_search — using the model
 * credentials AutoRAG already resolves, plus the always-keyless routes
 * (Perplexity anonymous ask, Parallel MCP). All HTTP is stubbed through the
 * provider `fetch` injection seam; no network.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	getModelNativeSearchAuth,
	modelNativeAuthFromAgentModel,
	resolveModelNativeCredential,
	setModelNativeSearchAuth,
} from "../../src/web/search/model-auth.ts";
import { AnthropicProvider } from "../../src/web/search/providers/anthropic.ts";
import type { FetchImpl } from "../../src/web/search/providers/base.ts";
import { CodexProvider } from "../../src/web/search/providers/codex.ts";
import { GeminiProvider } from "../../src/web/search/providers/gemini.ts";
import { ParallelProvider } from "../../src/web/search/providers/parallel.ts";
import { PerplexityProvider } from "../../src/web/search/providers/perplexity.ts";
import { XaiProvider } from "../../src/web/search/providers/xai.ts";
import { isSearchProviderId, SEARCH_PROVIDER_ORDER } from "../../src/web/search/types.ts";

const MODEL_ENV_NAMES = [
	"ANTHROPIC_API_KEY",
	"OPENAI_API_KEY",
	"GEMINI_API_KEY",
	"GOOGLE_API_KEY",
	"XAI_API_KEY",
] as const;
const originalEnv = Object.fromEntries(MODEL_ENV_NAMES.map((name) => [name, process.env[name]]));

beforeEach(() => {
	for (const name of MODEL_ENV_NAMES) delete process.env[name];
	setModelNativeSearchAuth(undefined);
});

afterEach(() => {
	setModelNativeSearchAuth(undefined);
	for (const name of MODEL_ENV_NAMES) {
		const value = originalEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
});

/** Stub fetch that records the request and returns the given JSON body. */
function stubFetchJson(
	body: unknown,
	init?: { status?: number },
): { fetch: FetchImpl; calls: Array<{ url: string; init?: RequestInit }> } {
	const calls: Array<{ url: string; init?: RequestInit }> = [];
	const fetchImpl: FetchImpl = async (input, requestInit) => {
		calls.push({ url: String(input), init: requestInit });
		return new Response(JSON.stringify(body), {
			status: init?.status ?? 200,
			headers: { "content-type": "application/json" },
		});
	};
	return { fetch: fetchImpl, calls };
}

function stubFetchSse(events: unknown[]): { fetch: FetchImpl; calls: Array<{ url: string; init?: RequestInit }> } {
	const calls: Array<{ url: string; init?: RequestInit }> = [];
	const payload = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
	const fetchImpl: FetchImpl = async (input, requestInit) => {
		calls.push({ url: String(input), init: requestInit });
		return new Response(payload, { status: 200, headers: { "content-type": "text/event-stream" } });
	};
	return { fetch: fetchImpl, calls };
}

describe("chain order: credential-reusing and keyless providers lead", () => {
	it("model-native and keyless providers precede the scraped engines", () => {
		const scraped = SEARCH_PROVIDER_ORDER.indexOf("startpage");
		for (const id of ["gemini", "anthropic", "codex", "xai", "perplexity", "parallel"] as const) {
			expect(isSearchProviderId(id), `${id} must be a recognized provider id`).toBe(true);
			expect(SEARCH_PROVIDER_ORDER.indexOf(id), `${id} must precede scraped engines`).toBeLessThan(scraped);
			expect(SEARCH_PROVIDER_ORDER.indexOf(id), `${id} must precede the public fan-out`).toBeLessThan(
				SEARCH_PROVIDER_ORDER.indexOf("public"),
			);
		}
		// Model-native reuse outranks the anonymous/keyless fallbacks.
		expect(SEARCH_PROVIDER_ORDER.indexOf("anthropic")).toBeLessThan(SEARCH_PROVIDER_ORDER.indexOf("perplexity"));
	});
});

describe("model-native auth resolution", () => {
	it("returns undefined with no injected auth and no env", () => {
		expect(resolveModelNativeCredential("anthropic")).toBeUndefined();
		expect(resolveModelNativeCredential("gemini")).toBeUndefined();
		expect(resolveModelNativeCredential("codex")).toBeUndefined();
		expect(resolveModelNativeCredential("xai")).toBeUndefined();
	});

	it("uses the injected agent model credential for the matching provider family", () => {
		setModelNativeSearchAuth({ provider: "anthropic", apiKey: "sk-ant-injected", modelId: "claude-haiku-4-5" });
		expect(getModelNativeSearchAuth()?.apiKey).toBe("sk-ant-injected");
		expect(resolveModelNativeCredential("anthropic")?.apiKey).toBe("sk-ant-injected");
		expect(resolveModelNativeCredential("gemini")).toBeUndefined();
	});

	it("maps pi model provider ids onto search providers", () => {
		setModelNativeSearchAuth({ provider: "openai", apiKey: "sk-openai" });
		expect(resolveModelNativeCredential("codex")?.apiKey).toBe("sk-openai");
		setModelNativeSearchAuth({ provider: "google", apiKey: "gm-key" });
		expect(resolveModelNativeCredential("gemini")?.apiKey).toBe("gm-key");
		setModelNativeSearchAuth({ provider: "xai", apiKey: "xai-key" });
		expect(resolveModelNativeCredential("xai")?.apiKey).toBe("xai-key");
	});

	it("falls back to the provider's model environment key", () => {
		process.env.ANTHROPIC_API_KEY = "sk-ant-env";
		expect(resolveModelNativeCredential("anthropic")?.apiKey).toBe("sk-ant-env");
		process.env.GOOGLE_API_KEY = "gm-env";
		expect(resolveModelNativeCredential("gemini")?.apiKey).toBe("gm-env");
	});

	it("prefers the injected credential over the environment", () => {
		process.env.ANTHROPIC_API_KEY = "sk-ant-env";
		setModelNativeSearchAuth({ provider: "anthropic", apiKey: "sk-ant-injected" });
		expect(resolveModelNativeCredential("anthropic")?.apiKey).toBe("sk-ant-injected");
	});

	it("derives injectable auth from a resolved agent model", () => {
		expect(
			modelNativeAuthFromAgentModel({ provider: "openai", apiKey: "sk-openai", modelId: "gpt-5" }),
		).toMatchObject({ provider: "openai", apiKey: "sk-openai", modelId: "gpt-5" });
		expect(modelNativeAuthFromAgentModel({ provider: "openrouter", apiKey: "sk-or" })).toBeUndefined();
		expect(modelNativeAuthFromAgentModel({ provider: "anthropic", apiKey: undefined })).toBeUndefined();
	});
});

describe("PerplexityProvider (anonymous ask)", () => {
	it("is always available with zero credentials", () => {
		expect(new PerplexityProvider().isAvailable()).toBe(true);
	});

	it("posts to the consumer ask endpoint and maps SSE blocks to a response", async () => {
		const { fetch: fetchImpl, calls } = stubFetchSse([
			{
				blocks: [
					{
						intended_usage: "web_results",
						web_result_block: {
							web_results: [
								{ name: "Result A", url: "https://a.example/page", snippet: "snip", timestamp: "2026-09-01" },
							],
						},
					},
				],
				status: "PENDING",
			},
			{
				blocks: [{ intended_usage: "ask_text", markdown_block: { chunks: ["Synthesized answer"] } }],
				display_model: "turbo",
				final: true,
				uuid: "req-1",
			},
		]);
		const response = await new PerplexityProvider().search({ query: "autorag", fetch: fetchImpl });
		expect(response.provider).toBe("perplexity");
		expect(response.authMode).toBe("anonymous");
		expect(response.answer).toContain("Synthesized answer");
		expect(response.sources.map((source) => source.url)).toContain("https://a.example/page");
		expect(calls[0]?.url).toContain("perplexity_ask");
		const body = JSON.parse(String(calls[0]?.init?.body)) as { query_str?: string };
		expect(body.query_str).toBe("autorag");
	});

	it("parses CRLF-delimited SSE streams", async () => {
		const payload =
			'data: {"blocks":[{"intended_usage":"web_results","web_result_block":{"web_results":[{"name":"R","url":"https://r.example/"}]}}]}\r\n\r\n' +
			'data: {"blocks":[{"intended_usage":"ask_text","markdown_block":{"chunks":["CRLF answer"]}}],"final":true}\r\n\r\n';
		const fetchImpl: FetchImpl = async () =>
			new Response(payload, { status: 200, headers: { "content-type": "text/event-stream" } });
		const response = await new PerplexityProvider().search({ query: "q", fetch: fetchImpl });
		expect(response.answer).toBe("CRLF answer");
		expect(response.sources.map((source) => source.url)).toContain("https://r.example/");
	});

	it("treats the anonymous sign-up deflection as a provider failure", async () => {
		const { fetch: fetchImpl } = stubFetchSse([
			{
				blocks: [{ intended_usage: "ask_text", markdown_block: { chunks: ["Sign up and repeat your request."] } }],
				final: true,
			},
		]);
		await expect(new PerplexityProvider().search({ query: "q", fetch: fetchImpl })).rejects.toThrow(/deflected/);
	});
});

describe("ParallelProvider (keyless MCP)", () => {
	it("is always available with zero credentials", () => {
		expect(new ParallelProvider().isAvailable()).toBe(true);
	});

	it("calls the keyless MCP endpoint with a tools/call web_search request", async () => {
		const { fetch: fetchImpl, calls } = stubFetchJson({
			jsonrpc: "2.0",
			id: 1,
			result: {
				structuredContent: {
					results: [{ title: "Result A", url: "https://a.example/page", excerpts: ["an excerpt"] }],
				},
			},
		});
		const response = await new ParallelProvider().search({ query: "autorag", fetch: fetchImpl });
		expect(response.provider).toBe("parallel");
		expect(response.sources[0]?.url).toBe("https://a.example/page");
		expect(calls[0]?.url).toContain("search.parallel.ai");
		const body = JSON.parse(String(calls[0]?.init?.body)) as { method?: string; params?: { name?: string } };
		expect(body.method).toBe("tools/call");
		expect(body.params?.name).toBe("web_search");
	});
});

describe("AnthropicProvider (model-native web_search)", () => {
	it("is unavailable without credentials and available with an env key", () => {
		expect(new AnthropicProvider().isAvailable()).toBe(false);
		process.env.ANTHROPIC_API_KEY = "sk-ant-env";
		expect(new AnthropicProvider().isAvailable()).toBe(true);
	});

	it("calls /v1/messages with the web_search tool and maps content blocks", async () => {
		process.env.ANTHROPIC_API_KEY = "sk-ant-env";
		const { fetch: fetchImpl, calls } = stubFetchJson({
			id: "msg_1",
			model: "claude-haiku-4-5",
			content: [
				{ type: "server_tool_use", id: "srvtoolu_1", name: "web_search", input: { query: "autorag" } },
				{
					type: "web_search_tool_result",
					tool_use_id: "srvtoolu_1",
					content: [{ type: "web_search_result", title: "T1", url: "https://a.example", page_age: "2 days ago" }],
				},
				{
					type: "text",
					text: "Answer text",
					citations: [
						{ type: "web_search_result_location", url: "https://a.example", title: "T1", cited_text: "cited" },
					],
				},
			],
			usage: { input_tokens: 100, output_tokens: 50, server_tool_use: { web_search_requests: 1 } },
		});
		const response = await new AnthropicProvider().search({ query: "autorag", fetch: fetchImpl });
		expect(response.provider).toBe("anthropic");
		expect(response.answer).toBe("Answer text");
		expect(response.sources[0]).toMatchObject({ title: "T1", url: "https://a.example" });
		expect(response.citations?.[0]?.citedText).toBe("cited");
		expect(response.usage?.searchRequests).toBe(1);
		expect(calls[0]?.url).toContain("/v1/messages");
		const headers = new Headers(calls[0]?.init?.headers);
		expect(headers.get("x-api-key")).toBe("sk-ant-env");
		const body = JSON.parse(String(calls[0]?.init?.body)) as { tools?: Array<{ type?: string }> };
		expect(body.tools?.[0]?.type).toBe("web_search_20250305");
	});

	it("uses the injected agent credential over the environment", async () => {
		process.env.ANTHROPIC_API_KEY = "sk-ant-env";
		setModelNativeSearchAuth({ provider: "anthropic", apiKey: "sk-ant-injected" });
		const { fetch: fetchImpl, calls } = stubFetchJson({ id: "msg_1", model: "m", content: [], usage: {} });
		await new AnthropicProvider().search({ query: "q", fetch: fetchImpl });
		expect(new Headers(calls[0]?.init?.headers).get("x-api-key")).toBe("sk-ant-injected");
	});
});

describe("GeminiProvider (model-native grounding)", () => {
	it("is unavailable without credentials and available with an env key", () => {
		expect(new GeminiProvider().isAvailable()).toBe(false);
		process.env.GEMINI_API_KEY = "gm-env";
		expect(new GeminiProvider().isAvailable()).toBe(true);
	});

	it("calls generateContent with google_search and maps grounding metadata", async () => {
		process.env.GEMINI_API_KEY = "gm-env";
		const { fetch: fetchImpl, calls } = stubFetchJson({
			candidates: [
				{
					content: { role: "model", parts: [{ text: "Grounded answer" }] },
					groundingMetadata: {
						groundingChunks: [{ web: { uri: "https://a.example", title: "A" } }],
						webSearchQueries: ["autorag"],
					},
				},
			],
			usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 20, totalTokenCount: 30 },
		});
		const response = await new GeminiProvider().search({ query: "autorag", fetch: fetchImpl });
		expect(response.provider).toBe("gemini");
		expect(response.answer).toBe("Grounded answer");
		expect(response.sources[0]).toMatchObject({ title: "A", url: "https://a.example" });
		expect(response.searchQueries).toContain("autorag");
		expect(calls[0]?.url).toContain("generateContent");
		expect(new Headers(calls[0]?.init?.headers).get("x-goog-api-key")).toBe("gm-env");
		const body = JSON.parse(String(calls[0]?.init?.body)) as { tools?: Array<Record<string, unknown>> };
		expect(body.tools?.[0]).toHaveProperty("google_search");
	});
});

describe("CodexProvider (OpenAI Responses web_search)", () => {
	it("is unavailable without credentials and available with an env key", () => {
		expect(new CodexProvider().isAvailable()).toBe(false);
		process.env.OPENAI_API_KEY = "sk-openai-env";
		expect(new CodexProvider().isAvailable()).toBe(true);
	});

	it("calls /v1/responses with web_search and maps output items", async () => {
		process.env.OPENAI_API_KEY = "sk-openai-env";
		const { fetch: fetchImpl, calls } = stubFetchJson({
			id: "resp_1",
			model: "gpt-5",
			status: "completed",
			output: [
				{
					type: "web_search_call",
					id: "ws_1",
					status: "completed",
					action: { type: "search", query: "autorag", sources: [{ url: "https://a.example", title: "A" }] },
				},
				{
					type: "message",
					id: "m_1",
					status: "completed",
					role: "assistant",
					content: [
						{
							type: "output_text",
							text: "Answer with citation",
							annotations: [
								{ type: "url_citation", url: "https://b.example", title: "B", start_index: 0, end_index: 6 },
							],
						},
					],
				},
			],
			usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
		});
		const response = await new CodexProvider().search({ query: "autorag", fetch: fetchImpl });
		expect(response.provider).toBe("codex");
		expect(response.answer).toBe("Answer with citation");
		expect(response.sources.map((source) => source.url)).toEqual(
			expect.arrayContaining(["https://a.example/", "https://b.example/"]),
		);
		expect(calls[0]?.url).toContain("/v1/responses");
		expect(new Headers(calls[0]?.init?.headers).get("authorization")).toBe("Bearer sk-openai-env");
		const body = JSON.parse(String(calls[0]?.init?.body)) as { tools?: Array<{ type?: string }> };
		expect(body.tools?.[0]?.type).toBe("web_search");
	});
});

describe("XaiProvider (xAI Responses web_search)", () => {
	it("is unavailable without credentials and available with an env key", () => {
		expect(new XaiProvider().isAvailable()).toBe(false);
		process.env.XAI_API_KEY = "xai-env";
		expect(new XaiProvider().isAvailable()).toBe(true);
	});

	it("calls api.x.ai /responses with web_search and maps output items", async () => {
		process.env.XAI_API_KEY = "xai-env";
		const { fetch: fetchImpl, calls } = stubFetchJson({
			id: "resp_1",
			model: "grok-4.5",
			output: [
				{
					type: "message",
					id: "m_1",
					role: "assistant",
					content: [
						{
							type: "output_text",
							text: "xAI answer",
							annotations: [{ type: "url_citation", url: "https://x.example", title: "X" }],
						},
					],
				},
			],
		});
		const response = await new XaiProvider().search({ query: "autorag", fetch: fetchImpl });
		expect(response.provider).toBe("xai");
		expect(response.answer).toBe("xAI answer");
		expect(response.sources.map((source) => source.url)).toContain("https://x.example/");
		expect(calls[0]?.url).toContain("api.x.ai");
		expect(new Headers(calls[0]?.init?.headers).get("authorization")).toBe("Bearer xai-env");
	});
});
