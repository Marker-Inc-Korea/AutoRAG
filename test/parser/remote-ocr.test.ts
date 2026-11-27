import { describe, expect, it, vi } from "vitest";
import { resolveVisualParseSyncOptions } from "../../src/agent/visual-parse-runtime.ts";
import {
	createRemoteApiOcrExtractor,
	findOfficialRemoteOcrModel,
	isOfficialRemoteOcrProvider,
	officialRemoteOcrModelsFor,
} from "../../src/parser/remote-ocr.ts";

const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const input = { virtualPath: "/docs/scan.pdf", page: 3, image: { data: png, mimeType: "image/png" as const } };

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("official remote OCR API extractors", () => {
	it("lists only dedicated OCR / document-parsing models per provider", () => {
		expect(isOfficialRemoteOcrProvider("mistral")).toBe(true);
		expect(isOfficialRemoteOcrProvider("openrouter")).toBe(false);
		expect(officialRemoteOcrModelsFor("nvidia").map((m) => m.id)).toEqual(["nvidia/nemotron-parse"]);
		expect(officialRemoteOcrModelsFor("upstage").map((m) => m.id)).toEqual(["document-parse", "ocr"]);
		expect(officialRemoteOcrModelsFor("zai").map((m) => m.id)).toEqual(["glm-ocr"]);
		expect(findOfficialRemoteOcrModel("mistral", "mistral-ocr-latest")?.defaultApiKeyEnv).toBe("MISTRAL_API_KEY");
		expect(findOfficialRemoteOcrModel("mistral", "mistral-large-latest")).toBeUndefined();
	});

	it("Mistral OCR: posts the page as an image_url data URI to /v1/ocr and joins page markdown", async () => {
		const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body));
			expect(body.model).toBe("mistral-ocr-latest");
			expect(body.document.type).toBe("image_url");
			expect(body.document.image_url).toMatch(/^data:image\/png;base64,/u);
			expect((init?.headers as Record<string, string> | undefined)?.authorization).toBe("Bearer k");
			return json({ pages: [{ index: 0, markdown: "# 청구서\nTotal 420" }], model: "mistral-ocr-latest" });
		});
		const extractor = createRemoteApiOcrExtractor({
			provider: "mistral",
			model: "mistral-ocr-latest",
			apiKey: "k",
			fetch: fetchMock as unknown as typeof fetch,
		});
		expect(extractor.kind).toBe("ocr");
		expect(extractor.remote).toBe(true);
		const result = await extractor.extract(input);
		expect(fetchMock.mock.calls[0]?.[0]).toBe("https://api.mistral.ai/v1/ocr");
		expect(result.markdown).toBe("# 청구서\nTotal 420");
		expect(result.extractorModel).toBe("mistral/mistral-ocr-latest");
	});

	it("NVIDIA Nemotron Parse: image-only markdown_no_bbox tool call, text read from tool_calls arguments", async () => {
		const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body));
			expect(body.model).toBe("nvidia/nemotron-parse");
			expect(body.tools).toEqual([{ type: "function", function: { name: "markdown_no_bbox" } }]);
			expect(body.messages[0].content).toHaveLength(1);
			expect(body.messages[0].content[0].type).toBe("image_url");
			return json({
				choices: [
					{
						message: {
							tool_calls: [
								{
									function: {
										name: "markdown_no_bbox",
										arguments: JSON.stringify([{ text: "Quarterly revenue" }, { text: "| Q1 | 12 |" }]),
									},
								},
							],
						},
					},
				],
			});
		});
		const extractor = createRemoteApiOcrExtractor({
			provider: "nvidia",
			model: "nvidia/nemotron-parse",
			apiKey: "nvapi-x",
			fetch: fetchMock as unknown as typeof fetch,
		});
		const result = await extractor.extract(input);
		expect(fetchMock.mock.calls[0]?.[0]).toBe("https://integrate.api.nvidia.com/v1/chat/completions");
		expect(result.markdown).toBe("Quarterly revenue\n\n| Q1 | 12 |");
	});

	it("Upstage Document Parse: multipart upload with forced OCR, reads content.markdown", async () => {
		const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
			const form = init?.body as FormData;
			expect(form).toBeInstanceOf(FormData);
			expect(form.get("model")).toBe("document-parse");
			expect(form.get("ocr")).toBe("force");
			expect(form.get("document")).toBeInstanceOf(Blob);
			return json({ api: "2.0", content: { markdown: "## 계약서\n제1조", html: "<h2>x</h2>" } });
		});
		const extractor = createRemoteApiOcrExtractor({
			provider: "upstage",
			model: "document-parse",
			apiKey: "up-k",
			fetch: fetchMock as unknown as typeof fetch,
		});
		const result = await extractor.extract(input);
		expect(fetchMock.mock.calls[0]?.[0]).toBe("https://api.upstage.ai/v1/document-digitization");
		expect(result.markdown).toBe("## 계약서\n제1조");
	});

	it("Z.AI GLM-OCR: layout_parsing lives on /api/paas/v4 even when the pi provider baseUrl is the coding endpoint", async () => {
		const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
			const body = JSON.parse(String(init?.body));
			expect(body.model).toBe("glm-ocr");
			expect(body.file).toMatch(/^data:image\/png;base64,/u);
			return json({ md_results: "손글씨 메모: 3시 김 대리 통화", layout_details: [] });
		});
		const extractor = createRemoteApiOcrExtractor({
			provider: "zai",
			model: "glm-ocr",
			apiKey: "z-k",
			baseUrl: "https://api.z.ai/api/coding/paas/v4",
			fetch: fetchMock as unknown as typeof fetch,
		});
		const result = await extractor.extract(input);
		expect(fetchMock.mock.calls[0]?.[0]).toBe("https://api.z.ai/api/paas/v4/layout_parsing");
		expect(result.markdown).toBe("손글씨 메모: 3시 김 대리 통화");
	});

	it("retries 429/503 with backoff but fails fast on 401", async () => {
		let calls = 0;
		const delays: number[] = [];
		const flaky = vi.fn(async () => {
			calls += 1;
			if (calls === 1) return json({ error: "rate limit" }, 429);
			if (calls === 2) return json({ error: "unavailable" }, 503);
			return json({ pages: [{ markdown: "ok" }] });
		});
		const extractor = createRemoteApiOcrExtractor({
			provider: "mistral",
			model: "mistral-ocr-latest",
			apiKey: "k",
			retryBaseMs: 50,
			sleep: async (ms) => {
				delays.push(ms);
			},
			fetch: flaky as unknown as typeof fetch,
		});
		expect((await extractor.extract(input)).markdown).toBe("ok");
		expect(delays).toEqual([50, 100]);

		const unauthorized = vi.fn(async () => json({ error: "bad key" }, 401));
		const strict = createRemoteApiOcrExtractor({
			provider: "mistral",
			model: "mistral-ocr-latest",
			apiKey: "bad",
			fetch: unauthorized as unknown as typeof fetch,
		});
		await expect(strict.extract(input)).rejects.toThrow(/401/u);
		expect(unauthorized).toHaveBeenCalledTimes(1);
	});

	it("fingerprints change with provider, model, and base URL so the visual mirror re-extracts", () => {
		const make = (provider: "mistral" | "zai", model: string, baseUrl?: string) =>
			createRemoteApiOcrExtractor({ provider, model, apiKey: "k", ...(baseUrl === undefined ? {} : { baseUrl }) })
				.fingerprint;
		expect(make("mistral", "mistral-ocr-latest")).not.toBe(make("mistral", "mistral-ocr-2512"));
		expect(make("mistral", "mistral-ocr-latest")).not.toBe(make("zai", "glm-ocr"));
		expect(make("mistral", "mistral-ocr-latest")).not.toBe(
			make("mistral", "mistral-ocr-latest", "https://proxy.example/v1"),
		);
	});

	it("resolveVisualParseSyncOptions resolves official OCR API models using provider env keys", async () => {
		const resolved = await resolveVisualParseSyncOptions({
			visualParse: {
				enabled: true,
				gate: {
					fingerprint: "test-gate",
					judge: async () => ({
						route: "ocr",
						visuallyRich: true,
						probability: 0.9,
						probabilities: { vlm: 0.05, ocr: 0.9, plain: 0.05 },
						model: "jev-1.13",
						backend: "openrouter",
					}),
				},
				ocr: {
					engine: "api",
					model: { provider: "upstage", id: "document-parse", name: "Upstage Document Parse" },
				},
			},
			workspaceRoot: "/tmp/autorag-test-ws",
			languages: ["ko", "en"],
			env: { AUTORAG_AUTH_MODE: "environment", UPSTAGE_API_KEY: "up-secret" },
		});
		expect(resolved?.ocrExtractor?.kind).toBe("ocr");
		expect(resolved?.ocrExtractor?.remote).toBe(true);
		expect(resolved?.ocrExtractor?.modelId).toBe("upstage/document-parse");

		await expect(
			resolveVisualParseSyncOptions({
				visualParse: {
					enabled: true,
					gate: {
						fingerprint: "test-gate",
						judge: async () => ({
							route: "ocr",
							visuallyRich: true,
							probability: 0.9,
							probabilities: { vlm: 0.05, ocr: 0.9, plain: 0.05 },
							model: "jev-1.13",
							backend: "openrouter",
						}),
					},
					ocr: {
						engine: "api",
						model: { provider: "nvidia", id: "nvidia/nemotron-parse" },
					},
				},
				workspaceRoot: "/tmp/autorag-test-ws",
				languages: ["ko", "en"],
				env: { AUTORAG_AUTH_MODE: "environment" },
			}),
		).rejects.toThrow(/Missing credentials for remote OCR provider "nvidia"/u);
	});
});
