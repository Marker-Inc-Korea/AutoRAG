import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { createOcrVisualPageExtractor, createVlmVisualPageExtractor } from "../../src/parser/visual-extractor.ts";

function fakeVisionChatModel(): Model<Api> {
	return {
		id: "google/gemini-3.5-flash-lite",
		name: "Gemini 3.5 Flash Lite",
		api: "openai-completions",
		provider: "openrouter",
		baseUrl: "https://openrouter.ai/api/v1",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0.3, output: 1.2, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 8192,
	};
}

describe("visual page extractors (VLM & OCR)", () => {
	const samplePng = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

	it("VLM extractor sends the page image and extracted-text context and returns markdown", async () => {
		const complete = vi.fn(
			async (
				model: Model<Api>,
				context: { systemPrompt?: string; messages: Array<{ role: string; content: unknown }> },
			): Promise<AssistantMessage> => {
				expect(model.id).toBe("google/gemini-3.5-flash-lite");
				const userMessage = context.messages[0];
				expect(Array.isArray(userMessage?.content)).toBe(true);
				const blocks = userMessage?.content as Array<{
					type: string;
					mimeType?: string;
					data?: string;
					text?: string;
				}>;
				expect(blocks.some((b) => b.type === "image" && b.mimeType === "image/png")).toBe(true);
				expect(blocks.some((b) => b.type === "text" && b.text?.includes("Figure 2"))).toBe(true);
				return {
					role: "assistant",
					content: [{ type: "text", text: "| Quarter | Revenue |\n| --- | --- |\n| Q1 | 12M |\n| Q4 | 31M |" }],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: {
						input: 500,
						output: 40,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 540,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: "stop",
					timestamp: Date.now(),
				};
			},
		);

		const extractor = createVlmVisualPageExtractor({
			model: fakeVisionChatModel(),
			apiKey: "test-key",
			complete,
		});
		const result = await extractor.extract({
			virtualPath: "/docs/report.pdf",
			page: 4,
			image: { data: samplePng, mimeType: "image/png" },
			extractedText: "Figure 2: Annual Revenue",
		});

		expect(result.extractorKind).toBe("vlm");
		expect(result.extractorModel).toBe("openrouter/google/gemini-3.5-flash-lite");
		expect(result.markdown).toContain("| Q4 | 31M |");
		expect(extractor.fingerprint).toContain("vlm");
		expect(extractor.fingerprint).toContain("google/gemini-3.5-flash-lite");
	});

	it("rejects a non-vision chat model for VLM extraction", () => {
		const textOnlyModel: Model<Api> = { ...fakeVisionChatModel(), input: ["text"] };
		expect(() => createVlmVisualPageExtractor({ model: textOnlyModel, apiKey: "k" })).toThrow(/image|vision/i);
	});

	it("OCR extractor delegates to the configured OCR provider and normalizes output", async () => {
		const ocrProvider = vi.fn(async (_bytes: Uint8Array, pageNumber: number, mimeType: string) => {
			expect(pageNumber).toBe(2);
			expect(mimeType).toBe("image/png");
			return "Scanned Chart Label\r\nTotal: 420 USD  ";
		});

		const extractor = createOcrVisualPageExtractor({
			languages: ["en", "ko"],
			ocrProvider,
		});
		const result = await extractor.extract({
			virtualPath: "/docs/scan.pdf",
			page: 2,
			image: { data: samplePng, mimeType: "image/png" },
		});

		expect(result.extractorKind).toBe("ocr");
		expect(result.markdown).toBe("Scanned Chart Label\nTotal: 420 USD");
		expect(extractor.fingerprint).toContain("ocr");
		expect(ocrProvider).toHaveBeenCalledTimes(1);
	});
});
