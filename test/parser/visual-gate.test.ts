import type { ClassifierContext, ClassifierModel, ClassifierResult } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import {
	BIVIA_DEFAULT_MODEL,
	BIVIA_PROVIDER_ID,
	type DecisionModelRuntime,
} from "../../src/agent/decision-backends.ts";
import { DazziJevBackend, toNativeBody } from "../../src/cloud/jev-backend.ts";
import {
	createVisualPageGate,
	DEFAULT_VISUAL_GATE_THRESHOLD,
	isVisionDecisionModel,
	resolveVisualGateBackend,
	VISION_DECISION_PROVIDER_DEFAULT_MODELS,
} from "../../src/parser/visual-gate.ts";

function fakeClassifierModel(
	provider: string,
	id: string,
	input: Array<"text" | "image"> = ["text", "image"],
): ClassifierModel<"typesafe-system-one"> {
	return {
		type: "classifier",
		id,
		name: id,
		api: "typesafe-system-one",
		provider,
		baseUrl: "https://example.invalid/v1",
		input,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 64000,
	};
}

describe("visual page decision gate", () => {
	const samplePng = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

	it("marks a page as visually rich when probability meets the threshold", async () => {
		const classify = vi.fn(
			async (
				_model: ClassifierModel<"typesafe-system-one">,
				context: ClassifierContext,
			): Promise<ClassifierResult> => {
				expect(context.images).toHaveLength(1);
				expect(context.images?.[0]).toEqual({
					type: "image",
					mimeType: "image/png",
					data: Buffer.from(samplePng).toString("base64"),
				});
				expect(String(context.state)).toContain("Revenue Q1-Q4");
				return {
					api: "typesafe-system-one",
					provider: BIVIA_PROVIDER_ID,
					model: BIVIA_DEFAULT_MODEL,
					stopReason: "stop",
					timestamp: Date.now(),
					answers: { visually_rich: { type: "bool", probability: 0.94 } },
				};
			},
		);
		const runtime: DecisionModelRuntime = {
			getModelsOfType: () => [fakeClassifierModel(BIVIA_PROVIDER_ID, BIVIA_DEFAULT_MODEL)],
			classify,
		};

		const gate = createVisualPageGate({
			backend: BIVIA_PROVIDER_ID,
			model: BIVIA_DEFAULT_MODEL,
			runtime,
		});
		const verdict = await gate.judge({
			virtualPath: "/docs/report.pdf",
			page: 3,
			image: { data: samplePng, mimeType: "image/png" },
			extractedText: "Revenue Q1-Q4",
		});

		expect(verdict).toEqual({
			visuallyRich: true,
			probability: 0.94,
			model: BIVIA_DEFAULT_MODEL,
			backend: BIVIA_PROVIDER_ID,
		});
		expect(classify).toHaveBeenCalledTimes(1);
		expect(gate.fingerprint).toContain(BIVIA_PROVIDER_ID);
		expect(gate.fingerprint).toContain(BIVIA_DEFAULT_MODEL);
	});

	it("marks plain text pages as not visually rich below threshold", async () => {
		const runtime: DecisionModelRuntime = {
			getModelsOfType: () => [fakeClassifierModel("openrouter", "typesafe/jev-1.14")],
			classify: async () => ({
				api: "typesafe-system-one",
				provider: "openrouter",
				model: "typesafe/jev-1.14",
				stopReason: "stop",
				timestamp: Date.now(),
				answers: { visually_rich: { type: "bool", probability: 0.08 } },
			}),
		};

		const gate = createVisualPageGate({
			backend: "openrouter",
			model: "typesafe/jev-1.14",
			threshold: 0.6,
			runtime,
		});
		const verdict = await gate.judge({
			virtualPath: "/docs/contract.pdf",
			page: 1,
			image: { data: samplePng, mimeType: "image/png" },
			extractedText: "Standard agreement clauses.",
		});

		expect(verdict.visuallyRich).toBe(false);
		expect(verdict.probability).toBe(0.08);
	});

	it("rejects a text-only classifier model loudly instead of silently dropping the image", async () => {
		const runtime: DecisionModelRuntime = {
			getModelsOfType: () => [fakeClassifierModel("openrouter", "upstage/solar-decide", ["text"])],
			classify: vi.fn(),
		};

		const gate = createVisualPageGate({
			backend: "openrouter",
			model: "upstage/solar-decide",
			runtime,
		});

		await expect(
			gate.judge({
				virtualPath: "/docs/chart.pdf",
				page: 1,
				image: { data: samplePng, mimeType: "image/png" },
			}),
		).rejects.toThrow(/image|vision/i);
	});

	it("sends multimodal image_url state when using the hosted dazzi backend", async () => {
		let capturedBody: Record<string, unknown> | undefined;
		const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
			capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
			return new Response(
				JSON.stringify({
					model: "typesafe/jev-1.14",
					answers: { visually_rich: { type: "noul", noul: 0.87 } },
					usage: { input_tokens: 400, output_tokens: 10 },
				}),
				{ status: 200, headers: { "Content-Type": "application/json" } },
			);
		}) as unknown as typeof fetch;

		const gate = createVisualPageGate({
			backend: "dazzi",
			model: "typesafe/jev-1.14",
			dazzi: { apiKey: "tok_test", baseUrl: "https://api.dazziapp.com", fetchImpl },
		});
		const verdict = await gate.judge({
			virtualPath: "/docs/deck.pdf",
			page: 2,
			image: { data: samplePng, mimeType: "image/png" },
			extractedText: "Quarterly metrics",
		});

		expect(verdict.visuallyRich).toBe(true);
		expect(verdict.probability).toBe(0.87);
		expect(Array.isArray(capturedBody?.state)).toBe(true);
		const parts = capturedBody?.state as unknown[];
		expect(parts[0]).toEqual({
			type: "image_url",
			image_url: { url: `data:image/png;base64,${Buffer.from(samplePng).toString("base64")}` },
		});
		expect(typeof parts[1]).toBe("string");
	});

	it("preserves multimodal image_url state arrays in DazziJevBackend and toNativeBody", () => {
		const multimodalState = [
			{ type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
			"Extracted text",
		];
		const body = toNativeBody(
			multimodalState as unknown as string,
			[{ id: "q", type: "noul", question: "has chart?" }],
			"typesafe/jev-1.14",
		);
		expect(body.state).toEqual(multimodalState);
		expect(DazziJevBackend).toBeDefined();
	});

	it("resolves vision decision backend defaults across bivia, openrouter, openai, typesafe, cloudflare, and dazzi", () => {
		expect(DEFAULT_VISUAL_GATE_THRESHOLD).toBe(0.6);
		expect(VISION_DECISION_PROVIDER_DEFAULT_MODELS[BIVIA_PROVIDER_ID]).toBe(BIVIA_DEFAULT_MODEL);
		expect(VISION_DECISION_PROVIDER_DEFAULT_MODELS.openrouter).toBe("typesafe/jev-1.14");
		expect(VISION_DECISION_PROVIDER_DEFAULT_MODELS.dazzi).toBe("typesafe/jev-1.14");
		expect(isVisionDecisionModel(fakeClassifierModel("bivia", "dec-1", ["text", "image"]))).toBe(true);
		expect(isVisionDecisionModel(fakeClassifierModel("opencode", "jev-1.13", ["text"]))).toBe(false);

		const resolved = resolveVisualGateBackend({
			env: { BIVIA_API_KEY: "test-bivia-key" },
		});
		expect(resolved).toEqual({ backend: "bivia", model: "dec-1" });

		const inherited = resolveVisualGateBackend({
			inheritedJev: { backend: "openrouter", model: "typesafe/jev-1.14" },
			env: { OPENROUTER_API_KEY: "or-key" },
		});
		expect(inherited).toEqual({ backend: "openrouter", model: "typesafe/jev-1.14" });
	});
});
