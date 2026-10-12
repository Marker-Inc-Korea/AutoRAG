import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage, fauxToolCall, type Message } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { fitRequestLimits, IMAGE_OMITTED_NOTE } from "../../src/agent/request-limits.ts";

const IMAGE_DATA = "A".repeat(100_000);

function photoResult(label: string): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: `call-${label}`,
		toolName: "read",
		content: [
			{ type: "text", text: `Read image file ${label}` },
			{ type: "image", data: IMAGE_DATA, mimeType: "image/jpeg" },
		],
		isError: false,
		timestamp: 0,
	} as AgentMessage;
}

function imageCount(messages: readonly AgentMessage[]): number {
	return messages.flatMap((message) => {
		const content = (message as { content?: unknown }).content;
		return Array.isArray(content) ? content.filter((part) => part.type === "image") : [];
	}).length;
}

function texts(message: AgentMessage | Message | undefined): string[] {
	const content = (message as { content?: unknown } | undefined)?.content;
	return Array.isArray(content) ? content.filter((part) => part.type === "text").map((part) => part.text) : [];
}

describe("fitRequestLimits", () => {
	const transcript = [
		{ role: "user", content: [{ type: "text", text: "find the photos" }], timestamp: 0 } as AgentMessage,
		photoResult("one"),
		photoResult("two"),
		photoResult("three"),
	];

	it("drops the oldest images until the transcript fits maxRequestBytes", () => {
		const fitted = fitRequestLimits(transcript, { maxRequestBytes: 250_000 });

		expect(fitted).toBeDefined();
		expect(Buffer.byteLength(JSON.stringify(fitted))).toBeLessThanOrEqual(250_000);
		expect(imageCount(fitted ?? [])).toBe(2);
		expect(texts(fitted?.[1])).toEqual(["Read image file one", IMAGE_OMITTED_NOTE]);
		expect(imageCount([fitted?.[3] as AgentMessage])).toBe(1);
	});

	it("keeps at most images.maxPerRequest images, newest last", () => {
		const fitted = fitRequestLimits(transcript, { images: { maxPerRequest: 1 } });

		expect(imageCount(fitted ?? [])).toBe(1);
		expect(imageCount([fitted?.[3] as AgentMessage])).toBe(1);
	});

	it("leaves a request that fits untouched", () => {
		expect(fitRequestLimits(transcript, { maxRequestBytes: 10_000_000 })).toBeUndefined();
		expect(fitRequestLimits(transcript, undefined)).toBeUndefined();
	});

	it("never removes text, even when text alone exceeds the limit", () => {
		const big = [{ role: "user", content: [{ type: "text", text: "x".repeat(5000) }], timestamp: 0 } as AgentMessage];

		expect(fitRequestLimits(big, { maxRequestBytes: 100 })).toBeUndefined();
	});

	it("does not mutate the transcript it fits", () => {
		const before = JSON.stringify(transcript);
		fitRequestLimits(transcript, { maxRequestBytes: 150_000 });

		expect(JSON.stringify(transcript)).toBe(before);
	});
});

describe("pi session request limits", () => {
	let root: string;

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-request-limits-"));
	});

	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	it("keeps a search that reads several photos under the model's request size limit", async () => {
		const photoTool: AgentTool = {
			name: "view_photo",
			label: "View photo",
			description: "Returns a photo",
			parameters: Type.Object({ name: Type.String() }),
			async execute(_id, params) {
				const { name } = params as { name: string };
				return {
					content: [
						{ type: "text", text: `photo ${name}` },
						{ type: "image", data: IMAGE_DATA, mimeType: "image/jpeg" },
					],
					details: {},
				};
			},
		};
		const maxRequestBytes = 250_000;
		const registration = registerFauxProvider({
			api: `faux-limits-${Date.now()}`,
			models: [{ id: "limited-model", input: ["text", "image"], inputLimits: { maxRequestBytes } }],
		});
		const requests: Message[][] = [];
		const record = (reply: AssistantMessage) => (context: { messages: Message[] }) => {
			requests.push(context.messages);
			return reply;
		};
		registration.setResponses([
			record(fauxAssistantMessage(fauxToolCall("view_photo", { name: "one" }), { stopReason: "toolUse" })),
			record(fauxAssistantMessage(fauxToolCall("view_photo", { name: "two" }), { stopReason: "toolUse" })),
			record(fauxAssistantMessage(fauxToolCall("view_photo", { name: "three" }), { stopReason: "toolUse" })),
			record(fauxAssistantMessage(fauxToolCall("view_photo", { name: "four" }), { stopReason: "toolUse" })),
			record(fauxAssistantMessage("done")),
		]);
		const session = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			apiKey: "test-key",
			getSystemPrompt: () => "AutoRAG system prompt",
			customTools: [photoTool],
		});
		try {
			await session.session.prompt("show me the photos");

			expect(requests).toHaveLength(5);
			const last = requests.at(-1) ?? [];
			expect(Buffer.byteLength(JSON.stringify(last.filter((m) => m.role !== "system")))).toBeLessThanOrEqual(
				maxRequestBytes,
			);
			const photoResults = last.filter((m) => m.role === "toolResult");
			expect(photoResults.map((m) => texts(m)[0])).toEqual(["photo one", "photo two", "photo three", "photo four"]);
			expect(texts(photoResults[0])).toContain(IMAGE_OMITTED_NOTE);
			expect(imageCount([photoResults.at(-1) as AgentMessage])).toBe(1);
			// The session record keeps every photo; only the request is trimmed.
			expect(imageCount(session.session.messages)).toBe(4);
		} finally {
			session.session.dispose();
			registration.unregister();
		}
	});
});
