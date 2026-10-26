import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage, fauxToolCall, getCurrentTools } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	INTENT_FIELD,
	intentOfMessage,
	normalizeIntent,
	schemaDefinesIntent,
	withIntentField,
} from "../../src/agent/intent-tracing.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { inertUi } from "../helpers/inert-ui.ts";

/** Narrows a schema node to a plain object so its members can be read without a cast. */
function record(value: unknown): Record<string, unknown> {
	if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("expected an object");
	return Object.fromEntries(Object.entries(value));
}

describe("intent field schema", () => {
	it("adds a required, first `i` string to an object schema", () => {
		const schema = Type.Object({ query: Type.String() });
		const wire = record(withIntentField(schema));
		const properties = record(wire.properties);
		expect(Object.keys(properties)).toEqual([INTENT_FIELD, "query"]);
		expect(wire.required).toEqual(expect.arrayContaining(["query", INTENT_FIELD]));
		expect(properties[INTENT_FIELD]).toMatchObject({ type: "string" });
	});

	it("leaves a tool that owns `i` alone", () => {
		const schema = Type.Object({ i: Type.Number() });
		expect(schemaDefinesIntent(schema)).toBe(true);
		expect(withIntentField(schema)).toBe(schema);
	});

	it("leaves a closed schema alone, since the tool would reject the field", () => {
		const schema = Type.Object({ query: Type.String() }, { additionalProperties: false });
		expect(withIntentField(schema)).toBe(schema);
	});

	it("pushes the field into every branch of a union schema", () => {
		const schema = Type.Union([Type.Object({ a: Type.String() }), Type.Object({ b: Type.String() })]);
		const branches = record(withIntentField(schema)).anyOf;
		if (!Array.isArray(branches)) throw new Error("expected a union");
		expect(branches.map((branch) => Object.keys(record(record(branch).properties))[0])).toEqual([
			INTENT_FIELD,
			INTENT_FIELD,
		]);
	});
});

describe("intent text", () => {
	it("trims whitespace and trailing dots", () => {
		expect(normalizeIntent("  Checking\nKakao   for the invoice... ")).toBe("Checking Kakao for the invoice");
	});

	it("rejects non-strings, blanks, and payload-sized values", () => {
		expect(normalizeIntent(42)).toBeUndefined();
		expect(normalizeIntent("  . ")).toBeUndefined();
		expect(normalizeIntent("x".repeat(201))).toBeUndefined();
	});

	it("takes the last tool call's intent from an assistant message", () => {
		const message = {
			role: "assistant",
			content: [
				{ type: "toolCall", id: "1", name: "a", arguments: { i: "first" } },
				{ type: "toolCall", id: "2", name: "b", arguments: { i: "second" } },
				{ type: "toolCall", id: "3", name: "c", arguments: {} },
			],
		};
		expect(intentOfMessage(message)).toBe("second");
		expect(intentOfMessage({ role: "user", content: "hi" })).toBeUndefined();
	});
});

describe("intent tracing in a pi session", () => {
	let root: string;
	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-intent-"));
	});
	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	it("declares `i` to the model, strips it before the tool runs, and shows it as the working message", async () => {
		const received: unknown[] = [];
		const tool: AgentTool = {
			name: "search_custom",
			label: "Search custom",
			description: "Custom test tool",
			parameters: Type.Object({ query: Type.String() }),
			async execute(_id, params) {
				received.push(params);
				return { content: [{ type: "text", text: "found" }], details: {} };
			},
		};
		const declared: unknown[] = [];
		const registration = registerFauxProvider({
			api: `faux-intent-${randomUUID()}`,
			models: [{ id: "intent-model" }],
		});
		registration.setResponses([
			(context) => {
				const custom = getCurrentTools(context.messages).find((entry) => entry.name === "search_custom");
				declared.push(custom?.parameters);
				return fauxAssistantMessage(
					[fauxToolCall("search_custom", { i: "Checking the refund policy in Kakao.", query: "refund" })],
					{ stopReason: "toolUse" },
				);
			},
			fauxAssistantMessage("done", { stopReason: "stop" }),
		]);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		const working: (string | undefined)[] = [];
		try {
			const hosted = await createAutoRAGPiSession({
				cwd: root,
				agentDir,
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [tool],
			});
			await hosted.session.bindExtensions({
				uiContext: {
					...inertUi(),
					setWorkingMessage: (message?: string) => {
						working.push(message);
					},
				},
			});
			await hosted.session.prompt("what is the refund policy?");
			await hosted.dispose();
		} finally {
			registration.unregister();
		}

		expect(Object.keys(record(record(declared[0]).properties))[0]).toBe(INTENT_FIELD);
		// The tool sees its own arguments only.
		expect(received).toEqual([{ query: "refund" }]);
		expect(working).toContain("Checking the refund policy in Kakao");
		// Cleared when the run ends, so the next one starts from pi's default.
		expect(working.at(-1)).toBeUndefined();
	});
});
