import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";

/**
 * A chat-bridge extension shaped like `whatsapp-pi`: it declares an online flag, injects each
 * inbound chat message with `pi.sendUserMessage(text, { deliverAs: "followUp" })`, and relays every
 * assistant message back to the chat. The test drives `globalThis.__chatBridge` as the "phone".
 */
const BRIDGE_SOURCE = `
export default function (pi) {
	pi.registerFlag("whatsapp-pi-online", { type: "boolean", default: false });
	pi.registerFlag("whatsapp-group", { type: "string", default: "" });
	const bridge = { online: false, group: "", replies: [], send: (text) => pi.sendUserMessage(text, { deliverAs: "followUp" }) };
	globalThis.__chatBridge = bridge;
	pi.on("session_start", () => {
		bridge.online = pi.getFlag("whatsapp-pi-online") === true;
		bridge.group = pi.getFlag("whatsapp-group") || "";
	});
	pi.on("message_end", (event) => {
		if (event.message.role !== "assistant") return;
		const text = event.message.content.filter((c) => c.type === "text").map((c) => c.text).join("\\n");
		if (text) bridge.replies.push(text);
	});
}
`;

interface ChatBridge {
	online: boolean;
	group: string;
	replies: string[];
	send(text: string): Promise<void> | void;
}

declare global {
	var __chatBridge: ChatBridge | undefined;
}

function bridge(): ChatBridge {
	const current = globalThis.__chatBridge;
	if (current === undefined) throw new Error("the chat bridge extension did not load");
	return current;
}

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-wa-remote-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
	globalThis.__chatBridge = undefined;
});

function jevDirect(): JevBackend {
	return new MockBackend({
		[QUERY_ROUTE_QUESTION_ID]: {
			answer: "direct",
			distribution: { local: 0.1, web: 0.1, direct: 0.8 },
			confidence: 0.8,
		},
		[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
		[FOLLOW_UP_QUESTION_ID]: { answer: 0.1 },
	});
}

function noUi(): ExtensionUIContext {
	return new Proxy({} as ExtensionUIContext, {
		get: (_target, key) => (key === "onTerminalInput" ? () => () => undefined : () => undefined),
	});
}

function textOf(content: string | readonly { type: string; text?: string }[]): string {
	return typeof content === "string"
		? content
		: content.map((block) => (block.type === "text" ? (block.text ?? "") : "")).join("");
}

async function hostedAgent(responses: FauxResponseStep[], flags: Record<string, string | boolean> = {}) {
	const registration = registerFauxProvider({ api: `faux-wa-${randomUUID()}`, models: [{ id: "wa-model" }] });
	registration.setResponses(responses);
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	const model = registration.getModel();
	writeFileSync(
		join(agentDir, "settings.json"),
		JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
	);
	const bridgePath = join(root, "chat-bridge.mjs");
	writeFileSync(bridgePath, BRIDGE_SOURCE);
	const agent = new AutoRAGAgent({
		model,
		apiKey: "test-key",
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		jev: { backend: jevDirect() },
		piExtensionPaths: [bridgePath],
		piExtensionFlags: flags,
	});
	const hosted = await agent.createPiInteractiveRuntime();
	await hosted.runtime.session.bindExtensions({ uiContext: noUi() });
	return {
		agent,
		registration,
		session: hosted.runtime.session,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

describe("remote chat extension hosted by the interactive session", () => {
	it("hands the extension its CLI flag values", async () => {
		const host = await hostedAgent([], { "whatsapp-pi-online": true, "whatsapp-group": "120363@g.us" });
		try {
			expect(bridge().online).toBe(true);
			expect(bridge().group).toBe("120363@g.us");
		} finally {
			await host.dispose();
		}
	});

	it("runs a chat message through AutoRAG's own search and relays the answer back", async () => {
		const host = await hostedAgent([fauxAssistantMessage("Paris.", { stopReason: "stop" })]);
		try {
			await bridge().send("Message from Jeff (821012345678): What is the capital of France?");
			await host.agent.waitForIdle();
			expect(host.registration.state.callCount).toBe(1);
			expect(bridge().replies).toEqual(["Paris."]);
			// Only AutoRAG's search driver names the session after the query; a raw pi prompt would not.
			expect(host.session.sessionName).toContain("What is the capital of France?");
			const users = host.session.messages.flatMap((m) => (m.role === "user" ? [textOf(m.content)] : []));
			expect(users).toHaveLength(1);
		} finally {
			await host.dispose();
		}
	});

	it("queues a chat message that arrives mid-search and answers it as the next question", async () => {
		let markEntered = (): void => {};
		let release = (): void => {};
		const entered = new Promise<void>((resolve) => {
			markEntered = resolve;
		});
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const held: FauxResponseStep = async () => {
			markEntered();
			await gate;
			return fauxAssistantMessage("First answer.", { stopReason: "stop" });
		};
		const host = await hostedAgent([held, fauxAssistantMessage("Second answer.", { stopReason: "stop" })]);
		try {
			await bridge().send("Message from Jeff (1): first question");
			await entered;
			await bridge().send("Message from Jeff (1): second question");
			release();
			await host.agent.waitForIdle();
			expect(bridge().replies).toEqual(["First answer.", "Second answer."]);
			expect(host.registration.state.callCount).toBe(2);
		} finally {
			await host.dispose();
		}
	});

	it("does not re-route the search's own prompts as new chat questions", async () => {
		const host = await hostedAgent([fauxAssistantMessage("Only once.", { stopReason: "stop" })]);
		try {
			await bridge().send("Message from Jeff (1): ping");
			await host.agent.waitForIdle();
			// A re-routed own prompt would start a second search and a second model request.
			expect(host.registration.state.callCount).toBe(1);
			expect(bridge().replies).toEqual(["Only once."]);
		} finally {
			await host.dispose();
		}
	});
});
