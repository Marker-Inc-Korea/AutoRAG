/**
 * Live E2E: a real hosted Dazzi Models session answers when a tool returns more tokens than the window holds.
 *
 * Gated on AUTORAG_DAZZI_OVERFLOW_LIVE=1 so the offline suite and CI (no stored
 * `/login` credential) never touch the network. Run explicitly after `/login`:
 *
 *   AUTORAG_DAZZI_OVERFLOW_LIVE=1 [AUTORAG_DAZZI_AGENT_DIR=~/.pi/agent] bunx vitest run test/live-e2e/dazzi-context-overflow.test.ts
 *
 * A tool returns ~2.2M characters of PDF-like binary garbage. The upstream
 * provider counts that as ~1.2M tokens (0.55 per character, over the 1,048,576
 * window; the 2026-10-31 incident answered 400 `context_length_exceeded`), where
 * a chars/4 guess reads ~550k. The context guard counts with a real tokenizer
 * (~1.22M), so it trims the result before the request is sent: the server counts
 * the request inside the window and the session answers on its own.
 *
 * The login is copied into a temp agent dir, so the real one is never written.
 */
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { DEFAULT_DAZZI_CHAT_MODEL_ID } from "../../src/cloud/config.ts";
import { readDazziLoginToken } from "../../src/cloud/credential.ts";
import { registerDazziProvider } from "../../src/cloud/provider.ts";

const LIVE = process.env.AUTORAG_DAZZI_OVERFLOW_LIVE === "1";

// The vitest setup isolates PI_CODING_AGENT_DIR, so read the real login explicitly.
const REAL_AGENT_DIR = process.env.AUTORAG_DAZZI_AGENT_DIR ?? join(homedir(), ".pi", "agent");

/** 0.55 tokens per character measured against the real server: well over the 1,048,576-token window. */
const GARBAGE_CHARS = 2_200_000;
const GARBAGE = "\ufffd\u0000%PDF-1.4\u0001\u0002\ufffd".repeat(Math.ceil(GARBAGE_CHARS / 18)).slice(0, GARBAGE_CHARS);

const bulkFileTool: AgentTool = {
	name: "search_datasource_files",
	label: "Files",
	description: "Search files. Always call this once for any file question.",
	parameters: Type.Object({ query: Type.Optional(Type.String()) }),
	async execute() {
		return { content: [{ type: "text", text: GARBAGE }], details: {} };
	},
};

function isAssistantMessage(message: AgentMessage): message is AssistantMessage {
	return message.role === "assistant";
}

describe.skipIf(!LIVE)("Dazzi context overflow live e2e (real API)", () => {
	let agentDir: string;

	beforeAll(() => {
		expect(readDazziLoginToken(REAL_AGENT_DIR), "run `autorag tui` and /login first").toBeTruthy();
		agentDir = mkdtempSync(join(tmpdir(), "autorag-dazzi-overflow-live-"));
		mkdirSync(agentDir, { recursive: true });
		copyFileSync(join(REAL_AGENT_DIR, "auth.json"), join(agentDir, "auth.json"));
		if (existsSync(join(REAL_AGENT_DIR, "models-store.json"))) {
			copyFileSync(join(REAL_AGENT_DIR, "models-store.json"), join(agentDir, "models-store.json"));
		}
	});

	afterAll(() => {
		if (agentDir !== undefined) rmSync(agentDir, { recursive: true, force: true });
	});

	it("trims the oversized tool result before sending, so no request overflows, and answers", async () => {
		const runtime = await ModelRuntime.create({
			authPath: join(agentDir, "auth.json"),
			modelsPath: join(agentDir, "models.json"),
			allowModelNetwork: false,
		});
		await registerDazziProvider(runtime);
		const model = runtime.getModel("dazzi", DEFAULT_DAZZI_CHAT_MODEL_ID);
		expect(model, `${DEFAULT_DAZZI_CHAT_MODEL_ID} is missing from the stored Dazzi catalog`).toBeDefined();
		if (model === undefined) return;

		const hosted = await createAutoRAGPiSession({
			cwd: agentDir,
			agentDir,
			model,
			apiKey: readDazziLoginToken(agentDir),
			getSystemPrompt: () =>
				"You are a file librarian. Use the search tool once, then answer in one short sentence.",
			customTools: [bulkFileTool],
		});
		try {
			hosted.session.setThinkingLevel("low");
			await hosted.session.prompt("Find my graduation PDF with the search tool.");
			await hosted.session.agent.waitForIdle();

			// The guard sizes the request from real token counts, so no request is rejected, and what it
			// keeps is most of the budget rather than a token. The server's own count is the evidence.
			const turns = hosted.session.messages.filter(isAssistantMessage);
			expect(turns.filter((turn) => turn.stopReason === "error")).toEqual([]);
			const largestRequest = Math.max(...turns.map((turn) => turn.usage.input));
			expect(largestRequest).toBeGreaterThan(500_000);
			expect(largestRequest).toBeLessThanOrEqual(model.contextWindow);
			const last = turns.at(-1);
			expect(last?.stopReason).toBe("stop");
			expect(last?.content.some((block) => block.type === "text" && block.text.length > 0)).toBe(true);
		} finally {
			await hosted.dispose();
		}
	}, 600_000);
});
