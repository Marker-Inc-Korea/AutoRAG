import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";
import { type FakeRequest, type FakeResponse, type FakeServer, startFakeServer } from "../cloud/helpers/fake-server.ts";
import { responsesTextStream, responsesToolCallStream, SSE_HEADERS } from "../cloud/helpers/responses-sse.ts";

const CONTEXT_WINDOW = 100_000;
/** What the real upstream tokenizer would turn into more than the window; the server's rejection threshold. */
const REJECTED_BODY_CHARS = 200_000;
/** chars / 4 reads this as 60k tokens, comfortably inside the window, so only the server knows it is too big. */
const HUGE_RESULT = "\u0000\ufffd\u0001pdf bytes ".repeat(10_000);

/** What the hosted server answers when the upstream provider says the input is longer than the model's window. */
const CONTEXT_EXCEEDED = {
	error: {
		message:
			"This model's maximum context length is 1048576 tokens, but the request resolved to 1244118 input tokens.",
		type: "invalid_request_error",
		code: "context_length_exceeded",
	},
};

let root: string;
let agentDir: string;
let server: FakeServer | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-pi-dazzi-overflow-"));
	agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
});

afterEach(async () => {
	await server?.close();
	server = undefined;
	rmSync(root, { recursive: true, force: true });
});

function dazziModel(baseUrl: string): Model<Api> {
	return {
		id: "vendor/chat",
		name: "Vendor Chat",
		api: "openai-responses",
		provider: "dazzi",
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
		contextWindow: CONTEXT_WINDOW,
		maxTokens: 4_096,
	};
}

/** A signed-in agent dir whose only Dazzi model is {@link dazziModel}. */
function signIn(baseUrl: string): void {
	writeFileSync(
		join(agentDir, "auth.json"),
		JSON.stringify({ dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER } }),
	);
	const models = mapServerModels({
		object: "list",
		data: [
			{
				id: "vendor/chat",
				name: "Vendor Chat",
				context_window: CONTEXT_WINDOW,
				max_output_tokens: 4_096,
				input_modalities: ["text"],
				reasoning: false,
				pricing: { input: 1, output: 2, cache_read: 0, cache_write: 0 },
			},
		],
	});
	writeFileSync(
		join(agentDir, "models-store.json"),
		JSON.stringify({ dazzi: buildStoreEntry(models, "dazzi", baseUrl) }),
	);
}

const bulkFileTool: AgentTool = {
	name: "search_datasource_files",
	label: "Files",
	description: "Returns whatever bulky file text was found",
	parameters: Type.Object({}),
	async execute() {
		return { content: [{ type: "text", text: HUGE_RESULT }], details: {} };
	},
};

function isAssistantMessage(message: AgentMessage): message is AssistantMessage {
	return message.role === "assistant";
}

function lastAssistantText(messages: readonly AgentMessage[]): string {
	const last = [...messages].reverse().find(isAssistantMessage);
	return (last?.content ?? []).map((block) => (block.type === "text" ? block.text : "")).join("");
}

describe("pi session on Dazzi when a tool result overflows the context", () => {
	it("compacts, trims the oversized result and retries after the server's context_length_exceeded, then answers", async () => {
		const requests: FakeRequest[] = [];
		server = await startFakeServer((request): FakeResponse => {
			requests.push(request);
			if (request.url !== "/v1/responses") return { status: 404, body: {} };
			if (request.body.length > REJECTED_BODY_CHARS) return { status: 400, body: CONTEXT_EXCEEDED };
			const summarizing = request.body.includes("checkpoint");
			if (summarizing) return { headers: SSE_HEADERS, rawBody: responsesTextStream("## Goal\nfind the PDF") };
			const hasToolOutput = request.body.includes("function_call_output");
			return {
				headers: SSE_HEADERS,
				rawBody: hasToolOutput
					? responsesTextStream("final answer after recovery")
					: responsesToolCallStream("search_datasource_files", {}),
			};
		});
		const baseUrl = `${server.url}/v1`;
		signIn(baseUrl);
		const hosted = await createAutoRAGPiSession({
			cwd: root,
			agentDir,
			model: dazziModel(baseUrl),
			apiKey: "dz_login",
			providerApiKeys: { dazzi: "dz_login" },
			getSystemPrompt: () => "AutoRAG system prompt",
			customTools: [bulkFileTool],
		});
		try {
			const compactions: string[] = [];
			hosted.session.subscribe((event) => {
				if (event.type === "compaction_end")
					compactions.push(`${event.reason}:${event.willRetry}:${event.errorMessage ?? ""}`);
			});

			await hosted.session.prompt("PDF 찾아줘");
			await hosted.session.agent.waitForIdle();

			expect(compactions).toEqual(["overflow:true:"]);
			// prompt, the oversized follow-up the server rejects, the summary, the trimmed retry
			expect(requests.map((request) => request.body.length > REJECTED_BODY_CHARS)).toEqual([
				false,
				true,
				false,
				false,
			]);
			expect(lastAssistantText(hosted.session.messages)).toBe("final answer after recovery");
		} finally {
			await hosted.dispose();
		}
	});
});
