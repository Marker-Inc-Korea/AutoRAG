import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentMessage, AgentTool } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage, fauxToolCall, type Message } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { BackendRequest, BackendResponse, JevBackend } from "jev-use";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { createModelDecompositionCompleter } from "../../src/agent/query-decomposition.ts";
import { DatasourceChunkStore } from "../../src/datasource/chunk-store.ts";
import { createHttpEmbedder } from "../../src/embedding/http-embedder.ts";
import { loadMirrorIndex, syncParsedMirrors } from "../../src/mirror/index.ts";
import { createDefaultParserRegistry } from "../../src/parser/index.ts";
import { redactMessages, withSecretRedaction } from "../../src/privacy/apply.ts";
import { OpenRouterReranker } from "../../src/retrieval/rerank.ts";

// Built dynamically so no literal credential sits in test source.
const ANTHROPIC_KEY = `sk-ant-api03-${"Zy9-".repeat(25)}AA`;
const AWS_KEY = `AKIA${"WXYZ".repeat(4)}`;
const GITHUB_KEY = `ghp_${"aB3dE6gH9k".repeat(3)}abcdef`;
const OTP_EN = "482913";
const OTP_KO = "739201";

const tempRoots: string[] = [];
function makeTempDir(prefix: string): string {
	const dir = mkdtempSync(join(tmpdir(), prefix));
	tempRoots.push(dir);
	return dir;
}

afterEach(() => {
	delete process.env.AUTORAG_REDACT_SECRETS;
	while (tempRoots.length > 0) {
		rmSync(tempRoots.pop()!, { recursive: true, force: true });
	}
});

describe("redactMessages", () => {
	it("masks user text, tool results and tool-call arguments, and reuses untouched messages", () => {
		const clean: AgentMessage = { role: "user", content: "hello", timestamp: 1 };
		const messages = [
			clean,
			{ role: "user", content: [{ type: "text", text: `my key ${ANTHROPIC_KEY}` }], timestamp: 2 },
			{
				role: "assistant",
				content: [
					{ type: "thinking", thinking: "t", thinkingSignature: "sig-abc" },
					{
						type: "toolCall",
						id: "call_1",
						name: "bash",
						arguments: { command: `curl -H "x: ${ANTHROPIC_KEY}"` },
					},
				],
				timestamp: 3,
			},
			{
				role: "toolResult",
				toolCallId: "call_1",
				toolName: "bash",
				content: [{ type: "text", text: `OTP: ${OTP_EN} ${ANTHROPIC_KEY}` }],
				isError: false,
				timestamp: 4,
			},
		] as unknown as AgentMessage[];
		const out = redactMessages(messages);
		const json = JSON.stringify(out);
		expect(json).not.toContain("Zy9-Zy9-");
		expect(json).not.toContain(OTP_EN);
		expect(out[0]).toBe(clean);
		expect(JSON.stringify(out[2])).toContain("sig-abc");
		expect(JSON.stringify(out[2])).toContain("call_1");
		expect(messages[1]).toEqual({
			role: "user",
			content: [{ type: "text", text: `my key ${ANTHROPIC_KEY}` }],
			timestamp: 2,
		});
	});

	it("honors AUTORAG_REDACT_SECRETS=0 opt-out", () => {
		process.env.AUTORAG_REDACT_SECRETS = "0";
		const messages = [{ role: "user", content: `key ${AWS_KEY}`, timestamp: 1 }] as AgentMessage[];
		expect(redactMessages(messages)).toBe(messages);
	});
});

describe("pi-session model input redaction", () => {
	it("masks secrets in user prompt, system prompt, and tool results before the model sees them", async () => {
		const root = makeTempDir("autorag-privacy-session-");
		const leakTool: AgentTool = {
			name: "read_secret_doc",
			label: "Read secret doc",
			description: "Returns a document containing credentials",
			parameters: Type.Object({}),
			async execute() {
				return {
					content: [{ type: "text", text: `[Web발신] 인증번호 [${OTP_KO}] / token=${GITHUB_KEY}` }],
					details: {},
				};
			},
		};
		const registration = registerFauxProvider({
			api: "faux-privacy-session",
			models: [{ id: "privacy-test-model", contextWindow: 128_000, maxTokens: 4096 }],
		});
		const seenContexts: Array<{ systemPrompt?: string; messages: Message[] }> = [];
		const record = (reply: AssistantMessage) => (context: { systemPrompt?: string; messages: Message[] }) => {
			seenContexts.push({ systemPrompt: context.systemPrompt, messages: context.messages });
			return reply;
		};
		registration.setResponses([
			record(fauxAssistantMessage(fauxToolCall("read_secret_doc", {}), { stopReason: "toolUse" })),
			record(fauxAssistantMessage("Checked your notes safely.")),
		]);
		const session = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			apiKey: "test-key",
			getSystemPrompt: () => `AutoRAG system prompt with leaked ${AWS_KEY}`,
			customTools: [leakTool],
		});
		try {
			await session.session.prompt(`Check deploy note for ${ANTHROPIC_KEY} and OTP: ${OTP_EN}`);
			expect(seenContexts).toHaveLength(2);
			const wire = JSON.stringify(seenContexts);
			expect(wire).not.toContain(AWS_KEY);
			expect(wire).not.toContain("Zy9-Zy9-");
			expect(wire).not.toContain(GITHUB_KEY);
			expect(wire).not.toContain(OTP_EN);
			expect(wire).not.toContain(OTP_KO);
			expect(wire).toContain("[REDACTED:aws-access-token]");
			expect(wire).toContain("[REDACTED:anthropic-api-key]");
			expect(wire).toContain("[REDACTED:github-token]");
			expect(wire).toContain("[REDACTED:otp-code]");
		} finally {
			session.session.dispose();
			registration.unregister();
		}
	});
});

describe("decision backend redaction", () => {
	it("masks state and questions through withSecretRedaction", async () => {
		let seen: BackendRequest | undefined;
		const inner: JevBackend = {
			name: "fake",
			async judge(request: BackendRequest): Promise<BackendResponse> {
				seen = request;
				return { answers: [], model: "m", latencyMs: 0 };
			},
		};
		const wrapped = withSecretRedaction(inner);
		await wrapped.judge({
			state: `User pasted ${ANTHROPIC_KEY} and code OTP: ${OTP_EN}`,
			questions: [{ id: "q", type: "noul", question: `Is ${ANTHROPIC_KEY} safe?` }],
		} as BackendRequest);
		const json = JSON.stringify(seen);
		expect(json).not.toContain("Zy9-Zy9-");
		expect(json).not.toContain(OTP_EN);
		expect(wrapped.name).toBe("fake");
	});

	it("automatically wraps backends inside createJevJudge", async () => {
		let seen: BackendRequest | undefined;
		const fakeBackend: JevBackend = {
			name: "fake-jev",
			async judge(request: BackendRequest): Promise<BackendResponse> {
				seen = request;
				return {
					answers: request.questions.map(() => ({ answer: 0.9 })),
					model: "fake-model",
					latencyMs: 1,
				};
			},
		};
		const judge = createJevJudge({ backend: fakeBackend });
		await judge({ query: `login with ${AWS_KEY}`, snippet: `인증번호: ${OTP_KO}` }, [
			{ id: "relevant", type: "noul", question: `Does this match ${GITHUB_KEY}?` },
		]);
		const json = JSON.stringify(seen);
		expect(json).not.toContain(AWS_KEY);
		expect(json).not.toContain(GITHUB_KEY);
		expect(json).not.toContain(OTP_KO);
		expect(json).toContain("[REDACTED:aws-access-token]");
		expect(json).toContain("[REDACTED:otp-code]");
	});
});

describe("indexing and retrieval redaction", () => {
	it("masks secrets in DatasourceChunkStore content, titles, and metadata", () => {
		const root = makeTempDir("autorag-privacy-chunks-");
		const store = new DatasourceChunkStore({ skillName: "t", instanceId: "i", workspaceRoot: root });
		store.replaceDocuments([
			{
				docId: "d1",
				title: `note ${ANTHROPIC_KEY}`,
				content: `deploy with ${ANTHROPIC_KEY}\nverification code ${OTP_EN}`,
				metadata: { authorNote: `use ${AWS_KEY}` },
			},
		]);
		const stored = JSON.stringify(store.chunks());
		expect(stored).not.toContain("Zy9-Zy9-");
		expect(stored).not.toContain(AWS_KEY);
		expect(stored).not.toContain(OTP_EN);
		expect(store.search("deploy", 3)).toHaveLength(1);
	});

	it("masks secrets in parsed mirror markdown during syncParsedMirrors", async () => {
		const root = makeTempDir("autorag-privacy-mirror-");
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(
			join(docs, "runbook.txt"),
			`Production runbook\nAWS key: ${AWS_KEY}\nYour verification code is ${OTP_EN}.\nKeep this safe.\n`,
		);
		await syncParsedMirrors({ root, searchPaths: [docs], registry: createDefaultParserRegistry() });
		const entry = loadMirrorIndex(root).entries["/docs/runbook.txt"];
		expect(entry).toBeDefined();
		const mirrored = readFileSync(entry!.outputPath, "utf8");
		expect(mirrored).toContain("Production runbook");
		expect(mirrored).not.toContain(AWS_KEY);
		expect(mirrored).not.toContain(OTP_EN);
		expect(mirrored).toContain("[REDACTED:aws-access-token]");
		expect(mirrored).toContain("[REDACTED:otp-code]");
	});

	it("masks secrets before posting to a remote HTTP embedder", async () => {
		const sentBodies: string[] = [];
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl: "https://embed.example.test/v1",
			model: "embed-test",
			provider: "openai",
			fetchImpl: async (_input, init) => {
				sentBodies.push(String(init?.body ?? ""));
				return new Response(JSON.stringify({ data: [{ index: 0, embedding: [0.1, 0.2] }] }), {
					status: 200,
					headers: { "content-type": "application/json" },
				});
			},
		});
		await embedder.embed([`token ${GITHUB_KEY} and OTP: ${OTP_EN}`]);
		expect(sentBodies).toHaveLength(1);
		expect(sentBodies[0]).not.toContain(GITHUB_KEY);
		expect(sentBodies[0]).not.toContain(OTP_EN);
		expect(sentBodies[0]).toContain("[REDACTED:github-token]");
		expect(sentBodies[0]).toContain("[REDACTED:otp-code]");
	});

	it("masks query and documents before posting to a remote reranker", async () => {
		let sentBody: { query: string; documents: string[] } | undefined;
		const reranker = new OpenRouterReranker({
			apiKey: "test-key",
			client: {
				async rerank({ requestBody }) {
					sentBody = { query: requestBody.query, documents: requestBody.documents };
					return { results: [{ index: 0, relevanceScore: 0.95 }] };
				},
			},
		});
		const ranked = await reranker.rerank(`find ${AWS_KEY}`, [
			{
				id: "doc-1",
				source: "/docs/a.md",
				content: `인증번호: ${OTP_KO} with ${ANTHROPIC_KEY}`,
				score: 0.5,
				metadata: {},
			},
		]);
		expect(ranked).toHaveLength(1);
		expect(sentBody?.query).not.toContain(AWS_KEY);
		expect(sentBody?.documents[0]).not.toContain(OTP_KO);
		expect(sentBody?.documents[0]).not.toContain("Zy9-Zy9-");
	});

	it("masks secrets in query-decomposition prompts before calling the model", async () => {
		const registration = registerFauxProvider({
			api: "faux-privacy-decomp",
			models: [{ id: "decomp-model", contextWindow: 64_000, maxTokens: 1024 }],
		});
		let seenPrompt = "";
		registration.setResponses([
			(context) => {
				seenPrompt = JSON.stringify(context.messages);
				return fauxAssistantMessage('{"queries": ["deploy status"]}');
			},
		]);
		try {
			const complete = createModelDecompositionCompleter({ model: registration.getModel(), apiKey: "test" });
			await complete(`User question: rotate ${GITHUB_KEY} after OTP ${OTP_EN}`);
			expect(seenPrompt).not.toContain(GITHUB_KEY);
			expect(seenPrompt).not.toContain(OTP_EN);
			expect(seenPrompt).toContain("[REDACTED:github-token]");
			expect(seenPrompt).toContain("[REDACTED:otp-code]");
		} finally {
			registration.unregister();
		}
	});
});
