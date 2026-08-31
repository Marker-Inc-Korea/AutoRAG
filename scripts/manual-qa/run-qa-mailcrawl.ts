import { randomUUID } from "node:crypto";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall, registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { MailcrawlClient, MailcrawlSkill } from "../../src/datasource/skills/mailcrawl/index.ts";

const root = mkdtempSync(join(tmpdir(), "autorag-mailcrawl-live-"));
try {
	const binary = join(root, "mailcrawl");
	const log = join(root, "calls.jsonl");
	writeFileSync(binary, `#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(log)}, JSON.stringify({ args: process.argv.slice(2), dataDir: process.env.MAILCRAWL_DATA_DIR ?? null }) + "\\n");
const args = process.argv.slice(2);
if (args[0] === "sync") process.stdout.write(JSON.stringify({ added: 1, chunksAdded: 1 }));
else if (args[0] === "index") process.stdout.write(JSON.stringify({ embedded: 1, generation: "qa" }));
else process.stdout.write(JSON.stringify([{ chunkId: "msg-1:latest:0", messageId: "msg-1", threadId: "thread-1", accountId: "personal", mailbox: "INBOX", subject: "Refund approval", from: "finance@example.com", to: ["director@example.com"], date: "2026-08-31", snippet: "Director approval is required before payout.", score: 0.99, mode: "bm25" }]));
`);
	chmodSync(binary, 0o755);
	const client = new MailcrawlClient({ binaryPath: binary, workspacePath: root, account: "personal", mailbox: "INBOX" });
	const skill = new MailcrawlSkill({ client, instanceId: "personal" });
	const indexed = await skill.index();
	if (!indexed.ok || indexed.chunkCount !== 1) throw new Error("mailcrawl index failed");
	const results = await skill.retrievalMethods()[0]?.retrieve("refund approval", { topK: 5, allowedScopes: ["/mailcrawl/personal/**"] });
	if (results?.[0]?.source !== "/mailcrawl/personal/chunks/msg-1:latest:0") throw new Error("mailcrawl source mapping failed");
	const bad = await new MailcrawlClient({ binaryPath: join(root, "missing") }).sync();
	if (bad.ok || bad.reason !== "binary-missing") throw new Error("mailcrawl bad binary handling failed");
	const calls = readFileSync(log, "utf8").trim().split("\n").map((line) => JSON.parse(line) as { args: string[] });
	if (!calls.some((call) => call.args[0] === "sync") || !calls.some((call) => call.args[0] === "index")) throw new Error("mailcrawl lifecycle calls missing");
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "mailcrawl-qa" }] });
	registration.setResponses([
		fauxAssistantMessage([fauxToolCall("load_datasource_skill", { name: "datasource-mailcrawl" })], { stopReason: "toolUse" }),
		fauxAssistantMessage([fauxToolCall("search_datasource_documents", { query: "refund approval", topK: 5, scope: "/mailcrawl/personal/**" })], { stopReason: "toolUse" }),
		fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
			answer: "[1] Director approval is required before payout.",
			results: [{ number: 1, title: "Refund approval", summary: "Director approval is required before payout.", evidence: [{ excerpt: "Director approval is required before payout." }], confidence: 0.99 }],
			mapping: [{ number: 1, source: "/mailcrawl/personal/chunks/msg-1:latest:0", method: "mailcrawl-bm25", content: "Director approval is required before payout.", evidenceRefs: [{ method: "mailcrawl-bm25", source: "/mailcrawl/personal/chunks/msg-1:latest:0", content: "Director approval is required before payout." }] }],
		})], { stopReason: "toolUse" }),
	]);
	try {
		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: false,
			bm25: false,
			dupey: false,
			datasourceSkills: [skill],
			datasourceAccess: { allowedTags: ["mailcrawl"], allowedScopes: ["/mailcrawl/personal/**"] },
			searchTimeoutMs: 30_000,
		});
		const response = await agent.searchDocuments("Which refund exceptions require approval before payout?", { topK: 1, scope: "/mailcrawl/personal/**" });
		if (response.results[0]?.source !== "/mailcrawl/personal/chunks/msg-1:latest:0") throw new Error("AutoRAG agent source mapping failed");
		if (!response.answer.includes("Director approval")) throw new Error("AutoRAG agent answer failed");
	} finally {
		registration.unregister();
	}
	console.log("MAILCRAWL_AGENT_QA_PASS");
} finally {
	rmSync(root, { recursive: true, force: true });
}
