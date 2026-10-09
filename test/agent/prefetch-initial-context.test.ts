import { randomUUID } from "node:crypto";
import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@earendil-works/pi-ai";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { writeFakeMinSync } from "../helpers/fake-minsync.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-prefetch-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(
		join(docs, "refund-policy.txt"),
		"Refund exceptions require director approval before payout.\nFinance acknowledged the policy in the July review.\n",
	);
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function extractUserText(context: Context): string {
	const chunks: string[] = [];
	for (const message of context.messages) {
		if (message.role !== "user") continue;
		const content = message.content;
		if (typeof content === "string") chunks.push(content);
		else if (Array.isArray(content)) {
			for (const part of content) {
				if (part && typeof part === "object" && "text" in part && typeof part.text === "string") {
					chunks.push(part.text);
				}
			}
		}
	}
	return chunks.join("\n");
}

function writeFakeJikji(binaryPath: string): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "find") {
	console.log(JSON.stringify({
		answer_paths: ["refund-policy.txt"],
		paths: ["refund-policy.txt"],
		candidates: [{ path: "refund-policy.txt", next_read: "original", label: "refund" }],
		evidence_pack: [{ path: "refund-policy.txt", next_read: "original" }],
		handoff_action: "raw_fallback_after_retry",
		tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
		agent_should_not_rerank: false,
	}));
} else {
	console.log(JSON.stringify({ prepared: true }));
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

function emitModel(onPrompt?: (text: string) => void) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "prefetch-model" }] });
	// The model cites evidence by id; this local ref is the absolute path of a
	// real file the model verified itself, which the final emit resolves.
	const refundRef = realpathSync(join(docs, "refund-policy.txt"));
	registration.setResponses([
		(context) => {
			onPrompt?.(extractUserText(context));
			return fauxAssistantMessage([{ type: "text", text: "초기 후보를 확인하고 있습니다." }], {
				stopReason: "stop",
			});
		},
		(context) => {
			onPrompt?.(extractUserText(context));
			return fauxAssistantMessage(
				[
					fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
						answer: "[1] Refund exceptions require director approval.",
						results: [
							{
								number: 1,
								title: "Refund approval",
								summary: "Refund exceptions require director approval before payout.",
								evidence: [{ excerpt: "director approval" }],
								confidence: 0.9,
								refs: [refundRef],
							},
						],
					}),
				],
				{ stopReason: "toolUse" },
			);
		},
	]);
	registrations.push(registration);
	return registration.getModel();
}

describe("AutoRAGAgent prefetchInitialRetrievalContext", () => {
	it("injects MinSync source paths into the first search prompt", async () => {
		writeFakeJikji(join(root, "fake-jikji.mjs"));
		writeFakeMinSync(join(root, "fake-minsync.mjs"));
		const source = realpathSync(join(docs, "refund-policy.txt"));
		const prompts: string[] = [];
		const agent = new AutoRAGAgent({
			model: emitModel((text) => prompts.push(text)),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { binaryPath: join(root, "fake-minsync.mjs"), autoInstall: false },
			jikji: { binaryPath: join(root, "fake-jikji.mjs") },
			everything: false,
			fsearch: false,
		});
		await agent.refresh(true);
		const response = await agent.searchDocuments("refund director approval");
		expect(response.results).toHaveLength(1);
		expect(prompts.join("\n")).toContain(source);
	});

	it("never indexes or prepares during a query, even on a never-refreshed workspace", async () => {
		// Inference must only read prebuilt indexes. Index builds belong to
		// `autorag refresh` / `watch`, never to the turn a user is waiting on.
		const jikjiLog = join(root, "jikji.log");
		const minSyncLog = join(root, "minsync.log");
		writeFileSync(
			join(root, "fake-jikji.mjs"),
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(jikjiLog)}, process.argv[2] + "\\n");
console.log(JSON.stringify(process.argv[2] === "find"
	? { answer_paths: [], paths: [], candidates: [], evidence_pack: [], handoff_action: "raw_fallback_after_retry",
		tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] }, agent_should_not_rerank: false }
	: { prepared: true }));
`,
		);
		chmodSync(join(root, "fake-jikji.mjs"), 0o755);
		writeFakeMinSync(join(root, "fake-minsync.mjs"), minSyncLog);
		const agent = new AutoRAGAgent({
			model: emitModel(),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { binaryPath: join(root, "fake-minsync.mjs"), autoInstall: false },
			jikji: { binaryPath: join(root, "fake-jikji.mjs"), autoInstall: false },
			everything: false,
			fsearch: false,
		});

		await agent.searchDocuments("refund director approval");
		await agent.searchDocuments("refund director approval");
		// Let any detached background work spawn before reading the logs.
		// (ES2022 target: no Promise.withResolvers.)
		await new Promise((resolve) => setTimeout(resolve, 300));

		const readLog = (path: string): string[] =>
			existsSync(path) ? readFileSync(path, "utf8").trim().split("\n").filter(Boolean) : [];
		const minSyncCommands = readLog(minSyncLog).map((line) => (JSON.parse(line) as { args: string[] }).args[0]);
		expect(readLog(jikjiLog).filter((command) => command !== "find")).toEqual([]);
		expect(minSyncCommands.filter((command) => command !== "query")).toEqual([]);
	});

	it("indexes MinSync incrementally and prepares every Jikji root on refresh", async () => {
		const jikjiLog = join(root, "jikji.log");
		const minSyncLog = join(root, "minsync.log");
		writeFileSync(
			join(root, "fake-jikji.mjs"),
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(jikjiLog)}, process.argv[2] + " " + process.argv[3] + "\\n");
console.log(JSON.stringify({ prepared: true }));
`,
		);
		chmodSync(join(root, "fake-jikji.mjs"), 0o755);
		writeFakeMinSync(join(root, "fake-minsync.mjs"), minSyncLog);
		const second = join(root, "notes");
		mkdirSync(second, { recursive: true });
		writeFileSync(join(second, "memo.txt"), "memo\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs, second],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { binaryPath: join(root, "fake-minsync.mjs"), autoInstall: false },
			jikji: { binaryPath: join(root, "fake-jikji.mjs"), autoInstall: false },
			everything: false,
			fsearch: false,
		});

		await agent.refresh(false);
		await agent.refresh(false);

		const syncs = readFileSync(minSyncLog, "utf8")
			.trim()
			.split("\n")
			.map((line) => (JSON.parse(line) as { args: string[] }).args)
			.filter((args) => args[0] === "sync");
		expect(syncs).toHaveLength(2);
		// The first refresh builds the index; the second only syncs changes.
		expect(syncs[1]).not.toContain("--full");
		const prepares = readFileSync(jikjiLog, "utf8").trim().split("\n");
		expect(prepares.filter((line) => line.startsWith("prepare"))).toHaveLength(4);
		expect(new Set(prepares.map((line) => line.split(" ")[1]))).toEqual(
			new Set([realpathSync(docs), realpathSync(second)]),
		);
	});
	it("collapses duplicate MinSync chunks without dropping distinct candidates", async () => {
		// CDC chunking can surface the same (source, content) pair more than once.
		// Deduplication must drop only the repeat, never the whole candidate set.
		const agent = new AutoRAGAgent({
			model: emitModel(),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: false,
			jikji: false,
			everything: false,
			fsearch: false,
		});
		const duplicated = [
			{ id: "1", source: "/docs/a.md", content: "shared boilerplate header", score: 0.9, metadata: {} },
			{ id: "2", source: "/docs/a.md", content: "shared boilerplate header", score: 0.8, metadata: {} },
			{ id: "3", source: "/docs/b.md", content: "unique refund clause", score: 0.7, metadata: {} },
		];
		const internals = agent as unknown as {
			minSyncMethod: unknown;
			prefetchInitialRetrievalContext: (
				query: string,
				searchQueries: readonly string[],
				options: Record<string, unknown>,
			) => Promise<string>;
		};
		internals.minSyncMethod = {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async () => duplicated,
		};

		const context = await internals.prefetchInitialRetrievalContext("refund", ["refund"], {});

		expect(context).toContain("shared boilerplate header");
		expect(context).toContain("unique refund clause");
		expect(context).toContain("/docs/b.md");
		expect(context.match(/shared boilerplate header/gu)).toHaveLength(1);
	});

	it("still emits structured results when Jikji prefetch throws", async () => {
		const agent = new AutoRAGAgent({
			model: emitModel(),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: false,
			jikji: { binaryPath: join(root, "missing-jikji") },
			everything: false,
			fsearch: false,
		});
		(agent as unknown as { findJikji: () => Promise<never> }).findJikji = async () => {
			throw new Error("jikji prefetch boom");
		};
		await expect(agent.searchDocuments("refund director approval")).resolves.toMatchObject({
			results: [{ number: 1 }],
		});
	});
});
