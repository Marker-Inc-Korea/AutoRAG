import { randomUUID } from "node:crypto";
import { chmodSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
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
	registration.setResponses([
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
							},
						],
						mapping: [
							{ number: 1, source: "/docs/refund-policy.txt", method: "minsync", content: "director approval" },
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
		let prompt = "";
		const agent = new AutoRAGAgent({
			model: emitModel((text) => {
				prompt = text;
			}),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { binaryPath: join(root, "fake-minsync.mjs"), autoInstall: false },
			jikji: { binaryPath: join(root, "fake-jikji.mjs") },
		});
		await agent.refresh(true);
		const response = await agent.searchDocuments("refund director approval");
		expect(response.results).toHaveLength(1);
		expect(prompt).toContain(source);
	});

	it("still emits structured results when Jikji prefetch throws", async () => {
		const agent = new AutoRAGAgent({
			model: emitModel(),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: false,
			bm25: false,
			jikji: { binaryPath: join(root, "missing-jikji") },
		});
		(agent as unknown as { findJikji: () => Promise<never> }).findJikji = async () => {
			throw new Error("jikji prefetch boom");
		};
		await expect(agent.searchDocuments("refund director approval")).resolves.toMatchObject({
			results: [{ number: 1 }],
		});
	});
});
