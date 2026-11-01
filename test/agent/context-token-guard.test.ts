import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	createContextTokenGuardedTransform,
	estimateContextMessageTokens,
	guardAgentContextMessages,
	MAX_OUTPUT_TOKENS,
	withBoundedOutput,
} from "../../src/agent/context-token-guard.ts";
import { noteContextOverflow } from "../../src/cloud/context-overflow.ts";

const model = {
	id: "guard-test",
	name: "Guard test",
	api: "test-api",
	provider: "test-provider",
	baseUrl: "http://localhost",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1_200,
	maxTokens: 200,
} as Model<any>;

function retrievalMessageForTool(toolName: string, contents: readonly string[]): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: "call-1",
		toolName,
		content: [
			{
				type: "text",
				text: contents
					.map((content, index) => `[${index + 1}] /docs/${index}.md score=0.9\n${content}`)
					.join("\n\n"),
			},
		],
		isError: false,
		timestamp: Date.now(),
	};
}

function retrievalMessage(contents: readonly string[]): AgentMessage {
	return retrievalMessageForTool("search_all_documents", contents);
}

function resultText(message: AgentMessage | undefined): string {
	if (message?.role !== "toolResult") return "";
	return message.content
		.filter((block): block is { type: "text"; text: string } => block.type === "text")
		.map((block) => block.text)
		.join("");
}

describe("context token guard", () => {
	it("keeps complete high-ranked candidates and drops lower-ranked candidates with a marker", () => {
		const contents = ["first candidate ".repeat(40), "second candidate ".repeat(40), "third candidate ".repeat(40)];
		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, [
			retrievalMessage(contents),
		]);
		const text = resultText(guarded[0]);

		expect(text).toContain(contents[0]);
		expect(text).not.toContain(contents[2]);
		expect(text).toContain("AutoRAG context guard");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});
	it.each(["semantic_search_local_docs", "fsearch_search"])("guards oversized %s retrieval results", (toolName) => {
		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, [
			retrievalMessageForTool(toolName, ["large retrieval candidate ".repeat(200)]),
		]);

		expect(resultText(guarded[0])).toContain("AutoRAG context guard");
	});
	it("guards Jikji answer paths as whole candidates", () => {
		const messages: AgentMessage[] = [
			{
				role: "toolResult",
				toolCallId: "call-jikji",
				toolName: "jikji_find",
				content: [
					{
						type: "text",
						text: `answer_paths:\n- /docs/first.md ${"hint ".repeat(200)}\n- /docs/second.md ${"hint ".repeat(200)}\n\ndirective: direct_use`,
					},
				],
				isError: false,
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);
		const text = resultText(guarded[0]);

		expect(text).toContain("/docs/first.md");
		expect(text).not.toContain("/docs/second.md");
		expect(text).toContain("AutoRAG context guard");
	});

	it("guards baseline retrieval embedded in a user prompt", () => {
		const messages: AgentMessage[] = [
			{
				role: "user",
				content: `Baseline retrieval evidence (already gathered for you):\n[1] /docs/first.md\n${"first ".repeat(120)}\n[2] /docs/second.md\n${"second ".repeat(120)}\n\nProduce the best answer now.`,
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);
		const message = guarded[0];
		const text = message?.role === "user" && typeof message.content === "string" ? message.content : "";

		expect(text).toContain("first ".repeat(120));
		expect(text).not.toContain("second ".repeat(120));
		expect(text).toContain("AutoRAG context guard");
	});

	it("guards candidates the harness numbered with evidence ids like [e1]", () => {
		const contents = ["first candidate ".repeat(40), "second candidate ".repeat(40), "third candidate ".repeat(40)];
		const message: AgentMessage = {
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "search_all_documents",
			content: [
				{
					type: "text",
					text: `Merged results (cite evidence by its [eN] id):\n\n${contents
						.map((content, index) => `[e${index + 1}] /docs/${index}.md score=0.9\n${content}`)
						.join("\n\n")}`,
				},
			],
			isError: false,
			timestamp: Date.now(),
		};
		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, [message]);
		const text = resultText(guarded[0]);

		expect(text).toContain(contents[0]);
		expect(text).not.toContain(contents[2]);
		expect(text).toContain("Retrieval candidates omitted: ");
	});

	it("reserves the system prompt and tool declarations that are not in the messages", () => {
		const contents = ["first candidate ".repeat(20), "second candidate ".repeat(20)];
		const messages = [retrievalMessage(contents)];
		const window = { ...model, contextWindow: 500, maxTokens: 100 };

		expect(resultText(guardAgentContextMessages(window, messages)[0])).toContain(contents[1]);
		const reserved = guardAgentContextMessages(window, messages, true, 150);

		expect(resultText(reserved[0])).toContain(contents[0]);
		expect(resultText(reserved[0])).not.toContain(contents[1]);
		expect(resultText(reserved[0])).toContain("AutoRAG context guard");
	});

	it("returns the same array when the request already fits", () => {
		const messages = [retrievalMessage(["short evidence"])];
		expect(guardAgentContextMessages(model, messages)).toBe(messages);
	});

	it("bounds an oversized non-retrieval tool result with an explicit marker", () => {
		const messages: AgentMessage[] = [
			{
				role: "toolResult",
				toolCallId: "call-bash",
				toolName: "bash",
				content: [{ type: "text", text: "large command output ".repeat(1_000) }],
				isError: false,
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);

		expect(resultText(guarded[0])).toContain("AutoRAG context guard");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});

	it("composes with an existing context transform", async () => {
		const transform = createContextTokenGuardedTransform(
			{ ...model, contextWindow: 500, maxTokens: 100 },
			async (messages) => [{ role: "user", content: "memory hint", timestamp: Date.now() }, ...messages],
		);

		const guarded = await transform([retrievalMessage(["latest evidence ".repeat(200)])]);

		expect(guarded[0]?.role).toBe("user");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});
	it("resolves a model selected after the transform is created", async () => {
		let selectedModel: Model<any> | undefined;
		const transform = createContextTokenGuardedTransform(() => selectedModel);
		const messages = [retrievalMessage(["large evidence ".repeat(200)])];

		expect(await transform(messages)).toBe(messages);
		selectedModel = { ...model, contextWindow: 500, maxTokens: 100 };

		const guarded = await transform(messages);
		expect(resultText(guarded[0])).toContain("AutoRAG context guard");
	});
	it("never replaces the user's query with the marker when reported usage exceeds the budget", () => {
		const messages: AgentMessage[] = [
			{
				role: "assistant",
				content: [{ type: "text", text: "short settled response" }],
				api: "test-api",
				provider: "test-provider",
				model: "guard-test",
				usage: {
					input: 1_000,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 1_000,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			},
			{ role: "user", content: "latest question", timestamp: Date.now() },
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);
		const user = guarded[1];

		expect(user?.role === "user" && user.content).toBe("latest question");
	});

	it("shortens only the newest tool result, and leaves the query and earlier results whole, when it is what exhausts the budget", () => {
		const messages: AgentMessage[] = [
			{ role: "user", content: "장원영 집 잘사냐?", timestamp: Date.now() },
			{
				role: "toolResult",
				toolCallId: "call-1",
				toolName: "bash",
				content: [{ type: "text", text: "earlier command output ".repeat(20) }],
				isError: false,
				timestamp: Date.now(),
			},
			{
				role: "toolResult",
				toolCallId: "call-2",
				toolName: "bash",
				content: [{ type: "text", text: "more command output ".repeat(1_000) }],
				isError: false,
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);

		expect(guarded[0]).toBe(messages[0]);
		expect(guarded[1]).toBe(messages[1]);
		expect(resultText(guarded[2])).toContain("AutoRAG context guard");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});

	it("leaves everything whole when earlier turns alone exceed the budget, for pi's compaction to summarise", () => {
		const messages: AgentMessage[] = [
			{ role: "user", content: "장원영 집 잘사냐?", timestamp: Date.now() },
			{
				role: "toolResult",
				toolCallId: "call-1",
				toolName: "bash",
				content: [{ type: "text", text: "large command output ".repeat(1_000) }],
				isError: false,
				timestamp: Date.now(),
			},
			{
				role: "toolResult",
				toolCallId: "call-2",
				toolName: "bash",
				content: [{ type: "text", text: "more command output ".repeat(1_000) }],
				isError: false,
				timestamp: Date.now(),
			},
		];

		expect(guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages)).toBe(messages);
	});

	it("leaves a long user prompt whole when it alone exceeds the budget", () => {
		const query = "Answer this original query immediately: 장원영 집 잘사냐?";
		const messages: AgentMessage[] = [
			{ role: "user", content: `${query}\n\n${"filler rule ".repeat(5_000)}`, timestamp: Date.now() },
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages, false, 280);

		expect(guarded).toBe(messages);
	});

	it("does not let a catalog maxTokens near the context window zero the input budget", () => {
		const window = { ...model, contextWindow: 262_144, maxTokens: 235_929 };
		const messages: AgentMessage[] = [
			{ role: "user", content: "장원영 집 잘사냐?", timestamp: Date.now() },
			retrievalMessage(["evidence ".repeat(40)]),
		];

		const guarded = guardAgentContextMessages(window, messages, false, 10_000);

		expect(guarded).toBe(messages);
	});

	it("holds a catalog maxTokens near the context window to the output cap, leaving a smaller one as is", () => {
		const near = { ...model, contextWindow: 262_144, maxTokens: 235_929 };
		const small = { ...model, contextWindow: 262_144, maxTokens: 8_000 };

		expect(withBoundedOutput(near).maxTokens).toBe(MAX_OUTPUT_TOKENS);
		expect(withBoundedOutput(near).contextWindow).toBe(262_144);
		expect(withBoundedOutput(small)).toBe(small);
	});

	it("keeps input plus the bounded output inside the window that overflowed in production", () => {
		// 262,144 window, ~88k real input (text + tool declarations), catalog maxTokens 235,929:
		// pi asked for 195,111 output tokens and the request was 20,691 over the window.
		const bounded = withBoundedOutput({ ...model, contextWindow: 262_144, maxTokens: 235_929 });
		const input = 87_724;

		expect(input + bounded.maxTokens).toBeLessThanOrEqual(bounded.contextWindow);
	});

	describe("only retrieval chunks are dropped, lowest-ranked first", () => {
		const tight = { ...model, contextWindow: 1_200, maxTokens: 200 };
		// About 150 tokens each, so four fit the 800-token input budget and five do not.
		const chunks = (prefix: string, count: number): string[] =>
			Array.from({ length: count }, (_unused, index) => `${prefix}${index + 1} chunk `.repeat(55));

		it("keeps the top-ranked chunk of every retrieval result before any second-ranked one", () => {
			const older = chunks("older", 5);
			const newer = chunks("newer", 5);
			const guarded = guardAgentContextMessages(tight, [retrievalMessage(older), retrievalMessage(newer)]);
			const olderText = resultText(guarded[0]);
			const newerText = resultText(guarded[1]);

			expect(olderText).toContain(older[0] ?? "");
			expect(newerText).toContain(newer[0] ?? "");
			expect(olderText).not.toContain(older[4] ?? "");
			expect(newerText).not.toContain(newer[4] ?? "");
			// A chunk is only ever dropped after every chunk ranked above it, in any result, is gone.
			for (const rank of [1, 2, 3]) {
				const keptBelow = [olderText.includes(older[rank + 1] ?? ""), newerText.includes(newer[rank + 1] ?? "")];
				if (keptBelow.some(Boolean)) {
					expect(olderText).toContain(older[rank] ?? "");
					expect(newerText).toContain(newer[rank] ?? "");
				}
			}
			expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(800);
		});

		it("drops the lowest-ranked chunk of every source list, not the whole trailing list", () => {
			const minsync = chunks("minsync", 3);
			const datasource = chunks("datasource", 3);
			const list = (heading: string, items: readonly string[], offset: number): string =>
				`${heading}\n${items.map((item, index) => `[${offset + index}] /docs/${heading.length}-${index}.md\n${item}`).join("\n")}`;
			const query = "Answer this original query immediately: 장원영 집 잘사냐?";
			const rules = "Produce the best answer now.";
			const messages: AgentMessage[] = [
				{
					role: "user",
					content: `${query}\n\nBaseline retrieval evidence (already gathered for you):\n${list("MinSync semantic initial candidates:", minsync, 1)}\n\n${list("Datasource initial candidates:", datasource, 4)}\n\n${rules}`,
					timestamp: Date.now(),
				},
			];

			const guarded = guardAgentContextMessages(tight, messages, false);
			const message = guarded[0];
			const text = message?.role === "user" && typeof message.content === "string" ? message.content : "";

			expect(text.startsWith(query)).toBe(true);
			expect(text.endsWith(rules)).toBe(true);
			expect(text).toContain(minsync[0] ?? "");
			expect(text).toContain(datasource[0] ?? "");
			expect(text).not.toContain(minsync[2] ?? "");
			expect(text).not.toContain(datasource[2] ?? "");
			expect(text).toContain("Datasource initial candidates:");
			expect(text).toContain("AutoRAG context guard");
		});

		it("never edits the system prompt, an earlier user or assistant turn, an earlier tool result, or the latest query", () => {
			const assistant: AgentMessage = {
				role: "assistant",
				content: [{ type: "text", text: "first answer" }],
				api: "test-api",
				provider: "test-provider",
				model: "guard-test",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};
			const untouched: AgentMessage[] = [
				{ role: "system", content: "system instruction ".repeat(300), timestamp: Date.now() },
				{ role: "user", content: "first question 장원영 집 잘사냐?", timestamp: Date.now() },
				assistant,
				{
					role: "toolResult",
					toolCallId: "call-old",
					toolName: "bash",
					content: [{ type: "text", text: "older command output ".repeat(600) }],
					isError: false,
					timestamp: Date.now(),
				},
				{
					role: "user",
					content: [{ type: "image", mimeType: "image/png", data: "large-image" }],
					timestamp: Date.now(),
				},
			];
			const retrieval = retrievalMessage(chunks("evidence", 6));
			const latest: AgentMessage = { role: "user", content: "latest follow-up question", timestamp: Date.now() };
			const messages = [...untouched, retrieval, latest];

			const guarded = guardAgentContextMessages(tight, messages, false);

			expect(guarded).toHaveLength(messages.length);
			for (const [index, message] of untouched.entries()) expect(guarded[index]).toBe(message);
			expect(guarded.at(-1)).toBe(latest);
			expect(resultText(guarded[untouched.length])).toContain("AutoRAG context guard");
		});

		it("returns a transcript with no retrieval chunks untouched however far over budget it is", () => {
			const messages: AgentMessage[] = [
				{ role: "system", content: "system instruction ".repeat(1_000), timestamp: Date.now() },
				{ role: "user", content: "first question", timestamp: Date.now() },
				{ role: "user", content: `latest question\n\n${"background ".repeat(5_000)}`, timestamp: Date.now() },
			];

			expect(guardAgentContextMessages(tight, messages, false)).toBe(messages);
		});
	});

	describe("dense text a chars/4 guess undercounts", () => {
		// 12,000 window, 1.2 headroom, 1,000 output reserve: a 9,000-token input budget.
		const denseModel = { ...model, contextWindow: 12_000, maxTokens: 1_000 };
		const inputBudget = 9_000;
		/** Deterministic text with nothing for a tokenizer to merge: punctuation, control characters, digits, replacement characters. */
		const denseText = (length: number): string => {
			const alphabet = "!@#$%^&*()_+-=[]{};:'\",.<>/?\\|~`0123456789abcXYZ\u0000\u0001\ufffd";
			let state = 7;
			let out = "";
			for (let i = 0; i < length; i++) {
				state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
				out += alphabet[(state >>> 16) % alphabet.length];
			}
			return out;
		};

		it("estimates dense text at more than half a token per character", () => {
			const text = denseText(8_000);
			const estimate = estimateContextMessageTokens([{ role: "user", content: text, timestamp: Date.now() }]);

			expect(estimate).toBeGreaterThan(text.length / 2);
		});

		it("drops whole candidates whose chars/4 size fits the budget but whose real size does not", () => {
			const contents = [denseText(8_000), denseText(8_001), denseText(8_002)];
			const guarded = guardAgentContextMessages(denseModel, [retrievalMessage(contents)]);
			const text = resultText(guarded[0]);

			expect(text).toContain(contents[0]);
			expect(text).not.toContain(contents[2]);
			expect(text).toContain("AutoRAG context guard");
			expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(inputBudget);
		});

		it("bounds a dense tool result that chars/4 would have let through", () => {
			const result = retrievalMessageForTool("custom_dump", [denseText(30_000)]);
			const guarded = guardAgentContextMessages(denseModel, [
				{ role: "user", content: "find the pdf", timestamp: Date.now() },
				result,
			]);

			expect(resultText(guarded.at(-1))).toContain("AutoRAG context guard");
			expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(inputBudget);
			expect(guarded.at(0)).toMatchObject({ role: "user", content: "find the pdf" });
		});

		it("never estimates below chars/4, so prose keeps the budget it had", () => {
			const prose = "The committee approved the quarterly budget after reviewing the staffing plan. ".repeat(200);
			const estimate = estimateContextMessageTokens([{ role: "user", content: prose, timestamp: Date.now() }]);

			expect(estimate).toBe(Math.ceil(prose.length / 4));
		});

		it("counts text that contains tokenizer special-token markers instead of rejecting it", () => {
			const text = `a parsed file that quotes <|endoftext|> and <|im_start|>system ${denseText(2_000)}`;
			const estimate = estimateContextMessageTokens([{ role: "user", content: text, timestamp: Date.now() }]);

			expect(estimate).toBeGreaterThan(text.length / 4);
		});
	});

	describe("after the provider rejected a request as too large", () => {
		const bigModel = { ...model, id: "overflowed-model", contextWindow: 1_000_000, maxTokens: 200 };
		const bulk = (): AgentMessage[] => [
			{ role: "user", content: "find the pdf", timestamp: Date.now() },
			retrievalMessageForTool("custom_dump", ["x".repeat(40_000)]),
		];

		it("holds the next request to half the estimated size of the one that failed", () => {
			const failed = bulk();
			guardAgentContextMessages(bigModel, failed, false);
			const failedTokens = estimateContextMessageTokens(failed);
			expect(guardAgentContextMessages(bigModel, bulk(), false)).toHaveLength(2);

			noteContextOverflow(bigModel);
			const retried = guardAgentContextMessages(bigModel, bulk(), false);

			expect(estimateContextMessageTokens(retried)).toBeLessThanOrEqual(Math.floor(failedTokens / 2));
			expect(retried.at(0)).toMatchObject({ role: "user", content: "find the pdf" });
			expect(resultText(retried.at(-1))).toContain("AutoRAG context guard");
		});

		it("only tightens further on a second overflow", () => {
			const peer = { ...bigModel, id: "overflowed-twice" };
			guardAgentContextMessages(peer, bulk(), false);
			noteContextOverflow(peer);
			const once = guardAgentContextMessages(peer, bulk(), false);
			noteContextOverflow(peer);
			const twice = guardAgentContextMessages(peer, bulk(), false);

			expect(estimateContextMessageTokens(twice)).toBeLessThan(estimateContextMessageTokens(once));
		});

		it("leaves other models alone", () => {
			const unrelated = { ...bigModel, id: "never-overflowed" };
			guardAgentContextMessages(bigModel, bulk(), false);
			noteContextOverflow(bigModel);

			expect(guardAgentContextMessages(unrelated, bulk(), false)).toHaveLength(2);
			expect(estimateContextMessageTokens(guardAgentContextMessages(unrelated, bulk(), false))).toBe(
				estimateContextMessageTokens(bulk()),
			);
		});
	});
});
