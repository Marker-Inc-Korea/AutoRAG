import { describe, expect, it } from "vitest";
import type { AgentEvent } from "@earendil-works/pi-agent-core";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { createTuiPresenter, runTui, type TuiDeps, type TuiDriver } from "../../src/cli/commands/tui.ts";

const response: SearchDocumentsResponse = {
	sessionId: "session",
	query: "question",
	answer: "answer",
	results: [],
	searched: 1,
	warnings: [],
	diagnostics: [],
};

function context(): { ctx: CommandContext; stdout: string[]; stderr: string[] } {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		ctx: {
			positionals: [],
			flags: {},
			json: false,
			debug: false,
			cwd: process.cwd(),
			stdout: (line) => stdout.push(line),
			stderr: (line) => stderr.push(line),
		},
		stdout,
		stderr,
	};
}

function driver(submissions: string[]): TuiDriver {
	return {
		submissions,
		rendered: [],
		started: false,
		stopped: false,
		onSubmit: undefined,
		onExit: undefined,
	};
}

describe("runTui", () => {
	it("renders thinking, answer deltas, and datasource tool lifecycle", () => {
		const presenter = createTuiPresenter();
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "thinking_delta", delta: "checking indexes" },
		} as AgentEvent);
		presenter.handle({
			type: "tool_execution_start",
			toolCallId: "tool-1",
			toolName: "search_datasource_documents",
			args: { query: "question" },
		});
		presenter.handle({
			type: "tool_execution_end",
			toolCallId: "tool-1",
			toolName: "search_datasource_documents",
			result: { details: { count: 2 } },
			isError: false,
		});
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "text_delta", delta: "final answer" },
		} as AgentEvent);

		expect(presenter.lines()).toEqual([
			"thinking: checking indexes",
			"search_datasource_documents: started",
			"search_datasource_documents: done",
			"assistant: final answer",
		]);
	});

	it("searches submitted questions and exits on the driver exit signal", async () => {
		const { ctx } = context();
		const tui = driver(["question"]);
		const queries: string[] = [];
		let listener: ((event: AgentEvent, signal: AbortSignal) => void | Promise<void>) | undefined;
		const deps: TuiDeps = {
			agentFactory: () => ({
				searchDocuments: async (query) => {
					queries.push(query);
					void listener?.({
						type: "tool_execution_start",
						toolCallId: "tool-1",
						toolName: "search_datasource_documents",
						args: {},
					}, new AbortController().signal);
					void listener?.({
						type: "message_update",
						message: { role: "assistant" },
						assistantMessageEvent: { type: "text_delta", delta: "streamed" },
					} as AgentEvent, new AbortController().signal);
					return response;
				},
				subscribe: (callback) => {
					listener = callback;
					return () => {
						listener = undefined;
					};
				},
			}),
			tuiFactory: () => tui,
		};

		const running = runTui(ctx, deps);
		await tui.onSubmit?.("question");
		tui.onExit?.();

		expect(await running).toBe(0);
		expect(queries).toEqual(["question"]);
		expect(tui.rendered.join("\n")).toContain("answer");
		expect(tui.rendered.join("\n")).toContain("search_datasource_documents: started");
		expect(tui.rendered.join("\n")).toContain("assistant: streamed");
		expect(tui.stopped).toBe(true);
	});

	it("ignores blank submissions without calling search", async () => {
		const { ctx } = context();
		const tui = driver([]);
		let calls = 0;
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async () => {
					calls++;
					return response;
				},
			}),
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("  ");
		tui.onExit?.();

		expect(await running).toBe(0);
		expect(calls).toBe(0);
	});

	it("renders a failed search and accepts a later submission", async () => {
		const { ctx } = context();
		const tui = driver([]);
		let calls = 0;
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async (query) => {
					calls++;
					if (query === "broken") throw new Error("provider unavailable");
					return { ...response, query, answer: "recovered" };
				},
			}),
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("broken");
		await tui.onSubmit?.("recovered");
		tui.onExit?.();

		expect(await running).toBe(0);
		expect(calls).toBe(2);
		expect(tui.rendered.join("\n")).toContain("provider unavailable");
		expect(tui.rendered.join("\n")).toContain("recovered");
	});

	it("keeps one event subscription alive across multiple questions", async () => {
		const { ctx } = context();
		const tui = driver([]);
		let listener: ((event: AgentEvent, signal: AbortSignal) => void | Promise<void>) | undefined;
		let calls = 0;
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async (query) => {
					calls++;
					listener?.(
						{
							type: "tool_execution_start",
							toolCallId: `tool-${calls}`,
							toolName: query,
							args: {},
						},
						new AbortController().signal,
					);
					return { ...response, query };
				},
				subscribe: (callback) => {
					listener = callback;
					return () => {
						listener = undefined;
					};
				},
			}),
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("first");
		await tui.onSubmit?.("second");
		tui.onExit?.();

		expect(await running).toBe(0);
		expect(tui.rendered.join("\n")).toContain("first: started");
		expect(tui.rendered.join("\n")).toContain("second: started");
	});
});
