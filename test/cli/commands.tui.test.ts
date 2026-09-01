import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { AgentEvent } from "@earendil-works/pi-agent-core";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import {
	createTuiPresenter,
	parseSlashCommand,
	runTui,
	type TuiDeps,
	type TuiDriver,
} from "../../src/cli/commands/tui.ts";
import { createFileTuiSessionStore } from "../../src/cli/tui-session-store.ts";

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
	it("persists and reloads resumable TUI sessions", () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-tui-session-"));
		try {
			const store = createFileTuiSessionStore(join(root, "sessions.json"));
			store.save({
				id: "session-1",
				query: "old question",
				answer: "old answer",
				trace: "old trace",
				updatedAt: 1,
			});
			const reloaded = createFileTuiSessionStore(join(root, "sessions.json"));
			expect(reloaded.get("session-1")).toMatchObject({
				query: "old question",
				answer: "old answer",
				trace: "old trace",
			});
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("parses supported slash commands and rejects unknown commands", () => {
		expect(parseSlashCommand("/")).toEqual({ kind: "incomplete" });
		expect(parseSlashCommand("/quit")).toEqual({ kind: "quit" });
		expect(parseSlashCommand("/resume")).toEqual({ kind: "resume", sessionId: undefined });
		expect(parseSlashCommand("/resume abc")).toEqual({ kind: "resume", sessionId: "abc" });
		expect(parseSlashCommand("/unknown")).toEqual({ kind: "unknown", name: "unknown" });
	});

	it("handles slash commands without invoking search", async () => {
		const { ctx } = context();
		const tui = driver([]);
		let calls = 0;
		const session = {
			id: "old-session",
			query: "old question",
			answer: "old answer",
			trace: "old trace",
			updatedAt: 1,
		};
		const store = {
			list: () => [session],
			get: (id: string) => (id === session.id ? session : undefined),
			save: () => undefined,
		};
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async () => {
					calls++;
					return response;
				},
			}),
			sessionStore: store,
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("/");
		await tui.onSubmit?.("/does-not-exist");
		await tui.onSubmit?.("/resume");
		await tui.onSubmit?.("/resume old-session");
		await tui.onSubmit?.("/quit");

		expect(calls).toBe(0);
		expect(tui.rendered.join("\n")).toContain("commands: /quit, /resume [session-id]");
		expect(tui.rendered.join("\n")).toContain("unknown command: /does-not-exist");
		expect(tui.rendered.join("\n")).toContain("old answer");
		expect(tui.stopped).toBe(true);
		expect(await running).toBe(0);
	});

	it("renders thinking, answer deltas, and datasource tool lifecycle", () => {
		const presenter = createTuiPresenter();
		presenter.handle({ type: "agent_start" });
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "thinking_delta", delta: "checking indexes" },
		} as AgentEvent);
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "thinking_end", content: "checking indexes" },
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

		expect(presenter.working()).toBe(true);
		expect(presenter.lines()).toEqual([
			"working",
			"agent: started",
			"\u001b[90m\u001b[2mthinking: (collapsed)\u001b[0m",
			"search_datasource_documents: started",
			"search_datasource_documents: done",
			"assistant: final answer",
		]);
		presenter.handle({ type: "agent_end", messages: [] });
		expect(presenter.working()).toBe(false);
	});

	it("shows working during an active run and collapses thinking after completion", () => {
		const presenter = createTuiPresenter();
		expect(presenter.working()).toBe(false);
		presenter.handle({ type: "agent_start" });
		expect(presenter.working()).toBe(true);
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "thinking_delta", delta: "private reasoning" },
		} as AgentEvent);
		expect(presenter.lines().join("\n")).toContain("private reasoning");
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "thinking_end", content: "private reasoning" },
		} as AgentEvent);
		expect(presenter.lines().join("\n")).not.toContain("private reasoning");
		expect(presenter.lines().join("\n")).toContain("collapsed");
	});

	it("interrupts an active search and marks the TUI paused", async () => {
		const { ctx } = context();
		const tui = driver([]);
		let abortCalls = 0;
		let resolveSearch: ((value: SearchDocumentsResponse) => void) | undefined;
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: () =>
					new Promise((resolve) => {
						resolveSearch = resolve;
					}),
				abort: () => {
					abortCalls++;
					resolveSearch?.(response);
				},
			}),
			tuiFactory: () => tui,
		});

		const submission = tui.onSubmit?.("long query");
		tui.onInput?.("\u0003");

		expect(abortCalls).toBe(1);
		expect(tui.rendered.join("\n")).toContain("interrupted");
		await submission;
		expect(tui.rendered.join("\n")).not.toContain("answer");
		tui.onExit?.();
		expect(await running).toBe(0);
	});

	it("consumes Ctrl+C while idle and leaves Ctrl+D as the exit action", async () => {
		const { ctx } = context();
		const tui = driver([]);
		const running = runTui(ctx, {
			agentFactory: () => ({ searchDocuments: async () => response }),
			tuiFactory: () => tui,
		});

		tui.onInput?.("\u0003");

		expect(tui.stopped).toBe(false);
		expect(tui.rendered.join("\n")).toContain("paused");
		tui.onInput?.("\u0004");
		tui.onExit?.();
		expect(await running).toBe(0);
	});

	it("ignores late agent events after interrupt until the next run begins", () => {
		const presenter = createTuiPresenter();
		presenter.beginRun();
		presenter.handle({ type: "agent_start" });
		presenter.interrupt();
		presenter.handle({ type: "agent_start" });
		expect(presenter.working()).toBe(false);
		expect(presenter.lines()[0]).toBe("paused");
		presenter.beginRun();
		presenter.handle({ type: "agent_start" });
		expect(presenter.working()).toBe(true);
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

	it("unsubscribes the event bridge exactly when the TUI exits", async () => {
		const { ctx } = context();
		const tui = driver([]);
		let unsubscribeCalls = 0;
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async () => response,
				subscribe: () => () => {
					unsubscribeCalls++;
				},
			}),
			tuiFactory: () => tui,
		});

		tui.onExit?.();

		expect(await running).toBe(0);
		expect(unsubscribeCalls).toBe(1);
	});
});
