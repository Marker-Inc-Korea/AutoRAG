import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentEvent } from "@earendil-works/pi-agent-core";
import { describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import {
	createTuiPresenter,
	parseSlashCommand,
	runTui,
	type TuiDeps,
	type TuiDriver,
} from "../../src/cli/commands/tui.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import {
	createFileTuiSessionStore,
	createMergedTuiSessionStore,
	renderTuiSessionList,
} from "../../src/cli/tui-session-store.ts";

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

	it("uses injected agents without requiring a user runtime config", async () => {
		const { ctx } = context();
		const tui = driver([]);
		const running = runTui(ctx, {
			agentFactory: () => ({ searchDocuments: async () => response }),
			modelResolver: () => {
				throw new Error("model resolution should not run for injected agents");
			},
			tuiFactory: () => tui,
		});

		tui.onExit?.();

		expect(await running).toBe(0);
		expect(tui.started).toBe(true);
		expect(tui.stopped).toBe(true);
	});

	it("renders streaming progress before the final answer", async () => {
		const { ctx } = context();
		const tui = driver([]);
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async () => response,
				async *searchDocumentsStream() {
					yield { type: "progress" as const, sessionId: "session", query: "question", text: "현재 자료를 확인 중입니다." };
					yield { type: "complete" as const, response };
				},
			}),
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("question");
		await tui.onSubmit?.("/quit");
		await running;

		expect(tui.rendered.join("\n")).toContain("현재 자료를 확인 중입니다.");
		expect(tui.rendered.join("\n")).toContain("answer");
	});

	it("opens a resume picker with first-question titles and restores the selected session", async () => {
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
		await tui.onSubmit?.("1");
		await tui.onSubmit?.("/quit");

		expect(calls).toBe(0);
		expect(tui.rendered).toHaveLength(1);
		expect(tui.rendered[0]).toContain("old question");
		expect(tui.rendered[0]).toContain("old answer");
		expect(tui.rendered[0]).not.toContain("unknown command");
		expect(tui.stopped).toBe(true);
		expect(await running).toBe(0);
	});

	it("replaces the current view instead of mixing it with a resumed session", async () => {
		const { ctx } = context();
		const tui = driver([]);
		const session = {
			id: "old-session",
			query: "old question",
			answer: "old answer",
			trace: "old trace",
			updatedAt: 1,
		};
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async () => ({ ...response, query: "current question", answer: "current answer" }),
			}),
			sessionStore: {
				list: () => [session],
				get: (id: string) => (id === session.id ? session : undefined),
				save: () => undefined,
			},
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("current question");
		await tui.onSubmit?.("/resume");
		await tui.onSubmit?.("1");

		const currentView = tui.rendered.at(-1) ?? "";
		expect(currentView).toContain("old question");
		expect(currentView).toContain("old answer");
		expect(currentView).not.toContain("current question");
		expect(currentView).not.toContain("current answer");
		expect(tui.rendered).toHaveLength(1);
		tui.onExit?.();
		expect(await running).toBe(0);
	});

	it("updates one trace row in place and keeps the final answer last", () => {
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
		presenter.handle({
			type: "tool_execution_start",
			toolCallId: "tool-2",
			toolName: "verify_sources",
			args: {},
		});
		presenter.handle({
			type: "tool_execution_end",
			toolCallId: "tool-2",
			toolName: "verify_sources",
			result: {},
			isError: false,
		});
		presenter.handle({ type: "agent_end", messages: [] });

		expect(presenter.lines()).toEqual([
			"\u001b[90m\u001b[2mthinking: (collapsed)\u001b[0m",
			"✓ search_datasource_documents: done",
			"✓ verify_sources: done",
			"assistant: final answer",
		]);
		expect(presenter.working()).toBe(false);
	});

	it("clears the previous trace when a different session is restored", () => {
		const presenter = createTuiPresenter();
		presenter.handle({
			type: "tool_execution_start",
			toolCallId: "old-tool",
			toolName: "old_session_tool",
			args: {},
		});
		presenter.handle({
			type: "message_update",
			message: { role: "assistant" },
			assistantMessageEvent: { type: "text_delta", delta: "old answer" },
		} as AgentEvent);

		presenter.reset();
		presenter.handle({
			type: "tool_execution_start",
			toolCallId: "new-tool",
			toolName: "new_session_tool",
			args: {},
		});

		expect(presenter.lines()).toEqual(["new_session_tool: searching"]);
	});

	it("shows working and completed status text around an agent run", () => {
		const presenter = createTuiPresenter();
		expect(presenter.status()).toBe("ready");
		presenter.beginRun();
		expect(presenter.status()).toBe("working...");
		presenter.handle({ type: "agent_end", messages: [] });
		expect(presenter.status()).toBe("completed.");
	});

	it("merges workspace and global sessions by newest record", () => {
		const workspace = {
			list: () => [{ id: "same", query: "workspace question", answer: "a", trace: "", updatedAt: 1 }],
			get: () => undefined,
			save: () => undefined,
		};
		const global = {
			list: () => [
				{ id: "same", query: "global question", answer: "b", trace: "", updatedAt: 2 },
				{ id: "global-only", query: "first global question", answer: "c", trace: "", updatedAt: 3 },
			],
			get: () => undefined,
			save: () => undefined,
		};

		const merged = createMergedTuiSessionStore(workspace, global);

		expect(merged.list().map((session) => session.query)).toEqual(["first global question", "global question"]);
		expect(renderTuiSessionList(merged.list())).toContain("1. first global question");
	});

	it("writes new sessions to both workspace and global stores", () => {
		let workspaceSaved = 0;
		let globalSaved = 0;
		const session = { id: "saved", query: "q", answer: "a", trace: "", updatedAt: 1 };
		const makeStore = (save: () => void) => ({
			list: () => [],
			get: () => undefined,
			save,
		});
		const merged = createMergedTuiSessionStore(
			makeStore(() => workspaceSaved++),
			makeStore(() => globalSaved++),
		);

		merged.save(session);

		expect(workspaceSaved).toBe(1);
		expect(globalSaved).toBe(1);
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
					void listener?.(
						{
							type: "tool_execution_start",
							toolCallId: "tool-1",
							toolName: "search_datasource_documents",
							args: {},
						},
						new AbortController().signal,
					);
					void listener?.(
						{
							type: "message_update",
							message: { role: "assistant" },
							assistantMessageEvent: { type: "text_delta", delta: "streamed" },
						} as AgentEvent,
						new AbortController().signal,
					);
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
		expect(tui.rendered.join("\n")).toContain("search_datasource_documents");
		expect(tui.rendered.join("\n")).toContain("assistant: streamed");
		expect(tui.stopped).toBe(true);
	});

	it("persists completed lifecycle statuses instead of transient searching rows", async () => {
		const { ctx } = context();
		const tui = driver([]);
		const saved: string[] = [];
		let listener: ((event: AgentEvent, signal: AbortSignal) => void | Promise<void>) | undefined;
		const running = runTui(ctx, {
			agentFactory: () => ({
				searchDocuments: async () => {
					listener?.(
						{
							type: "tool_execution_start",
							toolCallId: "tool-1",
							toolName: "lookup",
							args: {},
						},
						new AbortController().signal,
					);
					listener?.(
						{
							type: "tool_execution_end",
							toolCallId: "tool-1",
							toolName: "lookup",
							result: {},
							isError: false,
						},
						new AbortController().signal,
					);
					return response;
				},
				subscribe: (callback) => {
					listener = callback;
					return () => {
						listener = undefined;
					};
				},
			}),
			sessionStore: {
				list: () => [],
				get: () => undefined,
				save: (session) => saved.push(session.trace),
			},
			tuiFactory: () => tui,
		});

		await tui.onSubmit?.("question");
		tui.onExit?.();

		expect(await running).toBe(0);
		expect(saved).toEqual(["✓ lookup: done"]);
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
		expect(tui.rendered.join("\n")).toContain("first");
		expect(tui.rendered.join("\n")).toContain("second");
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
