import { join } from "node:path";
import type { AgentEvent } from "@earendil-works/pi-agent-core";
import {
	CombinedAutocompleteProvider,
	Editor,
	ProcessTerminal,
	Text,
	type TUI,
	TuiMainScreen,
} from "@earendil-works/pi-tui";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../agent/agent.ts";
import type { SearchDocumentsResponse } from "../../agent/search-documents.ts";
import { resolveAutoRAGHome } from "../../config/home.ts";
import {
	buildAgentOptions,
	type CliConfig,
	type ResolvedAgentModel,
	resolveAgentModel,
	resolveConfig,
} from "../config.ts";
import { renderError, renderSearch } from "../output.ts";
import { createTuiSlashCommands, parseSlashCommand, renderSlashHelp } from "../tui-commands.ts";
import {
	createFileTuiSessionStore,
	createMergedTuiSessionStore,
	renderRestoredTuiSession,
	renderTuiSessionList,
	type TuiSessionRecord,
	type TuiSessionStore,
} from "../tui-session-store.ts";
import type { CommandContext } from "./types.ts";

export { parseSlashCommand } from "../tui-commands.ts";

export interface TuiDriver {
	submissions: string[];
	rendered: string[];
	started: boolean;
	stopped: boolean;
	onSubmit?: (text: string) => void | Promise<void>;
	onExit?: () => void;
	onEvent?: (event: AgentEvent) => void;
	onInput?: (data: string) => void;
}

export interface TuiPresenter {
	handle(event: AgentEvent): void;
	lines(): readonly string[];
	working(): boolean;
	setWorking(active: boolean): void;
	beginRun(): void;
	interrupt(): void;
	status(): string;
	reset(): void;
}

type TuiAgent = Pick<AutoRAGAgent, "searchDocuments"> & Partial<Pick<AutoRAGAgent, "subscribe" | "abort">>;

export interface TuiDeps {
	agentFactory?: (opts: AutoRAGAgentOptions) => TuiAgent;
	modelResolver?: (config: CliConfig) => ResolvedAgentModel;
	tuiFactory?: (ctx: CommandContext) => TuiDriver;
	sessionStore?: TuiSessionStore;
}

export function createTuiPresenter(): TuiPresenter {
	const output: string[] = [];
	let thinkingIndex: number | undefined;
	let answer = "";
	let active = false;
	let paused = false;
	let suppressEvents = false;
	const lifecycleRows = new Map<string, number>();
	let statusText = "ready";

	const append = (line: string): void => {
		output.push(line);
	};
	const appendDelta = (prefix: string, delta: string, index: "thinking" | "answer"): void => {
		if (index === "answer") {
			answer = `${answer.length === 0 ? prefix : answer}${delta}`;
			return;
		}
		const currentIndex = thinkingIndex;
		if (currentIndex === undefined) {
			append(dimGray(`${prefix}${delta}`));
			thinkingIndex = output.length - 1;
			return;
		}
		output[currentIndex] = dimGray(`${stripAnsi(output[currentIndex])}${delta}`);
	};
	const updateLifecycle = (key: string, started: string, completed: string): void => {
		const index = lifecycleRows.get(key);
		if (index === undefined) {
			lifecycleRows.set(key, output.length);
			append(started);
			return;
		}
		output[index] = completed;
	};

	return {
		working: () => active,
		status: () => statusText,
		reset: () => {
			output.length = 0;
			thinkingIndex = undefined;
			answer = "";
			active = false;
			paused = false;
			suppressEvents = false;
			lifecycleRows.clear();
			statusText = "ready";
		},
		setWorking: (value) => {
			active = value;
			statusText = value ? "working..." : "completed.";
		},
		beginRun: () => {
			active = true;
			statusText = "working...";
			paused = false;
			suppressEvents = false;
		},
		interrupt: () => {
			active = false;
			statusText = "stopped.";
			paused = true;
			suppressEvents = true;
		},
		handle(event) {
			if (suppressEvents) return;
			switch (event.type) {
				case "agent_start":
					active = true;
					statusText = "working...";
					paused = false;
					return;
				case "agent_end":
					active = false;
					statusText = "completed.";
					return;
				case "turn_start":
					return;
				case "turn_end":
					return;
				case "message_start":
					return;
				case "message_end":
					return;
				case "message_update": {
					const streamEvent = event.assistantMessageEvent;
					if (streamEvent.type === "thinking_start") {
						thinkingIndex = undefined;
					} else if (streamEvent.type === "thinking_delta") {
						appendDelta("thinking: ", streamEvent.delta, "thinking");
					} else if (streamEvent.type === "thinking_end" && thinkingIndex !== undefined) {
						output[thinkingIndex] = dimGray("thinking: (collapsed)");
						thinkingIndex = undefined;
					} else if (streamEvent.type === "text_start") {
						answer = "";
					} else if (streamEvent.type === "text_delta") {
						appendDelta("assistant: ", streamEvent.delta, "answer");
					}
					return;
				}
				case "tool_execution_start":
					updateLifecycle(`tool:${event.toolCallId}`, `${event.toolName}: searching`, `✓ ${event.toolName}: done`);
					return;
				case "tool_execution_update":
					updateLifecycle(
						`tool:${event.toolCallId}`,
						`${event.toolName}: searching`,
						`${event.toolName}: searching`,
					);
					return;
				case "tool_execution_end":
					updateLifecycle(
						`tool:${event.toolCallId}`,
						`${event.toolName}: searching`,
						`${event.isError ? "✗" : "✓"} ${event.toolName}: ${event.isError ? "error" : "done"}`,
					);
					return;
			}
		},
		lines: () => (paused ? ["paused"] : []).concat(output, answer.length > 0 ? [answer] : []),
	};
}

const ANSI_RESET = "\u001b[0m";
const ANSI_DIM_GRAY = "\u001b[90m\u001b[2m";

function dimGray(text: string): string {
	return `${ANSI_DIM_GRAY}${text}${ANSI_RESET}`;
}

function stripAnsi(text: string): string {
	return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function createAgent(ctx: CommandContext, deps: TuiDeps) {
	const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	const resolvedModel = (deps.modelResolver ?? resolveAgentModel)(config);
	const options: AutoRAGAgentOptions = {
		...buildAgentOptions(config),
		model: resolvedModel.model,
		...(resolvedModel.apiKey !== undefined ? { apiKey: resolvedModel.apiKey } : {}),
		...(resolvedModel.providerApiKeys !== undefined ? { providerApiKeys: resolvedModel.providerApiKeys } : {}),
	};
	return deps.agentFactory ? deps.agentFactory(options) : new AutoRAGAgent(options);
}

function createRealTui(): TUI {
	return new TuiMainScreen(new ProcessTerminal());
}

function sessionForSelection(store: TuiSessionStore, input: string): TuiSessionRecord | undefined {
	const sessions = store.list();
	const index = Number.parseInt(input, 10);
	if (Number.isInteger(index) && index > 0) return sessions[index - 1];
	return store.get(input);
}

const TUI_TITLE = "AutoRAG librarian - Ctrl+C or Ctrl+D to exit";

export function renderTuiSessionView(session: TuiSessionRecord | undefined, input: string): string {
	const content = session === undefined ? `resume: session not found: ${input}` : renderRestoredTuiSession(session);
	return `${TUI_TITLE}\n\n${content}`;
}

function defaultSessionStore(ctx: CommandContext): TuiSessionStore {
	const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	const workspace = createFileTuiSessionStore(join(config.workspacePath, ".autorag", "tui-sessions.json"));
	const global = createFileTuiSessionStore(join(resolveAutoRAGHome(), "tui-sessions.json"));
	return createMergedTuiSessionStore(workspace, global);
}

function runRealTui(ctx: CommandContext, agent: TuiAgent, store: TuiSessionStore): Promise<number> {
	const tui = createRealTui();
	let transcriptHistory = TUI_TITLE;
	const transcript = new Text(transcriptHistory);
	const presenter = createTuiPresenter();
	let finalAnswer = "";
	const renderTranscript = (): void => {
		const trace = presenter.lines();
		const sections = [
			transcriptHistory,
			trace.length > 0 ? `--- live trace ---\n${trace.join("\n")}` : "",
			finalAnswer,
		];
		transcript.setText(sections.filter((section) => section.length > 0).join("\n\n"));
	};
	const status = new Text(presenter.status());
	const renderStatus = (): void => {
		status.setText(presenter.status());
	};
	const editor = new Editor(tui, {
		borderColor: (value) => value,
		selectList: {
			selectedPrefix: (value) => value,
			selectedText: (value) => value,
			description: (value) => value,
			scrollInfo: (value) => value,
			noMatch: (value) => value,
		},
	});
	editor.setAutocompleteProvider(new CombinedAutocompleteProvider(createTuiSlashCommands(store), ctx.cwd));
	let settled = false;
	let queryRunning = false;
	let interrupted = false;
	let resumeSelection = false;
	let completedResponse: SearchDocumentsResponse | undefined;
	let completedAnswer = "";
	return new Promise((resolve) => {
		const finish = () => {
			if (settled) return;
			settled = true;
			unsubscribe?.();
			tui.stop();
			resolve(0);
		};
		const unsubscribe = agent.subscribe?.((event) => {
			presenter.handle(event);
			renderStatus();
			renderTranscript();
			tui.requestRender();
		});
		const handleInput = (data: string): void => {
			if (data === "\u0003") {
				if (queryRunning) {
					interrupted = true;
					presenter.interrupt();
					agent.abort?.();
					transcriptHistory = `${transcriptHistory}\ninterrupted`;
				} else {
					presenter.interrupt();
					transcriptHistory = `${transcriptHistory}\npaused`;
				}
				renderStatus();
				renderTranscript();
				tui.requestRender();
				return;
			}
			if (data === "\u0004") finish();
		};
		editor.onSubmit = (raw) => {
			const query = raw.trim();
			editor.setText("");
			if (query.length === 0) return;
			if (resumeSelection && !query.startsWith("/")) {
				resumeSelection = false;
				const session = sessionForSelection(store, query);
				presenter.reset();
				finalAnswer = "";
				transcriptHistory = renderTuiSessionView(session, query);
				renderTranscript();
				tui.requestRender();
				return;
			}
			resumeSelection = false;
			const command = parseSlashCommand(query);
			if (command !== undefined) {
				if (command.kind === "quit") {
					finish();
					return;
				}
				if (command.kind === "incomplete" || command.kind === "unknown") {
					transcriptHistory = `${transcriptHistory}\n\n${command.kind === "unknown" ? `unknown command: /${command.name}` : renderSlashHelp()}`;
					renderTranscript();
					tui.requestRender();
					return;
				}
				if (command.sessionId === undefined) {
					presenter.reset();
					finalAnswer = "";
					resumeSelection = store.list().length > 0;
					transcriptHistory = `${TUI_TITLE}\n\n${renderTuiSessionList(store.list())}`;
				} else {
					const session = sessionForSelection(store, command.sessionId);
					presenter.reset();
					finalAnswer = "";
					transcriptHistory = renderTuiSessionView(session, command.sessionId);
				}
				renderTranscript();
				tui.requestRender();
				return;
			}
			queryRunning = true;
			interrupted = false;
			presenter.beginRun();
			editor.disableSubmit = true;
			presenter.setWorking(true);
			renderStatus();
			finalAnswer = "";
			transcriptHistory = `${transcriptHistory}\n\n> ${query}\nsearch: preparing retrieval`;
			renderTranscript();
			tui.requestRender();
			void agent
				.searchDocuments(query)
				.then((response) => {
					const answer = renderSearch(response, {
						json: false,
						debug: ctx.debug,
					});
					completedResponse = response;
					completedAnswer = answer;
					finalAnswer = answer;
					renderTranscript();
				})
				.catch((error) => {
					if (interrupted) return;
					transcriptHistory = `${transcriptHistory}\n\n${renderError(error, {
						json: false,
						debug: ctx.debug,
					})}`;
					renderTranscript();
				})
				.finally(() => {
					queryRunning = false;
					presenter.setWorking(false);
					renderStatus();
					if (completedResponse !== undefined) {
						store.save({
							id: completedResponse.sessionId,
							query,
							answer: completedAnswer,
							trace: presenter.lines().join("\n"),
							updatedAt: Date.now(),
						});
						completedResponse = undefined;
					}
					renderTranscript();
					editor.disableSubmit = false;
					tui.requestRender();
				});
		};
		tui.addChild(transcript);
		tui.addChild(status);
		tui.addChild(editor);
		tui.setFocus(editor);
		tui.addInputListener((data) => {
			handleInput(data);
			return { consume: data === "\u0003" || data === "\u0004" };
		});
		tui.start();
	});
}

export async function runTui(ctx: CommandContext, deps: TuiDeps = {}): Promise<number> {
	try {
		const agent = createAgent(ctx, deps);
		const store = deps.sessionStore ?? defaultSessionStore(ctx);
		if (!deps.tuiFactory) return runRealTui(ctx, agent, store);
		const tui = deps.tuiFactory(ctx);
		const presenter = createTuiPresenter();
		let queryRunning = false;
		let interrupted = false;
		let resumeSelection = false;
		let completedResponse: SearchDocumentsResponse | undefined;
		let completedAnswer = "";
		const unsubscribe = agent.subscribe?.((event) => {
			presenter.handle(event);
			tui.rendered.push(presenter.lines().join("\n"));
			tui.onEvent?.(event);
		});
		tui.onSubmit = async (raw) => {
			const query = raw.trim();
			if (query.length === 0) return;
			if (resumeSelection && !query.startsWith("/")) {
				resumeSelection = false;
				const session = sessionForSelection(store, query);
				presenter.reset();
				tui.rendered.length = 0;
				tui.rendered.push(
					session === undefined ? `resume: session not found: ${query}` : renderRestoredTuiSession(session),
				);
				return;
			}
			resumeSelection = false;
			const command = parseSlashCommand(query);
			if (command !== undefined) {
				if (command.kind === "quit") {
					tui.onExit?.();
					return;
				}
				if (command.kind === "incomplete") {
					tui.rendered.push(renderSlashHelp());
				} else if (command.kind === "unknown") {
					tui.rendered.push(`unknown command: /${command.name}`);
				} else if (command.sessionId === undefined) {
					presenter.reset();
					tui.rendered.length = 0;
					resumeSelection = store.list().length > 0;
					tui.rendered.push(renderTuiSessionList(store.list()));
				} else {
					const session = sessionForSelection(store, command.sessionId);
					presenter.reset();
					tui.rendered.length = 0;
					tui.rendered.push(
						session === undefined
							? `resume: session not found: ${command.sessionId}`
							: renderRestoredTuiSession(session),
					);
				}
				return;
			}
			queryRunning = true;
			interrupted = false;
			presenter.beginRun();
			try {
				const response: SearchDocumentsResponse = await agent.searchDocuments(query);
				if (!interrupted) {
					const answer = renderSearch(response, { json: false, debug: ctx.debug });
					tui.rendered.push(answer);
					completedResponse = response;
					completedAnswer = answer;
				}
			} catch (error) {
				if (!interrupted) {
					tui.rendered.push(renderError(error, { json: false, debug: ctx.debug }));
				}
			} finally {
				queryRunning = false;
				presenter.setWorking(false);
				if (completedResponse !== undefined) {
					store.save({
						id: completedResponse.sessionId,
						query,
						answer: completedAnswer,
						trace: presenter.lines().join("\n"),
						updatedAt: Date.now(),
					});
					completedResponse = undefined;
				}
			}
		};
		tui.onInput = (data) => {
			if (data === "\u0003") {
				if (queryRunning) {
					interrupted = true;
					presenter.interrupt();
					agent.abort?.();
					tui.rendered.push("interrupted");
				} else {
					presenter.interrupt();
					tui.rendered.push("paused");
				}
				return;
			}
			if (data === "\u0004") tui.onExit?.();
		};
		tui.onExit = () => {
			unsubscribe?.();
			tui.stopped = true;
		};
		tui.started = true;
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
