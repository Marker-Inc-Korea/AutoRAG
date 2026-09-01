import { join } from "node:path";
import {
	CombinedAutocompleteProvider,
	Editor,
	ProcessTerminal,
	Text,
	TuiMainScreen,
	type TUI,
} from "@earendil-works/pi-tui";
import type { AgentEvent } from "@earendil-works/pi-agent-core";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../agent/agent.ts";
import type { SearchDocumentsResponse } from "../../agent/search-documents.ts";
import {
	buildAgentOptions,
	type CliConfig,
	type ResolvedAgentModel,
	resolveAgentModel,
	resolveConfig,
} from "../config.ts";
import { renderError, renderSearch } from "../output.ts";
import {
	createTuiSlashCommands,
	parseSlashCommand,
	renderSlashHelp,
} from "../tui-commands.ts";
import {
	createFileTuiSessionStore,
	renderRestoredTuiSession,
	renderTuiSessionList,
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
}

type TuiAgent = Pick<AutoRAGAgent, "searchDocuments"> &
	Partial<Pick<AutoRAGAgent, "subscribe" | "abort">>;

export interface TuiDeps {
	agentFactory?: (opts: AutoRAGAgentOptions) => TuiAgent;
	modelResolver?: (config: CliConfig) => ResolvedAgentModel;
	tuiFactory?: (ctx: CommandContext) => TuiDriver;
	sessionStore?: TuiSessionStore;
}

export function createTuiPresenter(): TuiPresenter {
	const output: string[] = [];
	let thinkingIndex: number | undefined;
	let answerIndex: number | undefined;
	let active = false;
	let paused = false;
	let suppressEvents = false;

	const append = (line: string): void => {
		output.push(line);
	};
	const appendDelta = (prefix: string, delta: string, index: "thinking" | "answer"): void => {
		const currentIndex = index === "thinking" ? thinkingIndex : answerIndex;
		if (currentIndex === undefined) {
			append(index === "thinking" ? dimGray(`${prefix}${delta}`) : `${prefix}${delta}`);
			if (index === "thinking") thinkingIndex = output.length - 1;
			else answerIndex = output.length - 1;
			return;
		}
		output[currentIndex] =
			index === "thinking"
				? dimGray(`${stripAnsi(output[currentIndex])}${delta}`)
				: `${output[currentIndex]}${delta}`;
	};

	return {
		working: () => active,
		setWorking: (value) => {
			active = value;
		},
		beginRun: () => {
			active = true;
			paused = false;
			suppressEvents = false;
		},
		interrupt: () => {
			active = false;
			paused = true;
			suppressEvents = true;
		},
		handle(event) {
			if (suppressEvents) return;
			switch (event.type) {
				case "agent_start":
					active = true;
					paused = false;
					append("agent: started");
					return;
				case "agent_end":
					active = false;
					append("agent: done");
					return;
				case "turn_start":
					append("turn: started");
					return;
				case "turn_end":
					append("turn: done");
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
						answerIndex = undefined;
					} else if (streamEvent.type === "text_delta") {
						appendDelta("assistant: ", streamEvent.delta, "answer");
					}
					return;
				}
				case "tool_execution_start":
					append(`${event.toolName}: started`);
					return;
				case "tool_execution_update":
					append(`${event.toolName}: searching`);
					return;
				case "tool_execution_end":
					append(`${event.toolName}: ${event.isError ? "error" : "done"}`);
					return;
			}
		},
		lines: () => (active ? ["working"] : paused ? ["paused"] : []).concat(output),
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

function defaultSessionStore(ctx: CommandContext): TuiSessionStore {
	const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	return createFileTuiSessionStore(join(config.workspacePath, ".autorag", "tui-sessions.json"));
}

function runRealTui(
	ctx: CommandContext,
	agent: TuiAgent,
	store: TuiSessionStore,
): Promise<number> {
	const tui = createRealTui();
	let transcriptHistory = "AutoRAG librarian - Ctrl+C or Ctrl+D to exit";
	const transcript = new Text(transcriptHistory);
	const presenter = createTuiPresenter();
	const renderTranscript = (): void => {
		const trace = presenter.lines();
		transcript.setText(
			trace.length > 0
				? `${transcriptHistory}\n\n--- live trace ---\n${trace.join("\n")}`
				: transcriptHistory,
		);
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
					transcriptHistory = `${transcriptHistory}\n\n${renderTuiSessionList(store.list())}`;
				} else {
					const session = store.get(command.sessionId);
					transcriptHistory = `${transcriptHistory}\n\n${session === undefined ? `resume: session not found: ${command.sessionId}` : renderRestoredTuiSession(session)
						}`;
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
					store.save({
						id: response.sessionId,
						query,
						answer,
						trace: presenter.lines().join("\n"),
						updatedAt: Date.now(),
					});
					transcriptHistory = `${transcriptHistory}\n\n${answer}`;
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
					renderTranscript();
					editor.disableSubmit = false;
					tui.requestRender();
				});
		};
		tui.addChild(transcript);
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
		const unsubscribe = agent.subscribe?.((event) => {
			presenter.handle(event);
			tui.rendered.push(presenter.lines().join("\n"));
			tui.onEvent?.(event);
		});
		tui.onSubmit = async (raw) => {
			const query = raw.trim();
			if (query.length === 0) return;
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
					tui.rendered.push(renderTuiSessionList(store.list()));
				} else {
					const session = store.get(command.sessionId);
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
					store.save({
						id: response.sessionId,
						query,
						answer,
						trace: presenter.lines().join("\n"),
						updatedAt: Date.now(),
					});
				}
			} catch (error) {
				if (!interrupted) {
					tui.rendered.push(renderError(error, { json: false, debug: ctx.debug }));
				}
			} finally {
				queryRunning = false;
				presenter.setWorking(false);
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
