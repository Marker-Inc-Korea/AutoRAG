import { Editor, ProcessTerminal, Text, TuiMainScreen, type TUI } from "@earendil-works/pi-tui";
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
import type { CommandContext } from "./types.ts";

export interface TuiDriver {
	submissions: string[];
	rendered: string[];
	started: boolean;
	stopped: boolean;
	onSubmit?: (text: string) => void | Promise<void>;
	onExit?: () => void;
	onEvent?: (event: AgentEvent) => void;
}

export interface TuiPresenter {
	handle(event: AgentEvent): void;
	lines(): readonly string[];
}

export interface TuiDeps {
	agentFactory?: (
		opts: AutoRAGAgentOptions,
	) => Pick<AutoRAGAgent, "searchDocuments"> & Partial<Pick<AutoRAGAgent, "subscribe">>;
	modelResolver?: (config: CliConfig) => ResolvedAgentModel;
	tuiFactory?: (ctx: CommandContext) => TuiDriver;
}

export function createTuiPresenter(): TuiPresenter {
	const output: string[] = [];
	let thinkingIndex: number | undefined;
	let answerIndex: number | undefined;

	const append = (line: string): void => {
		output.push(line);
	};
	const appendDelta = (prefix: string, delta: string, index: "thinking" | "answer"): void => {
		const currentIndex = index === "thinking" ? thinkingIndex : answerIndex;
		if (currentIndex === undefined) {
			append(`${prefix}${delta}`);
			if (index === "thinking") thinkingIndex = output.length - 1;
			else answerIndex = output.length - 1;
			return;
		}
		output[currentIndex] = `${output[currentIndex]}${delta}`;
	};

	return {
		handle(event) {
			switch (event.type) {
				case "agent_start":
					append("agent: started");
					return;
				case "agent_end":
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
		lines: () => [...output],
	};
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

function runRealTui(
	ctx: CommandContext,
	agent: Pick<AutoRAGAgent, "searchDocuments"> & Partial<Pick<AutoRAGAgent, "subscribe">>,
): Promise<number> {
	const tui = createRealTui();
	let transcriptText = "AutoRAG librarian - Ctrl+C or Ctrl+D to exit";
	const transcript = new Text(transcriptText);
	const presenter = createTuiPresenter();
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
	let settled = false;
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
			transcriptText = `${transcriptText.split("\n\n--- live trace ---")[0]}\n\n--- live trace ---\n${presenter
				.lines()
				.join("\n")}`;
			transcript.setText(transcriptText);
			tui.requestRender();
		});
		editor.onSubmit = (raw) => {
			const query = raw.trim();
			editor.setText("");
			if (query.length === 0) return;
			editor.disableSubmit = true;
			transcriptText = `${transcriptText}\n\n> ${query}\nsearch: preparing retrieval`;
			transcript.setText(transcriptText);
			tui.requestRender();
			void agent
				.searchDocuments(query)
				.then((response) => {
					transcriptText = `${transcriptText}\n\n${renderSearch(response, { json: false, debug: ctx.debug })}`;
					transcript.setText(transcriptText);
				})
				.catch((error) => {
					transcriptText = `${transcriptText}\n\n${renderError(error, { json: false, debug: ctx.debug })}`;
					transcript.setText(transcriptText);
				})
				.finally(() => {
					editor.disableSubmit = false;
					tui.requestRender();
				});
		};
		tui.addChild(transcript);
		tui.addChild(editor);
		tui.setFocus(editor);
		tui.addInputListener((data) => {
			if (data === "\u0003" || data === "\u0004") {
				finish();
				return { consume: true };
			}
			return undefined;
		});
		tui.start();
	});
}

export async function runTui(ctx: CommandContext, deps: TuiDeps = {}): Promise<number> {
	try {
		const agent = createAgent(ctx, deps);
		if (!deps.tuiFactory) return runRealTui(ctx, agent);
		const tui = deps.tuiFactory(ctx);
		const presenter = createTuiPresenter();
		const unsubscribe = agent.subscribe?.((event) => {
			presenter.handle(event);
			tui.rendered.push(presenter.lines().join("\n"));
			tui.onEvent?.(event);
		});
		tui.onSubmit = async (raw) => {
			const query = raw.trim();
			if (query.length === 0) return;
			try {
				const response: SearchDocumentsResponse = await agent.searchDocuments(query);
				tui.rendered.push(renderSearch(response, { json: false, debug: ctx.debug }));
			} catch (error) {
				tui.rendered.push(renderError(error, { json: false, debug: ctx.debug }));
			}
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
