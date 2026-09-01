import { Editor, ProcessTerminal, Text, TuiMainScreen, type TUI } from "@earendil-works/pi-tui";
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
}

export interface TuiDeps {
	agentFactory?: (opts: AutoRAGAgentOptions) => Pick<AutoRAGAgent, "searchDocuments">;
	modelResolver?: (config: CliConfig) => ResolvedAgentModel;
	tuiFactory?: (ctx: CommandContext) => TuiDriver;
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

function runRealTui(ctx: CommandContext, agent: Pick<AutoRAGAgent, "searchDocuments">): Promise<number> {
	const tui = createRealTui();
	let transcriptText = "AutoRAG librarian - Ctrl+C or Ctrl+D to exit";
	const transcript = new Text(transcriptText);
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
			tui.stop();
			resolve(0);
		};
		editor.onSubmit = (raw) => {
			const query = raw.trim();
			editor.setText("");
			if (query.length === 0) return;
			editor.disableSubmit = true;
			transcriptText = `${transcriptText}\n\n> ${query}\nSearching...`;
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
		tui.onSubmit = async (raw) => {
			const query = raw.trim();
			if (query.length === 0) return;
			const response: SearchDocumentsResponse = await agent.searchDocuments(query);
			tui.rendered.push(renderSearch(response, { json: false, debug: ctx.debug }));
		};
		tui.onExit = () => {
			tui.stopped = true;
		};
		tui.started = true;
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
