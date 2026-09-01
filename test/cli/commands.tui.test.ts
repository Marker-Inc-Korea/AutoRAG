import { describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { runTui, type TuiDeps, type TuiDriver } from "../../src/cli/commands/tui.ts";

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
	it("searches submitted questions and exits on the driver exit signal", async () => {
		const { ctx } = context();
		const tui = driver(["question"]);
		const queries: string[] = [];
		const deps: TuiDeps = {
			agentFactory: () => ({
				searchDocuments: async (query) => {
					queries.push(query);
					return response;
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
});
