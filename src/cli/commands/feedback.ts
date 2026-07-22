import { RetrievalMemory } from "../../memory/memory.ts";
import { resolveConfig } from "../config.ts";
import { renderError, renderFeedback } from "../output.ts";
import type { CommandContext } from "./types.ts";

interface NumberedFeedbackItem {
	readonly number: number;
	readonly useful: boolean;
}

function parseCsvInts(value: string | boolean | undefined): number[] {
	if (typeof value !== "string" || value.trim() === "") return [];
	return value
		.split(",")
		.map((part) => part.trim())
		.filter((part) => part !== "")
		.map((part) => Number(part))
		.filter((n) => Number.isFinite(n))
		.map((n) => Math.trunc(n));
}

/**
 * Run the `autorag feedback` command. Records per-result useful/not-useful
 * signals against an existing curated-results session. Returns 0 when at least
 * one feedback signal was applied, 2 for usage errors or an unknown session,
 * 1 for runtime/IO errors.
 */
export async function runFeedback(ctx: CommandContext): Promise<number> {
	const sessionId = ctx.positionals[0];
	if (sessionId === undefined || sessionId.trim() === "") {
		ctx.stderr(
			renderError(new Error("Usage: autorag feedback <sessionId> [--useful 1,2] [--not-useful 3]"), {
				json: ctx.json,
			}),
		);
		return 2;
	}

	const usefulNumbers = parseCsvInts(ctx.flags.useful);
	const notUsefulNumbers = parseCsvInts(ctx.flags["not-useful"]);
	if (usefulNumbers.length === 0 && notUsefulNumbers.length === 0) {
		ctx.stderr(
			renderError(
				new Error(
					"Usage: autorag feedback <sessionId> [--useful 1,2] [--not-useful 3] — supply at least one of --useful or --not-useful",
				),
				{ json: ctx.json },
			),
		);
		return 2;
	}

	const feedback: NumberedFeedbackItem[] = [
		...usefulNumbers.map((number) => ({ number, useful: true })),
		...notUsefulNumbers.map((number) => ({ number, useful: false })),
	];

	let memoryPath: string;
	try {
		memoryPath = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd }).memoryPath;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 2;
	}

	let mem: RetrievalMemory;
	try {
		mem = new RetrievalMemory({ storagePath: memoryPath });
		mem.load();
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 1;
	}

	let applied: boolean;
	try {
		applied = mem.recordNumberedFeedback({ sessionId, query: "", feedback });
		if (applied) mem.save();
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 1;
	}

	ctx.stdout(renderFeedback({ applied, sessionId }, { json: ctx.json }));
	return applied ? 0 : 2;
}
