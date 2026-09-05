import type { EvidenceChunkRecord, MemorySchemaV4 } from "../../memory/memory.ts";
import { RetrievalMemory } from "../../memory/memory.ts";
import { resolveConfig } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

interface EvidenceView {
	readonly stableEvidenceId: string;
	readonly method: string;
	readonly source: string;
	readonly excerpt?: string;
	readonly content?: string;
	readonly retrievalResultId?: string;
	readonly chunkIndex?: number;
	readonly lineNumber?: number;
	readonly metadata?: Record<string, unknown>;
}

function parseResultNumber(value: string | boolean | undefined): number | undefined {
	if (typeof value !== "string" || value.trim() === "") return undefined;
	const number = Number(value);
	return Number.isInteger(number) && number > 0 ? number : undefined;
}

function evidenceFor(schema: MemorySchemaV4, sessionId: string, resultNumber?: number) {
	const results = schema.curatedResults
		.filter((result) => result.sessionId === sessionId)
		.filter((result) => resultNumber === undefined || result.number === resultNumber)
		.sort((a, b) => a.number - b.number)
		.map((result) => {
			const chunks: EvidenceView[] = result.evidenceIds
				.map((id) => schema.evidenceChunks.find((chunk) => chunk.stableEvidenceId === id))
				.filter((chunk): chunk is EvidenceChunkRecord => chunk !== undefined)
				.map((chunk) => ({
					stableEvidenceId: chunk.stableEvidenceId,
					method: chunk.method,
					source: chunk.source,
					...(chunk.excerpt !== undefined ? { excerpt: chunk.excerpt } : {}),
					...(chunk.content !== undefined ? { content: chunk.content } : {}),
					...(chunk.retrievalResultId !== undefined ? { retrievalResultId: chunk.retrievalResultId } : {}),
					...(chunk.chunkIndex !== undefined ? { chunkIndex: chunk.chunkIndex } : {}),
					...(chunk.lineNumber !== undefined ? { lineNumber: chunk.lineNumber } : {}),
					...(chunk.metadata !== undefined ? { metadata: chunk.metadata } : {}),
				}));
			return {
				number: result.number,
				title: result.title,
				summary: result.summary,
				chunks,
			};
		});
	return { sessionId, results };
}

function renderEvidence(view: ReturnType<typeof evidenceFor>, json: boolean): string {
	if (json) return JSON.stringify(view, null, 2);
	const lines = [`session: ${view.sessionId}`];
	for (const result of view.results) {
		lines.push(`[${result.number}] ${result.title}`);
		for (const chunk of result.chunks) {
			lines.push(`  source: ${chunk.source}`);
			lines.push(`  method: ${chunk.method}`);
			lines.push(`  evidenceId: ${chunk.stableEvidenceId}`);
			if (chunk.chunkIndex !== undefined) lines.push(`  chunkIndex: ${chunk.chunkIndex}`);
			if (chunk.lineNumber !== undefined) lines.push(`  lineNumber: ${chunk.lineNumber}`);
			if (chunk.excerpt !== undefined) lines.push(`  excerpt: ${chunk.excerpt}`);
			if (chunk.content !== undefined) lines.push(`  content: ${chunk.content}`);
		}
	}
	return lines.join("\n");
}

/** Show the persisted source/chunk evidence attached to a search session. */
export async function runEvidence(ctx: CommandContext): Promise<number> {
	const sessionId = ctx.positionals[0]?.trim();
	if (!sessionId) {
		ctx.stderr(renderError(new Error("Usage: autorag evidence <sessionId> [--result N]"), { json: ctx.json }));
		return 2;
	}
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const memory = new RetrievalMemory({ storagePath: config.memoryPath });
		memory.load();
		const view = evidenceFor(memory.getSchema(), sessionId, parseResultNumber(ctx.flags.result));
		if (view.results.length === 0) {
			ctx.stderr(renderError(new Error(`No evidence found for session ${sessionId}.`), { json: ctx.json }));
			return 2;
		}
		ctx.stdout(renderEvidence(view, ctx.json));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 1;
	}
}
