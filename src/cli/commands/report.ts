import { readFileSync } from "node:fs";
import { Value } from "typebox/value";
import { type AutoRAGResultsDetails, emitResultsSchema } from "../../agent/emit-results-tool.ts";
import { createAutoRAGLite } from "../../core.ts";
import { ConfigError } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

function readReportInput(input: string | boolean | undefined): string {
	if (typeof input === "string" && input.length > 0) return readFileSync(input, "utf8");
	return readFileSync(0, "utf8");
}

function validateReport(value: unknown): AutoRAGResultsDetails {
	if (!Value.Check(emitResultsSchema, value)) {
		throw new Error("Invalid report: expected the emit_autorag_results JSON schema");
	}
	const parsed = Value.Parse(emitResultsSchema, value);
	const resultNumbers = parsed.results.map((result) => result.number);
	const mappingNumbers = parsed.mapping.map((entry) => entry.number);
	const allNumbers = [...resultNumbers, ...mappingNumbers];
	const uniqueResultNumbers = new Set(resultNumbers);
	const uniqueMappingNumbers = new Set(mappingNumbers);
	const oneToOne =
		resultNumbers.length === mappingNumbers.length &&
		uniqueResultNumbers.size === resultNumbers.length &&
		uniqueMappingNumbers.size === mappingNumbers.length &&
		resultNumbers.every((number) => uniqueMappingNumbers.has(number));
	if (!oneToOne || allNumbers.some((number) => number < 1)) {
		throw new Error("Invalid report: results and mapping numbers must be one-to-one positive numbers");
	}
	if (
		parsed.results.some(
			(result) => !Number.isFinite(result.confidence) || result.confidence < 0 || result.confidence > 1,
		)
	) {
		throw new Error("Invalid report: result confidence must be in [0, 1]");
	}
	if (
		parsed.mapping.some((entry) =>
			(entry.evidenceRefs ?? []).some(
				(reference) =>
					reference.confidence !== undefined &&
					(!Number.isFinite(reference.confidence) || reference.confidence < 0 || reference.confidence > 1),
			),
		)
	) {
		throw new Error("Invalid report: evidence confidence must be in [0, 1]");
	}
	return {
		answer: parsed.answer,
		results: parsed.results,
		mapping: parsed.mapping.map((entry) => ({
			...entry,
			evidenceRefs: entry.evidenceRefs ?? [{ method: entry.method, source: entry.source, content: entry.content }],
		})),
		warnings: parsed.warnings ?? [],
	};
}

function renderReport(details: AutoRAGResultsDetails, sessionId: string, query: string, json: boolean): string {
	const envelope = { ok: true, sessionId, query, answer: details.answer, resultCount: details.results.length };
	if (json) return JSON.stringify(envelope, null, 2);
	return `report: ok\n  sessionId: ${sessionId}\n  results: ${details.results.length}`;
}

function renderReportError(error: unknown, json: boolean, debug: boolean): string {
	if (!json) return renderError(error, { json, debug });
	const message = error instanceof Error ? error.message : String(error);
	return JSON.stringify({ ok: false, error: message });
}

/** Persist a structured report submitted by an external curator. */
export async function runReport(ctx: CommandContext): Promise<number> {
	const query = ctx.positionals.join(" ").trim();
	if (query.length === 0) {
		ctx.stderr(renderError(new Error("Usage: autorag lite report <query> [--input FILE]"), { json: ctx.json }));
		return 2;
	}

	let details: AutoRAGResultsDetails;
	try {
		const text = readReportInput(ctx.flags.input);
		const parsed: unknown = JSON.parse(text);
		details = validateReport(parsed);
	} catch (error) {
		ctx.stderr(renderReportError(error, ctx.json, ctx.debug));
		return 2;
	}

	try {
		const lite = createAutoRAGLite({ flags: ctx.flags, cwd: ctx.cwd });
		const response = lite.recordReport(query, details);
		ctx.stdout(renderReport(details, response.sessionId, query, ctx.json));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return error instanceof ConfigError ? 2 : 1;
	}
}
