import type { RecordCut } from "../retrieval/record-cut.ts";
import { planReadBudget } from "../retrieval/record-cut.ts";
import type { RetrievalResult } from "../retrieval/types.ts";
import type { EvidenceLedger } from "./evidence-ledger.ts";

/** One retrieved chunk as a search tool shows it to the model. */
export interface EvidenceRow {
	/** Ledger id, cited as `[eN]`. */
	readonly id: string;
	/** Everything after the id on the first line: source, score, location. */
	readonly header: string;
	readonly content: string;
	/** A search earlier in this run already put this chunk's text in front of the model. */
	readonly repeat: boolean;
}

export interface RenderedEvidence {
	readonly text: string;
	/** Ids whose text is in `text`; the caller marks these as shown so a later search does not resend them. */
	readonly delivered: ReadonlySet<string>;
}

export const REPEAT_NOTE = "(already shown earlier in this run; text not repeated)";
export const NOT_READ_NOTE = "(not read: result budget used up; narrow the query to read it)";

const collapse = (content: string): string => content.replace(/\s+/gu, " ");

/**
 * Render search results for the model.
 *
 * Without a record cut every row is shown in full, exactly as before. With one,
 * a chunk an earlier search of this run already showed is named without its
 * text, and the rest share `recordCut.resultChars` max-min fairly: a long chunk
 * is cut and says how much, a short one keeps all of it, and when the budget
 * cannot give every chunk a useful share the lowest-ranked ones are listed as
 * not read rather than every chunk being clipped to a stub.
 */
export function renderEvidenceRows(rows: readonly EvidenceRow[], recordCut: RecordCut | undefined): RenderedEvidence {
	const delivered = new Set<string>();
	if (recordCut === undefined) {
		for (const row of rows) delivered.add(row.id);
		return {
			text: rows.map((row) => `[${row.id}] ${row.header}\n${collapse(row.content)}`).join("\n\n"),
			delivered,
		};
	}

	const fresh = rows.filter((row) => !row.repeat);
	const contents = fresh.map((row) => collapse(row.content));
	const framing = rows.reduce((sum, row) => sum + `[${row.id}] ${row.header}\n`.length + 2, 0);
	const plan = planReadBudget(
		contents.map((content) => content.length),
		recordCut.resultChars - framing,
		recordCut.minReadChars,
	);

	const bodyById = new Map<string, string>();
	for (const [index, row] of fresh.entries()) {
		const content = contents[index]!;
		const limit = plan[index];
		if (limit === undefined) {
			bodyById.set(row.id, NOT_READ_NOTE);
			continue;
		}
		delivered.add(row.id);
		bodyById.set(
			row.id,
			content.length <= limit
				? content
				: `${content.slice(0, limit)} …[+${content.length - limit} chars cut to fit the result budget]`,
		);
	}
	return {
		text: rows
			.map((row) => `[${row.id}] ${row.header}\n${row.repeat ? REPEAT_NOTE : bodyById.get(row.id)}`)
			.join("\n\n"),
		delivered,
	};
}

/** Options a search tool takes for the record cut; absent means the legacy rendering. */
export interface RecordCutToolOptions {
	readonly recordCut?: RecordCut;
}

/**
 * Register `results` in the ledger and render them. With a record cut, a chunk an earlier search of this run
 * already showed is not resent, and only chunks whose text reached the model are marked as shown.
 */
export function registerAndRender(
	ledger: EvidenceLedger,
	tool: string,
	results: readonly RetrievalResult[],
	header: (result: RetrievalResult) => string,
	recordCut: RecordCut | undefined,
): { readonly evidenceIds: string[]; readonly text: string } {
	const evidenceIds = results.map((result) => ledger.registerResult(tool, result));
	const rendered = renderEvidenceRows(
		results.map((result, index) => ({
			id: evidenceIds[index]!,
			header: header(result),
			content: result.content,
			repeat: recordCut !== undefined && ledger.hasShown(evidenceIds[index]!),
		})),
		recordCut,
	);
	if (recordCut !== undefined) for (const id of rendered.delivered) ledger.markShown(id);
	return { evidenceIds, text: rendered.text };
}
