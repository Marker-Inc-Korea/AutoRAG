import { isAbsolute } from "node:path";
import { answerMarkers } from "./citations.ts";
import { type EvidenceLedger, isRegularFile } from "./evidence-ledger.ts";
import type {
	AutoRAGEmittedResult,
	AutoRAGEvidenceRef,
	AutoRAGMappingEntry,
	AutoRAGResultsDetails,
} from "./results.ts";
import type { SearchDocumentDiagnostic } from "./search-documents.ts";

/** Longest evidence excerpt kept on a derived result. */
const MAX_EXCERPT_CHARS = 400;
/** Longest claim sentence kept as a derived result's summary. */
const MAX_CLAIM_CHARS = 280;
/** How far back from a citation to look for the sentence it backs. */
const CLAIM_LOOKBACK_CHARS = 600;

const SENTENCE_END = /[.!?。！？]/u;
const LEADING_LIST_MARK = /^[\s\-*•>]+/u;
const EVIDENCE_MARKER_TEXT = /\[(?:e\d+(?:,\s*)?)+\]|\[file:[^\]]*\]/gu;
const LEADING_BLANK = /^[ \t]*/u;

export interface DerivedAnswer {
	readonly details: AutoRAGResultsDetails;
	readonly diagnostics: readonly SearchDocumentDiagnostic[];
}

interface CitedSource {
	readonly number: number;
	readonly refs: AutoRAGEvidenceRef[];
	readonly chunkKeys: Set<string>;
	readonly claim: string;
}

/** The sentence ending at `end`, stripped of list marks and other citation markers. */
function claimBefore(text: string, end: number): string {
	const window = text.slice(Math.max(0, end - CLAIM_LOOKBACK_CHARS), end);
	const lineStart = window.lastIndexOf("\n") + 1;
	const line = window.slice(lineStart).replace(EVIDENCE_MARKER_TEXT, "");
	let start = 0;
	for (let index = line.length - 2; index >= 0; index--) {
		if (SENTENCE_END.test(line.charAt(index)) && /\s/u.test(line.charAt(index + 1))) {
			start = index + 2;
			break;
		}
	}
	const claim = line.slice(start).replace(LEADING_LIST_MARK, "").replace(/\s+/gu, " ").trim();
	return claim.length > MAX_CLAIM_CHARS ? `${claim.slice(0, MAX_CLAIM_CHARS)}…` : claim;
}

/**
 * Turn the model's plain answer into the structured results the rest of the
 * harness consumes. The model cites with the evidence ids the retrieval tools
 * printed (`[e3]`, `[e3, e7]`) or `[file:<abs path>]` for a file it opened
 * itself. Each id is resolved against the run's evidence ledger, so a source,
 * method, and chunk always come from recorded evidence, never from model text.
 *
 * Results are one per cited source, numbered by first appearance, and every
 * marker in the returned answer is rewritten to its result number (`[1]`).
 * A marker that matches no evidence, and any numeric marker the model wrote on
 * its own, is dropped and reported: it can never point at the wrong result.
 */
export function deriveResultsFromAnswer(
	text: string,
	ledger: EvidenceLedger,
	options: { readonly allowLocalFiles: boolean },
): DerivedAnswer {
	const isFile = (path: string): boolean => options.allowLocalFiles && isAbsolute(path) && isRegularFile(path);
	const sources = new Map<string, CitedSource>();
	const dropped = new Set<string>();
	let answer = "";
	let cursor = 0;
	let lastEmitted: { readonly number: number; readonly end: number } | undefined;

	const cite = (reference: string, claimEnd: number): number | undefined => {
		const refs = ledger.lookup(reference, options);
		const [primary] = refs;
		if (primary === undefined) {
			dropped.add(reference);
			return undefined;
		}
		let cited = sources.get(primary.source);
		if (cited === undefined) {
			cited = { number: sources.size + 1, refs: [], chunkKeys: new Set(), claim: claimBefore(text, claimEnd) };
			sources.set(primary.source, cited);
		}
		for (const ref of refs) {
			const chunkKey = `${ref.source}\0${ref.content ?? ""}`;
			if (cited.chunkKeys.has(chunkKey)) continue;
			cited.chunkKeys.add(chunkKey);
			cited.refs.push(ref);
		}
		return cited.number;
	};

	for (const marker of answerMarkers(text, isFile)) {
		answer += text.slice(cursor, marker.start);
		cursor = marker.end;
		if (marker.kind === "number") {
			dropped.add(`[${marker.number}]`);
			continue;
		}
		const references = marker.kind === "evidence" ? marker.ids : [`file:${marker.path}`];
		const numbers: number[] = [];
		for (const reference of references) {
			const number = cite(marker.kind === "file" ? marker.path : reference, marker.start);
			if (number !== undefined && !numbers.includes(number)) numbers.push(number);
		}
		if (numbers.length === 0) continue;
		// A source cited twice in a row ("[e3][e7]" on one file, or two sentences
		// ending on the same source) must not read "[1][1]". Only a marker that
		// directly follows the previous one (whitespace aside) is collapsed.
		const adjacent = lastEmitted !== undefined && text.slice(lastEmitted.end, marker.start).trim() === "";
		const fresh = numbers.filter((number) => !(adjacent && lastEmitted?.number === number));
		lastEmitted = { number: numbers[numbers.length - 1] as number, end: marker.end };
		if (fresh.length === 0) continue;
		const leading = LEADING_BLANK.exec(text.slice(marker.start, marker.end))?.[0] ?? "";
		answer += leading + fresh.map((number) => `[${number}]`).join("");
	}
	answer += text.slice(cursor);

	const results: AutoRAGEmittedResult[] = [];
	const mapping: AutoRAGMappingEntry[] = [];
	for (const [source, cited] of sources) {
		const [primary] = cited.refs;
		if (primary === undefined) continue;
		results.push({
			number: cited.number,
			title:
				source
					.split("/")
					.filter((part) => part.length > 0)
					.pop() ?? source,
			summary: cited.claim,
			evidence: cited.refs.map((ref) => {
				const excerpt = (ref.content ?? "").replace(/\s+/gu, " ").trim().slice(0, MAX_EXCERPT_CHARS);
				return ref.lineNumber !== undefined ? { excerpt, lineNumber: ref.lineNumber } : { excerpt };
			}),
		});
		mapping.push({
			number: cited.number,
			source,
			method: primary.method,
			content: primary.content ?? "",
			evidenceRefs: cited.refs,
		});
	}

	const diagnostics: SearchDocumentDiagnostic[] =
		dropped.size === 0
			? []
			: [
					{
						code: "citation-without-result",
						severity: "warning",
						message: `Dropped ${dropped.size} answer citation(s) that matched no evidence retrieved in this run: ${[...dropped].join(", ")}. Every remaining citation resolves to a recorded source.`,
						source: "agent",
					},
				];
	return { details: { answer: answer.trim(), results, mapping, warnings: [] }, diagnostics };
}
