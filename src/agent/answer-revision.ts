/**
 * Corrections to an answer the user has already read.
 *
 * The fast phase's answer reaches the user first. The verification phase may find that some of it was wrong, and a
 * correction is clearest when it says which sentence it replaces. The harness therefore splits the fast answer into
 * numbered claims (`c1`, `c2`, ...) the model can point at, and applies the model's revisions to those claims itself:
 * the text a revision strikes out is the text the harness kept, never the model's retelling of it.
 */

/** One sentence, bullet, table, or code block of an answer, with the whitespace that followed it. */
export interface Claim {
	readonly id: string;
	readonly text: string;
	/** The exact whitespace between this claim and the next; empty for the last claim. */
	readonly sep: string;
}

interface Span {
	readonly start: number;
	readonly end: number;
}

const FENCE = /^\s*(?:```|~~~)/u;
const TABLE_ROW = /^\s*\|/u;
const LIST_ITEM = /^\s*(?:[-*+•]|\d+[.)])\s/u;
/** A line that is a claim of its own: a list item, a heading, or a quote. */
const BLOCK_ITEM = /^\s*(?:[-*+•]|\d+[.)]|#{1,6}|>)\s/u;
const CJK_STOP = /[。！？]/u;
const ASCII_STOP = /[.!?]/u;
const CLOSER = /[)"'”’」』]/u;
/**
 * One model-form citation, with the spaces before it: `[e3]`, `[e3, e4]` or `[file:path]`. Citations that directly follow a
 * sentence's full stop back the sentence, not the next one, so {@link citationRunEnd} takes them one after another. The
 * runs are matched one citation at a time (never as a repeated group) so no input can make the matcher backtrack.
 */
const CITATION = /[ \t]*\[(?:e\d+(?:[ \t]*,[ \t]*e\d+)*|file:[^\]\n]*)\]/uy;

/** Where the run of citations that starts at `from` ends; `from` itself when none does. */
function citationRunEnd(text: string, from: number): number {
	let end = from;
	for (;;) {
		CITATION.lastIndex = end;
		const match = CITATION.exec(text);
		if (match === null) return end;
		end += match[0].length;
	}
}

function lineSpans(text: string): Span[] {
	const spans: Span[] = [];
	let start = 0;
	for (;;) {
		const newline = text.indexOf("\n", start);
		if (newline === -1) {
			spans.push({ start, end: text.length });
			return spans;
		}
		spans.push({ start, end: newline });
		start = newline + 1;
	}
}

/** The end of the sentence that begins at `from`: its full stop, closing quotes, and trailing citations. */
function sentenceEnd(text: string, from: number, limit: number): number {
	for (let index = from; index < limit; index++) {
		const char = text.charAt(index);
		let stop: number;
		if (CJK_STOP.test(char)) {
			stop = index + 1;
		} else if (ASCII_STOP.test(char)) {
			let after = index + 1;
			while (after < limit && ASCII_STOP.test(text.charAt(after))) after++;
			let closed = after;
			while (closed < limit && CLOSER.test(text.charAt(closed))) closed++;
			// "3.5" and "b.txt" continue; a stop ends the sentence only before whitespace or the end of the line.
			if (closed < limit && !/\s/u.test(text.charAt(closed))) {
				index = after - 1;
				continue;
			}
			stop = closed;
		} else {
			continue;
		}
		while (stop < limit && CLOSER.test(text.charAt(stop))) stop++;
		return Math.min(limit, citationRunEnd(text, stop));
	}
	return limit;
}

function sentenceSpans(text: string, start: number, end: number): Span[] {
	const spans: Span[] = [];
	let from = start;
	while (from < end) {
		const stop = sentenceEnd(text, from, end);
		spans.push({ start: from, end: stop });
		from = stop;
		while (from < end && /\s/u.test(text.charAt(from))) from++;
	}
	return spans;
}

function claimSpans(text: string): Span[] {
	const lines = lineSpans(text);
	const slice = (index: number): string => text.slice(lines[index]?.start ?? 0, lines[index]?.end ?? 0);
	const contentEnd = (index: number): number => (lines[index]?.start ?? 0) + slice(index).trimEnd().length;
	const spans: Span[] = [];
	let index = 0;
	while (index < lines.length) {
		const line = slice(index);
		if (line.trim() === "") {
			index++;
			continue;
		}
		const start = (lines[index]?.start ?? 0) + (line.length - line.trimStart().length);
		let last = index;
		if (FENCE.test(line)) {
			last = index + 1;
			while (last < lines.length && !FENCE.test(slice(last))) last++;
			last = Math.min(last, lines.length - 1);
		} else if (TABLE_ROW.test(line)) {
			while (last + 1 < lines.length && TABLE_ROW.test(slice(last + 1))) last++;
		} else if (BLOCK_ITEM.test(line)) {
			// An indented line that is not itself an item wraps the item above it.
			while (last + 1 < lines.length) {
				const next = slice(last + 1);
				if (next.trim() === "" || !/^\s/u.test(next) || BLOCK_ITEM.test(next)) break;
				last++;
			}
		} else {
			spans.push(...sentenceSpans(text, start, contentEnd(index)));
			index++;
			continue;
		}
		spans.push({ start, end: contentEnd(last) });
		index = last + 1;
	}
	return spans;
}

/**
 * Splits an answer into the claims a revision can point at: one per list item, heading, sentence, table, and code
 * block. The split is conservative: text it cannot split cleanly stays one claim. Concatenating every claim and its
 * separator gives back the trimmed answer exactly.
 */
export function splitClaims(answer: string): Claim[] {
	const text = answer.trim();
	const spans = claimSpans(text);
	return spans.map((span, index) => ({
		id: `c${index + 1}`,
		text: text.slice(span.start, span.end),
		sep: index + 1 < spans.length ? text.slice(span.end, spans[index + 1]?.start ?? span.end) : "",
	}));
}

export type RevisionOp = "confirm" | "retract" | "replace" | "add";

/** What the model asks for through `revise_answer`. */
export interface RevisionRequest {
	readonly op: RevisionOp;
	readonly claimId?: string;
	readonly text?: string;
	readonly reason: string;
}

/** An applied revision, in the form the user reads: citations are the `[n]` numbers the user was shown. */
export interface Revision {
	readonly op: RevisionOp;
	readonly claimId?: string;
	/** The claim as the user saw it; absent for an added finding. */
	readonly original?: string;
	/** The text that replaces the claim, or the added finding; absent for a confirmation or a retraction. */
	readonly replacement?: string;
	readonly reason: string;
}

/** The text of a {@link Revision} as the model wrote it, with its own `[eN]` ids instead of the user's `[n]`. */
export interface WrittenRevision {
	readonly original?: string;
	readonly replacement?: string;
}

export type RevisionOutcome =
	| { readonly ok: true; readonly revision: Revision; readonly written: WrittenRevision }
	| { readonly ok: false; readonly message: string };

/** What a revision session needs from the run's evidence: which citations resolve, and how the user sees them. */
export interface RevisionCitations {
	/** Citation markers in `text`: how many resolve to retrieved evidence, and the ones that do not. */
	inspect(text: string): { readonly resolved: number; readonly unresolved: readonly string[] };
	/** `text` with its model-form citations rewritten to the numbers the user is shown. */
	display(text: string): string;
}

type ClaimStatus = "open" | "confirmed" | "retracted" | "replaced";

interface ClaimState {
	readonly claim: Claim;
	status: ClaimStatus;
	replacement?: string;
}

/**
 * The revisions of one fast answer. A claim is retracted or replaced once; a confirmed claim may still be corrected
 * later. Every revision names a reason, and any text it adds must cite evidence the run retrieved.
 */
export class AnswerRevisionSession {
	private readonly states: ClaimState[];
	private readonly additions: string[] = [];
	private readonly applied: Revision[] = [];
	private readonly citations: RevisionCitations;

	constructor(fastAnswer: string, citations: RevisionCitations) {
		this.citations = citations;
		this.states = splitClaims(fastAnswer).map((claim) => ({ claim, status: "open" }));
	}

	/** The claims as the model wrote them, with its own `[eN]` citation ids, which stay valid for the whole run. */
	claims(): { readonly id: string; readonly text: string }[] {
		return this.states.map(({ claim }) => ({ id: claim.id, text: claim.text }));
	}

	/** Validates `request` exactly as {@link apply} would, without applying it or numbering anything for the user. */
	check(request: RevisionRequest): { readonly ok: true } | Rejection {
		const planned = this.plan(request);
		return planned.ok ? { ok: true } : planned;
	}

	apply(request: RevisionRequest): RevisionOutcome {
		const planned = this.plan(request);
		if (!planned.ok) return planned;
		const { revision, written } = planned.commit();
		this.applied.push(revision);
		return { ok: true, revision, written };
	}

	private plan(request: RevisionRequest): PlannedRevision | Rejection {
		const reason = request.reason.trim();
		if (reason === "") return reject("A revision needs a reason: one short line the user reads next to the change.");
		const text = request.text?.trim();
		const carriesText = request.op === "replace" || request.op === "add";
		if (carriesText && (text === undefined || text === "")) return reject(`${request.op} needs the new text.`);
		if (!carriesText && request.text !== undefined) return reject(`${request.op} takes no text.`);
		if (request.op === "add") {
			if (request.claimId !== undefined) return reject("add names no claim: it appends a new finding.");
			const problem = this.citationProblem(text ?? "", false);
			if (problem !== undefined) return reject(problem);
			return {
				ok: true,
				commit: () => {
					this.additions.push(text ?? "");
					return {
						revision: { op: "add", replacement: this.citations.display(text ?? ""), reason },
						written: { replacement: text ?? "" },
					};
				},
			};
		}
		if (request.claimId === undefined) return reject(`${request.op} needs the claimId of the claim it revises.`);
		const state = this.states.find(({ claim }) => claim.id === request.claimId);
		if (state === undefined) {
			return reject(
				`Unknown claim ${request.claimId}. The claims are: ${this.states.map(({ claim }) => claim.id).join(", ")}.`,
			);
		}
		if (state.status === "retracted" || state.status === "replaced") {
			return reject(`Claim ${request.claimId} was already ${state.status}; a claim is revised once.`);
		}
		if (request.op === "confirm" && state.status === "confirmed") {
			return reject(`Claim ${request.claimId} is already confirmed.`);
		}
		const claimId = request.claimId;
		if (request.op === "replace") {
			const problem = this.citationProblem(text ?? "", this.citations.inspect(state.claim.text).resolved > 0);
			if (problem !== undefined) return reject(problem);
			return {
				ok: true,
				commit: () => {
					state.status = "replaced";
					state.replacement = text ?? "";
					return {
						revision: {
							op: "replace",
							claimId,
							original: this.citations.display(state.claim.text),
							replacement: this.citations.display(text ?? ""),
							reason,
						},
						written: { original: state.claim.text, replacement: text ?? "" },
					};
				},
			};
		}
		const op = request.op;
		return {
			ok: true,
			commit: () => {
				state.status = op === "confirm" ? "confirmed" : "retracted";
				return {
					revision: { op, claimId, original: this.citations.display(state.claim.text), reason },
					written: { original: state.claim.text },
				};
			},
		};
	}

	/** The answer with every revision applied, still in model-form citations. */
	revisedText(): string {
		let kept = "";
		for (const { claim, status, replacement } of this.states) {
			if (status === "retracted") continue;
			kept += (status === "replaced" ? (replacement ?? "") : claim.text) + claim.sep;
		}
		let revised = kept.trim();
		for (const addition of this.additions) {
			const lastLine = revised.slice(revised.lastIndexOf("\n") + 1);
			const separator = revised === "" ? "" : LIST_ITEM.test(lastLine) && LIST_ITEM.test(addition) ? "\n" : "\n\n";
			revised += separator + addition;
		}
		return revised;
	}

	/** Claims the model has neither confirmed nor corrected. */
	unreviewed(): string[] {
		return this.states.filter(({ status }) => status === "open").map(({ claim }) => claim.id);
	}

	revisions(): readonly Revision[] {
		return this.applied;
	}

	/** True when the revised answer differs from the fast answer; a confirmation changes nothing. */
	changed(): boolean {
		return this.applied.some((revision) => revision.op !== "confirm");
	}

	private citationProblem(text: string, mustCite: boolean): string | undefined {
		const { resolved, unresolved } = this.citations.inspect(text);
		if (unresolved.length > 0) {
			return `These citations match no evidence retrieved in this conversation: ${unresolved.join(", ")}. Cite only the [eN] ids retrieval tools printed (or [file:<absolute path>] for a file you opened), never the [n] numbers the user sees.`;
		}
		if (mustCite && resolved === 0) {
			return "Cite at least one retrieved source in the text with its [eN] id, or retract the claim instead.";
		}
		return undefined;
	}
}

interface Rejection {
	readonly ok: false;
	readonly message: string;
}

/** A validated revision and the state change that applies it. */
interface PlannedRevision {
	readonly ok: true;
	/** Applies the revision and returns it, numbering any new source for the user only now. */
	readonly commit: () => { readonly revision: Revision; readonly written: WrittenRevision };
}

function reject(message: string): Rejection {
	return { ok: false, message };
}
