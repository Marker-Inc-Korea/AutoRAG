/**
 * Minimal markdown rendering for AI Search answers and evidence chunks.
 *
 * The reference (`design_handoff_ai_finder_v6`, DESIGN.md §5 AnswerCard) renders
 * answer text as paragraphs whose inline segments are plain text, `**bold**`,
 * and `[n]` citation chips. Evidence chunk bodies additionally render headings,
 * list items, quotes, and fenced code blocks (README §1, `proto:flags`).
 */

export type InlineSegment =
	| { readonly kind: "text"; readonly text: string }
	| { readonly kind: "bold"; readonly text: string }
	| { readonly kind: "code"; readonly text: string }
	| { readonly kind: "citation"; readonly number: number };

/** Split an inline string into text / bold / code / citation segments. */
export function parseInline(text: string): readonly InlineSegment[] {
	const tokens = [...text.matchAll(/\*\*([^*]+)\*\*|`([^`]+)`|\[([1-9]\d*)\]/gu)];
	if (tokens.length === 0) {
		return [{ kind: "text", text }];
	}
	const segments: InlineSegment[] = [];
	let cursor = 0;
	for (const match of tokens) {
		const index = match.index ?? 0;
		if (match[0] === "**" || (match[1] === undefined && match[2] === undefined && match[3] === undefined)) {
			continue;
		}
		if (index > cursor) {
			segments.push({ kind: "text", text: text.slice(cursor, index) });
		}
		if (match[1] !== undefined) {
			segments.push({ kind: "bold", text: match[1] });
		} else if (match[2] !== undefined) {
			segments.push({ kind: "code", text: match[2] });
		} else {
			segments.push({ kind: "citation", number: Number(match[3]) });
		}
		cursor = index + match[0].length;
	}
	if (cursor < text.length) {
		segments.push({ kind: "text", text: text.slice(cursor) });
	}
	return segments.length === 0 ? [{ kind: "text", text }] : segments;
}

/** Split a complete answer into display paragraphs on blank lines. */
export function splitAnswerParagraphs(answer: string): readonly string[] {
	return answer
		.split(/\n\s*\n/u)
		.map((chunk) => chunk.trim())
		.filter((chunk) => chunk.length > 0);
}

export type AnswerBlockKind = "heading" | "list" | "quote" | "code" | "paragraph";

export interface AnswerBlock {
	readonly kind: AnswerBlockKind;
	readonly text: string;
}

const HEADING = /^#{1,6}\s+/u;
const LIST = /^\s*(?:[-*•]|\d+[.)])\s+/u;
const QUOTE = /^>\s?/u;
const FENCE = /^```/u;

/**
 * Block-level markdown over an answer or an evidence excerpt. Contiguous plain
 * lines stay separate paragraphs: chunk excerpts use one line per sentence and
 * the reference renders each line as its own block.
 */
export function parseAnswerBlocks(markdown: string): readonly AnswerBlock[] {
	const blocks: AnswerBlock[] = [];
	const lines = markdown.split("\n");
	let inFence = false;
	let fence: string[] = [];
	for (const raw of lines) {
		if (FENCE.test(raw)) {
			if (inFence) {
				blocks.push({ kind: "code", text: fence.join("\n") });
				fence = [];
				inFence = false;
			} else {
				inFence = true;
			}
			continue;
		}
		if (inFence) {
			fence.push(raw);
			continue;
		}
		const line = raw.trim();
		if (line.length === 0) {
			continue;
		}
		if (HEADING.test(line)) {
			blocks.push({ kind: "heading", text: line.replace(HEADING, "") });
		} else if (QUOTE.test(line)) {
			blocks.push({ kind: "quote", text: line.replace(QUOTE, "") });
		} else if (LIST.test(line)) {
			blocks.push({ kind: "list", text: line.replace(LIST, "") });
		} else {
			blocks.push({ kind: "paragraph", text: line });
		}
	}
	if (inFence) {
		blocks.push({ kind: "code", text: fence.join("\n").replace(/\n+$/u, "") });
	}
	return blocks;
}
