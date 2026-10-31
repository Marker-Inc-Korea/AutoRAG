import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model, SystemMessage } from "@earendil-works/pi-ai";
import { calculateContextTokens, estimateTokens } from "@earendil-works/pi-coding-agent";
import { contextOverflowCeiling, recordRequestEstimate } from "../cloud/context-overflow.ts";
import { estimateTextTokens, tokensBeyondCharGuess } from "./token-count.ts";

const CONTEXT_HEADROOM = 1.2;
/**
 * Most output tokens ever held back from the input budget, and the most a
 * session request may ask for. pi clamps each request's `max_tokens` to the room
 * its own estimate says is left in the window, but that estimate omits the system
 * prompt and tool declarations, so a catalog `maxTokens` that approaches the
 * window (OpenRouter and Fireworks list many such models) overshoots it. Holding
 * the model to this cap keeps the request the guard budgeted for.
 */
export const MAX_OUTPUT_TOKENS = 32_000;
const GUARD_MARKER = "[AutoRAG context guard: omitted context to stay within the model context budget.]";
const RETRIEVAL_TOOL_PREFIX = "search_datasource_";
const RETRIEVAL_TOOL_NAMES: Record<string, true> = {
	jikji_find: true,
	search_all_documents: true,
	semantic_search_local_docs: true,
	search_single_datasource_documents: true,
	everything_search: true,
	fsearch_search: true,
};

type ContextTokenGuardModel = Model<any> | (() => Model<any> | undefined) | undefined;

/** One retrieved chunk as it appears in the text, with its position in the list it came from. */
interface CandidateBlock {
	readonly text: string;
	/** 0 for the best chunk of its own ranked list, 1 for the next. */
	readonly rank: number;
}

interface CandidateSection {
	readonly prefix: string;
	readonly suffix: string;
	readonly blocks: readonly CandidateBlock[];
}

/**
 * Compose an optional context transform with the aggregate token guard. The
 * guard runs on the final transcript immediately before each provider request,
 * so it sees the system prompt, tool declarations, user query, memory context,
 * and every prior tool result.
 */
export function createContextTokenGuardedTransform(
	model: ContextTokenGuardModel,
	base?: (messages: AgentMessage[]) => Promise<AgentMessage[]>,
	fixedOverheadTokens: () => number = () => 0,
): (messages: AgentMessage[]) => Promise<AgentMessage[]> {
	return async (messages) => {
		const transformed = base === undefined ? messages : await base(messages);
		const currentModel = typeof model === "function" ? model() : model;
		return guardAgentContextMessages(currentModel, transformed, transformed === messages, fixedOverheadTokens());
	};
}

/** The model with its output limit held to {@link MAX_OUTPUT_TOKENS}; the same model when already within it, or when absent. */
export function withBoundedOutput<T extends Model<any> | undefined>(model: T): T {
	if (model === undefined || (Number.isFinite(model.maxTokens) && model.maxTokens <= MAX_OUTPUT_TOKENS)) return model;
	return { ...model, maxTokens: MAX_OUTPUT_TOKENS };
}

/**
 * Keep the request inside the model budget by dropping retrieval chunks only: the
 * lowest-ranked ones go first, across every ranked list in the transcript (every
 * list's best chunk stays before any list's second), each with an explicit marker.
 * The user's query, earlier turns, and the system prompt are never edited. When no
 * chunk is left to drop, only the newest tool result is shortened; anything still
 * too large is left for the provider to reject and pi's compaction to recover.
 */
export function guardAgentContextMessages(
	model: Model<any> | undefined,
	messages: AgentMessage[],
	useReportedUsage = true,
	fixedOverheadTokens = 0,
): AgentMessage[] {
	const guarded = guardWithinBudget(model, messages, useReportedUsage, fixedOverheadTokens);
	if (model !== undefined && guarded.length > 0) {
		recordRequestEstimate(model, estimateContextMessageTokens(guarded, false) + fixedOverheadTokens);
	}
	return guarded;
}

function guardWithinBudget(
	model: Model<any> | undefined,
	messages: AgentMessage[],
	useReportedUsage = true,
	fixedOverheadTokens = 0,
): AgentMessage[] {
	if (
		model === undefined ||
		!Number.isFinite(model.contextWindow) ||
		model.contextWindow <= 0 ||
		messages.length === 0
	) {
		return messages;
	}

	const usableContext = Math.floor(model.contextWindow / CONTEXT_HEADROOM);
	const outputReserve = Math.min(
		Math.floor(usableContext / 4),
		MAX_OUTPUT_TOKENS,
		Number.isFinite(model.maxTokens) ? Math.max(0, model.maxTokens) : 0,
	);
	// After an overflow the model's own estimate proved too low for this transcript, so the
	// request is held to a fraction of the size that failed.
	const overflowCeiling = contextOverflowCeiling(model);
	const inputBudget = Math.min(
		Math.max(0, usableContext - outputReserve),
		overflowCeiling ?? Number.POSITIVE_INFINITY,
	);
	// The context event's messages exclude the system prompt and tool declarations, and
	// provider-reported usage counts them while a plain estimate does not. Every
	// post-edit estimate is a plain one, so the overhead comes off the budget.
	const budget = Math.max(0, inputBudget - fixedOverheadTokens);
	const reported = useReportedUsage && findLastAssistantUsage(messages) !== undefined;
	if (estimateContextMessageTokens(messages, useReportedUsage) <= (reported ? inputBudget : budget)) {
		return messages;
	}

	const candidates: CandidateSection[] = [];
	const candidateIndexes: number[] = [];
	const base = messages.slice();
	for (const [index, message] of base.entries()) {
		const text = textContent(message);
		if (text === undefined) continue;
		const section = parseCandidateSection(message, text);
		if (section === undefined) continue;
		candidates.push(section);
		candidateIndexes.push(index);
		base[index] = replaceTextContent(message, `${section.prefix}${section.suffix}`);
	}

	let guarded = messages;
	if (candidates.length > 0) {
		const entries = candidates.flatMap((section, sectionIndex) =>
			section.blocks.map((block, blockIndex) => ({ block, sectionIndex, blockIndex })),
		);
		const omissionMarker = (count: number): string => `\n\n${GUARD_MARKER} Retrieval candidates omitted: ${count}.\n`;
		// Every message that loses a chunk carries its own marker, so each one is reserved.
		const markerTokens = candidates.length * Math.ceil(omissionMarker(entries.length).length / 4);
		let remaining = Math.max(0, budget - estimateContextMessageTokens(base, false) - markerTokens);
		// Best chunk of every list first, then the second of every list; the older result wins a tie.
		entries.sort(
			(a, b) => a.block.rank - b.block.rank || a.sectionIndex - b.sectionIndex || a.blockIndex - b.blockIndex,
		);
		const kept = new Set<CandidateBlock>();
		for (const { block } of entries) {
			const tokens = Math.max(1, estimateTextTokens(block.text));
			if (tokens > remaining) break;
			remaining -= tokens;
			kept.add(block);
		}

		if (kept.size < entries.length) {
			const sectionByMessage = new Map(candidateIndexes.map((messageIndex, i) => [messageIndex, candidates[i]]));
			guarded = base.map((message, index) => {
				const section = sectionByMessage.get(index);
				if (section === undefined) return message;
				const keptBlocks = section.blocks.filter((block) => kept.has(block));
				const omitted = section.blocks.length - keptBlocks.length;
				if (omitted === 0) return messages[index] ?? message;
				const marker = omissionMarker(omitted);
				return replaceTextContent(
					message,
					`${section.prefix}${keptBlocks.map((block) => block.text).join("")}${marker}${section.suffix}`,
				);
			});
		}
	}
	if (guarded !== messages && estimateContextMessageTokens(guarded, false) <= budget) return guarded;
	return fitNewestToolResult(guarded, budget);
}

/**
 * Estimate the current transcript: pi-coding-agent's chars/4 counter as the
 * floor, raised by a real BPE count for text that chars/4 undercounts (see
 * `token-count.ts`). `useReportedUsage` adds the last assistant turn's
 * provider-reported usage for the settled prefix; pass false when the messages
 * were just edited.
 */
export function estimateContextMessageTokens(messages: readonly AgentMessage[], useReportedUsage = false): number {
	if (useReportedUsage) {
		const usage = findLastAssistantUsage(messages);
		if (usage !== undefined) {
			return (
				calculateContextTokens(usage.usage) +
				messages.slice(usage.index + 1).reduce((total, message) => total + estimateMessageTokens(message), 0)
			);
		}
	}
	return messages.reduce((total, message) => total + estimateMessageTokens(message), 0);
}

/**
 * Last resort once every droppable chunk is gone and the request still does not fit:
 * shorten the newest tool result, the one that just arrived, so pi's compact-and-retry
 * has something smaller to send. Nothing else is edited: the query, earlier turns, and
 * the system prompt stay whole, and when they alone exceed the budget the request is
 * left as it is for the provider to reject and pi's compaction to summarise.
 */
function fitNewestToolResult(messages: AgentMessage[], inputBudget: number): AgentMessage[] {
	const lastIndex = messages.length - 1;
	const last = messages[lastIndex];
	if (last?.role !== "toolResult") return messages;
	const others = messages.reduce(
		(total, message, index) => (index === lastIndex ? total : total + estimateMessageTokens(message)),
		0,
	);
	if (others >= inputBudget) return messages;
	const fitted = fitMessageToTokens(last, inputBudget - others);
	return fitted === last ? messages : [...messages.slice(0, lastIndex), fitted];
}

/** Re-measure at most this many times after sizing an excerpt before giving up and keeping only the marker. */
const MAX_FIT_ATTEMPTS = 6;

function fitMessageToTokens(message: AgentMessage, maxTokens: number): AgentMessage {
	const original = textContent(message);
	if (original === undefined) return message;
	const marker = `\n\n${GUARD_MARKER}\n`;
	// Counting every probe of a binary search is far too slow for megabytes of dense text, so the
	// excerpt is sized from the message's own tokens per character and then confirmed with a real
	// count, shrinking while it still overshoots (head and tail can be denser than the average).
	const overhead = estimateMessageTokens(replaceTextContent(message, ""));
	const density = Math.max(0, estimateMessageTokens(message) - overhead) / Math.max(1, original.length);
	let chars = density > 0 ? Math.floor((maxTokens - overhead) / density) : original.length;
	for (let attempt = 0; attempt < MAX_FIT_ATTEMPTS && chars > 0; attempt++) {
		const candidate = replaceTextContent(message, excerpt(original, chars, marker));
		const tokens = estimateMessageTokens(candidate);
		if (tokens <= maxTokens) return candidate;
		const scaled = Math.floor(((chars * Math.max(0, maxTokens - overhead)) / Math.max(1, tokens - overhead)) * 0.98);
		chars = Math.min(chars - 1, scaled);
	}
	return replaceTextContent(message, marker);
}

function excerpt(text: string, maxChars: number, marker: string): string {
	if (text.length <= maxChars) return text;
	const available = Math.max(0, maxChars - marker.length);
	const head = Math.ceil(available * 0.7);
	const tail = Math.max(0, available - head);
	return `${text.slice(0, head)}${marker}${tail > 0 ? text.slice(-tail) : ""}`;
}

function estimateMessageTokens(message: AgentMessage): number {
	if (message.role === "system") return estimateSystemMessageTokens(message);
	const estimate = estimateTokens(message) + tokensBeyondCharGuess(messageTextParts(message));
	return message.role === "toolResult" ? estimate + Math.ceil(message.toolName.length / 4) : estimate;
}

function estimateSystemMessageTokens(message: SystemMessage): number {
	const parts: string[] = [];
	let chars = 0;
	if (typeof message.content === "string") parts.push(message.content);
	else {
		for (const block of message.content) {
			if (block.type === "text") parts.push(block.text);
			else if (block.type === "image") chars += 4_800;
		}
	}
	parts.push(
		JSON.stringify(message.sections ?? {}),
		JSON.stringify(message.toolsAdded ?? []),
		JSON.stringify(message.toolsRemoved ?? []),
	);
	for (const part of parts) chars += part.length;
	return Math.ceil(chars / 4) + tokensBeyondCharGuess(parts);
}

/** The text pi's chars/4 estimate counts for a non-system message, one string per piece. */
function messageTextParts(message: AgentMessage): string[] {
	switch (message.role) {
		case "user":
		case "custom":
		case "toolResult":
			return typeof message.content === "string"
				? [message.content]
				: message.content.flatMap((block) => (block.type === "text" ? [block.text] : []));
		case "assistant":
			return message.content.flatMap((block) => {
				if (block.type === "text") return [block.text];
				if (block.type === "thinking") return [block.thinking];
				if (block.type === "toolCall") return [`${block.name}${JSON.stringify(block.arguments)}`];
				return [];
			});
		case "bashExecution":
			return [message.command, message.output];
		case "branchSummary":
		case "compactionSummary":
			return [message.summary];
		default:
			return [];
	}
}

function findLastAssistantUsage(
	messages: readonly AgentMessage[],
): { readonly index: number; readonly usage: AssistantMessage["usage"] } | undefined {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message?.role !== "assistant" || message.stopReason === "aborted" || message.stopReason === "error") continue;
		if (calculateContextTokens(message.usage) > 0) return { index, usage: message.usage };
	}
	return undefined;
}

function textContent(message: AgentMessage): string | undefined {
	if (message.role === "system") {
		if (typeof message.content === "string") return message.content;
		return message.content
			.filter((block): block is { type: "text"; text: string } => block.type === "text")
			.map((block) => block.text)
			.join("");
	}
	if (message.role === "user" || message.role === "toolResult") {
		if (typeof message.content === "string") return message.content;
		let text = "";
		for (const block of message.content) {
			if (block.type !== "text") return undefined;
			text += block.text;
		}
		return text;
	}
	if (message.role === "assistant") {
		let text = "";
		for (const block of message.content) if (block.type === "text") text += block.text;
		return text.length > 0 ? text : undefined;
	}
	return undefined;
}

function replaceTextContent(message: AgentMessage, text: string): AgentMessage {
	if (message.role === "system") return { ...message, content: text };
	if (message.role === "user") {
		if (typeof message.content === "string") return { ...message, content: text };
		return { ...message, content: [{ type: "text", text }] };
	}
	if (message.role === "toolResult") return { ...message, content: [{ type: "text", text }] };
	if (message.role === "assistant") {
		const content: typeof message.content = [];
		let replaced = false;
		for (const block of message.content) {
			if (block.type !== "text") {
				content.push(block);
				continue;
			}
			if (replaced) continue;
			replaced = true;
			content.push({ ...block, text });
		}
		return replaced ? { ...message, content } : message;
	}
	return message;
}

function parseCandidateSection(message: AgentMessage, text: string): CandidateSection | undefined {
	if (message.role === "user") {
		const markerMatch =
			/(?:Baseline retrieval (?:evidence \(already gathered for you\)|context):|initial candidates:)/u.exec(text);
		if (markerMatch === null) return undefined;
		const sectionStart = (markerMatch.index ?? 0) + markerMatch[0].length;
		const sectionEnd = findBaselineSectionEnd(text, sectionStart);
		if (!/^\[e?\d+\] /mu.test(text.slice(sectionStart, sectionEnd))) return undefined;
		return splitCandidates(text, sectionStart, sectionEnd, { lists: true });
	}
	if (message.role !== "toolResult") return undefined;
	const isRetrieval =
		RETRIEVAL_TOOL_NAMES[message.toolName] === true || message.toolName.startsWith(RETRIEVAL_TOOL_PREFIX);
	if (!isRetrieval) return undefined;
	if (message.toolName === "jikji_find") {
		const directiveStart = text.indexOf("\n\ndirective:");
		return splitCandidates(text, 0, directiveStart >= 0 ? directiveStart : text.length, {
			pattern: /^- [^\n]*(?:\n|$)/gmu,
		});
	}
	const diagnosticsStart = text.indexOf("\n\nDiagnostics:");
	return splitCandidates(text, 0, diagnosticsStart >= 0 ? diagnosticsStart : text.length);
}
function findBaselineSectionEnd(text: string, startIndex: number): number {
	const endMarkers = [
		"\n\nProduce the best",
		"\n\nTreat candidates as unverified evidence",
		"\n\nFormatting and content rules",
	];
	const positions = endMarkers.map((marker) => text.indexOf(marker, startIndex)).filter((index) => index >= 0);
	return positions.length > 0 ? Math.min(...positions) : text.length;
}

/**
 * A baseline that was not reranked lists each source (Jikji, MinSync, datasources,
 * web) under its own "... initial candidates:" heading, every list in its own
 * retrieval order. The heading ends the previous chunk's text; it belongs to the next list.
 */
const LIST_HEADING_TAIL = /\n[^\n]*initial candidates[^\n]*:\n$/u;

function splitCandidates(
	text: string,
	sectionStart: number,
	sectionEnd: number,
	options: { readonly pattern?: RegExp; readonly lists?: boolean } = {},
): CandidateSection | undefined {
	const region = text.slice(sectionStart, sectionEnd);
	const matches = [...region.matchAll(options.pattern ?? /^\[e?\d+\] [^\n]*(?:\n|$)/gmu)];
	const firstStart = matches[0]?.index;
	if (firstStart === undefined) return undefined;
	const blocks: CandidateBlock[] = [];
	let rank = 0;
	let heading = "";
	for (const [index, match] of matches.entries()) {
		const start = match.index ?? 0;
		const end = index + 1 < matches.length ? (matches[index + 1]?.index ?? region.length) : region.length;
		const raw = region.slice(start, end);
		const tail = options.lists === true ? LIST_HEADING_TAIL.exec(raw) : null;
		blocks.push({ text: heading + (tail === null ? raw : raw.slice(0, tail.index)), rank });
		// The next list starts over at rank 0 and keeps its heading with its best chunk.
		rank = tail === null ? rank + 1 : 0;
		heading = tail === null ? "" : tail[0];
	}
	return {
		prefix: text.slice(0, sectionStart) + region.slice(0, firstStart),
		suffix: text.slice(sectionEnd),
		blocks,
	};
}
