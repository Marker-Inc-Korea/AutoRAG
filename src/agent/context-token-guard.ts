import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model, SystemMessage } from "@earendil-works/pi-ai";
import { calculateContextTokens, estimateTokens } from "@earendil-works/pi-coding-agent";

const CONTEXT_HEADROOM = 1.2;
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

interface CandidateSection {
	readonly prefix: string;
	readonly suffix: string;
	readonly blocks: readonly string[];
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
): (messages: AgentMessage[]) => Promise<AgentMessage[]> {
	return async (messages) => {
		const transformed = base === undefined ? messages : await base(messages);
		const currentModel = typeof model === "function" ? model() : model;
		return guardAgentContextMessages(currentModel, transformed, transformed === messages);
	};
}

/**
 * Keep complete retrieval candidates while the assembled request fits the
 * model budget. Removing a per-chunk cut is only safe because this drops whole
 * lower-ranked candidates (with an explicit marker) instead of hiding text.
 */
export function guardAgentContextMessages(
	model: Model<any> | undefined,
	messages: AgentMessage[],
	useReportedUsage = true,
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
	const outputReserve = Math.min(usableContext, Number.isFinite(model.maxTokens) ? Math.max(0, model.maxTokens) : 0);
	const inputBudget = Math.max(0, usableContext - outputReserve);
	const reportedContextOverBudget = estimateContextMessageTokens(messages, useReportedUsage) > inputBudget;
	if (!reportedContextOverBudget) return messages;

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
		const fixedTokens = estimateContextMessageTokens(base, false);
		const maximumOmitted = candidates.reduce((total, section) => total + section.blocks.length, 0);
		const omissionMarker = (count: number): string => `\n\n${GUARD_MARKER} Retrieval candidates omitted: ${count}.\n`;
		let remaining = Math.max(0, inputBudget - fixedTokens - Math.ceil(omissionMarker(maximumOmitted).length / 4));
		const kept: string[][] = candidates.map(() => []);
		let omitted = 0;
		let firstOmitted = -1;

		for (const [sectionIndex, section] of candidates.entries()) {
			for (const block of section.blocks) {
				const tokens = Math.max(1, Math.ceil(block.length / 4));
				if (tokens <= remaining) {
					kept[sectionIndex]?.push(block);
					remaining -= tokens;
					continue;
				}
				omitted += section.blocks.length - (kept[sectionIndex]?.length ?? 0);
				for (const later of candidates.slice(sectionIndex + 1)) omitted += later.blocks.length;
				firstOmitted = sectionIndex;
				break;
			}
			if (firstOmitted !== -1) break;
		}

		if (omitted > 0) {
			const candidateByMessage = new Map<number, number>();
			for (const [sectionIndex, messageIndex] of candidateIndexes.entries())
				candidateByMessage.set(messageIndex, sectionIndex);
			guarded = base.map((message, index) => {
				const sectionIndex = candidateByMessage.get(index);
				if (sectionIndex === undefined) return message;
				const section = candidates[sectionIndex];
				const marker = sectionIndex === firstOmitted ? omissionMarker(omitted) : "";
				return replaceTextContent(
					message,
					`${section.prefix}${kept[sectionIndex]?.join("") ?? ""}${marker}${section.suffix}`,
				);
			});
		}
	}
	if (estimateContextMessageTokens(guarded, false) <= inputBudget && guarded !== messages) return guarded;
	return trimFixedContext(guarded, inputBudget);
}

/**
 * Estimate the current transcript with pi-coding-agent's conservative chars/4
 * counter. `useReportedUsage` adds the last assistant turn's provider-reported
 * usage for the settled prefix; pass false when the messages were just edited.
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

function trimFixedContext(messages: AgentMessage[], inputBudget: number): AgentMessage[] {
	const trimmed = messages.slice();
	const lastIndex = trimmed.length - 1;
	const shrinkable = trimmed
		.map((message, index) => ({ message, index }))
		.filter(({ index, message }) => index !== lastIndex && message.role !== "system");

	for (const { message, index } of shrinkable) {
		const replacement = replaceMessageWithMarker(message);
		if (replacement === message || estimateMessageTokens(replacement) >= estimateMessageTokens(message)) continue;
		trimmed[index] = replacement;
		if (estimateContextMessageTokens(trimmed, false) <= inputBudget) return trimmed;
	}

	const lastMessage = trimmed[lastIndex];
	if (lastMessage !== undefined && lastMessage.role !== "system") {
		const replacement = replaceMessageWithMarker(lastMessage);
		if (replacement !== lastMessage) {
			trimmed[lastIndex] = replacement;
			if (estimateContextMessageTokens(trimmed, false) <= inputBudget) return trimmed;
		}
		const fixedWithoutLast = trimmed.reduce(
			(total, message, index) => (index === lastIndex ? total : total + estimateMessageTokens(message)),
			0,
		);
		if (fixedWithoutLast < inputBudget) {
			const fitted = fitMessageToTokens(lastMessage, inputBudget - fixedWithoutLast);
			if (fitted !== lastMessage) {
				trimmed[lastIndex] = fitted;
				if (estimateContextMessageTokens(trimmed, false) <= inputBudget) return trimmed;
			}
		}
	}

	for (const [index, message] of trimmed.entries()) {
		if (message.role !== "system") continue;
		const replacement = replaceMessageWithMarker(message);
		if (replacement === message || estimateMessageTokens(replacement) >= estimateMessageTokens(message)) continue;
		trimmed[index] = replacement;
		if (estimateContextMessageTokens(trimmed, false) <= inputBudget) return trimmed;
	}

	return trimmed;
}

function fitMessageToTokens(message: AgentMessage, maxTokens: number): AgentMessage {
	const original = textContent(message);
	if (original === undefined) return message;
	const marker = `\n\n${GUARD_MARKER}\n`;
	let low = 0;
	let high = original.length;
	let best = replaceTextContent(message, marker);
	while (low <= high) {
		const middle = Math.floor((low + high) / 2);
		const candidate = replaceTextContent(message, excerpt(original, middle, marker));
		if (estimateMessageTokens(candidate) <= maxTokens) {
			best = candidate;
			low = middle + 1;
		} else {
			high = middle - 1;
		}
	}
	return best;
}

function excerpt(text: string, maxChars: number, marker: string): string {
	if (text.length <= maxChars) return text;
	const available = Math.max(0, maxChars - marker.length);
	const head = Math.ceil(available * 0.7);
	const tail = Math.max(0, available - head);
	return `${text.slice(0, head)}${marker}${tail > 0 ? text.slice(-tail) : ""}`;
}

function replaceMessageWithMarker(message: AgentMessage): AgentMessage {
	return replaceTextContent(message, `\n\n${GUARD_MARKER}\n`);
}

function estimateMessageTokens(message: AgentMessage): number {
	if (message.role === "system") return estimateSystemMessageTokens(message);
	const estimate = estimateTokens(message);
	return message.role === "toolResult" ? estimate + Math.ceil(message.toolName.length / 4) : estimate;
}

function estimateSystemMessageTokens(message: SystemMessage): number {
	let chars = typeof message.content === "string" ? message.content.length : 0;
	if (Array.isArray(message.content)) {
		for (const block of message.content) {
			if (block.type === "text") chars += block.text.length;
			else if (block.type === "image") chars += 4_800;
		}
	}
	chars += JSON.stringify(message.sections ?? {}).length;
	chars += JSON.stringify(message.toolsAdded ?? []).length;
	chars += JSON.stringify(message.toolsRemoved ?? []).length;
	return Math.ceil(chars / 4);
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
		if (!/^\[\d+\] /mu.test(text.slice(sectionStart, sectionEnd))) return undefined;
		return splitCandidates(text, sectionStart, sectionEnd);
	}
	if (message.role !== "toolResult") return undefined;
	const isRetrieval =
		RETRIEVAL_TOOL_NAMES[message.toolName] === true || message.toolName.startsWith(RETRIEVAL_TOOL_PREFIX);
	if (!isRetrieval) return undefined;
	if (message.toolName === "jikji_find") {
		const directiveStart = text.indexOf("\n\ndirective:");
		return splitCandidates(text, 0, directiveStart >= 0 ? directiveStart : text.length, /^- [^\n]*(?:\n|$)/gmu);
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

function splitCandidates(
	text: string,
	sectionStart: number,
	sectionEnd: number,
	pattern: RegExp = /^\[\d+\] [^\n]*(?:\n|$)/gmu,
): CandidateSection | undefined {
	const region = text.slice(sectionStart, sectionEnd);
	const matches = [...region.matchAll(pattern)];
	const firstStart = matches[0]?.index;
	if (firstStart === undefined) return undefined;
	const blocks: string[] = [];
	for (const [index, match] of matches.entries()) {
		const start = match.index ?? 0;
		const end = index + 1 < matches.length ? (matches[index + 1]?.index ?? region.length) : region.length;
		blocks.push(region.slice(start, end));
	}
	return {
		prefix: text.slice(0, sectionStart) + region.slice(0, firstStart),
		suffix: text.slice(sectionEnd),
		blocks,
	};
}
