import {
	AutoRAGAgent,
	buildAgentOptions,
	resolveConfigReadOnly,
	type AutoRAGAgentOptions,
	type SearchDocumentsResponse,
} from "@autorag/librarian";
import { resolveAgentModel } from "@autorag/librarian/core";
import type {
	AnswerPhase,
	ChatMessageAssistant,
	ChatMessageUser,
	SearchBridge,
	SearchStreamEvent,
} from "../shared/search-contract";
import { SEARCH_CHANNELS } from "../shared/search-contract";
import type { ChatStore } from "./chat-store";

type SearchAgent = Pick<AutoRAGAgent, "searchDocumentsStream" | "recordFeedbackByNumbers"> & {
	readonly abort?: () => void;
	readonly createChatSession?: AutoRAGAgent["createChatSession"];
};

export interface SearchServiceDeps {
	readonly agentFactory: () => SearchAgent;
	readonly chatStore: ChatStore;
	readonly send: (channel: string, payload: SearchStreamEvent) => void;
	readonly now?: () => Date;
}

interface SearchRun {
	readonly searchId: string;
	readonly chatId: string;
	readonly agent: SearchAgent;
	readonly startedAt: number;
	readonly finish: Promise<void>;
	readonly resolveFinish: () => void;
	cancelled: boolean;
	quick: AnswerPhase | null;
	deep: AnswerPhase | null;
	sessionId: string;
}

function errorText(error: unknown): string {
	return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function mapPhase(response: SearchDocumentsResponse, elapsedMs: number): AnswerPhase {
	return {
		answer: response.answer,
		evidence: response.results.map((result) => ({
			number: result.number,
			title: result.title,
			summary: result.summary,
			source: result.source ?? null,
			excerpts: result.evidence.map((evidence) => evidence.excerpt),
			confidence: result.confidence,
			feedbackId: result.feedbackId,
		})),
		meta: `${(elapsedMs / 1000).toFixed(1)}s · ${response.searched} sources`,
	};
}

function assistantMessage(run: SearchRun, stopped: boolean, now: () => Date): ChatMessageAssistant {
	return {
		role: "assistant",
		sessionId: run.sessionId,
		quick: run.quick,
		deep: run.deep,
		stopped,
		at: now().toISOString(),
	};
}

export function createSearchService(deps: SearchServiceDeps): SearchBridge {
	const now = deps.now ?? (() => new Date());
	const active = new Map<string, SearchRun>();

	const emit = (event: SearchStreamEvent): void => deps.send(SEARCH_CHANNELS.event, event);

	async function saveRunMessage(run: SearchRun, stopped: boolean): Promise<void> {
		const existing = await deps.chatStore.get(run.chatId);
		if (existing === null) return;
		const messages = existing.messages.filter((message) => message.role !== "assistant");
		await deps.chatStore.save({
			...existing,
			messages: [...messages, assistantMessage(run, stopped, now)],
			updatedAt: now().toISOString(),
		});
	}

	async function consume(run: SearchRun, query: string): Promise<void> {
		try {
			for await (const event of run.agent.searchDocumentsStream(query)) {
				if (run.cancelled) return;
				switch (event.type) {
					case "progress":
						emit({ type: "progress", searchId: run.searchId, text: event.text });
						break;
					case "answer_delta":
						emit({
							type: event.phase === "preliminary" ? "quick-delta" : "deep-delta",
							searchId: run.searchId,
							text: event.text,
						});
						break;
					case "preliminary": {
						run.sessionId = event.response.sessionId;
						run.quick = mapPhase(event.response, now().getTime() - run.startedAt);
						emit({
							type: "quick",
							searchId: run.searchId,
							sessionId: run.sessionId,
							phase: run.quick,
						});
						await saveRunMessage(run, false);
						break;
					}
					case "complete": {
						run.sessionId = event.response.sessionId;
						run.deep = mapPhase(event.response, now().getTime() - run.startedAt);
						emit({
							type: "deep",
							searchId: run.searchId,
							sessionId: run.sessionId,
							phase: run.deep,
						});
						await saveRunMessage(run, false);
						break;
					}
					default:
						break;
				}
			}
		} catch (error) {
			if (!run.cancelled) {
				emit({ type: "error", searchId: run.searchId, message: errorText(error) });
			}
		} finally {
			active.delete(run.searchId);
			run.resolveFinish();
		}
	}

	return {
		start: async (searchId, chatId, query, attachments) => {
			const agent = deps.agentFactory();
			let resolveFinish = (): void => undefined;
			const finish = new Promise<void>((resolve) => {
				resolveFinish = resolve;
			});
			const at = now().toISOString();
			const userMessage: ChatMessageUser = {
				role: "user",
				text: query,
				attachments,
				at,
			};
			const run: SearchRun = {
				searchId,
				chatId,
				agent,
				startedAt: now().getTime(),
				finish,
				resolveFinish,
				cancelled: false,
				quick: null,
				deep: null,
				sessionId: "",
			};
			active.set(searchId, run);
			const existing = await deps.chatStore.get(chatId);
			await deps.chatStore.save({
				id: chatId,
				title: query,
				snippet: "",
				updatedAt: at,
				messages: [...(existing?.messages ?? []), userMessage],
			});
			void consume(run, query);
			await finish;
		},
		cancel: async (searchId) => {
			const run = active.get(searchId);
			if (run === undefined || run.cancelled) return;
			run.cancelled = true;
			run.agent.abort?.();
			emit({ type: "cancelled", searchId });
			await saveRunMessage(run, true);
			active.delete(searchId);
			run.resolveFinish();
		},
		feedback: async (sessionId, useful, notUseful) => {
			deps.agentFactory().recordFeedbackByNumbers(sessionId, [...useful], [...notUseful]);
		},
		historyList: () => deps.chatStore.list(),
		historyGet: (chatId) => deps.chatStore.get(chatId),
		historySearch: (query) => deps.chatStore.search(query),
		historyClear: () => deps.chatStore.clear(),
		onEvent: () => () => undefined,
	};
}

export function createDefaultAgentFactory(): () => SearchAgent {
	return () => {
		const config = resolveConfigReadOnly({ flags: {}, env: process.env, cwd: process.cwd() });
		const resolved = resolveAgentModel(config);
		const options: AutoRAGAgentOptions = {
			...buildAgentOptions(config),
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
		};
		return new AutoRAGAgent(options) as SearchAgent;
	};
}
