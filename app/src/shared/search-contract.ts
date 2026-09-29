/**
 * Search/chat contract shared by the Electron main process and the renderer.
 *
 * The main process owns one AutoRAGAgent instance (the "main search agent")
 * and streams `searchDocumentsStream` events to the renderer over IPC.
 * Quick = the stream's `preliminary` event (emit_fast_answer fast phase);
 * Deep = the stream's `complete` event (full SearchDocumentsResponse).
 * All payloads crossing IPC are plain serializable shapes defined here —
 * never pi/librarian class instances.
 */

export const SEARCH_CHANNELS = {
	start: "search:start",
	cancel: "search:cancel",
	feedback: "search:feedback",
	event: "search:event",
	historyList: "search:historyList",
	historyGet: "search:historyGet",
	historySearch: "search:historySearch",
	historyClear: "search:historyClear",
} as const;

/** A file or contact attached to a chat message. */
export interface ChatAttachment {
	readonly kind: "file" | "contact";
	/** Absolute path for files; contact id for contacts. */
	readonly id: string;
	/** Display name. */
	readonly name: string;
}

export interface CitationEvidence {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	/** Real source path (local files) or datasource virtual id. */
	readonly source: string | null;
	readonly excerpts: readonly string[];
	readonly confidence: number;
	readonly feedbackId: string;
}

/** One assistant answer phase mapped for the UI. */
export interface AnswerPhase {
	readonly answer: string;
	readonly evidence: readonly CitationEvidence[];
	/** e.g. "0.9s · 2 sources". Computed in the main process. */
	readonly meta: string;
}

export type SearchStreamEvent =
	| { readonly type: "progress"; readonly searchId: string; readonly text: string }
	| { readonly type: "quick"; readonly searchId: string; readonly sessionId: string; readonly phase: AnswerPhase }
	| { readonly type: "deep"; readonly searchId: string; readonly sessionId: string; readonly phase: AnswerPhase }
	| { readonly type: "error"; readonly searchId: string; readonly message: string }
	| { readonly type: "cancelled"; readonly searchId: string };

export interface ChatMessageUser {
	readonly role: "user";
	readonly text: string;
	readonly attachments: readonly ChatAttachment[];
	readonly at: string;
}

export interface ChatMessageAssistant {
	readonly role: "assistant";
	readonly sessionId: string;
	readonly quick: AnswerPhase | null;
	readonly deep: AnswerPhase | null;
	readonly stopped: boolean;
	readonly at: string;
}

export type ChatMessage = ChatMessageUser | ChatMessageAssistant;

export interface ChatSummary {
	readonly id: string;
	readonly title: string;
	/** First Quick answer with markup stripped. */
	readonly snippet: string;
	readonly updatedAt: string;
}

export interface ChatRecord extends ChatSummary {
	readonly messages: readonly ChatMessage[];
}

export interface SearchBridge {
	/** Start a Quick+Deep search. Events stream to the sender via SEARCH_CHANNELS.event. */
	start(searchId: string, chatId: string, query: string, attachments: readonly ChatAttachment[]): Promise<void>;
	cancel(searchId: string): Promise<void>;
	/** Evidence thumbs feedback; delegates to agent.recordFeedbackByNumbers. */
	feedback(sessionId: string, useful: readonly number[], notUseful: readonly number[]): Promise<void>;
	historyList(): Promise<readonly ChatSummary[]>;
	historyGet(chatId: string): Promise<ChatRecord | null>;
	historySearch(query: string): Promise<readonly ChatSummary[]>;
	historyClear(): Promise<void>;
	readonly onEvent?: (listener: (event: SearchStreamEvent) => void) => () => void;
}
