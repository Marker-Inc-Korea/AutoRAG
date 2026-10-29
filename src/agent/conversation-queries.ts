import { Type } from "typebox";
import { Value } from "typebox/value";

/** Custom session entry type under which a conversation persists what the user typed. */
export const CONVERSATION_QUERIES_ENTRY_TYPE = "autorag.queries";

/** What a session persists under {@link CONVERSATION_QUERIES_ENTRY_TYPE}. */
const persistedQueriesSchema = Type.Object({ entries: Type.Array(Type.Unknown()) });

/** The queries a session persisted under {@link CONVERSATION_QUERIES_ENTRY_TYPE}; a malformed entry is skipped. */
export function parseQueryEntries(data: unknown): string[] {
	if (!Value.Check(persistedQueriesSchema, data)) return [];
	return data.entries.filter((entry): entry is string => typeof entry === "string");
}

/**
 * Everything the user typed in one conversation: each question and each
 * steer, as typed. The transcript cannot stand in for it, because the first
 * turn's user message there is the harness prompt that wraps the question
 * with baseline evidence and formatting rules.
 *
 * The hosting session persists what {@link flush} hands to the sink and
 * rebuilds the list with {@link restore} when a conversation is resumed or
 * a new one starts, the way the evidence ledger and result numbers are.
 */
export class ConversationQueries {
	private queries: string[] = [];
	/** Queries added since the last {@link flush}. */
	private unsaved: string[] = [];
	private sink: ((entries: string[]) => void) | undefined;

	add(query: string): void {
		this.queries.push(query);
		this.unsaved.push(query);
	}

	/** Every query of the conversation, oldest first. */
	all(): readonly string[] {
		return this.queries;
	}

	/** Where {@link flush} writes queries: the hosting session. Replaces any earlier sink. */
	setSink(sink: ((entries: string[]) => void) | undefined): void {
		this.sink = sink;
	}

	/** Persist queries added since the last save. With no sink installed they stay unsaved for the next save. */
	flush(): void {
		if (this.sink === undefined || this.unsaved.length === 0) return;
		const entries = this.unsaved;
		this.unsaved = [];
		this.sink(entries);
	}

	/** Replace the conversation's queries with already-persisted ones; an empty list forgets everything. */
	restore(queries: readonly string[]): void {
		this.queries = [...queries];
		this.unsaved = [];
	}

	/** Forget every query, saved or not: a run that owns no conversation starts over. */
	reset(): void {
		this.restore([]);
	}
}
