import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { CitationNumbers } from "../../src/agent/citation-numbers.ts";
import {
	CONVERSATION_QUERIES_ENTRY_TYPE,
	ConversationQueries,
	parseQueryEntries,
} from "../../src/agent/conversation-queries.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { restoreConversationState } from "../../src/agent/evidence-persistence.ts";

function customEntry(id: string, entries: unknown[]): SessionEntry {
	return { type: "custom", id, customType: CONVERSATION_QUERIES_ENTRY_TYPE, data: { entries } } as SessionEntry;
}

describe("ConversationQueries", () => {
	it("keeps everything the user typed, in order", () => {
		const queries = new ConversationQueries();
		queries.add("what is alpha?");
		queries.add("and beta?");
		expect(queries.all()).toEqual(["what is alpha?", "and beta?"]);
	});

	it("hands only the queries added since the last flush to the sink", () => {
		const queries = new ConversationQueries();
		const saved: string[][] = [];
		queries.setSink((entries) => saved.push(entries));

		queries.add("one");
		queries.flush();
		queries.flush();
		queries.add("two");
		queries.add("three");
		queries.flush();

		expect(saved).toEqual([["one"], ["two", "three"]]);
	});

	it("keeps unsaved queries for the next save when no session is attached", () => {
		const queries = new ConversationQueries();
		queries.add("typed before a session existed");
		queries.flush();

		const saved: string[][] = [];
		queries.setSink((entries) => saved.push(entries));
		queries.flush();

		expect(saved).toEqual([["typed before a session existed"]]);
	});

	it("forgets everything on reset, saved or not", () => {
		const queries = new ConversationQueries();
		const saved: string[][] = [];
		queries.setSink((entries) => saved.push(entries));
		queries.add("old conversation");

		queries.reset();
		queries.flush();

		expect(queries.all()).toEqual([]);
		expect(saved).toEqual([]);
	});

	it("restores what a session saved and does not save it again", () => {
		const queries = new ConversationQueries();
		const saved: string[][] = [];
		queries.setSink((entries) => saved.push(entries));
		queries.add("typed in the previous conversation");

		queries.restore(["earlier one", "earlier two"]);
		queries.flush();

		expect(queries.all()).toEqual(["earlier one", "earlier two"]);
		expect(saved).toEqual([]);
	});
});

describe("parseQueryEntries", () => {
	it("keeps strings and skips anything else", () => {
		expect(parseQueryEntries({ entries: ["a", 3, null, "b", { text: "c" }] })).toEqual(["a", "b"]);
	});

	it("returns nothing for data that is not a saved query list", () => {
		expect(parseQueryEntries(undefined)).toEqual([]);
		expect(parseQueryEntries({ entries: "a" })).toEqual([]);
	});
});

describe("restoreConversationState and typed queries", () => {
	function sessionWith(entries: SessionEntry[], branch: SessionEntry[]) {
		return { getEntries: () => entries, getBranch: () => branch };
	}

	it("restores the queries saved on the branch in view, oldest first", () => {
		const first = customEntry("q1", ["what is alpha?"]);
		const second = customEntry("q2", ["and beta?", "correct: only 2025"]);
		const queries = new ConversationQueries();

		restoreConversationState(
			new EvidenceLedger(),
			new CitationNumbers(),
			queries,
			sessionWith([first, second], [first, second]),
		);

		expect(queries.all()).toEqual(["what is alpha?", "and beta?", "correct: only 2025"]);
	});

	it("leaves out queries typed on another branch of the conversation", () => {
		const shared = customEntry("q1", ["what is alpha?"]);
		const otherBranch = customEntry("q2", ["something typed on a branch the user left"]);
		const queries = new ConversationQueries();

		restoreConversationState(
			new EvidenceLedger(),
			new CitationNumbers(),
			queries,
			sessionWith([shared, otherBranch], [shared]),
		);

		expect(queries.all()).toEqual(["what is alpha?"]);
	});

	it("starts empty for a session with no saved queries, so a new conversation never inherits the last one", () => {
		const queries = new ConversationQueries();
		queries.add("typed in the conversation before /new");

		restoreConversationState(new EvidenceLedger(), new CitationNumbers(), queries, sessionWith([], []));

		expect(queries.all()).toEqual([]);
	});
});
