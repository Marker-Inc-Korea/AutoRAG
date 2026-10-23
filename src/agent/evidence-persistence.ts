import type { ExtensionAPI, ExtensionFactory, SessionEntry } from "@earendil-works/pi-coding-agent";
import {
	CITATION_NUMBERS_ENTRY_TYPE,
	type CitationNumberEntry,
	type CitationNumbers,
	highestCitationNumber,
	parseCitationNumberEntries,
} from "./citation-numbers.ts";
import {
	EVIDENCE_ENTRY_TYPE,
	type EvidenceLedger,
	highestOrdinal,
	type LedgerEntry,
	parseLedgerEntries,
} from "./evidence-ledger.ts";

/** The part of pi's session manager the ledger is rebuilt from. */
interface SessionEntries {
	getBranch(): SessionEntry[];
	getEntries(): SessionEntry[];
}

/**
 * Rebuild `ledger` and `numbers` for the conversation branch `session` is on.
 * Evidence and result numbers from other branches stay out of reach (their
 * transcripts are not in context), but their ids and numbers still count: new
 * evidence and new results never reuse one issued anywhere in the conversation.
 */
export function restoreConversationState(
	ledger: EvidenceLedger,
	numbers: CitationNumbers,
	session: SessionEntries,
): void {
	const branchIds = new Set(session.getBranch().map((entry) => entry.id));
	const onBranch: LedgerEntry[] = [];
	const numbersOnBranch: CitationNumberEntry[] = [];
	let highest = 0;
	let highestNumber = 0;
	for (const entry of session.getEntries()) {
		if (entry.type !== "custom") continue;
		if (entry.customType === EVIDENCE_ENTRY_TYPE) {
			const saved = parseLedgerEntries(entry.data);
			highest = Math.max(highest, highestOrdinal(saved));
			if (branchIds.has(entry.id)) onBranch.push(...saved);
		} else if (entry.customType === CITATION_NUMBERS_ENTRY_TYPE) {
			const saved = parseCitationNumberEntries(entry.data);
			highestNumber = Math.max(highestNumber, highestCitationNumber(saved));
			if (branchIds.has(entry.id)) numbersOnBranch.push(...saved);
		}
	}
	ledger.restore(onBranch, highest);
	numbers.restore(numbersOnBranch, highestNumber);
}

/**
 * pi extension that keeps the conversation's evidence ledger and result
 * numbers in the session file, next to the transcript that shows the model
 * its `[eN]` ids and the user its `[n]` results. Evidence is appended as the
 * tools return it and result numbers as answers are derived; switching,
 * resuming, forking, or navigating the session tree rebuilds both from the
 * branch now in view. An id the model saw therefore resolves, and an id or
 * result number is never reissued, whatever turns, tool calls, resumes, or
 * restarts happened in between.
 */
export function createEvidencePersistence(ledger: EvidenceLedger, numbers: CitationNumbers): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		const save = (): void => {
			const entries = ledger.takeUnsaved();
			if (entries.length > 0) pi.appendEntry(EVIDENCE_ENTRY_TYPE, { entries });
		};
		numbers.setSink((entries) => pi.appendEntry(CITATION_NUMBERS_ENTRY_TYPE, { entries }));
		pi.on("tool_execution_end", save);
		pi.on("turn_end", save);
		pi.on("agent_end", save);
		pi.on("session_start", (_event, ctx) => restoreConversationState(ledger, numbers, ctx.sessionManager));
		pi.on("session_tree", (_event, ctx) => restoreConversationState(ledger, numbers, ctx.sessionManager));
	};
}
