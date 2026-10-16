import type { ExtensionAPI, ExtensionFactory, SessionEntry } from "@earendil-works/pi-coding-agent";
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
 * Rebuild `ledger` for the conversation branch `session` is on. Evidence from
 * other branches stays out of reach (their transcripts are not in context), but
 * its ids still count: new evidence never reuses an id issued anywhere in the
 * conversation.
 */
export function restoreEvidenceLedger(ledger: EvidenceLedger, session: SessionEntries): void {
	const branchIds = new Set(session.getBranch().map((entry) => entry.id));
	const onBranch: LedgerEntry[] = [];
	let highest = 0;
	for (const entry of session.getEntries()) {
		if (entry.type !== "custom" || entry.customType !== EVIDENCE_ENTRY_TYPE) continue;
		const saved = parseLedgerEntries(entry.data);
		highest = Math.max(highest, highestOrdinal(saved));
		if (branchIds.has(entry.id)) onBranch.push(...saved);
	}
	ledger.restore(onBranch, highest);
}

/**
 * pi extension that keeps the conversation's evidence ledger in the session
 * file, next to the transcript that shows the model its `[eN]` ids. Evidence is
 * appended as the tools return it; switching, resuming, forking, or navigating
 * the session tree rebuilds the ledger from the branch now in view. An id the
 * model saw therefore resolves, and is never reissued, whatever turns, tool
 * calls, resumes, or restarts happened in between.
 */
export function createEvidencePersistence(ledger: EvidenceLedger): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		const save = (): void => {
			const entries = ledger.takeUnsaved();
			if (entries.length > 0) pi.appendEntry(EVIDENCE_ENTRY_TYPE, { entries });
		};
		pi.on("tool_execution_end", save);
		pi.on("turn_end", save);
		pi.on("agent_end", save);
		pi.on("session_start", (_event, ctx) => restoreEvidenceLedger(ledger, ctx.sessionManager));
		pi.on("session_tree", (_event, ctx) => restoreEvidenceLedger(ledger, ctx.sessionManager));
	};
}
