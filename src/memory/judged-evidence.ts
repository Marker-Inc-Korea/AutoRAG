/**
 * Evidence the final answer cited and Jev judged to support the user's
 * question. This is the unit retrieval memory stores: a question, the search
 * query and method that surfaced the evidence, and the evidence itself.
 */
export interface JudgedEvidenceRecord {
	/** `<sessionId>:<stableEvidenceId>`; unique per search run and evidence. */
	readonly id: string;
	/** The search run (one `searchDocuments` call) that produced the evidence. */
	readonly sessionId: string;
	/** The agent conversation the run belonged to; "current session" memory is keyed on it. */
	readonly conversationId: string;
	/** The user's original question. */
	readonly question: string;
	/** The search query that surfaced the evidence (the question itself when no sub-query is known). */
	readonly searchQuery: string;
	/** Retrieval method or tool that surfaced the evidence. */
	readonly method: string;
	/** Opaque source identifier of the evidence. */
	readonly source: string;
	readonly stableEvidenceId: string;
	/** Number and title of the curated result the evidence backs. */
	readonly resultNumber: number;
	readonly title: string;
	/** Bounded evidence text. */
	readonly excerpt: string;
	/** Jev's P(this evidence supports the question), at or above {@link EVIDENCE_SUPPORT_THRESHOLD}. */
	readonly probability: number;
	readonly createdAt: number;
}

/** Evidence is kept in memory only when Jev's P(supports the question) reaches this. */
export const EVIDENCE_SUPPORT_THRESHOLD = 0.7;
