import { type Static, Type } from "typebox";
import { Value } from "typebox/value";

/** Custom session entry type under which a conversation persists its result numbers. */
export const CITATION_NUMBERS_ENTRY_TYPE = "autorag.citation-numbers";

const citationNumberEntrySchema = Type.Object({
	source: Type.String(),
	number: Type.Integer({ minimum: 1 }),
});

/** One source and the result number (`[n]`) the user was shown for it. */
export type CitationNumberEntry = Static<typeof citationNumberEntrySchema>;

/** What a session persists under {@link CITATION_NUMBERS_ENTRY_TYPE}. */
const persistedCitationNumbersSchema = Type.Object({ entries: Type.Array(Type.Unknown()) });

/**
 * Entries a session persisted under {@link CITATION_NUMBERS_ENTRY_TYPE}. A
 * malformed entry is skipped: its source is then numbered afresh, never
 * under a number it was not shown with.
 */
export function parseCitationNumberEntries(data: unknown): CitationNumberEntry[] {
	if (!Value.Check(persistedCitationNumbersSchema, data)) return [];
	return data.entries.filter((entry): entry is CitationNumberEntry => Value.Check(citationNumberEntrySchema, entry));
}

/** Highest result number among `entries`; 0 when there are none. */
export function highestCitationNumber(entries: readonly CitationNumberEntry[]): number {
	let highest = 0;
	for (const entry of entries) highest = Math.max(highest, entry.number);
	return highest;
}

/**
 * The result numbers (`[1]`, `[2]`, ...) one conversation has shown the user.
 *
 * A source keeps the number it was first shown with in every later answer of
 * the conversation, and a new source gets the next unused number. Numbers are
 * never reissued, mirroring the `e<N>` ordinals of the `EvidenceLedger`: the
 * quick and verification answers of one turn, and every turn after them,
 * share one numbering, so a `[n]` a user saw never names a different source.
 *
 * The hosting session persists what {@link takeUnsaved} returns (through the
 * sink installed with {@link setSink}) and rebuilds the numbers with
 * {@link restore} when a conversation is resumed.
 */
export class CitationNumbers {
	private readonly bySource = new Map<string, number>();
	/** Entries issued since the last {@link takeUnsaved}. */
	private unsaved: CitationNumberEntry[] = [];
	private sink: ((entries: CitationNumberEntry[]) => void) | undefined;
	/**
	 * The last number issued. Survives {@link clear} and {@link restore}: a
	 * conversation keeps earlier answers in view, so restarting at 1 would let
	 * an old `[n]` silently name a different source.
	 */
	private lastNumber = 0;

	/** The number for `source`: the one it already has, else the next unused number. */
	numberFor(source: string): number {
		const known = this.bySource.get(source);
		if (known !== undefined) return known;
		const number = ++this.lastNumber;
		this.bySource.set(source, number);
		this.unsaved.push({ source, number });
		return number;
	}

	/** Forget every source. Numbers are never reissued, so a previous conversation's `[n]` stays unused. */
	clear(): void {
		this.bySource.clear();
		this.unsaved = [];
	}

	/** Forget every source and restart at 1: for a run that owns no conversation, where no earlier `[n]` is in view. */
	reset(): void {
		this.clear();
		this.lastNumber = 0;
	}

	/** Numbers issued since the last call, for the hosting session to persist. */
	takeUnsaved(): CitationNumberEntry[] {
		const entries = this.unsaved;
		this.unsaved = [];
		return entries;
	}

	/** Where {@link flush} writes numbers: the hosting session. Replaces any earlier sink. */
	setSink(sink: ((entries: CitationNumberEntry[]) => void) | undefined): void {
		this.sink = sink;
	}

	/**
	 * Persist numbers issued since the last save right now. Answers are derived
	 * after a turn's last session event, so without this the final answer's new
	 * numbers would not reach the session file until the next turn. With no
	 * sink installed the numbers stay unsaved for the next save.
	 */
	flush(): void {
		if (this.sink === undefined || this.unsaved.length === 0) return;
		this.sink(this.takeUnsaved());
	}

	/**
	 * Rebuild the numbers from persisted entries, such as the ones on the
	 * resumed conversation branch; a later entry for the same source replaces an
	 * earlier one. Numbers keep counting past every restored number and past
	 * `numberFloor`, the highest number issued anywhere in the conversation
	 * (other branches included).
	 */
	restore(entries: readonly CitationNumberEntry[], numberFloor = 0): void {
		this.clear();
		for (const entry of entries) this.bySource.set(entry.source, entry.number);
		this.lastNumber = Math.max(this.lastNumber, numberFloor, highestCitationNumber(entries));
	}
}
