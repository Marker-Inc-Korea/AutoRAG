/**
 * Record cut: keep one long document from taking every slot of a ranked list.
 *
 * Hybrid retrieval ranks and cuts by passage. A long document that matches the
 * query in dozens of passages fills the top N, and a document only the keyword
 * leg found lands just past the cut. With the cut on, a search asks the backend
 * for `depth` times what the caller wants (never past {@link OVERFETCH_ABSOLUTE_CAP})
 * and keeps `limit` hits spread across documents: the best half with at most
 * `perRecord` passages from any one document, then the best passage of each
 * document not yet shown, then rank order for whatever room is left.
 *
 * The same module sizes what the model is *shown*: {@link planReadBudget} splits
 * a result budget max-min fairly across the results of one tool call, so the
 * first large result cannot spend all of it and leave the rest empty.
 *
 * The technique and its defaults follow PipesHub's `FUSION_RECORD_CUT`
 * (`backend/python/app/modules/retrieval/record_cut.py`).
 */

/** Most hits ever asked of a backend for the deeper fusion. */
export const OVERFETCH_ABSOLUTE_CAP = 300;

/** Past ten the cap binds for any limit above 30, and a small limit gains nothing from fusing hundreds of hits. */
export const MAX_FUSION_DEPTH = 10;

/** Bounds the per-record knobs: past the over-fetch cap a larger value changes nothing. */
export const MAX_PER_RECORD = 300;

/** Characters one search tool result may hold when the cut is on. */
export const DEFAULT_RESULT_CHARS = 60_000;

/** Below this share a result is announced as not read instead of being clipped to a stub. */
export const DEFAULT_MIN_READ_CHARS = 2_000;

export interface CutByRecordOptions<T> {
	/** Most hits one record may hold in the head of the cut. */
	readonly perRecord: number;
	/** Slots at the end of `limit` kept for the best hit of each record not yet shown. */
	readonly reserve: number;
	/** The record a hit belongs to; undefined counts as a record of its own. */
	readonly recordOf: (hit: T) => string | undefined;
}

/**
 * Keep `limit` of `results` (best first), spread across records.
 *
 * The best `limit - reserve` results come first, at most `perRecord` from any
 * one record. The reserve then goes to the best result of each record not yet
 * represented, and anything left is filled in rank order, so a search confined
 * to a few records still returns `limit`. Survivors keep their original order.
 * A `limit` below 1 keeps everything.
 */
export function cutByRecord<T>(results: readonly T[], limit: number, options: CutByRecordOptions<T>): T[] {
	if (limit < 1 || results.length <= limit) return [...results];
	const perRecord = Math.max(options.perRecord, 1);
	const reserve = Math.min(Math.max(options.reserve, 0), limit);
	const keys = results.map((result, index) => options.recordOf(result) ?? `\0unkeyed:${index}`);
	const chosen = new Set<number>();
	const perKey = new Map<string, number>();
	const take = (index: number): void => {
		chosen.add(index);
		const key = keys[index]!;
		perKey.set(key, (perKey.get(key) ?? 0) + 1);
	};

	const head = limit - reserve;
	for (const [index, key] of keys.entries()) {
		if (chosen.size >= head) break;
		if ((perKey.get(key) ?? 0) < perRecord) take(index);
	}
	for (const [index, key] of keys.entries()) {
		if (chosen.size >= limit) break;
		if (!perKey.has(key)) take(index);
	}
	for (let index = 0; index < results.length && chosen.size < limit; index++) {
		if (!chosen.has(index)) take(index);
	}
	return [...chosen].sort((a, b) => a - b).map((index) => results[index]!);
}

/**
 * What a caller may tune; every field is optional and clamped by {@link recordCutFromOptions}. The cut is on unless
 * `enabled` is false.
 */
export interface RecordCutOptions {
	/** Set false to turn the record cut off. Default true. */
	readonly enabled?: boolean;
	/** Fusion depth: the backend is asked for `depth` x the limit. 1-10, default 3. */
	readonly depth?: number;
	/** Share of the limit kept for documents not yet shown. 0-1, default 0.5. */
	readonly newRecordShare?: number;
	/** One record keeps at most `max(perRecordFloor, limit / perRecordDivisor)` of the head. Default 4. */
	readonly perRecordFloor?: number;
	/** See `perRecordFloor`. Default 6. */
	readonly perRecordDivisor?: number;
	/** Characters one search tool result may hold. Default {@link DEFAULT_RESULT_CHARS}. */
	readonly resultChars?: number;
	/** Smallest share worth reading; a result below it is announced as not read. Default {@link DEFAULT_MIN_READ_CHARS}. */
	readonly minReadChars?: number;
}

/** How deep a search fuses, how its hits are cut back by record, and how a result is budgeted. */
export class RecordCut {
	readonly depth: number;
	readonly newRecordShare: number;
	readonly perRecordFloor: number;
	readonly perRecordDivisor: number;
	readonly resultChars: number;
	readonly minReadChars: number;

	constructor(options: RecordCutOptions = {}) {
		this.depth = options.depth ?? 3;
		this.newRecordShare = options.newRecordShare ?? 0.5;
		this.perRecordFloor = options.perRecordFloor ?? 4;
		this.perRecordDivisor = options.perRecordDivisor ?? 6;
		this.resultChars = options.resultChars ?? DEFAULT_RESULT_CHARS;
		this.minReadChars = options.minReadChars ?? DEFAULT_MIN_READ_CHARS;
	}

	/** What to ask the backend for: `depth` x `limit`, within `cap`. The cap never lowers the caller's own limit. */
	fusedLimit(limit: number, cap: number): number {
		return Math.max(limit, Math.min(limit * this.depth, cap));
	}

	/** Passages one record may hold in the head of a cut to `limit` (8 at a limit of 50). */
	perRecord(limit: number): number {
		return Math.max(this.perRecordFloor, Math.floor(limit / this.perRecordDivisor));
	}

	cut<T>(results: readonly T[], limit: number, recordOf: (hit: T) => string | undefined): T[] {
		return cutByRecord(results, limit, {
			perRecord: this.perRecord(limit),
			reserve: Math.floor(limit * this.newRecordShare),
			recordOf,
		});
	}
}

function integerInRange(value: number | undefined, name: string, low: number, high: number): void {
	if (value !== undefined && (!Number.isInteger(value) || value < low || value > high)) {
		throw new RangeError(`${name} must be an integer from ${low} to ${high}`);
	}
}

/** The cut these options describe: on unless `enabled` is false, then undefined. Out-of-range tuning throws. */
export function recordCutFromOptions(options: RecordCutOptions | undefined): RecordCut | undefined {
	if (options?.enabled === false) return undefined;
	return validatedRecordCut(options ?? {});
}

function validatedRecordCut(options: RecordCutOptions): RecordCut {
	integerInRange(options.depth, "depth", 1, MAX_FUSION_DEPTH);
	integerInRange(options.perRecordFloor, "perRecordFloor", 1, MAX_PER_RECORD);
	integerInRange(options.perRecordDivisor, "perRecordDivisor", 1, MAX_PER_RECORD);
	if (options.newRecordShare !== undefined && !(options.newRecordShare >= 0 && options.newRecordShare <= 1)) {
		throw new RangeError("newRecordShare must be a number from 0 to 1");
	}
	return new RecordCut(options);
}

/**
 * Split `total` across results of the given sizes, max-min fair: a result that
 * needs less than an equal split gets exactly what it needs and what it leaves
 * is divided among the larger ones. Shares follow input order and never sum
 * past `total`.
 */
export function fairShares(sizes: readonly number[], total: number): number[] {
	const shares = sizes.map(() => 0);
	let remaining = Math.max(0, total);
	const pending = sizes.map((_, index) => index).sort((a, b) => Math.max(0, sizes[a]!) - Math.max(0, sizes[b]!));
	while (pending.length > 0) {
		const equal = Math.floor(remaining / pending.length);
		const index = pending[0]!;
		const need = Math.max(0, sizes[index]!);
		if (need <= equal) {
			shares[index] = need;
			remaining -= need;
			pending.shift();
			continue;
		}
		for (const rest of pending) shares[rest] = equal;
		break;
	}
	return shares;
}

/**
 * Per-result character limits for one tool call, in input order (best first).
 * `undefined` means the result is not read: reading it would push a better
 * result's fair share under `minRead`.
 *
 * The largest prefix whose every member can still get `min(minRead, size)`
 * characters is read, split max-min fairly, so the first large result cannot
 * spend the budget and leave the rest empty. At least the first result is
 * always read, with whatever the budget allows.
 */
export function planReadBudget(sizes: readonly number[], total: number, minRead: number): (number | undefined)[] {
	let shares = fairShares(sizes, total);
	let count = sizes.length;
	const starved = (): boolean => shares.some((share, index) => share < Math.min(minRead, Math.max(0, sizes[index]!)));
	while (count > 1 && starved()) {
		count--;
		shares = fairShares(sizes.slice(0, count), total);
	}
	return sizes.map((_, index) => (index < count ? shares[index] : undefined));
}
