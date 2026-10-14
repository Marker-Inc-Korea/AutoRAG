/**
 * Pure ranking primitives shared by every crawler sidecar: float32 vector
 * (de)serialization, cosine top-k, and reciprocal rank fusion. No I/O.
 */

/** Standard RRF constant (Cormack et al.); damps the head of each list. */
const RRF_K = 60;

export function encodeVector(vector: readonly number[]): Uint8Array {
	const floats = Float32Array.from(vector);
	return new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
}

/** Inverse of {@link encodeVector}; undefined when the blob is not a whole number of float32s. */
export function decodeVector(blob: Uint8Array): Float32Array | undefined {
	if (blob.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) return undefined;
	// Copy: a SQLite-returned view may be unaligned for Float32Array. No Array.from: this runs for every
	// stored vector on every semantic query, and boxing into a number[] dominated the scan cost.
	return new Float32Array(new Uint8Array(blob).buffer);
}

/** Cosine similarity; 0 (never NaN) for zero-length, zero-norm, or mismatched-dimension inputs. */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	if (a.length === 0 || a.length !== b.length) return 0;
	let dot = 0;
	let normA = 0;
	let normB = 0;
	for (let index = 0; index < a.length; index += 1) {
		const x = a[index] ?? 0;
		const y = b[index] ?? 0;
		dot += x * y;
		normA += x * x;
		normB += y * y;
	}
	if (normA === 0 || normB === 0) return 0;
	return dot / Math.sqrt(normA * normB);
}

export interface KeyedVector {
	readonly key: string;
	readonly vector: ArrayLike<number>;
}

export interface ScoredKey {
	readonly key: string;
	readonly score: number;
}

/** Best `k` positive-similarity keys, descending; ties broken by key for determinism. */
export function topKByCosine(query: ArrayLike<number>, rows: Iterable<KeyedVector>, k: number): ScoredKey[] {
	if (k <= 0) return [];
	const scored: ScoredKey[] = [];
	for (const row of rows) {
		const score = cosine(query, row.vector);
		if (score > 0) scored.push({ key: row.key, score });
	}
	scored.sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
	return scored.slice(0, k);
}

/**
 * Reciprocal rank fusion over ranked key lists (best first). A key repeated
 * inside one list counts once, at its best rank. Ties keep first-seen order,
 * so earlier lists win.
 */
export function reciprocalRankFusion(lists: readonly (readonly string[])[]): ScoredKey[] {
	const scores = new Map<string, number>();
	for (const list of lists) {
		const seen = new Set<string>();
		for (const [index, key] of list.entries()) {
			if (seen.has(key)) continue;
			seen.add(key);
			scores.set(key, (scores.get(key) ?? 0) + 1 / (RRF_K + index + 1));
		}
	}
	return [...scores.entries()].map(([key, score]) => ({ key, score })).sort((a, b) => b.score - a.score);
}
