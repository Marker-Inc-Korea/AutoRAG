/**
 * Map `items` through `fn` with at most `limit` calls in flight, preserving
 * result order. `fn` must not reject: a rejection stops new work and rejects
 * the whole call.
 */
export async function mapWithConcurrency<T, R>(
	items: readonly T[],
	limit: number,
	fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
	const results = new Array<R>(items.length);
	let next = 0;
	const worker = async (): Promise<void> => {
		while (next < items.length) {
			const index = next++;
			results[index] = await fn(items[index] as T, index);
		}
	};
	const width = Math.max(1, Math.min(Math.floor(limit) || 1, items.length));
	await Promise.all(Array.from({ length: width }, worker));
	return results;
}
