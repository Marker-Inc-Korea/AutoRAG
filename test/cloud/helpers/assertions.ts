/** Await a promise that must reject and hand back the reason as an `Error`. */
export async function rejectionOf(promise: Promise<unknown>): Promise<Error> {
	try {
		await promise;
	} catch (error) {
		return error instanceof Error ? error : new Error(String(error));
	}
	throw new Error("expected the promise to reject, but it resolved");
}
