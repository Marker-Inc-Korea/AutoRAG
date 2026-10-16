/** The rejection for work its caller aborted: the signal's reason when it is an error, else an `AbortError`. */
export function abortReason(signal: AbortSignal): Error {
	return signal.reason instanceof Error ? signal.reason : new DOMException("The operation was aborted", "AbortError");
}
