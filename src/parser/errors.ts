export class ParseError extends Error {
	readonly parserName: string;
	readonly virtualPath: string;

	constructor(parserName: string, virtualPath: string, cause: unknown) {
		super(`Parser "${parserName}" failed for ${virtualPath}: ${messageFrom(cause)}`);
		this.name = "ParseError";
		this.parserName = parserName;
		this.virtualPath = virtualPath;
		this.cause = cause;
	}
}

function messageFrom(cause: unknown): string {
	if (cause instanceof Error) return cause.message;
	if (typeof cause === "string") return cause;
	return "unknown parser failure";
}

export type ParseIsolationFailureCode = "PARSE_TIMEOUT" | "WORKER_OOM" | "WORKER_CRASHED";

/**
 * Why an isolated parse worker was lost. Carried as the `cause` of a
 * {@link ParseError} so a timeout, an out-of-memory kill, and a crash stay
 * distinguishable from a parser's own failure.
 */
export class ParseIsolationError extends Error {
	readonly code: ParseIsolationFailureCode;

	constructor(code: ParseIsolationFailureCode, message: string) {
		super(message);
		this.name = "ParseIsolationError";
		this.code = code;
	}
}
