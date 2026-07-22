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
