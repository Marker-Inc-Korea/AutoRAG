/**
 * MinSync is the only retrieval method that searches document content
 * semantically, so AutoRAG treats it as required: when it cannot be used there
 * is no degraded mode that answers from other retrieval paths. This error is
 * the one failure that is never swallowed per-method; it ends the operation and
 * reaches the operator with the reason attached.
 */
export class MinSyncRequiredError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "MinSyncRequiredError";
	}
}

/** Raised when the `minsync` executable cannot be resolved. Path-free on purpose. */
export function minSyncBinaryMissingError(): MinSyncRequiredError {
	return new MinSyncRequiredError(
		"MinSync is required but the `minsync` binary was not found on PATH or in the workspace .autorag/bin cache. " +
			"Install it with `cargo install minsync`, or leave `minSync.autoInstall` enabled and run `autorag refresh`.",
	);
}

/** Raised when auto-install ran and failed; carries the installer's own message. */
export function minSyncInstallFailedError(cause: unknown): MinSyncRequiredError {
	const detail = cause instanceof Error ? cause.message : String(cause);
	return new MinSyncRequiredError(
		`MinSync is required but installing the \`minsync\` binary failed: ${detail}. ` +
			"Install it manually with `cargo install minsync`.",
	);
}
