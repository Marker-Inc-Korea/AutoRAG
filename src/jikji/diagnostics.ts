import type { JikjiFailureReason, JikjiPrepareResult } from "./types.ts";

/**
 * Diagnostic codes for the optional Jikji indexing/preparation layer. These are
 * a subset of the search-level diagnostic code union and are intentionally
 * path-free so they can be surfaced through public search/refresh diagnostics
 * without leaking filesystem or binary paths.
 */
export type JikjiDiagnosticCode = "jikji-unavailable" | "jikji-prepare-failed";

export interface JikjiDiagnostic {
	readonly code: JikjiDiagnosticCode;
	readonly severity: "warning" | "error";
	readonly message: string;
	readonly source: "jikji";
}

/**
 * Reasons that mean the Jikji binary could not be started at all — i.e. the
 * optional dependency is unavailable rather than the run itself failing.
 */
const UNAVAILABLE_REASONS: ReadonlySet<JikjiFailureReason> = new Set(["spawn-error"]);

function describeReason(reason: JikjiFailureReason): string {
	switch (reason) {
		case "spawn-error":
			return "the Jikji binary could not be started (missing or not executable)";
		case "timeout":
			return "the Jikji prepare run timed out";
		case "aborted":
			return "the Jikji prepare run was aborted";
		case "nonzero-exit":
			return "the Jikji prepare run exited with a nonzero status";
		case "stdout-too-large":
			return "the Jikji prepare run produced too much stdout";
		case "stderr-too-large":
			return "the Jikji prepare run produced too much stderr";
		default:
			return "the Jikji prepare run failed";
	}
}

/**
 * Maps a {@link JikjiPrepareResult} to a path-opaque degraded-mode diagnostic.
 * Successful results map to `undefined`. The message never echoes the binary
 * path, stderr, or any filesystem path so it is safe for public diagnostics.
 */
export function jikjiPrepareDiagnostic(result: JikjiPrepareResult): JikjiDiagnostic | undefined {
	if (result.ok) return undefined;
	const unavailable = UNAVAILABLE_REASONS.has(result.reason);
	return {
		code: unavailable ? "jikji-unavailable" : "jikji-prepare-failed",
		severity: "warning",
		message: `Jikji indexing is degraded: ${describeReason(result.reason)}.`,
		source: "jikji",
	};
}
