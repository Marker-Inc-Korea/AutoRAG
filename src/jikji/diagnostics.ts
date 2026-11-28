import type { JikjiDoctorResult, JikjiFailureReason, JikjiFindResult, JikjiPrepareResult } from "./types.ts";

/**
 * Diagnostic codes for the optional Jikji indexing/preparation layer. These are
 * a subset of the search-level diagnostic code union. Index-health diagnostics
 * retain native doctor details, including paths, so users can identify repairs.
 */
export type JikjiDiagnosticCode =
	| "jikji-unavailable"
	| "jikji-prepare-failed"
	| "jikji-cloud-root-skipped"
	| "jikji-find-failed"
	| "jikji-discover-failed"
	| "jikji-not-prepared"
	| "jikji-index-unhealthy"
	| "jikji-documents-failed"
	| "jikji-doctor-failed"
	| "jikji-unsupported-version";

/** Index-health diagnostics retain native doctor errors and warnings verbatim. */
export function jikjiDoctorDiagnostics(result: JikjiDoctorResult): JikjiDiagnostic[] {
	const diagnostics: JikjiDiagnostic[] = [];
	if (result.state === "not-prepared") {
		diagnostics.push({
			code: "jikji-not-prepared",
			severity: "info",
			message: "Jikji index is not prepared.",
			source: "jikji",
		});
	}
	for (const message of result.errors) {
		diagnostics.push({
			code: result.state === "unavailable" ? "jikji-doctor-failed" : "jikji-index-unhealthy",
			severity: "warning",
			message,
			source: "jikji",
		});
	}
	for (const message of result.warnings) {
		diagnostics.push({ code: "jikji-index-unhealthy", severity: "warning", message, source: "jikji" });
	}
	if (result.state === "needs-reprepare" && result.errors.length === 0) {
		diagnostics.push({
			code: "jikji-index-unhealthy",
			severity: "warning",
			message: "Jikji index needs re-prepare.",
			source: "jikji",
		});
	}
	if (result.docsFailed !== undefined && result.docsFailed > 0) {
		diagnostics.push({
			code: "jikji-documents-failed",
			severity: "warning",
			message: `Jikji manifest reports docs_failed=${result.docsFailed}.`,
			source: "jikji",
		});
	}
	return diagnostics;
}

export interface JikjiDiagnostic {
	readonly code: JikjiDiagnosticCode;
	readonly severity: "info" | "warning" | "error";
	readonly message: string;
	readonly source: "jikji";
}

/**
 * Reasons that mean the Jikji binary could not be started at all — i.e. the
 * optional dependency is unavailable rather than the run itself failing.
 */
const UNAVAILABLE_REASONS: ReadonlySet<JikjiFailureReason> = new Set(["spawn-error"]);

function describeReason(reason: JikjiFailureReason, action: "prepare" | "find" | "discover" = "prepare"): string {
	switch (reason) {
		case "spawn-error":
			return "the Jikji binary could not be started (missing or not executable)";
		case "timeout":
			return `the Jikji ${action} run timed out`;
		case "aborted":
			return `the Jikji ${action} run was aborted`;
		case "nonzero-exit":
			return `the Jikji ${action} run exited with a nonzero status`;
		case "stdout-too-large":
			return `the Jikji ${action} run produced too much stdout`;
		case "stderr-too-large":
			return `the Jikji ${action} run produced too much stderr`;
		case "bad-answer-pack":
			return "the Jikji find run returned an unparseable answer pack";
		default:
			return `the Jikji ${action} run failed`;
	}
}

/**
 * Maps a {@link JikjiPrepareResult} to a path-opaque degraded-mode diagnostic.
 * Successful results map to `undefined`. The message never echoes the binary
 * path, stderr, or any filesystem path so it is safe for public diagnostics.
 */
export function jikjiPrepareDiagnostic(result: JikjiPrepareResult): JikjiDiagnostic | undefined {
	if (result.ok) return undefined;
	if (result.reason === "cloud-root-skipped") {
		return {
			code: "jikji-cloud-root-skipped",
			severity: "info",
			message:
				"Jikji was not run for this root because it is a cloud or network folder; its files are indexed through the parsed mirror, and reading them with Jikji would download released placeholders again or re-read a network share.",
			source: "jikji",
		};
	}
	if (result.reason === "unsupported-version") return unsupportedVersionDiagnostic(result.stderr);
	const unavailable = UNAVAILABLE_REASONS.has(result.reason);
	return {
		code: unavailable ? "jikji-unavailable" : "jikji-prepare-failed",
		severity: "warning",
		message: `Jikji indexing is degraded: ${describeReason(result.reason)}.`,
		source: "jikji",
	};
}

/**
 * Maps a {@link JikjiFindResult} to a path-opaque degraded-mode diagnostic.
 * Successful results map to `undefined`. The message never echoes the binary
 * path, stderr, or any filesystem path so it is safe for public diagnostics.
 * The one exception is an unsupported version, see {@link unsupportedVersionDiagnostic}.
 */
export function jikjiFindDiagnostic(result: JikjiFindResult): JikjiDiagnostic | undefined {
	if (result.ok) return undefined;
	if (result.reason === "unsupported-version") return unsupportedVersionDiagnostic(result.stderr);
	const unavailable = UNAVAILABLE_REASONS.has(result.reason);
	return {
		code: unavailable ? "jikji-unavailable" : "jikji-find-failed",
		severity: "warning",
		message: `Jikji find is degraded: ${describeReason(result.reason, "find")}.`,
		source: "jikji",
	};
}

/**
 * Maps a failed `discover` retry step to a diagnostic. Unlike find/prepare
 * diagnostics it is deliberately NOT path-opaque: the retry step is the one
 * place AutoRAG runs jikji on the model's behalf after a miss, so a failure
 * must be debuggable and carries the CLI exit status and stderr verbatim,
 * real paths included. Successful results map to `undefined`.
 */
export function jikjiDiscoverDiagnostic(result: JikjiFindResult): JikjiDiagnostic | undefined {
	if (result.ok) return undefined;
	if (UNAVAILABLE_REASONS.has(result.reason)) {
		return {
			code: "jikji-unavailable",
			severity: "warning",
			message: `Jikji discover is degraded: ${describeReason(result.reason, "discover")}. ${result.stderr}`.trim(),
			source: "jikji",
		};
	}
	const reason =
		result.reason === "bad-retry-step"
			? "jikji's retry step was issued for another root and was not run"
			: describeReason(result.reason, "discover");
	const status = result.code === null ? "no exit status" : `exit status ${result.code}`;
	const stderr = result.stderr.trim();
	return {
		code: "jikji-discover-failed",
		severity: "warning",
		message: `Jikji discover retry failed: ${reason} (${status})${stderr.length > 0 ? `: ${stderr}` : ""}`,
		source: "jikji",
	};
}

/**
 * `jikji_retry` named follow-up commands AutoRAG could not turn into a
 * `discover` step. The commands are quoted verbatim so the shape can be fixed.
 */
export function jikjiUnusableRetryDiagnostic(nextCommands: readonly string[]): JikjiDiagnostic {
	return {
		code: "jikji-discover-failed",
		severity: "warning",
		message: `Jikji asked for a discover retry that AutoRAG could not run: ${nextCommands.join(" ; ")}`,
		source: "jikji",
	};
}

/**
 * An unsupported version carries the client's own message verbatim: the detected version,
 * the binary's real path and the required floor. That text is operator-actionable ("which
 * jikji is stale, and where") and AGENTS.md "Error Transparency" forbids replacing it with
 * a generic sentence or dropping it because it contains a path.
 */
function unsupportedVersionDiagnostic(message: string): JikjiDiagnostic {
	return {
		code: "jikji-unsupported-version",
		severity: "warning",
		message: `Jikji is degraded: ${message}`,
		source: "jikji",
	};
}
