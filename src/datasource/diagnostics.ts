/**
 * Path-opaque diagnostic mapping for the datasource layer.
 *
 * Datasource operations surface {@link DatasourceDiagnostic} entries (with
 * datasource-specific codes). The agent/search pipeline consumes
 * {@link RetrievalDiagnostic}; these helpers bridge the two while preserving
 * the path-opacity invariant: no real filesystem path may appear in a mapped
 * diagnostic's `source` or `message`.
 *
 * The datasource codes themselves are path-opaque by construction (they
 * describe failure modes, not locations), and the `source` field is reduced
 * to a short opaque label — preferring the instance id, falling back to a
 * sanitized component label, and finally to the generic `"datasource"` label
 * — so a stray path can never leak into the agent-facing diagnostic stream.
 */

import type { RetrievalDiagnostic, RetrievalDiagnosticCode } from "../retrieval/types.ts";
import type { DatasourceDiagnostic, DatasourceDiagnosticCode } from "./types.ts";

/**
 * Map a single {@link DatasourceDiagnostic} to a path-opaque
 * {@link RetrievalDiagnostic}.
 *
 * The original datasource code is preserved as a `[datasource:<code>]` prefix
 * in the message (datasource codes collapse onto the generic
 * `retrieval-method-failed` code in the retrieval stream's closed union).
 */
export function mapDatasourceDiagnostic(ds: DatasourceDiagnostic): RetrievalDiagnostic {
	return {
		code: datasourceCodeToRetrieval(ds.code),
		severity: ds.severity,
		message: `[datasource:${ds.code}] ${sanitizeDiagnosticMessage(ds.message)}`,
		source: opaqueSourceLabel(ds.source, ds.instanceId),
	};
}

/** Map a list of datasource diagnostics to retrieval diagnostics. */
export function mapDatasourceDiagnostics(diagnostics: readonly DatasourceDiagnostic[]): RetrievalDiagnostic[] {
	return diagnostics.map(mapDatasourceDiagnostic);
}

/**
 * Reduce a datasource diagnostic `source` to a path-opaque label.
 *
 * Preference order:
 *  1. `datasource:<instanceId>` — when an instance id is present and is not
 *     itself a path (instance ids are short slugs like `acct-1`).
 *  2. the original `source` label, when it contains no path separators.
 *  3. the generic `"datasource"` label.
 *
 * Any value containing `/` or `\`, or a Windows drive letter, is treated as
 * a path leak and discarded in favor of the generic label.
 */
export function opaqueSourceLabel(source: string | undefined, instanceId?: string): string | undefined {
	if (
		typeof instanceId === "string" &&
		instanceId.length > 0 &&
		!instanceId.includes("/") &&
		!instanceId.includes("\\")
	) {
		return `datasource:${instanceId}`;
	}
	if (
		typeof source === "string" &&
		source.length > 0 &&
		!source.includes("/") &&
		!source.includes("\\") &&
		!looksLikeDriveLetter(source)
	) {
		return source;
	}
	return "datasource";
}

/**
 * Whether a mapped datasource diagnostic carries a path-opaque `source`.
 * Useful for agent/search-layer assertions that no path leaked.
 */
export function isPathOpaqueRetrievalSource(source: string | undefined): boolean {
	if (source === undefined) return true;
	if (source.includes("/") || source.includes("\\")) return false;
	return !looksLikeDriveLetter(source);
}

function looksLikeDriveLetter(value: string): boolean {
	return /^[a-zA-Z]:[\\/]/u.test(value);
}

function sanitizeDiagnosticMessage(message: string): string {
	if (message.includes("/") || message.includes("\\") || looksLikeDriveLetter(message)) {
		return "Datasource operation failed; details suppressed for datasource privacy.";
	}
	return message;
}

function datasourceCodeToRetrieval(code: DatasourceDiagnosticCode): RetrievalDiagnosticCode {
	// Datasource codes have no direct counterpart in the retrieval diagnostic
	// union; collapse onto the generic failure code. The original code is
	// preserved in the mapped message via the `[datasource:<code>]` prefix.
	switch (code) {
		case "datasource-unavailable":
		case "datasource-cli-error":
		case "datasource-empty":
		case "datasource-rate-limited":
		case "datasource-auth-error":
		case "datasource-embedding-egress-rejected":
		case "datasource-index-failed":
		case "datasource-permission-denied":
		case "datasource-remote-embedding-rejected":
			return "retrieval-method-failed";
	}
}
