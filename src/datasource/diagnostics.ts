/**
 * Diagnostic mapping for the datasource layer.
 *
 * Datasource operations surface {@link DatasourceDiagnostic} entries (with
 * datasource-specific codes). The agent/search pipeline consumes
 * {@link RetrievalDiagnostic}; these helpers bridge the two. Messages and
 * sources are passed through verbatim.
 */

import type { RetrievalDiagnostic, RetrievalDiagnosticCode } from "../retrieval/types.ts";
import type { DatasourceDiagnostic, DatasourceDiagnosticCode } from "./types.ts";

/**
 * Map a single {@link DatasourceDiagnostic} to a {@link RetrievalDiagnostic}.
 *
 * The original datasource code is preserved as a `[datasource:<code>]` prefix
 * in the message (datasource codes collapse onto the generic
 * `retrieval-method-failed` code in the retrieval stream's closed union).
 */
export function mapDatasourceDiagnostic(ds: DatasourceDiagnostic): RetrievalDiagnostic {
	return {
		code: datasourceCodeToRetrieval(ds.code),
		severity: ds.severity,
		message: `[datasource:${ds.code}] ${ds.message}`,
		source: ds.source ?? (ds.instanceId ? `datasource:${ds.instanceId}` : "datasource"),
	};
}

/** Map a list of datasource diagnostics to retrieval diagnostics. */
export function mapDatasourceDiagnostics(diagnostics: readonly DatasourceDiagnostic[]): RetrievalDiagnostic[] {
	return diagnostics.map(mapDatasourceDiagnostic);
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
