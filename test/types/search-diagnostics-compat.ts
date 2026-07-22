// Compile-only type-compatibility fixture (NOT a vitest file — vitest only runs
// *.test.ts). Typechecked by `npm run typecheck` because tsconfig includes
// test/**/*.ts. Proves the #21 public API migration is source-compatible:
//  - all new diagnostic types are importable from the PACKAGE ROOT,
//  - `diagnostics` is OPTIONAL on SearchDocumentsResponse (may be omitted),
//  - the legacy `warnings` union is unchanged.
import type {
	SearchDocumentDiagnostic,
	SearchDocumentDiagnosticCode,
	SearchDocumentDiagnosticSeverity,
	SearchDocumentsResponse,
	SearchDocumentWarning,
} from "../../src/index.ts";

// Legacy narrow warning union is unchanged and still assignable.
export const legacyWarning: SearchDocumentWarning = "empty-query";

// diagnostics OMITTED — must still compile (optional for the compatibility window).
export const responseWithoutDiagnostics: SearchDocumentsResponse = {
	sessionId: "s",
	query: "q",
	results: [],
	answer: "",
	searched: 0,
	warnings: ["empty-query"],
};

const severity: SearchDocumentDiagnosticSeverity = "warning";
const code: SearchDocumentDiagnosticCode = "unknown-warning";
const diagnostic: SearchDocumentDiagnostic = { code, severity, message: "m", source: "sanitizer" };

// Runtime-shaped response WITH diagnostics populated.
export const responseWithDiagnostics: SearchDocumentsResponse = {
	sessionId: "s",
	query: "q",
	results: [],
	answer: "",
	searched: 0,
	warnings: [],
	diagnostics: [diagnostic],
};
