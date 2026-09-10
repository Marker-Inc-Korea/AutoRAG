import { type Static, Type } from "typebox";

// ---------------------------------------------------------------------------
// Refusal codes (typed union)
// ---------------------------------------------------------------------------

export const PeerRefusalCodeSchema = Type.Union([
	Type.Literal("auth-error"),
	Type.Literal("replay-rejected"),
	Type.Literal("rate-limited"),
	Type.Literal("injection-detected"),
	Type.Literal("policy-denied"),
	Type.Literal("queue-full"),
	Type.Literal("internal-error"),
]);

export type PeerRefusalCode = Static<typeof PeerRefusalCodeSchema>;

const PeerDiagnosticSchema = Type.Object({
	code: Type.String({ minLength: 1, description: "Diagnostic or refusal code" }),
	message: Type.String({ description: "Human-readable diagnostic message" }),
});

// ---------------------------------------------------------------------------
// File response entry
// ---------------------------------------------------------------------------

const PeerFileEntrySchema = Type.Object({
	source: Type.String({
		description: "Canonical retrieval source identifier",
	}),
	contentBase64: Type.String({ description: "Base64-encoded file content" }),
	redacted: Type.Boolean({ description: "Whether the content was PII-redacted" }),
});

// ---------------------------------------------------------------------------
// Result entry
// ---------------------------------------------------------------------------

const PeerResultEntrySchema = Type.Object({
	number: Type.Integer({ minimum: 1, description: "1-based result number" }),
	title: Type.String({ minLength: 1, description: "Short title of the result" }),
	summary: Type.String({ description: "Key insight summary" }),
	source: Type.String({
		description: "Canonical retrieval source identifier",
	}),
	excerpt: Type.String({ description: "Supporting excerpt" }),
});

// ---------------------------------------------------------------------------
// PeerQueryRequest
// ---------------------------------------------------------------------------

export const PeerQueryRequestSchema = Type.Object({
	v: Type.Literal(1, { description: "Protocol version" }),
	query: Type.String({ minLength: 1, maxLength: 4096, description: "Search query (max 4096 chars)" }),
	topK: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, description: "Max results to return (max 20)" })),
	scope: Type.Optional(Type.String({ description: "Optional search scope hint" })),
});

export type PeerQueryRequest = Static<typeof PeerQueryRequestSchema>;

export const PeerQueryResponseSchema = Type.Object({
	v: Type.Literal(1, { description: "Protocol version" }),
	status: Type.Union([Type.Literal("ok"), Type.Literal("rejected")], {
		description: "Response status",
	}),
	answer: Type.String({ description: "Curated answer text" }),
	results: Type.Array(PeerResultEntrySchema, {
		description: "Curated result entries with opaque source ids",
	}),
	files: Type.Array(PeerFileEntrySchema, {
		description: "Attached original files with opaque source ids",
	}),
	diagnostics: Type.Array(PeerDiagnosticSchema, {
		description: "Diagnostic messages (degraded-mode, rejection reasons, etc.)",
	}),
});

export type PeerQueryResponse = Static<typeof PeerQueryResponseSchema>;
/** P2P carries the same canonical source identifier used by retrieval and policy. */
export function wireSourceId(source: string): string {
	if (source.length === 0) throw new Error("wireSourceId: source must not be empty");
	return source;
}

/**
 * Retained as a no-op test/setup hook while the public wire source is the
 * retrieval source itself.
 */
export function resetWireMapping(): void {
}
