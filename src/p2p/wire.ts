import { createHash } from "node:crypto";
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
		description: "Opaque wire source id (slugged virtual path matching the wire source regex)",
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
		pattern: "^/[a-z0-9-]+(/|$)",
		description: "Opaque wire source id — slugged virtual path matching ^/[a-z0-9-]+(/|$)",
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
// src/datasource/connector.ts:86-99)
// ---------------------------------------------------------------------------

/**
 * Slugify a single path segment: lowercase, replace runs of non-[a-z0-9] with
 * '-', trim leading/trailing '-', and append an 8-char sha256 hex suffix of the
 * original segment when slugging actually changed it.
 */
function slugSegment(segment: string): string {
	const lowered = segment.toLowerCase();
	const slugged = lowered.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
	if (slugged === lowered && slugged.length > 0) {
		// unchanged and non-empty — return as-is
		return slugged;
	}
	const hash = createHash("sha256").update(segment).digest("hex").slice(0, 8);
	return slugged.length > 0 ? `${slugged}-${hash}` : hash;
}

// ---------------------------------------------------------------------------
// Mapping stores: wireSourceId -> virtualPath, virtualPath -> wireSourceId
// ---------------------------------------------------------------------------

const wireToVirtual = new Map<string, string>();
const virtualToWire = new Map<string, string>();

/**
 * Compute the wire source id for a virtual path and record the mapping for
 * reverse resolution.
 *
 * wireSourceId(virtualPath) = '/' + slug(rootPrefix) + '/' + slug(relativePath)
 *
 * The result is a deterministic opaque id that matches ^/[a-z0-9-]+(/|$) and
 * can be round-tripped via wireSourceIdToVirtualPath.
 *
 * Throws if the virtual path does not start with '/'.
 */
export function wireSourceId(virtualPath: string): string {
	if (!virtualPath.startsWith("/")) {
		throw new Error(`wireSourceId: virtual path must start with '/', got: ${virtualPath}`);
	}

	if (virtualToWire.has(virtualPath)) {
		return virtualToWire.get(virtualPath)!;
	}

	// Normalize: collapse repeated slashes, trim trailing slash
	const normalized = virtualPath.replace(/\/+/g, "/").replace(/\/$/, "");
	if (normalized === "/" || normalized.length === 0) {
		const id = "/";
		virtualToWire.set(virtualPath, id);
		wireToVirtual.set(id, virtualPath);
		return id;
	}

	// Split into segments: strip leading /
	const segments = normalized.slice(1).split("/");
	const sluggedSegments = segments.map(slugSegment);
	const id = `/${sluggedSegments.join("/")}`;

	virtualToWire.set(virtualPath, id);
	wireToVirtual.set(id, virtualPath);
	return id;
}

/**
 * Reverse-resolve a wire source id back to its original virtual path.
 * Returns undefined for unknown ids.
 */
export function wireSourceIdToVirtualPath(wireId: string): string | undefined {
	// Normalize: collapse slashes, trim trailing
	const normalized = wireId.replace(/\/+/g, "/").replace(/\/$/, "") || "/";
	return wireToVirtual.get(normalized);
}

/**
 * Reset the wire id mapping. Useful for testing isolation.
 */
export function resetWireMapping(): void {
	wireToVirtual.clear();
	virtualToWire.clear();
}
