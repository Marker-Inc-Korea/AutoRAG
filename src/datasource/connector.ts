/**
 * Connector contract shared by the built-in datasource skills (Slack, Discord,
 * Notion, GitHub, Google Drive, Gmail/IMAP, local mail exports, Obsidian,
 * RSS/news).
 *
 * A *connector* is the trusted, server-configured bridge to one external
 * system. It fetches documents for indexing and NEVER throws — every failure
 * is reported as a discriminated `ok: false` result with a coarse
 * {@link ConnectorFailureReason} and an already-sanitized, path/PII-opaque
 * message. Connectors are constructed from server-supplied configuration only;
 * model/tool arguments never reach a connector.
 */

import { createHash } from "node:crypto";
import type { DatasourceDiagnosticCode } from "./types.ts";

/** Coarse failure classes a connector can report. */
export type ConnectorFailureReason =
	| "not-configured"
	| "auth"
	| "permission"
	| "rate-limited"
	| "unavailable"
	| "api-error"
	| "invalid-data"
	| "empty";

/**
 * One document fetched from the external system. `docId` and `hierarchy`
 * segments are sanitized into opaque slash-safe segments before they become
 * part of a source path.
 */
export interface ConnectorDocument {
	/** Stable, unique id within the connector instance (pre-sanitization). */
	readonly docId: string;
	/**
	 * Optional hierarchy segments under the instance root, e.g.
	 * `["channels", "general"]` for a Slack channel or
	 * `["folders", "projects"]` for a vault folder.
	 */
	readonly hierarchy?: readonly string[];
	readonly title?: string;
	readonly content: string;
	/** Epoch milliseconds of the document's own timestamp when known. */
	readonly publishedAt?: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ConnectorFetchOk {
	readonly ok: true;
	readonly documents: readonly ConnectorDocument[];
	/** Sanitized, path/PII-opaque warning strings. */
	readonly warnings?: readonly string[];
}

export interface ConnectorFetchFail {
	readonly ok: false;
	readonly reason: ConnectorFailureReason;
	/** Sanitized, path/PII-opaque failure message. */
	readonly message?: string;
}

export type ConnectorFetchResult = ConnectorFetchOk | ConnectorFetchFail;

/**
 * The trusted bridge to one external system. Implementations never throw from
 * `fetch` — they return an `ok: false` result instead.
 */
export interface DatasourceConnector {
	fetch(signal?: AbortSignal): Promise<ConnectorFetchResult>;
}

/** Map a connector failure reason onto the datasource diagnostic union. */
export function connectorFailureToDiagnosticCode(reason: ConnectorFailureReason): DatasourceDiagnosticCode {
	switch (reason) {
		case "auth":
			return "datasource-auth-error";
		case "permission":
			return "datasource-permission-denied";
		case "rate-limited":
			return "datasource-rate-limited";
		case "unavailable":
		case "not-configured":
			return "datasource-unavailable";
		case "empty":
			return "datasource-empty";
		case "api-error":
		case "invalid-data":
			return "datasource-index-failed";
	}
}

const OPAQUE_SUPPRESSED = "datasource operation failed; details suppressed for datasource privacy";

/**
 * Sanitize free-form diagnostic text so it stays path- and PII-opaque.
 * Anything that looks like a filesystem path, URL, email address, or long
 * token is replaced with a generic suppression notice.
 */
export function sanitizeOpaqueText(value: string): string {
	const trimmed = value.trim();
	if (trimmed.length === 0) return OPAQUE_SUPPRESSED;
	if (
		trimmed.includes("/") ||
		trimmed.includes("\\") ||
		trimmed.includes("@") ||
		/[A-Za-z]:[\\/]/u.test(trimmed) ||
		/https?:/iu.test(trimmed) ||
		/[A-Za-z0-9_-]{30,}/u.test(trimmed)
	) {
		return OPAQUE_SUPPRESSED;
	}
	return trimmed.length > 200 ? `${trimmed.slice(0, 200)}…` : trimmed;
}

/**
 * Sanitize an arbitrary string into a slash-safe source-path segment.
 * Unicode letters/digits (e.g. Hangul channel or folder names) are kept so
 * scopes stay human-readable; separators, punctuation, and control chars
 * collapse to `-`. When sanitization loses information, a short hash of the
 * original value is appended so distinct inputs stay distinct.
 */
export function sanitizeIdSegment(value: string): string {
	const cleaned = value.replace(/[^\p{L}\p{N}._-]+/gu, "-").replace(/^[-.]+|[-.]+$/gu, "");
	if (cleaned === value && cleaned.length > 0 && cleaned.length <= 80) return cleaned;
	const hash = createHash("sha256").update(value).digest("hex").slice(0, 8);
	const stem = cleaned.slice(0, 60);
	return stem.length > 0 ? `${stem}-${hash}` : hash;
}
