/**
 * Error types for the hosted AutoRAG (Dazzi) provider.
 *
 * Messages never contain credentials, authorization headers, or provider
 * payloads; they are safe to surface in the CLI UI and in test assertions.
 */

/** Base class for every error raised by the AutoRAG provider. */
export class AutoragError extends Error {
	constructor(message: string, options?: { cause?: unknown }) {
		super(message, options);
		this.name = "AutoragError";
	}
}

/** Raised for OAuth/PKCE/callback failures. */
export class AutoragOAuthError extends AutoragError {
	constructor(message: string, options?: { cause?: unknown }) {
		super(message, options);
		this.name = "AutoragOAuthError";
	}
}

/**
 * Raised when an HTTP call to AutoRAG fails.
 *
 * `status` is the HTTP status code, `code`/`type` come from the OpenAI-shaped
 * error body when the server supplies them.
 */
export class AutoragHttpError extends AutoragError {
	readonly status: number;
	readonly code?: string;
	readonly type?: string;

	constructor(message: string, options: { status: number; code?: string; type?: string; cause?: unknown }) {
		super(message, { cause: options.cause });
		this.name = "AutoragHttpError";
		this.status = options.status;
		if (options.code !== undefined) this.code = options.code;
		if (options.type !== undefined) this.type = options.type;
	}
}

export interface ErrorPayload {
	message?: string;
	code?: string;
	type?: string;
}

/**
 * Extract `{ message, code, type }` from an OpenAI-shaped error body:
 * `{ "error": { "message": ..., "type": ..., "code": ... } }`.
 * Also tolerates `{ "error": "message" }` and a bare `{ "message": ... }`.
 */
export function extractErrorPayload(body: unknown): ErrorPayload {
	if (typeof body !== "object" || body === null) return {};
	const record = body as Record<string, unknown>;
	const error = record.error;
	if (typeof error === "string") return error.length > 0 ? { message: error } : {};
	if (typeof error === "object" && error !== null) {
		const inner = error as Record<string, unknown>;
		const payload: ErrorPayload = {};
		if (typeof inner.message === "string") payload.message = inner.message;
		if (typeof inner.code === "string") payload.code = inner.code;
		if (typeof inner.type === "string") payload.type = inner.type;
		return payload;
	}
	return typeof record.message === "string" ? { message: record.message } : {};
}
