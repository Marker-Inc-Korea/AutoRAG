/**
 * Minimal HTTP helper shared by the API-backed datasource connectors.
 *
 * Every request is bounded by a timeout, honors an optional outer abort
 * signal, and maps failures onto the coarse {@link ConnectorFailureReason}
 * union with short, path/PII-opaque messages (never URLs, tokens, or bodies).
 */

import type { ConnectorFailureReason } from "./connector.ts";

export interface HttpRequestOptions {
	readonly fetchImpl?: typeof fetch;
	readonly timeoutMs?: number;
	readonly signal?: AbortSignal;
	readonly headers?: Readonly<Record<string, string>>;
	readonly method?: string;
	readonly body?: string;
}

export type HttpJsonResult =
	| { readonly ok: true; readonly status: number; readonly json: unknown }
	| { readonly ok: false; readonly reason: ConnectorFailureReason; readonly message: string };

export type HttpTextResult =
	| { readonly ok: true; readonly status: number; readonly text: string }
	| { readonly ok: false; readonly reason: ConnectorFailureReason; readonly message: string };

const DEFAULT_TIMEOUT_MS = 15_000;

function combineSignals(timeoutMs: number, outer?: AbortSignal): AbortSignal {
	const timeout = AbortSignal.timeout(timeoutMs);
	return outer === undefined ? timeout : AbortSignal.any([timeout, outer]);
}

/** Map an HTTP status code onto a connector failure reason. */
export function statusToReason(status: number): ConnectorFailureReason {
	if (status === 401) return "auth";
	if (status === 403) return "permission";
	if (status === 429) return "rate-limited";
	return "api-error";
}

async function requestText(url: string, options: HttpRequestOptions): Promise<HttpTextResult> {
	const fetchImpl = options.fetchImpl ?? globalThis.fetch;
	const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
	let response: Response;
	try {
		response = await fetchImpl(url, {
			method: options.method ?? "GET",
			headers: options.headers,
			body: options.body,
			signal: combineSignals(timeoutMs, options.signal),
		});
	} catch {
		return { ok: false, reason: "unavailable", message: "request failed or timed out" };
	}
	if (!response.ok) {
		return { ok: false, reason: statusToReason(response.status), message: `http-${response.status}` };
	}
	let text: string;
	try {
		text = await response.text();
	} catch {
		return { ok: false, reason: "invalid-data", message: "response body unreadable" };
	}
	return { ok: true, status: response.status, text };
}

/** GET/POST a URL and return the body text; never throws. */
export function httpText(url: string, options: HttpRequestOptions = {}): Promise<HttpTextResult> {
	return requestText(url, options);
}

/** GET/POST a URL and parse the body as JSON; never throws. */
export async function httpJson(url: string, options: HttpRequestOptions = {}): Promise<HttpJsonResult> {
	const result = await requestText(url, options);
	if (!result.ok) return result;
	try {
		return { ok: true, status: result.status, json: JSON.parse(result.text) };
	} catch {
		return { ok: false, reason: "invalid-data", message: "response was not valid JSON" };
	}
}

/** Narrowing helpers for defensive JSON traversal. */
export function asRecord(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

export function asArray(value: unknown): readonly unknown[] {
	return Array.isArray(value) ? value : [];
}

export function asString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

export function asNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** Parse a date-ish string to epoch ms, or undefined. */
export function parseEpochMs(value: unknown): number | undefined {
	if (typeof value !== "string" || value.length === 0) return undefined;
	const parsed = Date.parse(value);
	return Number.isNaN(parsed) ? undefined : parsed;
}

/** Resolve an auth token from explicit config or an environment variable name. */
export function resolveToken(token: string | undefined, tokenEnv: string | undefined): string | undefined {
	if (token !== undefined && token.length > 0) return token;
	if (tokenEnv !== undefined && tokenEnv.length > 0) {
		const fromEnv = process.env[tokenEnv];
		if (fromEnv !== undefined && fromEnv.length > 0) return fromEnv;
	}
	return undefined;
}
