import { AutoragOAuthError } from "./errors.ts";

/** Authorization response extracted from either the loopback server or a manual paste. */
export interface CallbackResult {
	code?: string;
	error?: string;
	state?: string;
}

/** Read an authorization response out of query parameters. */
export function parseCallbackQuery(params: URLSearchParams): CallbackResult {
	const result: CallbackResult = {};
	const code = params.get("code");
	const error = params.get("error");
	const state = params.get("state");
	if (code !== null && code !== "") result.code = code;
	if (error !== null && error !== "") result.error = error;
	if (state !== null && state !== "") result.state = state;
	return result;
}

/**
 * Parse what the user pasted when the browser could not reach the loopback
 * server: a full redirect URL, a bare `code=...&state=...` query, or just the
 * authorization code.
 */
export function parseCallbackInput(input: string): CallbackResult {
	const trimmed = input.trim();
	if (trimmed === "") {
		throw new AutoragOAuthError("AutoRAG sign-in failed: nothing was pasted.");
	}
	if (/^https?:\/\//i.test(trimmed)) {
		try {
			return parseCallbackQuery(new URL(trimmed).searchParams);
		} catch {
			throw new AutoragOAuthError("AutoRAG sign-in failed: the pasted URL is not valid.");
		}
	}
	const query = trimmed.startsWith("?") ? trimmed.slice(1) : trimmed;
	if (!query.includes(" ")) {
		const params = new URLSearchParams(query);
		if (params.has("code") || params.has("error") || params.has("state")) {
			return parseCallbackQuery(params);
		}
	}
	return { code: trimmed };
}

/**
 * Validate an authorization response against the expected `state` and return
 * the authorization code. `state` is compared whenever the response carries
 * one; a mismatch always aborts the login. A manual code paste carries no
 * `state`, so there is nothing to compare there.
 */
export function requireAuthorizationCode(result: CallbackResult, expectedState: string): { code: string } {
	if (result.state !== undefined && result.state !== expectedState) {
		throw new AutoragOAuthError(
			"AutoRAG sign-in failed: state mismatch (the callback did not originate from this login attempt).",
		);
	}
	if (result.error !== undefined) {
		throw new AutoragOAuthError(
			result.error === "access_denied"
				? "AutoRAG sign-in was cancelled in the browser."
				: `AutoRAG sign-in failed: ${result.error}.`,
		);
	}
	if (result.code === undefined) {
		throw new AutoragOAuthError("AutoRAG sign-in failed: the callback carried no authorization code.");
	}
	return { code: result.code };
}
