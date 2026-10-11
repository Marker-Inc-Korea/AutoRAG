import { normalizeBaseUrl, TOKEN_PATH } from "./config.ts";
import { AutoragHttpError, AutoragOAuthError, extractErrorPayload } from "./errors.ts";

export interface TokenResponse {
	accessToken: string;
	email?: string;
}

export interface ExchangeCodeOptions {
	/** Server root, e.g. `https://api.dazziapp.com`. */
	baseUrl: string;
	code: string;
	codeVerifier: string;
	redirectUri: string;
	signal?: AbortSignal;
	fetchImpl?: typeof fetch;
}

/** Map a failed token-exchange response to an error carrying the HTTP status and server code. */
export function mapTokenExchangeError(status: number, body: unknown): AutoragHttpError {
	const payload = extractErrorPayload(body);
	const reason = payload.message ?? "the server returned no error message";
	return new AutoragHttpError(`AutoRAG token exchange failed (HTTP ${status}): ${reason}`, {
		status,
		code: payload.code,
		type: payload.type,
	});
}

/**
 * Trade a PKCE authorization code for a long-lived API key.
 *
 * `POST <root>/api/desktop/token` with `{ code, code_verifier, redirect_uri }`.
 */
export async function exchangeCodeForToken(options: ExchangeCodeOptions): Promise<TokenResponse> {
	const doFetch = options.fetchImpl ?? fetch;
	let response: Response;
	try {
		response = await doFetch(`${normalizeBaseUrl(options.baseUrl)}${TOKEN_PATH}`, {
			method: "POST",
			headers: { "content-type": "application/json", accept: "application/json" },
			body: JSON.stringify({
				code: options.code,
				code_verifier: options.codeVerifier,
				redirect_uri: options.redirectUri,
			}),
			signal: options.signal,
		});
	} catch (cause) {
		if (options.signal?.aborted) {
			throw new AutoragOAuthError("AutoRAG sign-in was cancelled.", { cause });
		}
		throw new AutoragOAuthError(`AutoRAG token exchange could not reach ${normalizeBaseUrl(options.baseUrl)}.`, {
			cause,
		});
	}

	let body: unknown;
	try {
		body = await response.json();
	} catch {
		body = undefined;
	}

	if (!response.ok) throw mapTokenExchangeError(response.status, body);

	const accessToken = readAccessToken(body);
	if (accessToken === undefined) {
		throw new AutoragOAuthError("AutoRAG token exchange succeeded but returned no access token.");
	}
	const email = readEmail(body);
	return email === undefined ? { accessToken } : { accessToken, email };
}

function readAccessToken(body: unknown): string | undefined {
	if (typeof body !== "object" || body === null) return undefined;
	const value = (body as Record<string, unknown>).access_token;
	return typeof value === "string" && value !== "" ? value : undefined;
}

function readEmail(body: unknown): string | undefined {
	if (typeof body !== "object" || body === null) return undefined;
	const value = (body as Record<string, unknown>).email;
	return typeof value === "string" && value !== "" ? value : undefined;
}
