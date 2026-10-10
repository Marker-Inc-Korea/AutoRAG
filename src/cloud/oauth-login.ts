import type { OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import { buildAuthorizeUrl } from "./authorize.ts";
import { parseCallbackInput, parseCallbackQuery, requireAuthorizationCode } from "./callback.ts";
import { DEFAULT_LOGIN_TIMEOUT_MS, startCallbackServer } from "./callback-server.ts";
import { createPkcePair, createState } from "./pkce.ts";
import { exchangeCodeForToken } from "./token.ts";

export interface LoginOptions {
	/** Server root, e.g. `https://api.dazziapp.com`. */
	baseUrl: string;
	/** Loopback wait budget; defaults to {@link DEFAULT_LOGIN_TIMEOUT_MS}. */
	timeoutMs?: number;
	fetchImpl?: typeof fetch;
}

/**
 * AutoRAG tokens are long-lived revocable API keys: there is no refresh token
 * and no expiry, so a refresh returns the stored credentials unchanged.
 */
export function refreshAutoragCredentials(credentials: OAuthCredentials): OAuthCredentials {
	return credentials;
}

/**
 * OAuth2 authorization-code + PKCE (S256) login for native apps.
 *
 * The browser is sent to `<root>/desktop/authorize` with a loopback redirect
 * URI. When the browser cannot reach the loopback server, the user can paste
 * the final redirect URL (or the bare code) into the prompt instead; both paths
 * are raced and the first result wins.
 */
export async function loginAutorag(callbacks: OAuthLoginCallbacks, options: LoginOptions): Promise<OAuthCredentials> {
	const { verifier, challenge } = createPkcePair();
	const state = createState();
	const server = await startCallbackServer({
		timeoutMs: options.timeoutMs ?? DEFAULT_LOGIN_TIMEOUT_MS,
		signal: callbacks.signal,
	});

	try {
		callbacks.onAuth({
			url: buildAuthorizeUrl({
				baseUrl: options.baseUrl,
				redirectUri: server.redirectUri,
				state,
				codeChallenge: challenge,
			}),
			instructions: `Waiting for the AutoRAG sign-in callback on ${server.redirectUri}. If the browser cannot reach this machine, paste the final redirect URL or the code instead.`,
		});
		callbacks.onProgress?.(`Listening for the AutoRAG sign-in callback on ${server.redirectUri}`);

		const outcome = await Promise.race([
			server.waitForCallback().then((params) => ({ params }) as const),
			callbacks
				.onPrompt({
					message:
						"Complete the AutoRAG sign-in in your browser, or paste the final redirect URL / authorization code here:",
					placeholder: server.redirectUri,
					allowEmpty: true,
				})
				.then((value) => ({ value }) as const),
		]);

		const parsed = "params" in outcome ? parseCallbackQuery(outcome.params) : parseCallbackInput(outcome.value);
		const { code } = requireAuthorizationCode(parsed, state);

		const token = await exchangeCodeForToken({
			baseUrl: options.baseUrl,
			code,
			codeVerifier: verifier,
			redirectUri: server.redirectUri,
			signal: callbacks.signal,
			fetchImpl: options.fetchImpl,
		});

		return {
			access: token.accessToken,
			refresh: "",
			expires: Number.MAX_SAFE_INTEGER,
			...(token.email !== undefined ? { email: token.email } : {}),
		};
	} finally {
		server.close();
	}
}
