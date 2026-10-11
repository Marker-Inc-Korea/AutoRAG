import { AUTHORIZE_PATH, normalizeBaseUrl, OAUTH_CLIENT_ID } from "./config.ts";

export interface AuthorizeUrlParams {
	/** Server root, e.g. `https://api.dazziapp.com`. */
	baseUrl: string;
	/** Loopback redirect URI chosen by the login flow. */
	redirectUri: string;
	/** Random `state` echoed back on the redirect. */
	state: string;
	/** S256 code challenge derived from the PKCE verifier. */
	codeChallenge: string;
}

/** Build the browser authorization URL for the PKCE authorization-code flow. */
export function buildAuthorizeUrl(params: AuthorizeUrlParams): string {
	const url = new URL(`${normalizeBaseUrl(params.baseUrl)}${AUTHORIZE_PATH}`);
	url.searchParams.set("client", OAUTH_CLIENT_ID);
	url.searchParams.set("redirect_uri", params.redirectUri);
	url.searchParams.set("state", params.state);
	url.searchParams.set("code_challenge", params.codeChallenge);
	url.searchParams.set("code_challenge_method", "S256");
	return url.toString();
}
