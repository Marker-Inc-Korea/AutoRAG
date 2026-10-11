import { createHash, randomBytes } from "node:crypto";

/** RFC 7636 base64url encoding (unpadded, `-`/`_` alphabet). */
export function base64UrlEncode(bytes: Uint8Array): string {
	return Buffer.from(bytes).toString("base64url");
}

/**
 * `BASE64URL(SHA256(ASCII(code_verifier)))` — the S256 code challenge.
 *
 * RFC 7636 Appendix B reference vector: verifier
 * `dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk` yields challenge
 * `E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM`.
 */
export function createCodeChallenge(verifier: string): string {
	return base64UrlEncode(createHash("sha256").update(verifier, "ascii").digest());
}

/** Random high-entropy PKCE code verifier (43 chars for the default 32 bytes). */
export function createCodeVerifier(byteLength = 32): string {
	return base64UrlEncode(randomBytes(byteLength));
}

/** Random `state` value used to detect CSRF on the OAuth redirect. */
export function createState(byteLength = 24): string {
	return base64UrlEncode(randomBytes(byteLength));
}

export interface PkcePair {
	verifier: string;
	challenge: string;
}

/** Generate a verifier and its matching S256 challenge in one step. */
export function createPkcePair(byteLength = 32): PkcePair {
	const verifier = createCodeVerifier(byteLength);
	return { verifier, challenge: createCodeChallenge(verifier) };
}
