import { describe, expect, it } from "vitest";
import {
	base64UrlEncode,
	createCodeChallenge,
	createCodeVerifier,
	createPkcePair,
	createState,
} from "../../src/cloud/pkce.ts";

describe("PKCE", () => {
	it("matches the RFC 7636 appendix B reference vector", () => {
		expect(createCodeChallenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")).toBe(
			"E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
		);
	});

	it("encodes without padding and with the url-safe alphabet", () => {
		expect(base64UrlEncode(new Uint8Array([251, 255, 191]))).toBe("-_-_");
	});

	it("generates a 43-character verifier and a state of the requested length", () => {
		expect(createCodeVerifier()).toHaveLength(43);
		expect(createState()).toMatch(/^[A-Za-z0-9_-]+$/);
	});

	it("pairs a verifier with its S256 challenge", () => {
		const pair = createPkcePair();
		expect(pair.challenge).toBe(createCodeChallenge(pair.verifier));
	});
});
