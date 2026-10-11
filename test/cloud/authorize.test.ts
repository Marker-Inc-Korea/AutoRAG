import { describe, expect, it } from "vitest";
import { buildAuthorizeUrl } from "../../src/cloud/authorize.ts";
import { OAUTH_CLIENT_ID } from "../../src/cloud/config.ts";

describe("buildAuthorizeUrl", () => {
	it("builds the PKCE authorization URL with the agent client id", () => {
		const url = new URL(
			buildAuthorizeUrl({
				baseUrl: "https://api.dazziapp.com",
				redirectUri: "http://127.0.0.1:51234/callback",
				state: "state-1",
				codeChallenge: "challenge-1",
			}),
		);

		expect(OAUTH_CLIENT_ID).toBe("agent");
		expect(url.origin).toBe("https://api.dazziapp.com");
		expect(url.pathname).toBe("/desktop/authorize");
		expect(url.searchParams.get("client")).toBe("agent");
		expect(url.searchParams.get("redirect_uri")).toBe("http://127.0.0.1:51234/callback");
		expect(url.searchParams.get("state")).toBe("state-1");
		expect(url.searchParams.get("code_challenge")).toBe("challenge-1");
		expect(url.searchParams.get("code_challenge_method")).toBe("S256");
	});

	it("normalizes a trailing slash and keeps a base path", () => {
		const url = new URL(
			buildAuthorizeUrl({
				baseUrl: "https://staging.example.com/",
				redirectUri: "http://127.0.0.1:1/callback",
				state: "s",
				codeChallenge: "c",
			}),
		);
		expect(url.pathname).toBe("/desktop/authorize");

		const nested = new URL(
			buildAuthorizeUrl({
				baseUrl: "https://staging.example.com/tenant/",
				redirectUri: "http://127.0.0.1:1/callback",
				state: "s",
				codeChallenge: "c",
			}),
		);
		expect(nested.pathname).toBe("/tenant/desktop/authorize");
	});
});
