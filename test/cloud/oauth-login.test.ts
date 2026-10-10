import type { OAuthLoginCallbacks, OAuthPrompt } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { AutoragHttpError } from "../../src/cloud/errors.ts";
import { loginAutorag, refreshAutoragCredentials } from "../../src/cloud/oauth-login.ts";
import { createCodeChallenge } from "../../src/cloud/pkce.ts";
import { rejectionOf } from "./helpers/assertions.ts";
import { expectConnectionRefused, type FakeServer, startFakeServer } from "./helpers/fake-server.ts";

let tokenServer: FakeServer | undefined;

afterEach(async () => {
	await tokenServer?.close();
	tokenServer = undefined;
});

interface Probe {
	callbacks: OAuthLoginCallbacks;
	/** Every authorization URL the flow opened, in order. */
	urls: string[];
	/** Resolves with the first authorization URL. */
	authorized: Promise<string>;
	prompts: OAuthPrompt[];
}

function createProbe(overrides: Partial<OAuthLoginCallbacks> = {}): Probe {
	const urls: string[] = [];
	const prompts: OAuthPrompt[] = [];
	let resolveAuthorized!: (url: string) => void;
	const authorized = new Promise<string>((resolve) => {
		resolveAuthorized = resolve;
	});
	const callbacks: OAuthLoginCallbacks = {
		onAuth: (info) => {
			urls.push(info.url);
			if (urls.length === 1) resolveAuthorized(info.url);
		},
		onDeviceCode: () => {},
		onProgress: () => {},
		onPrompt: (prompt) => {
			prompts.push(prompt);
			return new Promise<string>(() => {});
		},
		onSelect: async () => undefined,
		...overrides,
	};
	return { callbacks, urls, authorized, prompts };
}

function redirectUriOf(authorizeUrl: string): string {
	return new URL(authorizeUrl).searchParams.get("redirect_uri")!;
}

describe("loginAutorag", () => {
	it("completes the PKCE flow through the loopback callback", async () => {
		tokenServer = await startFakeServer(() => ({
			body: { access_token: "dz_token", token_type: "Bearer", email: "user@example.com" },
		}));

		const probe = createProbe();
		const login = loginAutorag(probe.callbacks, { baseUrl: tokenServer.url });
		const authorizeUrl = new URL(await probe.authorized);

		expect(authorizeUrl.searchParams.get("client")).toBe("agent");
		expect(authorizeUrl.searchParams.get("code_challenge_method")).toBe("S256");
		const redirectUri = authorizeUrl.searchParams.get("redirect_uri")!;
		const challenge = authorizeUrl.searchParams.get("code_challenge")!;
		const state = authorizeUrl.searchParams.get("state")!;
		expect(probe.prompts[0]!.placeholder).toBe(redirectUri);

		const response = await fetch(`${redirectUri}?code=dz-code&state=${state}`);
		expect(await response.text()).toContain("Signed in to AutoRAG");

		const credentials = await login;
		expect(credentials).toMatchObject({ access: "dz_token", refresh: "", email: "user@example.com" });
		expect(credentials.expires).toBeGreaterThan(Date.now());

		const tokenRequest = tokenServer.requests[0]!;
		expect(tokenRequest.json).toMatchObject({ code: "dz-code", redirect_uri: redirectUri });
		const verifier =
			typeof tokenRequest.json === "object" && tokenRequest.json !== null && "code_verifier" in tokenRequest.json
				? tokenRequest.json.code_verifier
				: undefined;
		if (typeof verifier !== "string") throw new Error("the token request carried no code_verifier");
		expect(createCodeChallenge(verifier)).toBe(challenge);
		await expectConnectionRefused(redirectUri);
	});

	it("accepts a pasted redirect URL when the browser cannot reach loopback", async () => {
		tokenServer = await startFakeServer(() => ({ body: { access_token: "dz_manual" } }));

		const probe = createProbe({
			onPrompt: async (prompt) => {
				probe.prompts.push(prompt);
				const authorizeUrl = new URL(probe.urls[0]!);
				const redirectUri = authorizeUrl.searchParams.get("redirect_uri")!;
				const state = authorizeUrl.searchParams.get("state")!;
				return `${redirectUri}?code=manual-code&state=${state}`;
			},
		});

		const credentials = await loginAutorag(probe.callbacks, { baseUrl: tokenServer.url });

		expect(credentials.access).toBe("dz_manual");
		expect(probe.prompts).toHaveLength(1);
		expect(tokenServer.requests[0]!.json).toMatchObject({ code: "manual-code" });
		await expectConnectionRefused(redirectUriOf(await probe.authorized));
	});

	it("rejects a state mismatch", async () => {
		tokenServer = await startFakeServer(() => ({ body: { access_token: "nope" } }));
		const probe = createProbe();
		const login = loginAutorag(probe.callbacks, { baseUrl: tokenServer.url });
		const rejection = expect(login).rejects.toThrow(/state mismatch/);

		await fetch(`${redirectUriOf(await probe.authorized)}?code=dz-code&state=tampered`);

		await rejection;
		expect(tokenServer.requests).toHaveLength(0);
	});

	it("rejects an access_denied callback", async () => {
		tokenServer = await startFakeServer(() => ({ body: { access_token: "nope" } }));
		const probe = createProbe();
		const login = loginAutorag(probe.callbacks, { baseUrl: tokenServer.url });
		const rejection = expect(login).rejects.toThrow(/cancelled/);
		const authorizeUrl = await probe.authorized;

		await fetch(
			`${redirectUriOf(authorizeUrl)}?error=access_denied&state=${new URL(authorizeUrl).searchParams.get("state")!}`,
		);

		await rejection;
		expect(tokenServer.requests).toHaveLength(0);
	});

	it("surfaces a token-exchange failure", async () => {
		tokenServer = await startFakeServer(() => ({
			status: 400,
			body: { error: { message: "Invalid code", type: "invalid_request_error", code: "invalid_code" } },
		}));
		const probe = createProbe();
		const login = loginAutorag(probe.callbacks, { baseUrl: tokenServer.url });
		const settled = rejectionOf(login);
		const authorizeUrl = await probe.authorized;

		await fetch(
			`${redirectUriOf(authorizeUrl)}?code=dz-code&state=${new URL(authorizeUrl).searchParams.get("state")!}`,
		);

		const failure = await settled;
		expect(failure).toBeInstanceOf(AutoragHttpError);
		expect(failure).toMatchObject({ status: 400, code: "invalid_code" });
	});

	// Exercises the loopback wait budget itself, so a real (short) timer is required here.
	it("times out and closes the loopback listener", async () => {
		const probe = createProbe();
		const login = loginAutorag(probe.callbacks, { baseUrl: "http://127.0.0.1:1", timeoutMs: 30 });
		const rejection = expect(login).rejects.toThrow(/timed out/);
		const authorizeUrl = await probe.authorized;

		await rejection;
		await expectConnectionRefused(redirectUriOf(authorizeUrl));
	});

	it("aborts when the caller cancels", async () => {
		const controller = new AbortController();
		const probe = createProbe({ signal: controller.signal });
		const login = loginAutorag(probe.callbacks, { baseUrl: "http://127.0.0.1:1" });
		const rejection = expect(login).rejects.toThrow(/cancelled/);
		await probe.authorized;

		controller.abort();

		await rejection;
	});
});

describe("refreshAutoragCredentials", () => {
	it("returns the credential unchanged", () => {
		const credentials = { access: "dz", refresh: "", expires: Number.MAX_SAFE_INTEGER };
		expect(refreshAutoragCredentials(credentials)).toBe(credentials);
	});
});
