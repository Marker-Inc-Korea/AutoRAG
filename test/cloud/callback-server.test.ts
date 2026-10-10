import { describe, expect, it } from "vitest";
import { startCallbackServer } from "../../src/cloud/callback-server.ts";
import { AutoragOAuthError } from "../../src/cloud/errors.ts";
import { rejectionOf } from "./helpers/assertions.ts";
import { expectConnectionRefused } from "./helpers/fake-server.ts";

const state = "expected-state";

describe("startCallbackServer", () => {
	it("binds loopback and resolves the callback query", async () => {
		const server = await startCallbackServer({ state });
		expect(server.redirectUri).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/callback$/);

		const response = await fetch(`${server.redirectUri}?code=abc&state=${state}`);
		const html = await response.text();

		expect(response.status).toBe(200);
		expect(response.headers.get("content-type")).toContain("text/html");
		expect(html).toContain("Signed in to AutoRAG. You can close this tab.");

		const params = await server.waitForCallback();
		expect(params.get("code")).toBe("abc");
		expect(params.get("state")).toBe(state);
	});

	it("closes the listener after the first callback", async () => {
		const server = await startCallbackServer({ state });
		await fetch(`${server.redirectUri}?code=abc&state=${state}`);
		await server.waitForCallback();
		await expectConnectionRefused(server.redirectUri);
	});

	it("serves an error page and closes after an error callback", async () => {
		const server = await startCallbackServer({ state });

		const response = await fetch(`${server.redirectUri}?error=access_denied&state=${state}`);
		expect(await response.text()).toContain("did not complete");

		const params = await server.waitForCallback();
		expect(params.get("error")).toBe("access_denied");
		await expectConnectionRefused(server.redirectUri);
	});

	it("ignores requests to other paths", async () => {
		const server = await startCallbackServer({ state });
		const response = await fetch(new URL("/favicon.ico", server.redirectUri));
		expect(response.status).toBe(404);

		const callback = server.waitForCallback();
		await fetch(`${server.redirectUri}?code=still-works&state=${state}`);
		expect((await callback).get("code")).toBe("still-works");
	});

	it("rejects and closes on abort", async () => {
		const controller = new AbortController();
		const server = await startCallbackServer({ state, signal: controller.signal });
		const waiting = server.waitForCallback();

		controller.abort();

		await expect(waiting).rejects.toThrow(/cancelled/);
		await expectConnectionRefused(server.redirectUri);
	});

	it("rejects and closes on timeout", async () => {
		const server = await startCallbackServer({ state, timeoutMs: 30 });

		await expect(server.waitForCallback()).rejects.toThrow(/timed out/);
		await expectConnectionRefused(server.redirectUri);
	});

	it("rejects a pending wait when closed explicitly", async () => {
		const server = await startCallbackServer({ state });
		const waiting = server.waitForCallback();
		server.close();

		const failure = await rejectionOf(waiting);
		expect(failure).toBeInstanceOf(AutoragOAuthError);
		await expectConnectionRefused(server.redirectUri);
	});

	it("rejects malformed and forged callbacks without consuming the pending callback", async () => {
		const server = await startCallbackServer({ state });

		const malformed = [
			server.redirectUri,
			`${server.redirectUri}?code=abc`,
			`${server.redirectUri}?state=${state}`,
			`${server.redirectUri}?code=abc&state=other`,
			`${server.redirectUri}?code=abc&state=`,
			`${server.redirectUri}?error=access_denied&state=other`,
		];
		for (const url of malformed) {
			const response = await fetch(url);
			expect(response.status, url).toBe(400);
			expect(await response.text()).toContain("did not complete");
		}

		// The listener is still open and the genuine callback still wins.
		const params = await fetch(`${server.redirectUri}?code=genuine&state=${state}`).then((response) => {
			expect(response.status).toBe(200);
			return server.waitForCallback();
		});
		expect(params.get("code")).toBe("genuine");
		await expectConnectionRefused(server.redirectUri);
	});

	it("accepts an OAuth error callback that still echoes the state", async () => {
		const server = await startCallbackServer({ state });

		const response = await fetch(`${server.redirectUri}?error=invalid_scope&state=${state}`);
		expect(response.status).toBe(200);

		const params = await server.waitForCallback();
		expect(params.get("error")).toBe("invalid_scope");
	});
});
