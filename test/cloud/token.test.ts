import { afterEach, describe, expect, it } from "vitest";
import { AutoragHttpError, AutoragOAuthError } from "../../src/cloud/errors.ts";
import { exchangeCodeForToken, mapTokenExchangeError } from "../../src/cloud/token.ts";
import { rejectionOf } from "./helpers/assertions.ts";
import { type FakeServer, startFakeServer } from "./helpers/fake-server.ts";

let server: FakeServer | undefined;

afterEach(async () => {
	await server?.close();
	server = undefined;
});

const request = {
	code: "auth-code",
	codeVerifier: "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
	redirectUri: "http://127.0.0.1:5555/callback",
};

describe("exchangeCodeForToken", () => {
	it("posts the PKCE verifier and returns the access token", async () => {
		server = await startFakeServer(() => ({
			body: { access_token: "dz_secret", token_type: "Bearer", email: "user@example.com" },
		}));

		const token = await exchangeCodeForToken({ ...request, baseUrl: server.url });

		expect(token).toEqual({ accessToken: "dz_secret", email: "user@example.com" });
		const sent = server.requests[0]!;
		expect(sent.method).toBe("POST");
		expect(sent.url).toBe("/api/desktop/token");
		expect(sent.headers["content-type"]).toBe("application/json");
		expect(sent.json).toEqual({
			code: "auth-code",
			code_verifier: request.codeVerifier,
			redirect_uri: request.redirectUri,
		});
	});

	it("omits email when the server does not return one", async () => {
		server = await startFakeServer(() => ({ body: { access_token: "dz_x" } }));
		await expect(exchangeCodeForToken({ ...request, baseUrl: server.url })).resolves.toEqual({
			accessToken: "dz_x",
		});
	});

	it("normalizes a trailing slash on the base URL", async () => {
		server = await startFakeServer(() => ({ body: { access_token: "dz_x" } }));
		await exchangeCodeForToken({ ...request, baseUrl: `${server.url}/` });
		expect(server.requests[0]!.url).toBe("/api/desktop/token");
	});

	it("maps an OpenAI-shaped error body to an HTTP error", async () => {
		server = await startFakeServer(() => ({
			status: 400,
			body: { error: { message: "Invalid code", type: "invalid_request_error", code: "invalid_code" } },
		}));

		const failure = await rejectionOf(exchangeCodeForToken({ ...request, baseUrl: server.url }));

		expect(failure).toBeInstanceOf(AutoragHttpError);
		expect(failure).toMatchObject({ status: 400, code: "invalid_code", type: "invalid_request_error" });
		expect(failure.message).toContain("Invalid code");
	});

	it("handles a non-JSON error body (401)", async () => {
		server = await startFakeServer(() => ({ status: 401, rawBody: "nope" }));

		const failure = await rejectionOf(exchangeCodeForToken({ ...request, baseUrl: server.url }));

		expect(failure).toBeInstanceOf(AutoragHttpError);
		expect(failure).toMatchObject({ status: 401 });
		expect(failure.message).toContain("no error message");
	});

	it("rejects a success response without an access token", async () => {
		server = await startFakeServer(() => ({ body: { token_type: "Bearer" } }));
		await expect(exchangeCodeForToken({ ...request, baseUrl: server.url })).rejects.toThrow(/no access token/);
	});

	it("wraps connection failures", async () => {
		const failingFetch: typeof fetch = async () => {
			throw new Error("ECONNREFUSED");
		};

		await expect(
			exchangeCodeForToken({ ...request, baseUrl: "http://127.0.0.1:1", fetchImpl: failingFetch }),
		).rejects.toThrow(AutoragOAuthError);
	});

	it("reports cancellation through the abort signal", async () => {
		const controller = new AbortController();
		const abortingFetch: typeof fetch = async () => {
			controller.abort();
			throw new Error("aborted");
		};

		const failure = await rejectionOf(
			exchangeCodeForToken({
				...request,
				baseUrl: "http://127.0.0.1:1",
				fetchImpl: abortingFetch,
				signal: controller.signal,
			}),
		);

		expect(failure).toBeInstanceOf(AutoragOAuthError);
		expect(failure.message).toContain("cancelled");
	});
});

describe("mapTokenExchangeError", () => {
	it("uses the server message and keeps the code", () => {
		const error = mapTokenExchangeError(403, {
			error: { message: "Plan does not allow this", code: "forbidden", type: "permission_error" },
		});
		expect(error.message).toBe("AutoRAG token exchange failed (HTTP 403): Plan does not allow this");
		expect(error.code).toBe("forbidden");
		expect(error.type).toBe("permission_error");
	});
});
