import { describe, expect, it } from "vitest";
import { parseCallbackInput, parseCallbackQuery, requireAuthorizationCode } from "../../src/cloud/callback.ts";
import { AutoragOAuthError } from "../../src/cloud/errors.ts";

const state = "expected-state";

describe("parseCallbackQuery", () => {
	it("reads code, error and state", () => {
		expect(parseCallbackQuery(new URLSearchParams("code=abc&state=s"))).toEqual({ code: "abc", state: "s" });
		expect(parseCallbackQuery(new URLSearchParams("error=access_denied&state=s"))).toEqual({
			error: "access_denied",
			state: "s",
		});
		expect(parseCallbackQuery(new URLSearchParams(""))).toEqual({});
	});
});

describe("parseCallbackInput", () => {
	it("parses a full redirect URL", () => {
		expect(parseCallbackInput("http://127.0.0.1:5555/callback?code=abc&state=s")).toEqual({
			code: "abc",
			state: "s",
		});
	});

	it("parses a bare query string", () => {
		expect(parseCallbackInput("code=abc&state=s")).toEqual({ code: "abc", state: "s" });
		expect(parseCallbackInput("?code=abc")).toEqual({ code: "abc" });
	});

	it("treats a bare code as the authorization code", () => {
		expect(parseCallbackInput("  dz-code-123  ")).toEqual({ code: "dz-code-123" });
	});

	it("rejects empty input", () => {
		expect(() => parseCallbackInput("   ")).toThrow(AutoragOAuthError);
	});
});

describe("requireAuthorizationCode", () => {
	it("returns the code when state matches", () => {
		expect(requireAuthorizationCode({ code: "abc", state }, state)).toEqual({ code: "abc" });
	});

	it("rejects a state mismatch", () => {
		expect(() => requireAuthorizationCode({ code: "abc", state: "other" }, state)).toThrow(/state mismatch/);
	});

	it("rejects a state mismatch even together with an error", () => {
		expect(() => requireAuthorizationCode({ error: "access_denied", state: "other" }, state)).toThrow(
			/state mismatch/,
		);
	});

	it("maps access_denied to a cancellation message", () => {
		expect(() => requireAuthorizationCode({ error: "access_denied", state }, state)).toThrow(/cancelled/);
	});

	it("maps other provider errors through", () => {
		expect(() => requireAuthorizationCode({ error: "invalid_scope", state }, state)).toThrow(/invalid_scope/);
	});

	it("rejects a callback without a code or error", () => {
		expect(() => requireAuthorizationCode({ state }, state)).toThrow(/no authorization code/);
	});

	it("accepts a bare code paste that carries no state", () => {
		expect(requireAuthorizationCode({ code: "abc" }, state)).toEqual({ code: "abc" });
	});
});
