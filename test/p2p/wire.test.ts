import type { Static, TSchema } from "typebox";
import { Value } from "typebox/value";
import { beforeEach, describe, expect, it } from "vitest";
import {
	PeerQueryRequestSchema,
	PeerQueryResponseSchema,
	resetWireMapping,
	wireSourceId,
} from "../../src/p2p/wire.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function check<T extends TSchema>(schema: T, value: unknown): value is Static<T> {
	return Value.Check(schema, value);
}

// ---------------------------------------------------------------------------
// Schema acceptance — happy
// ---------------------------------------------------------------------------

beforeEach(() => {
	resetWireMapping();
});

describe("PeerQueryRequest schema", () => {
	it("accepts a minimal valid request", () => {
		const req = { v: 1, query: "what is autorag?" };
		expect(check(PeerQueryRequestSchema, req)).toBe(true);
	});

	it("accepts a request with optional topK and scope", () => {
		const req = { v: 1, query: "find me docs", topK: 5, scope: "research" };
		expect(check(PeerQueryRequestSchema, req)).toBe(true);
	});

	it("accepts topK = 20 (boundary)", () => {
		const req = { v: 1, query: "hello", topK: 20 };
		expect(check(PeerQueryRequestSchema, req)).toBe(true);
	});

	it("accepts a 4096-char query (boundary)", () => {
		const query = "a".repeat(4096);
		expect(check(PeerQueryRequestSchema, { v: 1, query })).toBe(true);
	});
});

describe("PeerQueryResponse schema", () => {
	const minimalResponse = (): Record<string, unknown> => ({
		v: 1,
		status: "ok",
		answer: "some answer",
		results: [
			{
				number: 1,
				title: "Doc title",
				summary: "Some summary",
				source: "/my-docs/abc123",
				excerpt: "relevant excerpt",
			},
		],
		files: [],
		diagnostics: [],
	});

	it("accepts a minimal valid response with status ok", () => {
		expect(check(PeerQueryResponseSchema, minimalResponse())).toBe(true);
	});

	it("accepts status rejected with diagnostics", () => {
		const resp = {
			...minimalResponse(),
			status: "rejected",
			diagnostics: [{ code: "auth-error", message: "Unauthenticated" }],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("accepts a response with files", () => {
		const resp = {
			...minimalResponse(),
			files: [
				{
					source: "/my-docs/doc123",
					contentBase64: "aGVsbG8=",
					redacted: false,
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("accepts a response with multiple results", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: 1,
					title: "First",
					summary: "Sum 1",
					source: "/root/aaa",
					excerpt: "x1",
				},
				{
					number: 2,
					title: "Second",
					summary: "Sum 2",
					source: "/root/bbb",
					excerpt: "x2",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// Schema rejection — failure cases
// ---------------------------------------------------------------------------

describe("PeerQueryRequest schema — rejections", () => {
	it("rejects query > 4096 chars", () => {
		const req = { v: 1, query: "a".repeat(4097) };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});

	it("rejects topK > 20", () => {
		const req = { v: 1, query: "test", topK: 21 };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});

	it("rejects negative topK", () => {
		const req = { v: 1, query: "test", topK: -1 };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});

	it("rejects missing v field", () => {
		const req = { query: "test" };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});

	it("rejects v !== 1", () => {
		const req = { v: 2, query: "test" };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});

	it("rejects missing query", () => {
		const req = { v: 1 };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});

	it("rejects empty string query", () => {
		const req = { v: 1, query: "" };
		expect(check(PeerQueryRequestSchema, req)).toBe(false);
	});
});

describe("PeerQueryResponse schema — rejections", () => {
	const minimalResponse = (): Record<string, unknown> => ({
		v: 1,
		status: "ok",
		answer: "some answer",
		results: [
			{
				number: 1,
				title: "Doc title",
				summary: "Some summary",
				source: "/my-docs/abc123",
				excerpt: "relevant excerpt",
			},
		],
		files: [],
		diagnostics: [],
	});

	it("rejects unknown status value", () => {
		const resp = { ...minimalResponse(), status: "maybe" };
		expect(check(PeerQueryResponseSchema, resp)).toBe(false);
	});

	it("rejects status ok when answer is missing", () => {
		const resp = {
			...minimalResponse(),
			answer: undefined,
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(false);
	});

	it("accepts any diagnostic code string (refusal code typing is semantic, not schema-enforced)", () => {
		const resp = {
			...minimalResponse(),
			status: "rejected",
			diagnostics: [{ code: "nope-nope", message: "bad" }],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("accepts source with uppercase path segments", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: 1,
					title: "Bad source",
					summary: "should fail",
					source: "/Users/me/docs/file.pdf", // uppercase U doesn't match ^/[a-z0-9-]+(/|$)
					excerpt: "x",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("accepts source with uppercase characters", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: 1,
					title: "Uppercase source",
					summary: "should fail",
					source: "/MyDocs/report",
					excerpt: "x",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("accepts source with underscore", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: 1,
					title: "Underscore source",
					summary: "should fail",
					source: "/my_docs/report",
					excerpt: "x",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("accepts a datasource or local source string without imposing wire remapping", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: 1,
					title: "No leading slash",
					summary: "should fail",
					source: "docs/file",
					excerpt: "x",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});

	it("rejects non-integer result number", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: 1.5,
					title: "Float",
					summary: "should fail",
					source: "/docs/aaa",
					excerpt: "x",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(false);
	});

	it("rejects negative result number", () => {
		const resp = {
			...minimalResponse(),
			results: [
				{
					number: -1,
					title: "Negative",
					summary: "should fail",
					source: "/docs/aaa",
					excerpt: "x",
				},
			],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(false);
	});
});

// ---------------------------------------------------------------------------
// wireSourceId and reverse mapping
// ---------------------------------------------------------------------------

describe("wireSourceId", () => {
	it("preserves the canonical source identifier without a second wire format", () => {
		const id = wireSourceId("/My Docs/report.final.pdf");
		expect(id).toBe("/My Docs/report.final.pdf");
	});

	it("keeps the source unchanged on the wire", () => {
		const virtualPath = "/My Docs/report.final.pdf";
		const id = wireSourceId(virtualPath);
		expect(id).toBe(virtualPath);
	});

	it("preserves distinct canonical source identifiers", () => {
		const id1 = wireSourceId("/My Docs/report");
		const id2 = wireSourceId("/my-docs/report");
		expect(id1).not.toBe(id2);
	});

	it("handles a simple path with no slugging needed", () => {
		const id = wireSourceId("/docs/simple");
		// /docs/simple has all lowercase already and no special chars, so the hash suffix
		// is only appended when slugging changes the segment.
		// Each segment unchanged: id = /docs/simple
		expect(id).toBe("/docs/simple");
	});

	it("handles root-only path", () => {
		const id = wireSourceId("/docs");
		expect(id).toBe("/docs");
	});

	it("accepts absolute local and slash-prefixed datasource sources", () => {
		expect(() => wireSourceId("")).toThrow();
		expect(wireSourceId("/kakao/personal/chunks/c-1")).toBe("/kakao/personal/chunks/c-1");
		expect(wireSourceId("/Users/me/docs/file.md")).toBe("/Users/me/docs/file.md");
	});
});

describe.each([
	["auth-error"],
	["replay-rejected"],
	["rate-limited"],
	["injection-detected"],
	["policy-denied"],
	["queue-full"],
	["internal-error"],
])("refusal code %s", (code) => {
	it("is a valid refusal code", () => {
		const resp = {
			v: 1,
			status: "rejected",
			answer: "",
			results: [],
			files: [],
			diagnostics: [{ code, message: "test" }],
		};
		expect(check(PeerQueryResponseSchema, resp)).toBe(true);
	});
});
