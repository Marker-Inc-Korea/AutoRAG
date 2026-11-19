import { describe, expect, it } from "vitest";
import { createWebTools } from "../../src/profiles.ts";

/** The shape a tool's TypeBox parameter schema reduces to, for machine-consumed assertions. */
interface ParameterSchema {
	readonly type?: string;
	readonly properties?: Record<string, unknown>;
}

interface WebFetchResult {
	readonly details: { readonly fetchMethod: string; readonly available: boolean };
}

type FetchExecutor = (toolCallId: string, params: { url: string }) => Promise<WebFetchResult>;

describe("createWebTools", () => {
	it("returns web_search then web_fetch by default", () => {
		expect(createWebTools().map((tool) => tool.name)).toEqual(["web_search", "web_fetch"]);
	});

	it("drops web_search when search is false", () => {
		expect(createWebTools({ search: false }).map((tool) => tool.name)).toEqual(["web_fetch"]);
	});

	it("drops web_fetch when fetch is false", () => {
		expect(createWebTools({ fetch: false }).map((tool) => tool.name)).toEqual(["web_search"]);
	});

	it("returns no tools when both are disabled", () => {
		expect(createWebTools({ search: false, fetch: false })).toEqual([]);
	});

	it("gives each tool an object schema with its query/url parameter", () => {
		const [search, fetch] = createWebTools();
		const searchSchema = search.parameters as ParameterSchema;
		const fetchSchema = fetch.parameters as ParameterSchema;
		expect(searchSchema.type).toBe("object");
		expect(Object.keys(searchSchema.properties ?? {})).toContain("query");
		expect(fetchSchema.type).toBe("object");
		expect(Object.keys(fetchSchema.properties ?? {})).toContain("url");
	});

	it("rejects a non-http URL in web_fetch without any network", async () => {
		const [fetch] = createWebTools({ search: false });
		const execute = fetch.execute as unknown as FetchExecutor;
		const result = await execute("test-call", { url: "file:///etc/passwd" });
		expect(result.details.fetchMethod).toBe("rejected");
		expect(result.details.available).toBe(false);
	});
});
