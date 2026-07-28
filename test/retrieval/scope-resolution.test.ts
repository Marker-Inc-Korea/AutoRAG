import { describe, expect, it } from "vitest";
import * as scopeModule from "../../src/retrieval/scope.js";

interface ScopeBinding {
	readonly virtualPrefix: string;
	readonly physicalRoots: readonly string[];
}

type ScopeResolver = (
	scope: string | undefined,
	bindings: readonly ScopeBinding[],
	platform?: NodeJS.Platform,
	passthroughVirtualPrefixes?: readonly string[],
) => string | undefined;

function resolver(): ScopeResolver {
	const candidate = (scopeModule as Record<string, unknown>).resolveRetrievalScope;
	expect(candidate).toBeTypeOf("function");
	return candidate as ScopeResolver;
}

describe("retrieval scope resolution", () => {
	it("preserves virtual scopes and maps POSIX physical roots", () => {
		const resolveScope = resolver();
		const bindings = [{ virtualPrefix: "/docs", physicalRoots: ["/Users/example/Documents"] }];

		expect(resolveScope("/docs/reports/**", bindings, "darwin")).toBe("/docs/reports/**");
		expect(resolveScope("/Users/example/Documents/reports/**", bindings, "darwin")).toBe("/docs/reports/**");
	});

	it("maps Windows drive, case, separator, and UNC aliases consistently", () => {
		const resolveScope = resolver();

		expect(
			resolveScope(
				String.raw`c:\documents\Reports\**`,
				[{ virtualPrefix: "/docs", physicalRoots: [String.raw`C:\Documents`] }],
				"win32",
			),
		).toBe("/docs/Reports/**");
		expect(
			resolveScope(
				String.raw`\\server\share\Documents\Reports`,
				[{ virtualPrefix: "/docs", physicalRoots: [String.raw`\\SERVER\SHARE\Documents`] }],
				"win32",
			),
		).toBe("/docs/Reports");
	});

	it("rejects unknown physical absolute scopes instead of returning zero results", () => {
		const resolveScope = resolver();

		expect(() =>
			resolveScope(
				"/definitely-nonexistent/autorag-private",
				[{ virtualPrefix: "/docs", physicalRoots: ["/srv/docs"] }],
				"darwin",
			),
		).toThrow("invalid-retrieval-scope");
	});

	it("preserves declared virtual datasource scopes on POSIX and Windows", () => {
		const resolveScope = resolver();
		const bindings = [{ virtualPrefix: "/docs", physicalRoots: ["/srv/docs"] }];

		expect(resolveScope("/datasource/account", bindings, "darwin", ["/datasource"])).toBe("/datasource/account");
		expect(resolveScope("/datasource/account", bindings, "win32", ["/datasource"])).toBe("/datasource/account");
	});

	it("selects the most specific physical root regardless of binding order", () => {
		const resolveScope = resolver();
		const bindings = [
			{ virtualPrefix: "/repo", physicalRoots: ["/srv/repo"] },
			{ virtualPrefix: "/docs", physicalRoots: ["/srv/repo/docs"] },
		];

		expect(resolveScope("/srv/repo/docs/guide.md", bindings, "darwin")).toBe("/docs/guide.md");
	});
});
