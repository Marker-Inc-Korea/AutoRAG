import { describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../../src/datasource/skills/factory.ts";

describe("datasource skill factory", () => {
	it("does not register Gmail REST datasources", () => {
		const { skills, unknown } = buildDatasourceSkills({
			inbox: {
				type: "gmail",
				connector: { tokenEnv: "GMAIL_ACCESS_TOKEN_TEST" },
			},
			archive: {
				type: "mailcrawl",
				connector: { binaryPath: "/opt/bin/company-mail-wrapper" },
			},
		});

		expect(skills.map((skill) => skill.describe().name)).toEqual(["archive"]);
		expect(unknown).toEqual(["inbox"]);
		expect(skills[0]?.describe().type).toBe("mailcrawl-archive");
	});

	it("treats leftover datasources.gmail config as unknown without crashing", () => {
		const { skills, unknown } = buildDatasourceSkills({
			gmail: { connector: { tokenEnv: "GMAIL_ACCESS_TOKEN" } },
			rss: { connector: { feeds: [{ url: "https://example.com/feed.xml" }] } },
		});

		expect(unknown).toEqual(["gmail"]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["rss"]);
	});

	it("builds a github-gist skill from a typed alias entry", () => {
		const { skills, unknown } = buildDatasourceSkills(
			{
				"my-gists": {
					type: "github-gist",
					connector: { tokenEnv: "GITHUB_GIST_TEST_UNSET" },
				},
			},
			undefined,
		);
		expect(unknown).toEqual([]);
		const gist = skills.find((skill) => skill.describe().name === "my-gists");
		expect(gist?.describe().type).toBe("github-gist");
	});

	describe("github-gist semantic index", () => {
		function gistMethodTypes(connector: Record<string, unknown>): string[] {
			const { skills } = buildDatasourceSkills(
				{ "my-gists": { type: "github-gist", connector: { tokenEnv: "GITHUB_GIST_TEST_UNSET", ...connector } } },
				undefined,
			);
			return skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().type));
		}

		it("keeps the semantic method by default", () => {
			expect(gistMethodTypes({})).toContain("vector");
		});

		it.each([
			["connector.semantic: false", { semantic: false }],
			["connector.semantic.enabled: false", { semantic: { enabled: false } }],
		])("leaves only the lexical method with %s", (_label, connector) => {
			expect(gistMethodTypes(connector)).not.toContain("vector");
			expect(gistMethodTypes(connector)).toContain("bm25");
		});

		it("does not turn it off for semantic: true or an object that does not disable it", () => {
			expect(gistMethodTypes({ semantic: true })).toContain("vector");
			expect(gistMethodTypes({ semantic: { enabled: true } })).toContain("vector");
		});
	});
});
