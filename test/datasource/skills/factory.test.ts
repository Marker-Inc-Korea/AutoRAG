import { describe, expect, it } from "vitest";
import {
	BUILTIN_DATASOURCE_SKILL_NAMES,
	buildDatasourceSkills,
	builtinDatasourceSkillNames,
} from "../../../src/datasource/skills/factory.ts";
import { withPlatform } from "../../helpers/platform.ts";

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

	it("treats retired datasources.mail-export config as unknown because mailcrawl --source files owns static mailboxes", () => {
		const { skills, unknown } = buildDatasourceSkills({
			"mail-export": { connector: { paths: ["/tmp/export.mbox"] } },
			"old-backup": { type: "mail-export", connector: { paths: ["/tmp/eml"] } },
			archive: {
				type: "mailcrawl",
				connector: { source: "files", paths: ["/tmp/export.mbox"] },
			},
		});

		expect(unknown).toEqual(["mail-export", "old-backup"]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["archive"]);
		expect(BUILTIN_DATASOURCE_SKILL_NAMES).not.toContain("mail-export");
	});

	it("treats datasources.spotlight config as unknown because Spotlight support was removed", () => {
		const { skills, unknown } = buildDatasourceSkills({
			spotlight: { connector: { queries: ["kMDItemContentTypeTree == 'public.text'"] } },
			"my-files": { type: "spotlight" },
			rss: { connector: { feeds: [{ url: "https://example.com/feed.xml" }] } },
		});

		expect(unknown).toEqual(["spotlight", "my-files"]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["rss"]);
		expect(BUILTIN_DATASOURCE_SKILL_NAMES).not.toContain("spotlight");
		expect(builtinDatasourceSkillNames("darwin")).not.toContain("spotlight");
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
		const embedder = {
			identity: async () => ({ provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 2 }),
			embed: async (texts: readonly string[]) => texts.map(() => [1, 0]),
		};
		function gistMethodTypes(
			connector: Record<string, unknown>,
			shared: typeof embedder | null = embedder,
		): string[] {
			const { skills } = buildDatasourceSkills(
				{ "my-gists": { type: "github-gist", connector: { tokenEnv: "GITHUB_GIST_TEST_UNSET", ...connector } } },
				undefined,
				shared === null ? {} : { embedder: shared },
			);
			return skills.flatMap((skill) => skill.retrievalMethods().map((method) => method.describe().type));
		}

		it("keeps the semantic method by default", () => {
			expect(gistMethodTypes({})).toContain("vector");
		});

		it("has no semantic method when no embedder is available, instead of building its own", () => {
			expect(gistMethodTypes({}, null)).not.toContain("vector");
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

describe("macOS-only kakao datasource (#1998)", () => {
	it("offers kakao only on darwin and every other built-in on every platform", () => {
		const darwin = builtinDatasourceSkillNames("darwin");
		expect(darwin).toContain("kakao");
		for (const platform of ["linux", "win32"] as const) {
			const names = builtinDatasourceSkillNames(platform);
			expect(names).not.toContain("kakao");
			expect(names).toEqual(darwin.filter((name) => name !== "kakao"));
		}
	});

	it("derives the exported list from the running platform", () => {
		expect(BUILTIN_DATASOURCE_SKILL_NAMES).toEqual(builtinDatasourceSkillNames(process.platform));
	});

	it("builds kakao on darwin", () => {
		const { skills, unknown } = withPlatform("darwin", () =>
			buildDatasourceSkills({ kakao: {}, "family-kakao": { type: "kakao" } }, "/ws"),
		);
		expect(unknown).toEqual([]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["kakao", "family-kakao"]);
	});

	it.each(["linux", "win32"] as const)("does not build kakao on %s, even when configured", (platform) => {
		const { skills, unknown } = withPlatform(platform, () =>
			buildDatasourceSkills(
				{
					kakao: { connector: { binaryPath: "/opt/lazykatok" } },
					"family-kakao": { type: "kakao" },
					rss: { connector: { feeds: [{ url: "https://example.com/feed.xml" }] } },
				},
				"/ws",
			),
		);
		expect(unknown).toEqual(["kakao", "family-kakao"]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["rss"]);
	});
});
