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

	it("treats leftover datasources.kakao config as unknown without crashing", () => {
		const { skills, unknown } = buildDatasourceSkills({
			kakao: { connector: { binaryPath: "/missing/katok" } },
			"family-kakao": { type: "kakao" },
			discord: true,
		});

		expect(unknown).toEqual(["kakao", "family-kakao"]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["discord"]);
	});
});
