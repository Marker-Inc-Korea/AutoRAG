import { describe, expect, it } from "vitest";
import { BUILTIN_DATASOURCE_SKILL_NAMES } from "../../src/datasource/skills/factory.ts";
import { DATASOURCE_TYPE_CATALOG, getDatasourceType, LOCAL_FOLDERS_ID, SOURCE_PICKER } from "../../src/ui/catalog.ts";

describe("datasource UI catalog", () => {
	it("covers every built-in datasource skill name", () => {
		const types = DATASOURCE_TYPE_CATALOG.map((entry) => entry.type).sort();
		expect(types).toEqual([...BUILTIN_DATASOURCE_SKILL_NAMES].sort());
	});

	it("keeps local folders as a separate non-skill surface", () => {
		expect(LOCAL_FOLDERS_ID).toBe("local-folders");
		expect(DATASOURCE_TYPE_CATALOG.some((entry) => entry.type === LOCAL_FOLDERS_ID)).toBe(false);
	});

	it("limits the add-source picker to datasource skills — local folders stay their own section", () => {
		expect(SOURCE_PICKER.map((entry) => entry.type).sort()).toEqual([...BUILTIN_DATASOURCE_SKILL_NAMES].sort());
		expect(
			SOURCE_PICKER.some(
				(entry) => entry.type === LOCAL_FOLDERS_ID || entry.type === "minsync" || entry.type === "jikji",
			),
		).toBe(false);
	});

	it("exposes only operator fields that persist names, paths, or lists — never secret values", () => {
		for (const entry of DATASOURCE_TYPE_CATALOG) {
			for (const field of entry.fields) {
				expect(["text", "path", "path-list", "env", "textarea", "select", "checkbox"]).toContain(field.kind);
				expect(field.kind).not.toBe("password");
				expect(field.key).not.toMatch(/(^|\.)(token|password|accessToken|apiKey|clientSecret)$/i);
			}
		}
	});

	it("gives KakaoTalk no extras and GitHub/Drive/mail optional extras", () => {
		const kakao = SOURCE_PICKER.find((entry) => entry.type === "kakao");
		expect(kakao?.supportsMultiple).toBe(false);
		expect(kakao?.extras).toEqual([]);
		const github = SOURCE_PICKER.find((entry) => entry.type === "github");
		expect(github?.supportsMultiple).toBe(true);
		expect(github?.extras.some((extra) => extra.key === "repos")).toBe(true);
		expect(SOURCE_PICKER.find((entry) => entry.type === "gmail")?.extras).toEqual([]);
		expect(getDatasourceType("gmail")?.fields.some((field) => field.key === "connector.backend")).toBe(false);
		expect(SOURCE_PICKER.find((entry) => entry.type === "mailcrawl")?.binaryName).toBe("mailcrawl");
		expect(
			SOURCE_PICKER.find((entry) => entry.type === "cloud-drive")?.extras.some(
				(extra) => extra.choices === "rclone-remotes",
			),
		).toBe(true);
		expect(
			SOURCE_PICKER.find((entry) => entry.type === "clawgallery")?.extras.some((extra) => extra.key === "path"),
		).toBe(true);
	});

	it("describes the GitHub, Drive, RSS, and Obsidian wizards a non-developer can fill", () => {
		const github = getDatasourceType("github");
		expect(github?.fields.map((field) => field.key)).toEqual(
			expect.arrayContaining(["connector.tokenEnv", "connector.repos"]),
		);
		expect(getDatasourceType("cloud-drive")?.fields.some((field) => field.key === "connector.remote")).toBe(true);
		expect(getDatasourceType("clawgallery")?.fields.some((field) => field.key === "connector.path")).toBe(true);
		expect(getDatasourceType("rss")?.fields.some((field) => field.key === "connector.feeds")).toBe(true);
		expect(getDatasourceType("obsidian")?.fields.some((field) => field.key === "connector.vaultPath")).toBe(true);
		expect(getDatasourceType("kakao")?.binaryName).toBe("katok");
	});
});
