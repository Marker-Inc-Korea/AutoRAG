import { describe, expect, it } from "vitest";
import { ConfigError } from "../../src/cli/config.ts";
import { buildRegistrationPrompt } from "../../src/ui/prompt.ts";

describe("registration prompt", () => {
	it("tells an agent to install katok and write trusted config without a CLI path from the user", () => {
		const result = buildRegistrationPrompt({
			type: "kakao",
			alias: "family-kakao",
			note: "Mom and dad family chat",
		});
		expect(result.title).toBe("KakaoTalk");
		expect(result.prompt).toContain("family-kakao");
		expect(result.prompt).toContain("Mom and dad family chat");
		expect(result.prompt).toContain("katok");
		expect(result.prompt).toContain("Do not ask me for a CLI path");
		expect(result.prompt).toContain("/family-kakao/**");
		expect(result.prompt).not.toContain("ghp_");
	});

	it("does not treat MinSync or Jikji as add-source types", () => {
		expect(() => buildRegistrationPrompt({ type: "minsync" })).toThrow(ConfigError);
		expect(() => buildRegistrationPrompt({ type: "jikji" })).toThrow(ConfigError);
	});

	it("asks the user for GitHub repos when they were not chosen", () => {
		const empty = buildRegistrationPrompt({ type: "github", alias: "work-github" });
		expect(empty.questions.some((item) => item.includes("owner/repo"))).toBe(true);
		expect(empty.prompt).toContain("Ask me these before writing config");
		const filled = buildRegistrationPrompt({
			type: "github",
			alias: "work-github",
			extras: { repos: "Marker-Inc-Korea/AutoRAG" },
		});
		expect(filled.questions).toEqual([]);
		expect(filled.prompt).toContain("Marker-Inc-Korea/AutoRAG");
		expect(filled.prompt).not.toContain("Ask me these before writing config");
	});

	it("does not ask which Kakao account to add", () => {
		const result = buildRegistrationPrompt({ type: "kakao", alias: "family-kakao" });
		expect(result.questions).toEqual([]);
		expect(result.prompt).toContain("single-account");
		expect(result.prompt).toContain("Do not ask which Kakao account");
	});

	it("rejects an unknown type", () => {
		expect(() => buildRegistrationPrompt({ type: "dropbox" })).toThrow(ConfigError);
	});
});
