import { describe, expect, it } from "vitest";
import { ConfigError } from "../../src/cli/config.ts";
import { buildRegistrationPrompt } from "../../src/ui/prompt.ts";

describe("registration prompt", () => {
	it("tells an agent to install discrawl and write trusted config without a CLI path from the user", () => {
		const result = buildRegistrationPrompt({
			type: "discord",
			alias: "work-discord",
			note: "Engineering Discord archive",
		});
		expect(result.title).toBe("Discord");
		expect(result.prompt).toContain("work-discord");
		expect(result.prompt).toContain("Engineering Discord archive");
		expect(result.prompt).toContain("discrawl");
		expect(result.prompt).toContain("Do not ask me for a CLI path");
		expect(result.prompt).toContain("/work-discord/**");
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

	it("rejects an unknown type", () => {
		expect(() => buildRegistrationPrompt({ type: "dropbox" })).toThrow(ConfigError);
	});
});
