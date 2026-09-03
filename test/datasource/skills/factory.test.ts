import { describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../../src/datasource/skills/factory.ts";

describe("datasource skill factory", () => {
	it("skips a gmail entry when connector.backend is himalaya", () => {
		const { skills, unknown } = buildDatasourceSkills({
			"legacy-imap": {
				type: "gmail",
				connector: { backend: "himalaya", account: "personal", folder: "INBOX" },
			},
			inbox: {
				type: "gmail",
				connector: { tokenEnv: "GMAIL_ACCESS_TOKEN_TEST" },
			},
			archive: {
				type: "mailcrawl",
				connector: { binaryPath: "/opt/bin/company-mail-wrapper" },
			},
		});

		expect(skills.map((skill) => skill.describe().name).sort()).toEqual(["archive", "inbox"]);
		expect(unknown).toEqual(["legacy-imap"]);
		expect(skills.find((skill) => skill.describe().name === "inbox")?.describe().type).toBe("gmail-account");
		expect(skills.find((skill) => skill.describe().name === "archive")?.describe().type).toBe("mailcrawl-archive");
	});
});
