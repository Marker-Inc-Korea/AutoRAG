import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { normalizeIndexingConfig } from "../../src/cli/config.ts";
import { renderSearch } from "../../src/cli/output.ts";
import { BUILTIN_DATASOURCE_SKILL_NAMES } from "../../src/datasource/skills/factory.ts";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

function readSkill(name: "autorag" | "autorag-setup"): string {
	return readFileSync(join(repoRoot, "skills", name, "SKILL.md"), "utf8").replace(/\r\n?/g, "\n");
}

const searchResponse: SearchDocumentsResponse = {
	sessionId: "session-1",
	query: "q",
	answer: "[1] answer",
	results: [
		{
			number: 1,
			title: "A",
			summary: "answer",
			evidence: [{ excerpt: "answer" }],
			confidence: 1,
			feedbackId: "session-1:1",
			source: "/docs/a.md",
		},
	],
	searched: 1,
	warnings: [],
	diagnostics: [],
};

describe("parent-agent skill docs", () => {
	it("keeps skill folder names aligned with frontmatter", () => {
		expect(readSkill("autorag")).toMatch(/^---\nname: autorag\n/m);
		expect(readSkill("autorag-setup")).toMatch(/^---\nname: autorag-setup\n/m);
	});

	it("documents MinSync auto-install as on by default", () => {
		expect(normalizeIndexingConfig({}).minSync.autoInstall).toBe(true);
		const setup = readSkill("autorag-setup");
		expect(setup).not.toMatch(/MinSync auto-install is off by default/);
		expect(setup).toMatch(/minSync\.autoInstall` defaults to\ntrue/s);
		expect(readSkill("autorag")).toMatch(/MinSync and Jikji\nauto-install on first use by default/s);
	});

	it("documents search JSON sessionId as debug-only", () => {
		const json = JSON.parse(renderSearch(searchResponse, { json: true, debug: false })) as {
			sessionId?: string;
		};
		const debugJson = JSON.parse(renderSearch(searchResponse, { json: true, debug: true })) as {
			sessionId?: string;
		};
		expect(json.sessionId).toBeUndefined();
		expect(debugJson.sessionId).toBe("session-1");

		const search = readSkill("autorag");
		expect(search).toContain("autorag search");
		expect(search).toContain("--json --debug");
		expect(search).toMatch(/`--json` alone omits `sessionId`/);
		expect(search).not.toMatch(/The response contains a `sessionId`/);
	});

	it("does not claim unknown datasource names fail config resolution", () => {
		const setup = readSkill("autorag-setup");
		expect(setup).not.toMatch(/Unknown skill names fail config resolution/);
		expect(setup).toContain("unknown-datasource-skill");
		expect(setup).toMatch(/they do not fail config resolution/);
	});

	it("lists builtin datasource templates and the loopback UI", () => {
		const setup = readSkill("autorag-setup");
		expect(setup).toContain("autorag ui --no-open");
		for (const name of BUILTIN_DATASOURCE_SKILL_NAMES) {
			expect(setup).toContain(name);
		}
	});

	it("presents hwp as a supported parsed format and not a legacy exclusion", () => {
		const setup = readSkill("autorag-setup");
		expect(setup).toMatch(/`hwp`, `hwpx`/);
		expect(setup).not.toMatch(/or `hwp` as fully supported parsed formats/);
		expect(setup).toMatch(/Do not present legacy\n`\.doc` or `\.xls` as supported parsed formats/s);
	});
});
