import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { normalizeIndexingConfig } from "../../src/cli/config.ts";
import { renderSearch } from "../../src/cli/output.ts";
import { BUILTIN_DATASOURCE_SKILL_NAMES } from "../../src/datasource/skills/factory.ts";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

const SKILL_NAMES = ["autorag", "autorag-setup", "autorag-lite-setup", "autorag-lite-search"] as const;

type SkillName = (typeof SKILL_NAMES)[number];

function readSkill(name: SkillName): string {
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
		for (const name of SKILL_NAMES) {
			expect(readSkill(name)).toMatch(new RegExp(`^---\\nname: ${name}\\n`, "m"));
		}
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

	it("ships every skill folder in the npm package", () => {
		const pkg = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8")) as {
			files: string[];
		};
		expect(pkg.files).toContain("skills");
	});

	it("documents the lite retrieve JSON contract and diagnostics codes", () => {
		const search = readSkill("autorag-lite-search");
		expect(search).toContain("autorag lite retrieve");
		expect(search).toContain("index-not-ready");
		expect(search).toContain("retrieval-method-failed");
		expect(search).toContain("minsync-unavailable");
		expect(search).toContain("--scope");
		expect(search).toContain("--tags");
		expect(search).toContain("sessionId");
		expect(search).toContain("autorag evidence");
		expect(search).toContain("autorag feedback");
	});

	it("documents the lite refresh method values and force flags", () => {
		const setup = readSkill("autorag-lite-setup");
		for (const method of ["parsed", "minsync", "datasources", "jikji", "all"]) {
			expect(setup).toContain(method);
		}
		expect(setup).toContain("--full");
		expect(setup).toContain("--force");
		expect(setup).toContain("unknown-datasource-skill");
		expect(setup).toContain("autorag lite watch --once");
	});

	it("presents hwp as a supported parsed format and not a legacy exclusion", () => {
		const setup = readSkill("autorag-setup");
		expect(setup).toMatch(/`hwp`, `hwpx`/);
		expect(setup).not.toMatch(/or `hwp` as fully supported parsed formats/);
		expect(setup).toMatch(/`xlsx`, `xls`, `hwp`, `hwpx`/);
		expect(setup).toMatch(/Do not present legacy\n`\.doc` as a supported parsed format/s);
	});
});
