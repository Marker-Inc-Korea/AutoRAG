import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { normalizeIndexingConfig } from "../../src/cli/config.ts";
import { renderSearch } from "../../src/cli/output.ts";
import { BUILTIN_DATASOURCE_SKILL_NAMES } from "../../src/datasource/skills/factory.ts";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

const SKILL_NAMES = [
	"autorag",
	"autorag-setup",
	"autorag-lite-setup",
	"autorag-lite-search",
	"autorag-doctor",
] as const;

type SkillName = (typeof SKILL_NAMES)[number];

function readSkill(name: SkillName | string): string {
	return readFileSync(join(repoRoot, "skills", name, "SKILL.md"), "utf8").replace(/\r\n?/g, "\n");
}

/** Every folder under `skills/`, which is what the npm package ships to parent agents. */
function skillFolders(): string[] {
	return readdirSync(join(repoRoot, "skills"), { withFileTypes: true })
		.filter((entry) => entry.isDirectory())
		.map((entry) => entry.name)
		.sort();
}

/** Parse the leading YAML frontmatter block of a SKILL.md into flat string fields. */
function parseFrontmatter(markdown: string): Record<string, string> | undefined {
	const match = /^---\n([\s\S]*?)\n---\n/.exec(markdown);
	if (!match) return undefined;
	const fields: Record<string, string> = {};
	for (const line of match[1].split("\n")) {
		const pair = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
		if (pair) fields[pair[1]] = pair[2].trim();
	}
	return fields;
}

/** The real CLI surface, parsed from the shipped dispatcher rather than restated here. */
function cliSurface(): { commands: Set<string>; flags: Set<string> } {
	const source = readFileSync(join(repoRoot, "src/cli/index.ts"), "utf8");
	const listOf = (name: string): string[] => {
		const block = new RegExp(`${name}\\s*=\\s*(?:new Set\\()?\\[([\\s\\S]*?)\\]`).exec(source);
		if (!block) throw new Error(`cannot parse ${name} from src/cli/index.ts`);
		return [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
	};
	return {
		commands: new Set(listOf("COMMANDS")),
		flags: new Set([...listOf("BOOLEAN_FLAGS"), ...listOf("VALUE_FLAGS")]),
	};
}

/** Collect `autorag <command> ... --flag` invocations out of markdown prose and fences. */
function autoragInvocations(markdown: string): { command: string; flags: string[] }[] {
	const calls: { command: string; flags: string[] }[] = [];
	for (const line of markdown.split("\n")) {
		const match = /(?:^|[`$\s(|])autorag\s+([a-z][a-z0-9-]*)((?:\s+[^`\n]*)?)/.exec(line);
		if (!match) continue;
		if (match[1] === "init" && line.includes("npm")) continue;
		const flags = [...match[2].matchAll(/\s--([a-z][a-z0-9-]*)/g)].map((m) => m[1]);
		calls.push({ command: match[1], flags });
	}
	return calls;
}

/** Extract one `## <heading>` section body from a markdown document. */
function section(markdown: string, heading: string): string {
	const pattern = new RegExp(`^## ${heading}\\s*$([\\s\\S]*?)(?=^## |\\Z)`, "m");
	return pattern.exec(markdown)?.[1] ?? "";
}

/** Every string literal used as a diagnostic/warning code anywhere in the shipped source. */
function sourceStringLiterals(): Set<string> {
	const literals = new Set<string>();
	const walk = (dir: string): void => {
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			const full = join(dir, entry.name);
			if (entry.isDirectory()) walk(full);
			else if (entry.name.endsWith(".ts") && statSync(full).isFile()) {
				for (const m of readFileSync(full, "utf8").matchAll(/"([a-z][a-z0-9]*(?:-[a-z0-9]+)+)"/g))
					literals.add(m[1]);
			}
		}
	};
	walk(join(repoRoot, "src"));
	return literals;
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

	it("lists builtin datasource templates and probes them wizard-style", () => {
		const setup = readSkill("autorag-setup");
		expect(setup).toContain("wizard-style");
		expect(setup).toContain("do not recommend it for datasource");
		expect(setup).not.toContain("autorag ui --no-open");
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

	it("keeps every shipped skill folder valid under the agent-skill frontmatter spec", () => {
		const folders = skillFolders();
		expect(folders).toEqual([...SKILL_NAMES].sort());
		for (const folder of folders) {
			const fields = parseFrontmatter(readSkill(folder));
			expect(fields, `${folder}: missing frontmatter`).toBeDefined();
			expect(fields?.name).toBe(folder);
			expect(folder).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
			expect(folder.length).toBeLessThanOrEqual(64);
			const description = fields?.description ?? "";
			expect(description.length, `${folder}: description length`).toBeGreaterThan(0);
			expect(description.length, `${folder}: description length`).toBeLessThanOrEqual(1024);
			expect(description, `${folder}: angle brackets are prohibited`).not.toMatch(/[<>]/);
		}
	});

	it("keeps doctor-skill and README troubleshooting commands inside the real CLI surface", () => {
		const { commands, flags } = cliSurface();
		const readme = readFileSync(join(repoRoot, "README.md"), "utf8").replace(/\r\n?/g, "\n");
		const troubleshooting = section(readme, "Troubleshooting");
		expect(troubleshooting.length, "README needs a ## Troubleshooting section").toBeGreaterThan(0);
		expect(troubleshooting).toContain("autorag-doctor");

		for (const [label, markdown] of [
			["skills/autorag-doctor/SKILL.md", readSkill("autorag-doctor")],
			["README ## Troubleshooting", troubleshooting],
		] as const) {
			const calls = autoragInvocations(markdown);
			expect(calls.length, `${label}: no autorag invocations found`).toBeGreaterThan(0);
			for (const call of calls) {
				expect(commands, `${label}: unknown command "autorag ${call.command}"`).toContain(call.command);
				for (const flag of call.flags) {
					expect(flags, `${label}: unknown flag "--${flag}" on autorag ${call.command}`).toContain(flag);
				}
			}
		}
	});

	it("cites only diagnostic codes that exist in the shipped source", () => {
		const literals = sourceStringLiterals();
		const codes = section(readSkill("autorag-doctor"), "Diagnostic codes");
		expect(codes.length, "doctor skill needs a ## Diagnostic codes section").toBeGreaterThan(0);
		const cited = [...codes.matchAll(/`([a-z][a-z0-9]*(?:-[a-z0-9]+)+)`/g)].map((m) => m[1]);
		expect(cited.length).toBeGreaterThan(4);
		for (const code of cited) {
			expect(literals, `doctor skill cites unknown diagnostic code "${code}"`).toContain(code);
		}
	});

	it("presents hwp as a supported parsed format and not a legacy exclusion", () => {
		const setup = readSkill("autorag-setup");
		expect(setup).toMatch(/`hwp`, `hwpx`/);
		expect(setup).not.toMatch(/or `hwp` as fully supported parsed formats/);
		expect(setup).toMatch(/`xlsx`, `xls`, `hwp`, `hwpx`/);
		expect(setup).toMatch(/Do not present legacy\n`\.doc` as a supported parsed format/s);
	});
});
