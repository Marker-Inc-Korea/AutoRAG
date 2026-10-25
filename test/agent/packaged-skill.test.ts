import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { datasourceSkillLocation, formatDatasourceSkillInvocation } from "../../src/agent/datasource-skill.ts";
import {
	loadPackagedSkill,
	ORGANIZE_SKILL_NAME,
	packagedSkillLocation,
	SETUP_SKILL_NAME,
} from "../../src/agent/packaged-skill.ts";
import { formatSkillInvocation } from "../../src/agent/skill-invocation.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-packaged-skill-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("loadPackagedSkill", () => {
	it("returns the shipped autorag-setup skill body without its front matter", () => {
		const skill = loadPackagedSkill(SETUP_SKILL_NAME);
		expect(skill.name).toBe("autorag-setup");
		expect(skill.content.startsWith("# AutoRAG setup")).toBe(true);
		expect(skill.content).toContain("## Configure one search model");
		expect(skill.content).not.toMatch(/^---/u);
		expect(skill.content).not.toContain("license: Apache-2.0");
	});

	it("returns the shipped organize skill: front matter stripped, rules intact", () => {
		const skill = loadPackagedSkill(ORGANIZE_SKILL_NAME);
		expect(skill.name).toBe("organize");
		expect(skill.content.startsWith("# Organizing")).toBe(true);
		expect(skill.content).toContain("old path → new path");
		expect(skill.content).toMatch(/trash/iu);
		expect(skill.content).not.toMatch(/^---/u);
		expect(skill.content).not.toContain("license: Apache-2.0");
	});

	it("gives every packaged skill a path-opaque location, never a filesystem path", () => {
		for (const name of [SETUP_SKILL_NAME, ORGANIZE_SKILL_NAME] as const) {
			const skill = loadPackagedSkill(name);
			expect(skill.location).toBe(`autorag-skill://${name}`);
			expect(skill.location).not.toContain("SKILL.md");
			expect(skill.location).not.toContain(root);
			expect(skill.location).not.toMatch(/\/(?:Users|Volumes|home|tmp)\//u);
		}
		expect(packagedSkillLocation(ORGANIZE_SKILL_NAME)).toBe("autorag-skill://organize");
	});

	it("reads an explicit skill path and keeps a body that has no front matter intact", () => {
		const path = join(root, "SKILL.md");
		writeFileSync(path, "# Plain skill\n\nbody\n");
		const skill = loadPackagedSkill(SETUP_SKILL_NAME, path);
		expect(skill.content).toBe("# Plain skill\n\nbody\n");
		expect(skill.location).toBe("autorag-skill://autorag-setup");
	});

	it("strips the front matter and the blank line after it from a CRLF checkout, as on Windows", () => {
		const path = join(root, "SKILL.md");
		writeFileSync(path, "---\r\nname: organize\r\n---\r\n\r\n# Organizing\r\n\r\nbody\r\n");
		expect(loadPackagedSkill(ORGANIZE_SKILL_NAME, path).content.startsWith("# Organizing")).toBe(true);
	});

	it("throws an error naming the skill and the missing file", () => {
		mkdirSync(join(root, "empty"));
		expect(() => loadPackagedSkill(SETUP_SKILL_NAME, join(root, "empty", "SKILL.md"))).toThrow(/SKILL\.md/u);
		expect(() => loadPackagedSkill("not-a-skill")).toThrow(/not-a-skill/u);
	});
});

describe("formatSkillInvocation", () => {
	it("matches the datasource skill loader's invocation shape exactly", () => {
		const datasource = {
			name: "kakao",
			description: "chat",
			content: "body\n",
			filePath: datasourceSkillLocation("kakao"),
		};
		expect(formatDatasourceSkillInvocation(datasource)).toBe(
			formatSkillInvocation({ name: "kakao", location: "datasource://kakao", content: "body\n" }),
		);
		expect(formatSkillInvocation(loadPackagedSkill(ORGANIZE_SKILL_NAME))).toBe(
			`<skill name="organize" location="autorag-skill://organize">\n${loadPackagedSkill(ORGANIZE_SKILL_NAME).content}\n</skill>`,
		);
	});
});
