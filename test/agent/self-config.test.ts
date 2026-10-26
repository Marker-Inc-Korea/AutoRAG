import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type PackagedAgentSkill, packagedSkillLocation, SETUP_SKILL_NAME } from "../../src/agent/packaged-skill.ts";
import { buildSelfConfigPrompt } from "../../src/agent/self-config.ts";
import { formatSkillInvocation } from "../../src/agent/skill-invocation.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-self-config-unit-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** A loaded skill as the config branch passes it in, with the packaged skill's name and location. */
function setupSkill(content: string): PackagedAgentSkill {
	return { name: SETUP_SKILL_NAME, location: packagedSkillLocation(SETUP_SKILL_NAME), content };
}

describe("buildSelfConfigPrompt", () => {
	it("carries the full skill, the active config path, the pi agent dir, and the user's request", () => {
		const skill = setupSkill("# Skill body\n\nstep one\n");
		const prompt = buildSelfConfigPrompt({
			query: "add an anthropic provider",
			configPath: "/tmp/qa/config.json",
			agentDir: "/tmp/qa/pi-agent",
			skill,
		});
		expect(prompt).toContain(formatSkillInvocation(skill));
		expect(prompt).toContain('<skill name="autorag-setup" location="autorag-skill://autorag-setup">');
		expect(prompt).toContain("# Skill body\n\nstep one\n");
		expect(prompt).toContain("/tmp/qa/config.json");
		expect(prompt).toContain("/tmp/qa/pi-agent");
		expect(prompt).toContain("add an anthropic provider");
	});

	it("names the tools the task uses and keeps retrieval, datasource, and web search out of it", () => {
		const prompt = buildSelfConfigPrompt({
			query: "x",
			configPath: "/tmp/qa/config.json",
			agentDir: "/tmp/qa/pi-agent",
			skill: setupSkill("# Skill\n"),
		});
		for (const tool of ["bash", "read", "edit", "write"]) expect(prompt).toContain(tool);
		expect(prompt).toMatch(/do NOT call the retrieval, datasource, or web search tools/u);
	});

	it("sends automation requests to manage_automations instead of editing config.json or rules by hand", () => {
		const prompt = buildSelfConfigPrompt({
			query: "x",
			configPath: "/tmp/qa/config.json",
			agentDir: "/tmp/qa/pi-agent",
			skill: setupSkill("# Skill\n"),
			automations: true,
		});
		expect(prompt).toContain("manage_automations");
		expect(prompt).toMatch(/file-triggered automation/iu);
		expect(prompt).toMatch(/never by hand-editing the rules file/iu);
	});

	it("leaves automations out of scope when the host did not register manage_automations", () => {
		const prompt = buildSelfConfigPrompt({
			query: "x",
			configPath: "/tmp/qa/config.json",
			agentDir: "/tmp/qa/pi-agent",
			skill: setupSkill("# Skill\n"),
		});
		expect(prompt).not.toMatch(/file-triggered automation requests/iu);
	});
});

describe("buildSelfConfigPrompt guidance learned from manual QA", () => {
	const prompt = buildSelfConfigPrompt({
		query: "x",
		configPath: "/tmp/qa/config.json",
		agentDir: "/tmp/qa/pi-agent",
		skill: setupSkill("# Skill\n"),
	});

	it("sends custom providers to the config model object, not a hand-written models.json", () => {
		expect(prompt).toContain("model.baseUrl");
		expect(prompt).toMatch(/do not (create|write|edit) .*models\.json/iu);
	});

	it("makes a provider-usability claim depend on that provider's own credential and a live health probe", () => {
		expect(prompt).toMatch(/test -n/u);
		expect(prompt).toMatch(/gateway/iu);
		expect(prompt).toContain("autorag health --json --config /tmp/qa/config.json");
	});

	it("tells the model what to do when the workspace directory is missing", () => {
		expect(prompt).toMatch(/workspace.*(does not exist|missing)/iu);
	});
});

describe("buildSelfConfigPrompt model-switch discipline", () => {
	const prompt = buildSelfConfigPrompt({
		query: "x",
		configPath: "/tmp/qa/config.json",
		agentDir: "/tmp/qa/pi-agent",
		skill: setupSkill("# Skill\n"),
	});

	it("forbids guessing a model id and requires the exact id from the catalog", () => {
		expect(prompt).toMatch(/never (guess|invent)/iu);
		expect(prompt).toContain("autorag models list --available --config /tmp/qa/config.json");
	});

	it("requires restoring the previous model when verification fails, so a broken config is never left behind", () => {
		expect(prompt).toMatch(/must not leave (a|the) (broken|failing)/iu);
		expect(prompt).toMatch(/restore/iu);
	});

	it("explains how to split a listed `provider/id` line, whose id may itself contain slashes", () => {
		expect(prompt).toContain("openrouter/z-ai/glm-5.3-flash");
		expect(prompt).toMatch(/before the first slash/iu);
	});
});
