import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildSelfConfigPrompt, loadSetupSkill } from "../../src/agent/self-config.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-self-config-unit-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("loadSetupSkill", () => {
	it("returns the shipped autorag-setup skill body without its front matter", () => {
		const skill = loadSetupSkill();
		expect(skill.startsWith("# AutoRAG setup")).toBe(true);
		expect(skill).toContain("## Configure one search model");
		expect(skill).not.toMatch(/^---/u);
		expect(skill).not.toContain("license: MIT");
	});

	it("reads an explicit skill path and keeps a body that has no front matter intact", () => {
		const path = join(root, "SKILL.md");
		writeFileSync(path, "# Plain skill\n\nbody\n");
		expect(loadSetupSkill(path)).toBe("# Plain skill\n\nbody\n");
	});

	it("throws a descriptive error for a missing skill file", () => {
		mkdirSync(join(root, "empty"));
		expect(() => loadSetupSkill(join(root, "empty", "SKILL.md"))).toThrow(/SKILL\.md/u);
	});
});

describe("buildSelfConfigPrompt", () => {
	it("carries the full skill, the active config path, the pi agent dir, and the user's request", () => {
		const prompt = buildSelfConfigPrompt({
			query: "add an anthropic provider",
			configPath: "/tmp/qa/config.json",
			agentDir: "/tmp/qa/pi-agent",
			skill: "# Skill body\n\nstep one\n",
		});
		expect(prompt).toContain("# Skill body\n\nstep one\n");
		expect(prompt).toContain("/tmp/qa/config.json");
		expect(prompt).toContain("/tmp/qa/pi-agent");
		expect(prompt).toContain("add an anthropic provider");
	});
});

describe("buildSelfConfigPrompt guidance learned from manual QA", () => {
	const prompt = buildSelfConfigPrompt({
		query: "x",
		configPath: "/tmp/qa/config.json",
		agentDir: "/tmp/qa/pi-agent",
		skill: "# Skill\n",
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
		skill: "# Skill\n",
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
