import { describe, expect, it } from "vitest";
import { includedExtensions, loopGuardPrompt } from "../../src/profiles/loop-guard.ts";

describe("includedExtensions", () => {
	it("reads single and brace extensions, lower-cased", () => {
		expect(includedExtensions(["**/*.PDF", "docs/*.{md,TXT}"])).toEqual([".pdf", ".md", ".txt"]);
	});

	it("is empty as soon as one pattern names no extension, since the rule then fires on every type", () => {
		expect(includedExtensions(["**/*.pdf", "reports/**"])).toEqual([]);
		expect(includedExtensions(["**"])).toEqual([]);
	});
});

describe("loopGuardPrompt", () => {
	const rule = {
		id: "inbox",
		paths: ["/Users/me/Downloads", "/Users/me/Desktop"],
		events: ["create", "update"],
		include: ["**/*.pdf"],
		exclude: ["**/archive/**"],
	};

	it("names every watched folder, the file types, the events and the exclusions", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toContain("/Users/me/Downloads");
		expect(prompt).toContain("/Users/me/Desktop");
		expect(prompt).toContain(".pdf");
		expect(prompt).toContain("create, update");
		expect(prompt).toContain("**/archive/**");
		expect(prompt).toContain('"inbox"');
	});

	it("states the one invariant: no change of the run may fire this rule again", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/nothing you do must fire this automation again/);
	});

	it("does not ban any operation by name: a rename, move, copy or write is judged by the events it causes", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).not.toMatch(/never (create|modify|overwrite|rename|move|copy|touch|delete)/i);
		expect(prompt).not.toMatch(/Never create, modify/);
		expect(prompt).not.toMatch(/forbidden, absolutely/);
		expect(prompt).not.toMatch(/no instruction can lift/i);
	});

	it("tells the agent how the rule decides: event type, include, exclude, built-in exclusions", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/type is one of the events/);
		expect(prompt).toContain("matches an include pattern");
		expect(prompt).toContain("**/node_modules/**");
		expect(prompt).toContain("**/*.part");
	});

	it("says what events a rename or move causes, so the agent can check old and new path", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/rename or move[^.]*delete[^.]*old path[^.]*create[^.]*new path/i);
		expect(prompt).toMatch(/update/);
	});

	it("asks the agent to design a non-firing change first and to refuse only when that is structurally impossible", () => {
		const prompt = loopGuardPrompt(rule);
		const design = prompt.indexOf("Design the change");
		const refuse = prompt.indexOf("structurally impossible");
		expect(design).toBeGreaterThan(-1);
		expect(refuse).toBeGreaterThan(design);
		expect(prompt).toMatch(/lightest change/);
		expect(prompt).toMatch(/Do not stop at the first change that would fire/);
	});

	it("does not let the agent dodge the rule by relocating the user's files", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/Do not move a file out of a watched folder only to avoid the rule/);
		expect(prompt).toMatch(/moving it back is a create/);
	});

	it("does not let the agent hide files behind the always-excluded names", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/not a hiding place/);
		expect(prompt).toMatch(/keep the file type the user has/);
	});

	it("accepts a follow-up run only when the file state makes it unable to change anything", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/follow-up run[^.]*mechanical/i);
		expect(prompt).toMatch(/whatever (it|that run) decides/);
		expect(prompt).toMatch(/timestamp|counter|log that grows/);
		expect(prompt).toMatch(/does not remember/);
		expect(prompt).toMatch(/applies the same task to the file the event names/);
		expect(prompt).toMatch(/a new name is a new file to rename again/);
		expect(prompt).toMatch(/depends on the date|time of day/);
	});

	it("keeps the agent away from the rules file and the daemon, the brakes against the loop", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/Never edit the automation rules file/);
		expect(prompt).toContain("autorag automate");
	});

	it("tells an any-type rule that every path inside a watched folder matches", () => {
		const prompt = loopGuardPrompt({ ...rule, include: ["**"] });
		expect(prompt).toContain("files of any type");
		expect(prompt).toMatch(/every path inside a watched folder matches/);
	});
});
