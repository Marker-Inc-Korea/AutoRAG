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

	it("forbids triggering the rule, editing it, and running the daemon", () => {
		const prompt = loopGuardPrompt(rule);
		expect(prompt).toMatch(/Triggering this automation is forbidden/);
		expect(prompt).toMatch(/Never edit the automation rules file/);
		expect(prompt).toContain("autorag automate");
	});

	it("tells an any-type rule to write nothing inside a watched folder", () => {
		const prompt = loopGuardPrompt({ ...rule, include: ["**"] });
		expect(prompt).toContain("files of any type");
		expect(prompt).toContain("write nothing inside a watched folder");
	});
});
