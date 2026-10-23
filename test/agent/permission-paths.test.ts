import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type ProposedCall, pathsNamedBy } from "../../src/agent/permission-paths.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-paths-")));
	writeFileSync(join(root, "report.md"), "x");
	mkdirSync(join(root, "sorted"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

const bash = (command: string): ProposedCall => ({ tool: "bash", input: { command }, cwd: root });
const paths = (command: string) => pathsNamedBy(bash(command)).map((entry) => entry.path);
const listed = (command: string) =>
	pathsNamedBy(bash(command))
		.filter((entry) => entry.always)
		.map((entry) => entry.path);

describe("pathsNamedBy", () => {
	it("finds absolute, ~/, ./, and ../ paths, resolved against the working directory", () => {
		const got = paths(`cat /etc/hosts ~/notes.md ./report.md ../elsewhere.txt`);
		expect(got).toContain("/etc/hosts");
		expect(got).toContain(join(homedir(), "notes.md"));
		expect(got).toContain(join(root, "report.md"));
		expect(got).toContain(join(dirname(root), "elsewhere.txt"));
	});

	it("finds a bare file name with an extension even when it does not exist yet", () => {
		expect(listed("cp report.md copy.md")).toEqual([join(root, "report.md"), join(root, "copy.md")]);
	});

	it("finds a bare word only when it names something that exists, but still tracks it as a candidate", () => {
		expect(listed("ls sorted")).toEqual([join(root, "sorted")]);
		const all = pathsNamedBy(bash("mkdir fresh && ls sorted"));
		const fresh = all.find((entry) => entry.path === join(root, "fresh"));
		expect(fresh).toEqual({ path: join(root, "fresh"), always: false });
		expect(all.find((entry) => entry.path === join(root, "sorted"))?.always).toBe(true);
	});

	it("treats a trailing slash as the folder itself", () => {
		expect(listed("ls sorted/ && cp report.md sorted/")).toContain(join(root, "sorted"));
	});

	it("does not take option flags, sed scripts, backup suffixes, or URLs for paths", () => {
		const got = paths("sed -i.bak 's/old/new/' report.md && curl -o out.pdf https://example.com/r.pdf --header=x");
		expect(got).toContain(join(root, "report.md"));
		expect(got).toContain(join(root, "out.pdf"));
		for (const path of got) {
			expect(path).not.toContain("-i.bak");
			expect(path).not.toContain("old/new");
			expect(path).not.toContain("example.com");
		}
	});

	it("takes a relative path with a slash when it exists or ends in an extension", () => {
		const everything = pathsNamedBy(bash("cp sorted/a.pdf sorted/b && ls sorted/sub"));
		const always = everything.filter((entry) => entry.always).map((entry) => entry.path);
		expect(always).toContain(join(root, "sorted", "a.pdf"));
		expect(always).not.toContain(join(root, "sorted", "b"));
		expect(everything.map((entry) => entry.path)).toContain(join(root, "sorted", "b"));
	});

	it("is capped, so a command with thousands of words stays cheap", () => {
		const many = Array.from({ length: 5000 }, (_unused, index) => `/tmp/file-${index}.txt`).join(" ");
		expect(paths(`rm ${many}`).length).toBeLessThanOrEqual(40);
	});

	it("keeps absolute and explicit paths ahead of bare words when the cap cuts", () => {
		const words = Array.from({ length: 80 }, (_unused, index) => `word${index}`).join(" ");
		expect(paths(`echo ${words} /tmp/the-one.txt`)).toContain("/tmp/the-one.txt");
	});

	it("names the one path a write or edit call targets, resolved and always listed", () => {
		expect(pathsNamedBy({ tool: "write", input: { path: "new.md", content: "x" }, cwd: root })).toEqual([
			{ path: join(root, "new.md"), always: true },
		]);
		expect(pathsNamedBy({ tool: "edit", input: { path: "~/n.md", edits: [] }, cwd: root })).toEqual([
			{ path: join(homedir(), "n.md"), always: true },
		]);
	});

	it("names nothing for a saved automation, which has no single target", () => {
		expect(
			pathsNamedBy({
				tool: "manage_automations",
				input: { action: "save", automation: { run: ["rm", "x"] } },
				cwd: root,
			}),
		).toEqual([]);
	});
});
