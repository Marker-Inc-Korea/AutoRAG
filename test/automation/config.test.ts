import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	type Automation,
	AutomationConfigError,
	automationsConfigPath,
	loadAutomations,
	matchesAutomation,
	parseAutomations,
} from "../../src/automation/config.ts";

let root: string;
const env = (home: string): NodeJS.ProcessEnv => ({ HOME: home });

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-automation-config-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function only(raw: Record<string, unknown>): Automation {
	const [automation] = parseAutomations({ automations: [raw] }, env(root));
	return automation;
}

describe("parseAutomations", () => {
	it("applies defaults and expands ~ in paths and run", () => {
		const automation = only({ id: "pdf", paths: ["~/Downloads"], run: ["node", "~/s.ts"] });
		expect(automation).toEqual({
			id: "pdf",
			paths: [join(root, "Downloads")],
			events: ["create", "update", "delete"],
			include: ["**"],
			exclude: [],
			run: ["node", join(root, "s.ts")],
			appendFiles: false,
			debounceMs: 1000,
			timeoutMs: 600_000,
		});
	});

	it("rejects relative watch paths", () => {
		expect(() => only({ id: "a", paths: ["Downloads"], run: ["x"] })).toThrow(/absolute/);
	});

	it("rejects duplicate ids", () => {
		const raw = { id: "a", paths: [root], run: ["x"] };
		expect(() => parseAutomations({ automations: [raw, raw] }, env(root))).toThrow(/duplicate/i);
	});

	it("rejects ids that are not safe log file names", () => {
		expect(() => only({ id: "../evil", paths: [root], run: ["x"] })).toThrow(AutomationConfigError);
	});

	it("rejects an empty run argv and unknown event types", () => {
		expect(() => only({ id: "a", paths: [root], run: [] })).toThrow(/run/);
		expect(() => only({ id: "a", paths: [root], run: ["x"], events: ["add"] })).toThrow(/events/);
	});

	it("rejects unknown fields so typos do not silently disable a rule", () => {
		expect(() => only({ id: "a", paths: [root], run: ["x"], inclde: ["*.pdf"] })).toThrow(/inclde/);
	});
});

describe("loadAutomations", () => {
	it("returns no automations when the file does not exist", () => {
		expect(loadAutomations(join(root, "missing.json"), env(root))).toEqual([]);
	});

	it("reports invalid JSON with the file path", () => {
		const path = join(root, "automations.json");
		writeFileSync(path, "{ nope");
		expect(() => loadAutomations(path, env(root))).toThrow(path);
	});

	it("lives under AUTORAG_HOME", () => {
		expect(automationsConfigPath({ AUTORAG_HOME: join(root, "h") })).toBe(join(root, "h", "automations.json"));
	});
});

describe("matchesAutomation", () => {
	const watched = (): string => {
		const dir = join(root, "watched");
		mkdirSync(dir, { recursive: true });
		return dir;
	};

	it("matches include globs against the path relative to the watched root, at any depth", () => {
		const dir = watched();
		const automation = only({ id: "a", paths: [dir], run: ["x"], include: ["**/*.pdf"] });
		expect(matchesAutomation(automation, dir, { type: "create", path: join(dir, "a.pdf") })).toBe(true);
		expect(matchesAutomation(automation, dir, { type: "create", path: join(dir, "x", "y", "a.pdf") })).toBe(true);
		expect(matchesAutomation(automation, dir, { type: "create", path: join(dir, "a.txt") })).toBe(false);
	});

	it("filters by event type", () => {
		const dir = watched();
		const automation = only({ id: "a", paths: [dir], run: ["x"], events: ["delete"] });
		expect(matchesAutomation(automation, dir, { type: "create", path: join(dir, "a") })).toBe(false);
		expect(matchesAutomation(automation, dir, { type: "delete", path: join(dir, "a") })).toBe(true);
	});

	it("applies user excludes and the built-in partial-download/VCS excludes", () => {
		const dir = watched();
		const automation = only({ id: "a", paths: [dir], run: ["x"], exclude: ["out/**"] });
		for (const rel of ["out/r.pdf", "f.crdownload", "f.part", ".git/HEAD", "node_modules/a/b.js", "~$doc.docx"]) {
			expect(matchesAutomation(automation, dir, { type: "create", path: join(dir, rel) }), rel).toBe(false);
		}
		expect(matchesAutomation(automation, dir, { type: "create", path: join(dir, "in", "r.pdf") })).toBe(true);
	});

	it("never matches the root itself or paths outside it", () => {
		const dir = watched();
		const automation = only({ id: "a", paths: [dir], run: ["x"] });
		expect(matchesAutomation(automation, dir, { type: "delete", path: dir })).toBe(false);
		expect(matchesAutomation(automation, dir, { type: "create", path: join(root, "other.pdf") })).toBe(false);
	});
});
