import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { parseArgs } from "../../../src/cli/index.ts";
import { runSetup, type SetupDeps } from "../../../src/cli/setup.ts";
import { buildDatasourceSkills, canonicalDatasourceTemplate } from "../../../src/datasource/skills/factory.ts";
import {
	defaultNogonConfigPath,
	findNogonEntries,
	nogonServiceName,
} from "../../../src/datasource/skills/nogon/setup.ts";

/**
 * Nogon was released as Hister until v0.21.0 (asciimoo/hister became asciimoo/nogon after a trademark complaint).
 * A machine upgraded in place keeps `type: "hister"` in its AutoRAG config, a `hister` binary until Homebrew
 * relinks it, `HISTER_CONFIG`, and a config under a `hister` directory. All of that has to keep working.
 */

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("legacy `hister` template name", () => {
	it("maps the old template to nogon and leaves every other name alone", () => {
		expect(canonicalDatasourceTemplate("hister")).toBe("nogon");
		expect(canonicalDatasourceTemplate("nogon")).toBe("nogon");
		expect(canonicalDatasourceTemplate("mailcrawl")).toBe("mailcrawl");
	});

	it("builds a skill from `type: hister`, and from a bare `hister` key, without reporting them unknown", () => {
		const { skills, unknown } = buildDatasourceSkills({
			browser: { type: "hister", connector: { baseUrl: "http://127.0.0.1:4433" } },
			hister: true,
		});
		expect(unknown).toEqual([]);
		expect(skills.map((skill) => skill.describe().type)).toEqual(["nogon-web-history", "nogon-web-history"]);
	});

	it("keeps the configured connection id, so existing `/hister/...` evidence identities stay valid", () => {
		const { skills } = buildDatasourceSkills({ hister: true });
		expect(skills[0]?.describe().datasourceId).toBe("hister");
		const fresh = buildDatasourceSkills({ nogon: true }).skills[0];
		expect(fresh?.describe().datasourceId).toBe("nogon");
	});

	it("finds legacy entries for setup, by type and by name", () => {
		const entries = findNogonEntries({
			datasources: {
				browser: { type: "hister", connector: { semantic: true } },
				hister: true,
				history: { type: "nogon" },
				off: false,
				mail: { type: "mailcrawl" },
			},
		});
		expect(entries.map((entry) => entry.name)).toEqual(["browser", "hister", "history"]);
		expect(entries[0]?.semantic).toBe(true);
	});
});

describe("defaultNogonConfigPath", () => {
	const home = "/home/u";
	const exists =
		(...present: string[]) =>
		(path: string) =>
			present.includes(path);

	it("prefers the Nogon config over a legacy Hister one when both exist", () => {
		const nogon = "/home/u/Library/Preferences/nogon/config.yml";
		const hister = "/home/u/Library/Preferences/hister/config.yml";
		expect(defaultNogonConfigPath({}, exists(hister, nogon), "darwin", home)).toBe(nogon);
	});

	it("falls back to the legacy Hister location on every platform Nogon itself searches", () => {
		const darwin = "/home/u/Library/Preferences/hister/config.yml";
		expect(defaultNogonConfigPath({}, exists(darwin), "darwin", home)).toBe(darwin);
		const appSupport = "/home/u/Library/Application Support/hister/config.yml";
		expect(defaultNogonConfigPath({}, exists(appSupport), "darwin", home)).toBe(appSupport);
		const linux = "/home/u/.config/hister/config.yml";
		expect(defaultNogonConfigPath({}, exists(linux), "linux", home)).toBe(linux);
		const localAppData = "C:\\Users\\u\\AppData\\Local";
		const win = join(localAppData, "hister", "config.yml");
		expect(defaultNogonConfigPath({ LOCALAPPDATA: localAppData }, exists(win), "win32", home)).toBe(win);
	});

	it("honours XDG_CONFIG_HOME for both names", () => {
		const nogon = "/xdg/nogon/config.yml";
		expect(defaultNogonConfigPath({ XDG_CONFIG_HOME: "/xdg" }, exists(nogon), "linux", home)).toBe(nogon);
		const hister = "/xdg/hister/config.yml";
		expect(defaultNogonConfigPath({ XDG_CONFIG_HOME: "/xdg" }, exists(hister), "linux", home)).toBe(hister);
	});

	it("reads NOGON_CONFIG first, then the legacy HISTER_CONFIG", () => {
		const both = defaultNogonConfigPath(
			{ NOGON_CONFIG: "/a/nogon.yml", HISTER_CONFIG: "/b/hister.yml" },
			exists("/a/nogon.yml", "/b/hister.yml"),
			"linux",
			home,
		);
		expect(both).toBe("/a/nogon.yml");
		const legacyOnly = defaultNogonConfigPath(
			{ HISTER_CONFIG: "/b/hister.yml" },
			exists("/b/hister.yml"),
			"linux",
			home,
		);
		expect(legacyOnly).toBe("/b/hister.yml");
	});

	it("returns undefined when nothing exists", () => {
		expect(defaultNogonConfigPath({}, exists(), "linux", home)).toBeUndefined();
	});
});

describe("nogonServiceName", () => {
	it("follows the program that was found, because Homebrew's service is named after the formula", () => {
		expect(nogonServiceName("/opt/homebrew/bin/nogon")).toBe("nogon");
		expect(nogonServiceName("/opt/homebrew/bin/hister")).toBe("hister");
		expect(nogonServiceName("C:\\bin\\nogon.exe")).toBe("nogon");
		expect(nogonServiceName(undefined)).toBe("nogon");
	});
});

describe("autorag setup --sync-hister (deprecated alias)", () => {
	it("is accepted and means --sync-nogon", () => {
		const parsed = parseArgs(["setup", "--sync-hister", "--json"]);
		expect("error" in parsed).toBe(false);
		if ("error" in parsed) return;
		expect(parsed.flags["sync-nogon"]).toBe(true);
		expect(parsed.flags.json).toBe(true);
	});
});

describe("setup with only the legacy `hister` binary installed", () => {
	const fakeLock = () => ({ path: "setup.lock", contents: "", assertOwned() {}, release() {} });

	function legacyFixture(): { configPath: string; root: string } {
		const root = mkdtempSync(join(tmpdir(), "autorag-setup-nogon-legacy-"));
		roots.push(root);
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				datasources: { browser: { type: "hister", connector: { baseUrl: "http://127.0.0.1:4433" } } },
			}),
		);
		mkdirSync(join(root, "docs"), { recursive: true });
		return { configPath, root };
	}

	function deps(root: string, binaries: readonly string[]): SetupDeps {
		return {
			env: { PATH: "/bin", HOME: root },
			executable: (name) => (binaries.includes(name) ? `/opt/homebrew/bin/${name}` : undefined),
			pathExists: () => true,
			acquireLock: fakeLock,
			nogon: { configPath: () => undefined, health: async () => ({ reachable: true }) },
		};
	}

	it("treats a `hister` binary as installed and does not skip the datasource for a missing binary", async () => {
		const f = legacyFixture();
		const report = await runSetup({
			configPath: f.configPath,
			workspacePath: f.root,
			deps: deps(f.root, ["hister"]),
		});
		expect(report.nogon).toMatchObject({ installed: true, running: true });
		expect(report.datasources.find((d) => d.name === "browser")?.reason ?? "").not.toMatch(/missing/);
	});

	it("names `nogon` in the install hint when neither program exists", async () => {
		const f = legacyFixture();
		const report = await runSetup({ configPath: f.configPath, workspacePath: f.root, deps: deps(f.root, []) });
		expect(report.nogon).toMatchObject({ installed: false });
		expect(report.nogon?.remediation).toContain("brew install nogon");
		expect(report.datasources.find((d) => d.name === "browser")).toMatchObject({
			state: "skipped",
			reason: "missing nogon binary",
		});
	});
});
