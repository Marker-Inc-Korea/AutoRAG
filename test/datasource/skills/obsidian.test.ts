import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	realpathSync,
	rmSync,
	statSync,
	symlinkSync,
	utimesSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ObsidianSkill } from "../../../src/datasource/skills/obsidian/index.ts";
import type { RetrievalMethod } from "../../../src/retrieval/types.ts";
import { type FakeMinSync, installFakeMinSync } from "../../fixtures/fake-minsync.ts";

let root: string;
let vault: string;
let fake: FakeMinSync;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-obsidian-")));
	vault = join(root, "vault");
	mkdirSync(vault, { recursive: true });
	fake = installFakeMinSync(join(root, "bin"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function note(relativePath: string, body: string): void {
	const path = join(vault, relativePath);
	mkdirSync(join(path, ".."), { recursive: true });
	writeFileSync(path, body);
}

function makeSkill(options: { readonly instanceId?: string } = {}): ObsidianSkill {
	return new ObsidianSkill({
		instanceId: options.instanceId ?? "vault-1",
		vaultPath: vault,
		workspaceRoot: root,
		minSync: { binaryPath: fake.binaryPath, autoInstall: false },
	});
}

function method(skill: ObsidianSkill, name: "obsidian-bm25" | "obsidian-semantic"): RetrievalMethod {
	const found = skill.retrievalMethods().find((candidate) => candidate.describe().name === name);
	if (found === undefined) throw new Error(`skill exposes no ${name} method`);
	return found;
}

function workspace(instanceId = "vault-1"): string {
	return join(root, ".autorag", "datasources", "obsidian", instanceId, "minsync");
}

function stagedFiles(instanceId = "vault-1"): string[] {
	const filesRoot = join(workspace(instanceId), "files");
	const walk = (directory: string): string[] =>
		readdirSync(directory, { withFileTypes: true }).flatMap((entry) =>
			entry.isDirectory() ? walk(join(directory, entry.name)) : [join(directory, entry.name)],
		);
	return existsSync(filesRoot) ? walk(filesRoot).map((path) => path.slice(filesRoot.length + 1)) : [];
}

function minsyncCalls(): { args: string[]; cwd: string }[] {
	const log = join(root, "bin", "calls.jsonl");
	return existsSync(log)
		? readFileSync(log, "utf8")
				.trim()
				.split("\n")
				.filter(Boolean)
				.map((line) => JSON.parse(line) as { args: string[]; cwd: string })
		: [];
}

describe("ObsidianSkill indexing through MinSync", () => {
	it("stages only markdown notes into its own MinSync workspace and leaves the vault untouched", async () => {
		note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta launches in October.");
		note("Daily/2026-10-01.markdown", "# Daily\nRead about postgres.");
		note("Projects/diagram.png", "not a note");
		note("Projects/data.txt", "plain text is not a vault note");
		note(".obsidian/workspace.json", '{"mobile":"settings"}');
		note(".trash/Deleted.md", "deleted note about mobile");
		note("node_modules/pkg/README.md", "a dependency readme about mobile");
		symlinkSync(join(root, "elsewhere"), join(vault, "linked"));
		const before = readdirSync(vault, { recursive: true }).sort();

		const result = await makeSkill().index();

		expect(result).toMatchObject({ ok: true, instanceId: "vault-1", skill: "obsidian", chunkCount: 2 });
		expect(stagedFiles().sort()).toEqual(["Daily/2026-10-01.markdown", "Projects/Roadmap.md"]);
		expect(minsyncCalls().map((call) => call.args[0])).toEqual(expect.arrayContaining(["init", "check", "sync"]));
		expect(new Set(minsyncCalls().map((call) => call.cwd))).toEqual(new Set([workspace()]));
		expect(readdirSync(vault, { recursive: true }).sort()).toEqual(before);
	});

	it("re-indexes incrementally: edits and deletions reach the index, unchanged notes are not rewritten", async () => {
		note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta launches in October.");
		note("Projects/Budget.md", "# Budget\nHiring is frozen.");
		const skill = makeSkill();
		await skill.index();
		const budgetStaged = join(workspace(), "files", "Projects", "Budget.md");
		const pinned = new Date("2020-01-01T00:00:00.000Z");
		utimesSync(budgetStaged, pinned, pinned);

		note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta slipped to November.");
		rmSync(join(vault, "Projects", "Budget.md"));
		note("Projects/New.md", "# New\nA brand new note about kubernetes.");
		const second = await skill.index();

		expect(second).toMatchObject({ ok: true, chunkCount: 2 });
		expect(stagedFiles().sort()).toEqual(["Projects/New.md", "Projects/Roadmap.md"]);
		const hits = await method(skill, "obsidian-bm25").retrieve("november", { topK: 5 });
		expect(hits.map((hit) => hit.source)).toEqual(["/obsidian/vault-1/Projects/Roadmap.md"]);
		expect(await method(skill, "obsidian-bm25").retrieve("frozen", { topK: 5 })).toEqual([]);

		const untouched = join(workspace(), "files", "Projects", "Roadmap.md");
		utimesSync(untouched, pinned, pinned);
		await skill.index();
		expect(statSync(untouched).mtime.getTime()).toBe(pinned.getTime());
	});

	it("keeps the previous index when the vault is unavailable instead of wiping it", async () => {
		note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta launches in October.");
		const skill = makeSkill();
		await skill.index();
		rmSync(vault, { recursive: true, force: true });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect(stagedFiles()).toEqual(["Projects/Roadmap.md"]);
	});

	it("does not treat a vault that turned empty as a mass delete", async () => {
		note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta launches in October.");
		const skill = makeSkill();
		await skill.index();
		rmSync(join(vault, "Projects"), { recursive: true });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-empty" });
		expect(stagedFiles()).toEqual(["Projects/Roadmap.md"]);
	});

	it("keeps notes under a folder it could not read and reports a warning", async () => {
		note("Open/Visible.md", "# Visible\nreadable note");
		note("Locked/Secret.md", "# Secret\nlocked note");
		const skill = makeSkill();
		await skill.index();
		chmodSync(join(vault, "Locked"), 0o000);
		try {
			const result = await skill.index();
			expect(result.ok).toBe(true);
			expect(result.diagnostics.some((diagnostic) => diagnostic.severity === "warning")).toBe(true);
			expect(stagedFiles().sort()).toEqual(["Locked/Secret.md", "Open/Visible.md"]);
		} finally {
			chmodSync(join(vault, "Locked"), 0o755);
		}
	});

	it("reports a missing minsync binary as an unavailable datasource instead of throwing", async () => {
		note("Projects/Roadmap.md", "# Roadmap");
		const skill = new ObsidianSkill({
			instanceId: "vault-1",
			vaultPath: vault,
			workspaceRoot: root,
			minSync: { binaryPath: join(root, "no-such-minsync"), autoInstall: false },
		});

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect(result.diagnostics[0]?.message).toContain("minsync");
	});

	it("reports a failing MinSync sync with MinSync's own reason", async () => {
		note("Projects/Roadmap.md", "# Roadmap");
		fake.breakSync();

		const result = await makeSkill().index();

		expect(result).toMatchObject({ ok: false, code: "datasource-index-failed" });
		expect(result.diagnostics[0]?.message).toContain("boom");
	});

	it("fails clearly when no vaultPath is configured", async () => {
		const result = await new ObsidianSkill({ workspaceRoot: root }).index();
		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
	});
});

describe("ObsidianSkill retrieval", () => {
	beforeEach(() => {
		note("Projects/Roadmap.md", "# Roadmap\nThe mobile app beta launches in October.");
		note("Projects/Nested/Plan.md", "# Plan\nMobile launch plan details.");
		note("Projects-old/Archive.md", "# Archive\nOld mobile launch notes.");
		note("Daily/2026-10-01.md", "# Daily\nMobile standup and postgres reading.");
		note("Work/Daily/2026-10-02.md", "# Work daily\nMobile release checklist.");
	});

	it("returns vault-relative opaque sources with the real note path in metadata", async () => {
		const skill = makeSkill();
		await skill.index();

		const hits = await method(skill, "obsidian-bm25").retrieve("beta october", { topK: 5 });

		expect(hits).toHaveLength(1);
		expect(hits[0]).toMatchObject({
			source: "/obsidian/vault-1/Projects/Roadmap.md",
			content: expect.stringContaining("mobile app beta"),
			metadata: {
				method: "obsidian-bm25",
				datasourceId: "obsidian",
				instanceId: "vault-1",
				file: "Projects/Roadmap.md",
				title: "Roadmap",
				path: join(vault, "Projects", "Roadmap.md"),
			},
		});
	});

	it("filters by folder through scope without leaking sibling folders that share a prefix", async () => {
		const skill = makeSkill();
		await skill.index();
		const bm25 = method(skill, "obsidian-bm25");

		const everything = await bm25.retrieve("mobile", { topK: 10 });
		expect(everything.map((hit) => hit.source).sort()).toEqual([
			"/obsidian/vault-1/Daily/2026-10-01.md",
			"/obsidian/vault-1/Projects-old/Archive.md",
			"/obsidian/vault-1/Projects/Nested/Plan.md",
			"/obsidian/vault-1/Projects/Roadmap.md",
			"/obsidian/vault-1/Work/Daily/2026-10-02.md",
		]);

		const projects = await bm25.retrieve("mobile", { topK: 10, scope: "/obsidian/vault-1/Projects" });
		expect(projects.map((hit) => hit.source).sort()).toEqual([
			"/obsidian/vault-1/Projects/Nested/Plan.md",
			"/obsidian/vault-1/Projects/Roadmap.md",
		]);

		// The shared scope matcher's `**/` needs at least one folder before `Daily`, so a top-level `Daily` is not included.
		const nestedDaily = await bm25.retrieve("mobile", { topK: 10, scope: "/obsidian/vault-1/**/Daily/**" });
		expect(nestedDaily.map((hit) => hit.source)).toEqual(["/obsidian/vault-1/Work/Daily/2026-10-02.md"]);

		const oneNote = await bm25.retrieve("mobile", { topK: 10, scope: "/obsidian/vault-1/Projects/Roadmap.md" });
		expect(oneNote.map((hit) => hit.source)).toEqual(["/obsidian/vault-1/Projects/Roadmap.md"]);
	});

	it("still fills topK from a small folder when many notes elsewhere rank higher", async () => {
		for (let index = 0; index < 40; index += 1) {
			note(`Noise/n${index}.md`, "mobile mobile mobile mobile mobile");
		}
		const skill = makeSkill();
		await skill.index();

		const scoped = await method(skill, "obsidian-bm25").retrieve("mobile", {
			topK: 3,
			scope: "/obsidian/vault-1/Daily",
		});

		expect(scoped.map((hit) => hit.source)).toEqual(["/obsidian/vault-1/Daily/2026-10-01.md"]);
	});

	it("queries MinSync in the mode each method names, against its own workspace", async () => {
		const skill = makeSkill();
		await skill.index();

		await method(skill, "obsidian-bm25").retrieve("mobile", { topK: 4 });
		await method(skill, "obsidian-semantic").retrieve("mobile", { topK: 4 });

		const queries = minsyncCalls().filter((call) => call.args[0] === "query");
		expect(queries.map((call) => call.args[call.args.indexOf("--mode") + 1])).toEqual(["bm25", "vector"]);
		expect(new Set(queries.map((call) => call.cwd))).toEqual(new Set([workspace()]));
		expect(method(skill, "obsidian-bm25").describe()).toMatchObject({
			type: "bm25",
			datasourceId: "obsidian",
			capabilities: expect.arrayContaining(["lexical", "scoped"]),
		});
		expect(method(skill, "obsidian-semantic").describe()).toMatchObject({
			type: "vector",
			datasourceId: "obsidian",
			capabilities: expect.arrayContaining(["semantic", "scoped"]),
		});
	});

	it("keeps distinct sources for notes whose names contain # or %, and matches Korean names written decomposed", async () => {
		const decomposed = "계획.md".normalize("NFD");
		note("Ideas/C# tips 100%.md", "# Tips\nspecialchars mobile");
		note(`프로젝트/${decomposed}`, "# 계획\n모바일 앱 출시 일정");
		const skill = makeSkill();
		await skill.index();

		const special = await method(skill, "obsidian-bm25").retrieve("specialchars", { topK: 5 });
		expect(special.map((hit) => hit.source)).toEqual(["/obsidian/vault-1/Ideas/C%23 tips 100%25.md"]);
		expect(special[0]?.metadata.file).toBe("Ideas/C# tips 100%.md");
		expect(
			await method(skill, "obsidian-bm25").retrieve("specialchars", {
				topK: 5,
				scope: "/obsidian/vault-1/Ideas/C%23 tips 100%25.md",
			}),
		).toHaveLength(1);

		const korean = await method(skill, "obsidian-bm25").retrieve("모바일", { topK: 5 });
		expect(korean.map((hit) => hit.source)).toEqual(["/obsidian/vault-1/프로젝트/계획.md".normalize("NFC")]);
		expect(korean[0]?.metadata.path).toBe(join(vault, "프로젝트", decomposed));
	});

	it("rejects with an actionable reason before the first successful index", async () => {
		await expect(method(makeSkill(), "obsidian-bm25").retrieve("mobile", { topK: 3 })).rejects.toThrow(
			/not indexed yet/u,
		);
	});

	it("rejects while a sync is rewriting the index instead of returning a partial view", async () => {
		const skill = makeSkill();
		await skill.index();
		note("Projects/Another.md", "# Another\nmobile");
		const syncing = skill.index();
		await expect(method(skill, "obsidian-bm25").retrieve("mobile", { topK: 3 })).rejects.toThrow(
			/sync is in progress/u,
		);
		await syncing;
	});

	it("returns nothing for an empty query without spawning MinSync", async () => {
		const skill = makeSkill();
		await skill.index();
		const before = minsyncCalls().length;

		expect(await method(skill, "obsidian-bm25").retrieve("   ", { topK: 3 })).toEqual([]);
		expect(minsyncCalls()).toHaveLength(before);
	});
});

describe("ObsidianSkill description", () => {
	it("announces a MinSync-backed vault datasource with folder scopes and no external CLI", () => {
		const skill = makeSkill({ instanceId: "my vault" });
		const descriptor = skill.describe();

		expect(descriptor).toMatchObject({ name: "obsidian", datasourceId: "obsidian", instanceId: "my vault" });
		expect(descriptor.requiresExternalCli).toBeUndefined();
		expect(descriptor.capabilities).toEqual(expect.arrayContaining(["lexical", "semantic", "incremental"]));
		expect(descriptor.capabilities).not.toContain("external-cli");
		expect(skill.describeSources().map((source) => source.source)).toEqual(["/obsidian/my vault"]);
		const manifest = skill.skillManifest();
		expect(manifest.content).toContain("search_datasource_obsidian");
		expect(manifest.content).toContain("/obsidian/my vault/<folder>");
		expect(manifest.content).not.toMatch(/qmd/iu);
	});

	it("keeps instances in separate MinSync workspaces", async () => {
		note("A.md", "# A\nalpha");
		await makeSkill({ instanceId: "one" }).index();
		await makeSkill({ instanceId: "two" }).index();

		expect(stagedFiles("one")).toEqual(["A.md"]);
		expect(stagedFiles("two")).toEqual(["A.md"]);
		expect(workspace("one")).not.toBe(workspace("two"));
		expect(existsSync(join(root, ".autorag", "minsync"))).toBe(false);
	});
});
