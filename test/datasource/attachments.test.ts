import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	AttachmentRootsSkill,
	formatAttachmentRoots,
	resolveAttachmentRoots,
} from "../../src/datasource/attachments.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import type { DatasourceAttachmentRoot, DatasourceSkill } from "../../src/datasource/types.ts";

let home: string;

beforeEach(() => {
	home = mkdtempSync(join(tmpdir(), "autorag-attach-"));
});
afterEach(() => {
	rmSync(home, { recursive: true, force: true });
});

const resolve = (template: string, connector?: Record<string, unknown>, name = template) =>
	resolveAttachmentRoots({
		template,
		registrationName: name,
		config: connector === undefined ? {} : { connector },
		home,
		env: { AUTORAG_HOME: join(home, ".autorag") },
	});

describe("resolveAttachmentRoots", () => {
	it("discord: AutoRAG's own per-connection download folder", () => {
		const roots = resolve("discord", undefined, "my server");
		expect(roots).toHaveLength(1);
		expect(roots[0]?.path).toMatch(/attachments[\\/]discord[\\/]my-server/u);
		expect(roots[0]?.description).toContain("download-status.json");
	});

	it("telegram: media folder next to the archive, honoring connector.databasePath", () => {
		expect(resolve("telegram")[0]?.path).toBe(join(home, ".telecrawl", "media"));
		expect(resolve("telegram", { databasePath: join(home, "x", "t.db") })[0]?.path).toBe(join(home, "x", "media"));
	});

	it("whatsapp: copied-media folder next to the archive", () => {
		const roots = resolve("whatsapp");
		expect(roots.map((root) => root.path)).toContain(join(home, ".wacrawl", "media"));
	});

	it("slack: media folder under cache_dir from slacrawl's own config, default ~/.slacrawl/cache", () => {
		expect(resolve("slack")[0]?.path).toBe(join(home, ".slacrawl", "cache", "media"));
		const configPath = join(home, "slack.toml");
		writeFileSync(configPath, `cache_dir = "${join(home, "elsewhere").replaceAll("\\", "\\\\")}"\n`);
		expect(resolve("slack", { configPath })[0]?.path).toBe(join(home, "elsewhere", "media"));
	});

	it("kakao: media folder under the katok data dir, honoring connector.workspacePath", () => {
		expect(resolve("kakao")[0]?.path).toBe(join(home, "Library", "Application Support", "katok", "media"));
		expect(resolve("kakao", { workspacePath: join(home, "k") })[0]?.path).toBe(join(home, "k", "media"));
	});

	it("datasources without attachment files report none", () => {
		for (const template of ["mailcrawl", "github", "rss", "notion", "unknown-thing"]) {
			expect(resolve(template)).toEqual([]);
		}
	});
});

describe("formatAttachmentRoots", () => {
	const root = (path: string): DatasourceAttachmentRoot => ({ id: "files", path, description: "Saved files." });

	it("is empty with no roots", () => {
		expect(formatAttachmentRoots([])).toBe("");
	});

	it("labels each folder missing, empty or present", () => {
		const present = join(home, "present");
		mkdirSync(present);
		writeFileSync(join(present, "a.png"), "x");
		const empty = join(home, "empty");
		mkdirSync(empty);
		const text = formatAttachmentRoots([root(present), root(empty), root(join(home, "nope"))]);
		expect(text).toContain(`${present} (has files)`);
		expect(text).toContain(`${empty} (empty)`);
		expect(text).toContain(`${join(home, "nope")} (not created yet)`);
		expect(text).toContain("Saved files.");
	});
});

function stubSkill(roots?: readonly DatasourceAttachmentRoot[]): DatasourceSkill {
	return {
		describe: () => ({
			name: "x",
			type: "chat",
			description: "d",
			capabilities: [],
			tags: [],
			status: "active",
			datasourceId: "x",
		}),
		polling: () => ({ mode: "none" }),
		index: async () => ({ ok: true, instanceId: "i", skill: "x", chunkCount: 0, indexedAt: 0, diagnostics: [] }),
		retrievalMethods: () => [],
		describeSources: () => [],
		skillManifest: () => ({ name: "datasource-x", description: "d", content: "# x" }),
		...(roots === undefined ? {} : { attachmentRoots: () => roots }),
	};
}

describe("AttachmentRootsSkill", () => {
	it("adds an Attachments section to the manifest and exposes the roots", () => {
		const roots = [{ id: "files", path: join(home, "f"), description: "Saved files." }];
		const skill = new AttachmentRootsSkill(stubSkill(), roots);
		expect(skill.attachmentRoots()).toEqual(roots);
		const manifest = skill.skillManifest();
		expect(manifest.content).toContain("# x");
		expect(manifest.content).toContain("## Attachments");
		expect(manifest.content).toContain(join(home, "f"));
	});

	it("keeps the manifest untouched when there are no roots", () => {
		expect(new AttachmentRootsSkill(stubSkill(), []).skillManifest().content).toBe("# x");
	});
});

describe("buildDatasourceSkills attachment wiring", () => {
	it("exposes roots on aliased and described connections, not on rootless ones", () => {
		const { skills } = buildDatasourceSkills(
			{
				"tg-work": {
					type: "telegram",
					description: "work telegram",
					connector: { databasePath: join(home, "t.db") },
				},
				rss: { connector: { feeds: [{ url: "https://example.com/feed.xml" }] } },
			},
			home,
		);
		const tg = skills.find((skill) => skill.describe().datasourceId === "tg-work");
		expect(tg?.attachmentRoots?.().map((root) => root.path)).toEqual([join(home, "media")]);
		expect(tg?.skillManifest().content).toContain("## Attachments");
		const rss = skills.find((skill) => skill.describe().datasourceId === "rss");
		expect(rss?.attachmentRoots?.() ?? []).toEqual([]);
	});
});
