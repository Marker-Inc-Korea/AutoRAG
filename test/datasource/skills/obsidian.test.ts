import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ObsidianConnector } from "../../../src/datasource/skills/obsidian/connector.ts";
import { ObsidianSkill } from "../../../src/datasource/skills/obsidian/skill.ts";

let vault: string;

beforeEach(() => {
	vault = mkdtempSync(join(tmpdir(), "autorag-obsidian-"));
});

afterEach(() => {
	rmSync(vault, { recursive: true, force: true });
});

const NOTE = [
	"---",
	"tags: [project, planning]",
	"---",
	"# Roadmap 2024",
	"",
	"We link to [[Budget]] and [[Team|the team]] and embed ![[diagram.png]].",
	"",
	"Ship the beta in June. #milestone",
].join("\n");

describe("ObsidianConnector", () => {
	it("returns not-configured without a vault path and unavailable for a missing vault", async () => {
		expect(await new ObsidianConnector({}).fetch()).toMatchObject({ ok: false, reason: "not-configured" });
		expect(await new ObsidianConnector({ vaultPath: join(vault, "nope") }).fetch()).toMatchObject({
			ok: false,
			reason: "unavailable",
		});
	});

	it("walks the vault, extracting frontmatter tags, inline tags, links, and embeds", async () => {
		mkdirSync(join(vault, "projects"), { recursive: true });
		mkdirSync(join(vault, ".obsidian"), { recursive: true });
		writeFileSync(join(vault, ".obsidian", "ignored.md"), "# ignored");
		writeFileSync(join(vault, "projects", "roadmap.md"), NOTE);
		writeFileSync(join(vault, "inbox.md"), "# Inbox\nQuick note without tags.");

		const result = await new ObsidianConnector({ vaultPath: vault }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2);
			const roadmap = result.documents.find((d) => d.docId === "projects__roadmap");
			expect(roadmap).toMatchObject({
				title: "Roadmap 2024",
				hierarchy: ["folders", "projects"],
			});
			expect(roadmap?.metadata).toMatchObject({
				tags: expect.arrayContaining(["project", "planning", "milestone"]),
				links: ["Budget", "Team"],
				embeds: ["diagram.png"],
				folder: "projects",
			});
			expect(roadmap?.content).not.toContain("tags: [project");
			const inbox = result.documents.find((d) => d.docId === "inbox");
			expect(inbox?.hierarchy).toEqual(["folders"]);
		}
	});

	it("supports dash-list frontmatter tags", async () => {
		writeFileSync(join(vault, "note.md"), "---\ntags:\n  - alpha\n  - beta\n---\nBody text.");
		const result = await new ObsidianConnector({ vaultPath: vault }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents[0]?.metadata).toMatchObject({ tags: ["alpha", "beta"] });
		}
	});

	it("skips oversized notes with a count warning that has no paths", async () => {
		writeFileSync(join(vault, "big.md"), "# Big\ncontent");
		const result = await new ObsidianConnector({ vaultPath: vault, maxBytesPerFile: 3 }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(0);
			expect(result.warnings?.[0]).toContain("size limit");
			expect(JSON.stringify(result.warnings)).not.toContain(vault);
		}
	});
});

describe("ObsidianSkill", () => {
	it("indexes and searches with opaque /obsidian sources", async () => {
		mkdirSync(join(vault, "projects"), { recursive: true });
		writeFileSync(join(vault, "projects", "roadmap.md"), NOTE);
		const skill = new ObsidianSkill({
			instanceId: "vault-1",
			connectorOptions: { vaultPath: vault },
		});
		expect(skill.describe()).toMatchObject({ name: "obsidian", datasourceId: "obsidian" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("ship beta June", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/obsidian\/vault-1\/chunks\//);
		expect(hits?.[0]?.source).not.toContain(vault);
		const sources = skill.describeSources();
		expect(sources.map((s) => s.source)).toContain("/obsidian/vault-1/folders/projects");
	});
});
