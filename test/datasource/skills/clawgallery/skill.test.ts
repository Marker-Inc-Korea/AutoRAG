import { describe, expect, it } from "vitest";
import { formatDatasourceSkillInvocation } from "../../../../src/agent/datasource-skill.ts";
import { ClawGallerySkill, type ClawGallerySkillClient } from "../../../../src/datasource/skills/clawgallery/skill.ts";
import type {
	ClawGalleryIndexResult,
	ClawGallerySearchResult,
	ClawGalleryVdrResult,
} from "../../../../src/datasource/skills/clawgallery/types.ts";

class StubClient implements ClawGallerySkillClient {
	async bootstrap(): Promise<ClawGalleryIndexResult> {
		return { ok: true, data: { indexed: 2, skipped: 5, pruned: 0 }, stdout: "", stderr: "", code: 0 };
	}
	async syncVisual(): Promise<ClawGalleryVdrResult> {
		return { ok: true, data: { processed: 2, skipped: 5, failed: 0 }, stdout: "", stderr: "", code: 0 };
	}
	async search(): Promise<ClawGallerySearchResult> {
		return {
			ok: true,
			hits: [{ imageId: "img-1", content: "Login error", score: 0.8 }],
			data: { hits: [{ imageId: "img-1", content: "Login error", score: 0.8 }] },
			stdout: "",
			stderr: "",
			code: 0,
		};
	}
}

describe("ClawGallerySkill", () => {
	it("indexes incrementally and exposes all retrieval modes", async () => {
		const skill = new ClawGallerySkill({ client: new StubClient(), instanceId: "personal" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual([
			"clawgallery-hybrid",
			"clawgallery-keyword",
			"clawgallery-lexical",
			"clawgallery-embedding",
		]);
	});

	it("maps hits to isolated screenshot scopes", async () => {
		const skill = new ClawGallerySkill({ client: new StubClient(), instanceId: "personal" });
		const results = await skill
			.retrievalMethods()[0]
			?.retrieve("login", { topK: 5, allowedScopes: ["/screenshots/personal/**"] });
		expect(results?.[0]).toMatchObject({
			source: "/screenshots/personal/images/img-1",
			id: "clawgallery:personal:img-1",
		});
		const denied = await skill.retrievalMethods()[0]?.retrieve("login", { topK: 5, scope: "/screenshots/work/**" });
		expect(denied).toEqual([]);
	});

	it("keeps existing search available when visual sync fails", async () => {
		const client = new StubClient();
		client.syncVisual = async () => ({ ok: false, reason: "nonzero-exit", stdout: "", stderr: "", code: 1 });
		const result = await new ClawGallerySkill({ client }).index();
		expect(result).toMatchObject({ ok: true });
		expect(result.diagnostics[0]?.severity).toBe("warning");
	});

	it("documents sparse, dense, and hybrid mode selection for the agent", () => {
		const skill = new ClawGallerySkill({ client: new StubClient() });
		const manifest = skill.skillManifest();
		expect(manifest.content).toContain("Never use `embedding` just because a V-SPLADE index exists");
		expect(manifest.content).toContain("`hybrid` is the default");
		expect(
			formatDatasourceSkillInvocation({
				name: manifest.name,
				description: manifest.description,
				content: manifest.content,
				filePath: "datasource://datasource-clawgallery",
			}),
		).toContain("V-SPLADE is sparse lexical retrieval");
	});
});
