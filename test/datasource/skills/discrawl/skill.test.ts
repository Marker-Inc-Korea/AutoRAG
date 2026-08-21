import { describe, expect, it } from "vitest";
import { DiscrawlSkill, type DiscrawlSkillClient } from "../../../../src/datasource/skills/discrawl/skill.ts";
import type {
	DiscrawlDoctorResult,
	DiscrawlEmbedResult,
	DiscrawlSearchHit,
	DiscrawlSearchResult,
	DiscrawlSyncResult,
} from "../../../../src/datasource/skills/discrawl/types.ts";
import { DEFAULT_DISCRAWL_EMBEDDING_MODEL } from "../../../../src/datasource/skills/discrawl/types.ts";

function okDoctor(overrides: Partial<Record<string, boolean | string>> = {}): DiscrawlDoctorResult {
	return {
		ok: true,
		stdout: "",
		stderr: "",
		code: 0,
		data: {
			ready: true,
			configOk: true,
			databaseOk: true,
			ftsOk: true,
			embeddingsOk: true,
			...overrides,
		} as DiscrawlDoctorResult extends { data: infer D } ? D : never,
	};
}

function okSync(messages: number): DiscrawlSyncResult {
	return { ok: true, stdout: "", stderr: "", code: 0, data: { messages, guilds: 1, channels: 2 } };
}

function okEmbed(failed = 0): DiscrawlEmbedResult {
	return {
		ok: true,
		stdout: "",
		stderr: "",
		code: 0,
		data: { processed: 10, succeeded: 10 - failed, failed, remainingBacklog: 0 },
	};
}

function okSearch(hits: readonly DiscrawlSearchHit[]): DiscrawlSearchResult {
	return { ok: true, hits, data: { hits }, stdout: "", stderr: "", code: 0 };
}

class StubClient {
	public doctorResult: DiscrawlDoctorResult = okDoctor();
	public syncResult: DiscrawlSyncResult = okSync(5);
	public embedResult: DiscrawlEmbedResult = okEmbed();
	public searchResult: DiscrawlSearchResult = okSearch([]);
	public searchCalls: { mode: string; query: string }[] = [];
	public embedCalled = false;

	async doctor(): Promise<DiscrawlDoctorResult> {
		return this.doctorResult;
	}
	async sync(): Promise<DiscrawlSyncResult> {
		return this.syncResult;
	}
	async embed(): Promise<DiscrawlEmbedResult> {
		this.embedCalled = true;
		return this.embedResult;
	}
	async search(mode: string, query: string): Promise<DiscrawlSearchResult> {
		this.searchCalls.push({ mode, query });
		return this.searchResult;
	}
}

function asClient(stub: StubClient): DiscrawlSkillClient {
	return stub as unknown as DiscrawlSkillClient;
}

const HITS: readonly DiscrawlSearchHit[] = [
	{
		messageId: "1513467741523415161",
		content: "공금 장부 잔액은 27,136원입니다",
		score: 0.87,
		channelName: "general",
		guildId: "1512354544628011068",
		authorName: "리누스형",
		timestamp: "2026-08-11T10:50:30Z",
	},
];

describe("DiscrawlSkill descriptor", () => {
	it("publishes discord id, archive type, and external-cli capabilities", () => {
		const descriptor = new DiscrawlSkill({ client: asClient(new StubClient()) }).describe();
		expect(descriptor).toMatchObject({
			id: "discord",
			name: "discord",
			type: "discord-archive",
			instanceId: "default",
			requiresExternalCli: true,
		});
		expect(descriptor.capabilities).toEqual(expect.arrayContaining(["fts5", "semantic", "hybrid", "incremental"]));
	});

	it("honors a custom instance id and tags", () => {
		const descriptor = new DiscrawlSkill({
			client: asClient(new StubClient()),
			instanceId: "guild-1",
			tags: ["discord", "internal"],
		}).describe();
		expect(descriptor).toMatchObject({ instanceId: "guild-1", tags: ["discord", "internal"] });
	});
});

describe("DiscrawlSkill index", () => {
	it("runs doctor then sync then embed and reports synced message count", async () => {
		const stub = new StubClient();
		const result = await new DiscrawlSkill({ client: asClient(stub) }).index();
		expect(result).toMatchObject({ ok: true, skill: "discord", chunkCount: 5 });
		expect(stub.embedCalled).toBe(true);
	});

	it("skips embed and warns when embeddings are not configured", async () => {
		const stub = new StubClient();
		stub.doctorResult = okDoctor({ embeddingsOk: false });
		const result = await new DiscrawlSkill({ client: asClient(stub) }).index();
		expect(result.ok).toBe(true);
		expect(stub.embedCalled).toBe(false);
		expect(result.diagnostics.some((d) => d.message.includes("embeddings are not configured"))).toBe(true);
	});

	it("fails when the archive database is not ready", async () => {
		const stub = new StubClient();
		stub.doctorResult = okDoctor({ databaseOk: false });
		expect(await new DiscrawlSkill({ client: asClient(stub) }).index()).toMatchObject({
			ok: false,
			code: "datasource-unavailable",
		});
	});

	it("maps a missing binary to datasource-unavailable instead of throwing", async () => {
		const stub = new StubClient();
		stub.doctorResult = { ok: false, reason: "binary-missing", stdout: "", stderr: "", code: null };
		expect(await new DiscrawlSkill({ client: asClient(stub) }).index()).toMatchObject({
			ok: false,
			code: "datasource-unavailable",
		});
	});

	it("maps a rejected user token to permission-denied", async () => {
		const stub = new StubClient();
		stub.doctorResult = {
			ok: false,
			reason: "user-token-rejected",
			stdout: "",
			stderr: "",
			code: null,
			violatingKey: "DISCORD_USER_TOKEN",
		};
		expect(await new DiscrawlSkill({ client: asClient(stub) }).index()).toMatchObject({
			ok: false,
			code: "datasource-permission-denied",
		});
	});

	it("warns when an English-only embedding model is configured", async () => {
		const stub = new StubClient();
		const result = await new DiscrawlSkill({
			client: asClient(stub),
			embeddingModel: "nomic-embed-text",
		}).index();
		expect(result.ok).toBe(true);
		const warning = result.diagnostics.find((d) => d.message.includes("English-only"));
		expect(warning).toBeDefined();
		expect(warning?.message).toContain(DEFAULT_DISCRAWL_EMBEDDING_MODEL);
	});

	it("warns for the actual English-only model reported by doctor", async () => {
		const stub = new StubClient();
		stub.doctorResult = okDoctor({ embeddingModel: "nomic-embed-text" });
		const result = await new DiscrawlSkill({ client: asClient(stub) }).index();

		expect(result.ok).toBe(true);
		expect(result.diagnostics.some((d) => d.message.includes('"nomic-embed-text" is English-only'))).toBe(true);
	});

	it("does not warn for a multilingual embedding model", async () => {
		const result = await new DiscrawlSkill({
			client: asClient(new StubClient()),
			embeddingModel: "bge-m3",
		}).index();
		expect(result.diagnostics.some((d) => d.message.includes("English-only"))).toBe(false);
	});

	it("reports an empty sync as an info diagnostic", async () => {
		const stub = new StubClient();
		stub.syncResult = okSync(0);
		const result = await new DiscrawlSkill({ client: asClient(stub) }).index();
		expect(result.ok).toBe(true);
		expect(result.diagnostics.some((d) => d.code === "datasource-empty")).toBe(true);
	});

	it("degrades a failed embed pass to a warning without failing the index", async () => {
		const stub = new StubClient();
		stub.embedResult = { ok: false, reason: "nonzero-exit", stdout: "", stderr: "", code: 1 };
		const result = await new DiscrawlSkill({ client: asClient(stub) }).index();
		expect(result.ok).toBe(true);
		expect(result.diagnostics.some((d) => d.message.includes("discrawl embed failed"))).toBe(true);
	});
});

describe("DiscrawlSkill retrieval methods", () => {
	it("orders hybrid first by default", () => {
		const names = new DiscrawlSkill({ client: asClient(new StubClient()) })
			.retrievalMethods()
			.map((m) => m.describe().name);
		expect(names[0]).toBe("discord-hybrid");
		expect(names).toEqual(expect.arrayContaining(["discord-fts", "discord-semantic"]));
	});

	it("honors an explicit fts default mode", () => {
		const names = new DiscrawlSkill({ client: asClient(new StubClient()), defaultMode: "fts" })
			.retrievalMethods()
			.map((m) => m.describe().name);
		expect(names[0]).toBe("discord-fts");
	});

	it("maps hits to traceable /discord sources with guild and channel metadata", async () => {
		const stub = new StubClient();
		stub.searchResult = okSearch(HITS);
		const skill = new DiscrawlSkill({ client: asClient(stub), instanceId: "guild-1" });
		const [hybrid] = skill.retrievalMethods();
		const results = await hybrid?.retrieve("공금 잔액", { topK: 5 });
		expect(results?.[0]?.source).toBe("/discord/guild-1/chunks/1513467741523415161");
		expect(results?.[0]?.metadata).toMatchObject({
			datasourceId: "discord",
			channelName: "general",
			authorName: "리누스형",
			mode: "hybrid",
		});
	});

	it("returns no results when the CLI fails, never throwing", async () => {
		const stub = new StubClient();
		stub.searchResult = { ok: false, reason: "nonzero-exit", stdout: "", stderr: "", code: 1 };
		const [hybrid] = new DiscrawlSkill({ client: asClient(stub) }).retrievalMethods();
		await expect(hybrid?.retrieve("anything", { topK: 5 })).resolves.toEqual([]);
	});

	it("skips the CLI entirely for an empty query", async () => {
		const stub = new StubClient();
		const [hybrid] = new DiscrawlSkill({ client: asClient(stub) }).retrievalMethods();
		await expect(hybrid?.retrieve("   ", { topK: 5 })).resolves.toEqual([]);
		expect(stub.searchCalls).toHaveLength(0);
	});

	it("drops hits outside the requested scope", async () => {
		const stub = new StubClient();
		stub.searchResult = okSearch(HITS);
		const [hybrid] = new DiscrawlSkill({ client: asClient(stub), instanceId: "guild-1" }).retrievalMethods();
		const results = await hybrid?.retrieve("공금", { topK: 5, scope: "/discord/other-guild" });
		expect(results).toEqual([]);
	});

	it("keeps hits inside an allowed scope", async () => {
		const stub = new StubClient();
		stub.searchResult = okSearch(HITS);
		const [hybrid] = new DiscrawlSkill({ client: asClient(stub), instanceId: "guild-1" }).retrievalMethods();
		const results = await hybrid?.retrieve("공금", { topK: 5, allowedScopes: ["/discord/guild-1/**"] });
		expect(results).toHaveLength(1);
	});
});

describe("DiscrawlSkill manifest", () => {
	it("documents hybrid default and the FTS newline caveat without leaking paths", () => {
		const manifest = new DiscrawlSkill({ client: asClient(new StubClient()) }).skillManifest();
		expect(manifest.name).toBe("datasource-discord");
		expect(manifest.content).toContain("Hybrid retrieval is the default");
		expect(manifest.content).toContain("line breaks");
		expect(manifest.content).not.toContain("/Users/");
	});
});
