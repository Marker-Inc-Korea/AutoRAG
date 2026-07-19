import { describe, expect, it } from "vitest";
import { KatokSkill, type KatokSkillClient } from "../../../../src/datasource/skills/katok/skill.ts";
import type { KatokFailureReason, KatokHit, KatokSearchResult } from "../../../../src/datasource/skills/katok/types.ts";

type StepResult =
	| { ok: true; stdout: string; stderr: string; code: number }
	| { ok: false; reason: KatokFailureReason; stdout: string; stderr: string; code: number | null };

class StubSkillClient {
	public doctorResult: StepResult = okStep();
	public syncResult: StepResult = okStep();
	public indexResult: StepResult = okStep();
	public searchResult: KatokSearchResult = okSearch([]);

	async doctor(): Promise<StepResult> {
		return this.doctorResult;
	}
	async sync(): Promise<StepResult> {
		return this.syncResult;
	}
	async index(): Promise<StepResult> {
		return this.indexResult;
	}
	async search(): Promise<KatokSearchResult> {
		return this.searchResult;
	}
}

function okStep(): StepResult {
	return { ok: true, stdout: "", stderr: "", code: 0 };
}

function failStep(reason: KatokFailureReason): StepResult {
	return { ok: false, reason, stdout: "", stderr: "katok: unavailable", code: 1 };
}

function okSearch(hits: readonly KatokHit[]): KatokSearchResult {
	return { ok: true, hits, data: { hits }, stdout: "", stderr: "", code: 0 };
}

function asClient(stub: StubSkillClient): KatokSkillClient {
	return stub as unknown as KatokSkillClient;
}

const HITS: readonly KatokHit[] = [
	{
		chunkId: "chunk-001",
		content: "refund policy approval workflow",
		score: 2.1,
		source: "/kakao/default/chunks/chunk-001",
		metadata: { room: "support-ops" },
	},
];

describe("KatokSkill descriptor", () => {
	it("publishes kakao id, kakaotalk type, default instance, and pii tags", () => {
		const skill = new KatokSkill({ client: asClient(new StubSkillClient()) });
		const descriptor = skill.describe();

		expect(descriptor).toMatchObject({
			id: "kakao",
			type: "kakaotalk",
			instanceId: "default",
			tags: ["kakaotalk", "personal", "pii"],
		});
	});

	it("honors a custom instance id and tags", () => {
		const skill = new KatokSkill({
			client: asClient(new StubSkillClient()),
			instanceId: "work",
			tags: ["kakaotalk", "team"],
		});

		expect(skill.describe()).toMatchObject({ id: "kakao", instanceId: "work" });
		expect(skill.describe().tags).toEqual(["kakaotalk", "team"]);
	});
});

describe("KatokSkill skillManifest (Pi agent-skill layer)", () => {
	it("exposes a progressive-disclosure manifest with path-opaque authorized scopes in content", () => {
		const skill = new KatokSkill({ client: asClient(new StubSkillClient()), instanceId: "work" });
		const manifest = skill.skillManifest();

		expect(manifest.name).toBe("datasource-kakao");
		expect(manifest.description.toLowerCase()).toContain("kakaotalk");
		expect(manifest.content).toContain("search_datasource_documents");
		expect(manifest.content).toContain("/kakao/work");
		expect(manifest.content).not.toContain("/Users/");
		expect(manifest.content).not.toContain("Library/Containers");
	});
});

describe("KatokSkill polling", () => {
	it("defaults to poll mode at a 15 minute interval with no lastIndexedAt", () => {
		const skill = new KatokSkill({ client: asClient(new StubSkillClient()) });
		const polling = skill.polling();

		expect(polling).toEqual({
			mode: "poll",
			intervalMs: 15 * 60 * 1000,
			lastIndexedAt: undefined,
		});
	});

	it("honors a custom polling interval and seeded lastIndexedAt", () => {
		const skill = new KatokSkill({
			client: asClient(new StubSkillClient()),
			pollingIntervalMs: 60_000,
			lastIndexedAt: 1_700_000_000_000,
		});

		expect(skill.polling()).toMatchObject({
			mode: "poll",
			intervalMs: 60_000,
			lastIndexedAt: 1_700_000_000_000,
		});
	});

	it("sets lastIndexedAt after a successful index", async () => {
		const skill = new KatokSkill({ client: asClient(new StubSkillClient()) });
		expect(skill.polling().lastIndexedAt).toBeUndefined();

		const result = await skill.index();

		expect(result.ok).toBe(true);
		expect(skill.polling().lastIndexedAt).toEqual(expect.any(Number));
		if (result.ok) {
			expect(result.indexedAt).toEqual(skill.polling().lastIndexedAt);
		}
	});
});

describe("KatokSkill index", () => {
	it("runs doctor -> sync -> index and returns ok with indexedAt", async () => {
		const stub = new StubSkillClient();
		const skill = new KatokSkill({ client: asClient(stub) });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: true });
		if (result.ok) {
			expect(result.indexedAt).toEqual(expect.any(Number));
		}
	});

	it("returns datasource-unavailable without throwing when doctor fails (binary missing)", async () => {
		const stub = new StubSkillClient();
		stub.doctorResult = failStep("binary-missing");
		const skill = new KatokSkill({ client: asClient(stub) });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect(JSON.stringify(result)).toContain("binary-missing");
	});

	it("maps remote embedding rejection to an egress policy diagnostic", async () => {
		const stub = new StubSkillClient();
		stub.doctorResult = failStep("remote-embedding-rejected");
		const skill = new KatokSkill({ client: asClient(stub) });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-embedding-egress-rejected" });
		expect(result.diagnostics[0]?.code).toBe("datasource-embedding-egress-rejected");
	});

	it("returns datasource-index-failed when sync fails", async () => {
		const stub = new StubSkillClient();
		stub.syncResult = failStep("nonzero-exit");
		const skill = new KatokSkill({ client: asClient(stub) });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-index-failed" });
	});

	it("returns datasource-index-failed when index fails (degraded)", async () => {
		const stub = new StubSkillClient();
		stub.indexResult = failStep("invalid-json");
		const skill = new KatokSkill({ client: asClient(stub) });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-index-failed" });
	});

	it("does not set lastIndexedAt on failure", async () => {
		const stub = new StubSkillClient();
		stub.doctorResult = failStep("binary-missing");
		const skill = new KatokSkill({ client: asClient(stub) });

		await skill.index();

		expect(skill.polling().lastIndexedAt).toBeUndefined();
	});

	it("returns datasource-unavailable without throwing when a step throws", async () => {
		const stub = new StubSkillClient();
		(stub as unknown as { doctor: () => Promise<never> }).doctor = async () => {
			throw new Error("spawn ENOENT");
		};
		const skill = new KatokSkill({ client: asClient(stub) });

		const result = await skill.index();

		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect(JSON.stringify(result)).not.toContain("spawn ENOENT");
	});
});

describe("KatokSkill retrievalMethods", () => {
	it("returns kakao-bm25 and kakao-semantic methods sharing the client and instance", async () => {
		const stub = new StubSkillClient();
		stub.searchResult = okSearch(HITS);
		const skill = new KatokSkill({
			client: asClient(stub),
			instanceId: "default",
		});

		const methods = skill.retrievalMethods();
		const descriptors = methods.map((m) => m.describe());

		expect(descriptors.map((d) => d.name)).toEqual(["kakao-bm25", "kakao-semantic"]);
		expect(descriptors.map((d) => d.type)).toEqual(["bm25", "vector"]);
		expect(descriptors.every((d) => d.datasourceId === "kakao")).toBe(true);

		const [bm25] = methods;
		const results = await bm25.retrieve("refund", { topK: 1 });
		expect(results.map((r) => r.source)).toEqual(["/kakao/default/chunks/chunk-001"]);
		expect(results[0]?.metadata).toMatchObject({ method: "kakao-bm25", instanceId: "default" });
	});

	it("degrades to [] when the underlying search fails", async () => {
		const stub = new StubSkillClient();
		stub.searchResult = {
			ok: false,
			reason: "binary-missing",
			hits: [],
			stdout: "",
			stderr: "katok: unavailable",
			code: null,
		};
		const skill = new KatokSkill({ client: asClient(stub) });

		const [bm25] = skill.retrievalMethods();
		const results = await bm25.retrieve("refund", {});

		expect(results).toEqual([]);
	});
});

describe("KatokSkill describeSources", () => {
	it("surfaces a single path-opaque instance source by default", () => {
		const skill = new KatokSkill({ client: asClient(new StubSkillClient()) });
		const sources = skill.describeSources();

		expect(sources).toHaveLength(1);
		expect(sources[0]?.source).toBe("/kakao/default");
		expect(sources[0]?.chunkId).toBeUndefined();
		expect(sources[0]?.metadata).toMatchObject({
			datasourceId: "kakao",
			instanceId: "default",
		});
	});

	it("surfaces one path-opaque source per configured instance", () => {
		const skill = new KatokSkill({
			client: asClient(new StubSkillClient()),
			instances: ["default", "work"],
		});
		const sources = skill.describeSources();

		expect(sources.map((s) => s.source)).toEqual(["/kakao/default", "/kakao/work"]);
		for (const source of sources) {
			expect(source.source).toMatch(/^\/kakao\/[^/]+$/u);
			expect(source.chunkId).toBeUndefined();
		}
	});

	it("keeps sources path-opaque with no chunk ids in instance sources", () => {
		const skill = new KatokSkill({ client: asClient(new StubSkillClient()) });
		const sources = skill.describeSources();

		for (const source of sources) {
			expect(source.source).not.toContain("chunks");
			expect(source.source).not.toMatch(/[A-Za-z]:[\\/]/u);
		}
	});
});
