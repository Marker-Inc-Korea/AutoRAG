import { describe, expect, it } from "vitest";
import { DatasourceAccessContext } from "../../src/datasource/access-context.ts";
import { DatasourceSkillRegistry, type RegisteredDatasourceSkill } from "../../src/datasource/registry.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	PollingMetadata,
	RetrievalMethod,
	SourceDescription,
} from "../../src/datasource/types.ts";

const polling: PollingMetadata = { mode: "poll", intervalMs: 60_000, lastIndexedAt: 0 };

const makeSkill = (
	overrides: Partial<DatasourceSkillDescriptor> = {},
	methods: readonly RetrievalMethod[] = [],
): DatasourceSkill => {
	const descriptor: DatasourceSkillDescriptor = {
		name: "kakao",
		type: "kakao",
		description: "KakaoTalk via katok",
		capabilities: ["chat"],
		tags: ["kakao"],
		status: "active",
		requiresExternalCli: true,
		datasourceId: "kakao",
		instances: ["acct-1"],
		...overrides,
	};
	return {
		describe: () => descriptor,
		polling: () => polling,
		index: async (): Promise<DatasourceIndexResult> => ({
			ok: true,
			instanceId: "acct-1",
			skill: "kakao",
			chunkCount: 0,
			indexedAt: 0,
			diagnostics: [],
		}),
		retrievalMethods: () => methods,
		describeSources: (): readonly SourceDescription[] => [],
		skillManifest: () => ({ name: "datasource-kakao", description: "KakaoTalk chats.", content: "# KakaoTalk" }),
	};
};

describe("DatasourceSkillRegistry", () => {
	describe("register / list", () => {
		it("registers and lists skills", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill());
			registry.register(makeSkill({ name: "slack", tags: ["slack"], instances: ["ws-1"] }));
			const list = registry.list();
			expect(list.map((e) => e.descriptor.name)).toEqual(["kakao", "slack"]);
		});

		it("caches the descriptor on the registered entry", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill());
			const entry: RegisteredDatasourceSkill | undefined = registry.get("kakao");
			expect(entry).toBeDefined();
			expect(entry?.descriptor.name).toBe("kakao");
		});
	});

	describe("duplicate skill id", () => {
		it("throws when registering a skill with an already-used name", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill());
			expect(() => registry.register(makeSkill())).toThrow(/already registered/);
		});

		it("throws when registering a skill with an empty name", () => {
			const registry = new DatasourceSkillRegistry();
			expect(() => registry.register(makeSkill({ name: "" }))).toThrow(/empty name/);
		});
	});

	describe("byTag", () => {
		it("returns only skills carrying the requested tag", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ name: "kakao", tags: ["kakao", "chat"] }));
			registry.register(makeSkill({ name: "slack", tags: ["slack", "chat"] }));
			registry.register(makeSkill({ name: "drive", tags: ["docs"] }));
			const chatSkills = registry.byTag("chat");
			expect(chatSkills.map((e) => e.descriptor.name).sort()).toEqual(["kakao", "slack"]);
			expect(registry.byTag("missing")).toEqual([]);
		});
	});

	describe("resolveInstances", () => {
		it("resolves only instances accessible under the context", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ name: "kakao", tags: ["kakao"], instances: ["acct-1", "acct-2"] }));
			registry.register(makeSkill({ name: "slack", tags: ["slack"], instances: ["ws-1"] }));
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
			});
			const instances = registry.resolveInstances(ctx);
			expect(instances.map((i) => i.id)).toEqual(["acct-1", "acct-2"]);
			expect(instances.every((i) => i.skill.describe().name === "kakao")).toBe(true);
		});

		it("builds opaque slash-hierarchical sourcePaths from trusted skill name + id", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ name: "kakao", tags: ["kakao"], instances: ["acct-1"] }));
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] });
			const [instance] = registry.resolveInstances(ctx);
			expect(instance.sourcePath).toBe("/kakao/acct-1");
			expect(instance.descriptor.name).toBe("kakao");
			expect(instance.polling).toBe(polling);
		});

		it("does not resolve instances when tags match but no trusted scopes are granted", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ name: "kakao", tags: ["kakao"], instances: ["acct-1"] }));
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(registry.resolveInstances(ctx)).toEqual([]);
		});

		it("resolves nothing under a deny-all context", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ instances: ["acct-1"] }));
			const ctx = new DatasourceAccessContext(); // deny-all
			expect(registry.resolveInstances(ctx)).toEqual([]);
		});

		it("resolves nothing for a skill with no declared instances", () => {
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ instances: [] }));
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1"] });
			expect(registry.resolveInstances(ctx)).toEqual([]);
		});
	});

	describe("accessibleMethods", () => {
		it("exposes retrieval methods only from accessible skills", () => {
			const kakaoMethod: RetrievalMethod = {
				describe: () => ({
					name: "kakao-vector",
					type: "vector",
					description: "",
					status: "active",
					capabilities: [],
					datasourceId: "kakao",
					tags: ["kakao"],
				}),
				retrieve: async () => [],
			};
			const slackMethod: RetrievalMethod = {
				describe: () => ({
					name: "slack-vector",
					type: "vector",
					description: "",
					status: "active",
					capabilities: [],
					datasourceId: "slack",
					tags: ["slack"],
				}),
				retrieve: async () => [],
			};
			const registry = new DatasourceSkillRegistry();
			registry.register(makeSkill({ name: "kakao", tags: ["kakao"], instances: ["acct-1"] }, [kakaoMethod]));
			registry.register(makeSkill({ name: "slack", tags: ["slack"], instances: ["ws-1"] }, [slackMethod]));
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			const methods = registry.accessibleMethods(ctx);
			expect(methods.map((m) => m.describe().name)).toEqual(["kakao-vector"]);
		});
	});
});
