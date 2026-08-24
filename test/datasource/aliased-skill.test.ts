import { describe, expect, it } from "vitest";
import { AliasedDatasourceSkill } from "../../src/datasource/aliased-skill.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";

function chatSkill(): DatasourceSkill {
	return {
		describe: () => ({
			name: "chat",
			id: "chat",
			type: "chat",
			description: "chat",
			capabilities: ["chat"],
			tags: ["chat"],
			status: "active",
			datasourceId: "chat",
			instanceId: "default",
			instances: ["default"],
		}),
		polling: () => ({ mode: "none" }),
		index: async () => ({
			ok: true,
			instanceId: "default",
			skill: "chat",
			chunkCount: 2,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [
			{
				describe: () => ({
					name: "chat-fts",
					type: "bm25",
					description: "chat search",
					status: "active",
					capabilities: ["chat"],
					datasourceId: "chat",
					tags: ["chat"],
				}),
				retrieve: async () => [
					{ id: "1", content: "family", source: "/chat/default/1", score: 1, metadata: { chatName: "가족방" } },
					{
						id: "2",
						content: "work",
						source: "/chat/default/2",
						score: 0.9,
						metadata: { channelName: "engineering" },
					},
				],
			},
		],
		describeSources: () => [
			{
				source: "/chat/default/**",
				datasourceId: "chat",
				skill: "chat",
				instanceId: "default",
				contentType: "chat",
				metadata: {},
			},
		],
		skillManifest: () => ({ name: "datasource-chat", description: "chat", content: "search /chat/default/**" }),
	};
}

it("rewrites alias scopes before delegating retrieval", async () => {
	const received: { scope?: string; allowedScopes?: readonly string[] }[] = [];
	const base = chatSkill();
	const originalMethod = base.retrievalMethods()[0];
	base.retrievalMethods = () => [
		{
			...originalMethod,
			retrieve: async (_query, options) => {
				received.push(options);
				return originalMethod.retrieve(_query, options);
			},
		},
	];

	const aliased = new AliasedDatasourceSkill(base, { alias: "account-a" });
	await aliased.retrievalMethods()[0].retrieve("query", {
		scope: "/account-a/channel-1",
		allowedScopes: ["/account-a/channel-1/**"],
	});

	expect(received).toEqual([
		{
			scope: "/chat/channel-1",
			allowedScopes: ["/chat/channel-1/**"],
		},
	]);
});

describe("AliasedDatasourceSkill", () => {
	it("searches every channel by default", async () => {
		const skill = new AliasedDatasourceSkill(chatSkill(), { alias: "all-chats" });
		const results = await skill.retrievalMethods()[0]?.retrieve("hello", { topK: 5 });
		expect(results).toHaveLength(2);
		expect(results?.every((result) => result.source.startsWith("/all-chats/"))).toBe(true);
	});

	it("restricts an alias to explicitly configured channel names", async () => {
		const skill = new AliasedDatasourceSkill(chatSkill(), { alias: "family-kakao", channelNames: ["가족방"] });
		const results = await skill.retrievalMethods()[0]?.retrieve("hello", { topK: 5 });
		expect(results?.map((result) => result.content)).toEqual(["family"]);
		expect(skill.skillManifest().content).toContain("가족방");
	});
});
