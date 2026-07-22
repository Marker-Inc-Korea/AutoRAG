import { describe, expect, it } from "vitest";
import { DiscordConnector } from "../../../src/datasource/skills/discord/connector.ts";
import { DiscordSkill } from "../../../src/datasource/skills/discord/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const CHANNELS = [
	{ id: "100", name: "general", type: 0 },
	{ id: "200", name: "voice-lounge", type: 2 },
	{ id: "300", name: "announcements", type: 5 },
];

const GENERAL_MESSAGES = [
	{
		id: "9001",
		content: "Server maintenance tonight at 10pm",
		timestamp: "2024-01-05T10:00:00.000Z",
		author: { username: "mod" },
	},
	{ id: "9000", content: "", timestamp: "2024-01-05T09:00:00.000Z", author: { username: "bot" } },
];

describe("DiscordConnector", () => {
	it("returns not-configured without token or guild id", async () => {
		expect(await new DiscordConnector({ tokenEnv: "DISCORD_TEST_UNSET" }).fetch()).toMatchObject({
			ok: false,
			reason: "not-configured",
		});
		expect(await new DiscordConnector({ token: "t" }).fetch()).toMatchObject({
			ok: false,
			reason: "not-configured",
		});
	});

	it("fetches text and announcement channels only, skipping empty messages", async () => {
		const mock = createMockFetch([
			{ match: "/guilds/g1/channels", json: CHANNELS },
			{ match: "before=", json: [] },
			{ match: "/channels/100/messages", json: GENERAL_MESSAGES },
			{ match: "/channels/300/messages", json: [] },
		]);
		const result = await new DiscordConnector({ token: "t", guildId: "g1", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(1);
			expect(result.documents[0]).toMatchObject({
				docId: "100-9001",
				hierarchy: ["channels", "general"],
				content: "mod: Server maintenance tonight at 10pm",
			});
		}
		expect(mock.requests.some((r) => r.includes("/channels/200/"))).toBe(false);
	});

	it("degrades per-channel 403 to a warning and keeps the rest", async () => {
		const mock = createMockFetch([
			{ match: "/guilds/g1/channels", json: CHANNELS },
			{ match: "before=", json: [] },
			{ match: "/channels/100/messages", status: 403 },
			{ match: "/channels/300/messages", json: GENERAL_MESSAGES },
		]);
		const result = await new DiscordConnector({ token: "t", guildId: "g1", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(1);
			expect(result.warnings?.some((w) => w.includes("permission"))).toBe(true);
		}
	});

	it("fails the whole fetch as rate-limited on 429", async () => {
		const mock = createMockFetch([
			{ match: "/guilds/g1/channels", json: CHANNELS },
			{ match: "/channels/100/messages", status: 429 },
		]);
		expect(
			await new DiscordConnector({ token: "t", guildId: "g1", fetchImpl: mock.fetchImpl }).fetch(),
		).toMatchObject({ ok: false, reason: "rate-limited" });
	});

	it("maps guild-level 401 to auth", async () => {
		const mock = createMockFetch([{ match: "/guilds/g1/channels", status: 401 }]);
		expect(
			await new DiscordConnector({ token: "t", guildId: "g1", fetchImpl: mock.fetchImpl }).fetch(),
		).toMatchObject({ ok: false, reason: "auth" });
	});
});

describe("DiscordSkill", () => {
	it("indexes and searches with opaque /discord sources", async () => {
		const mock = createMockFetch([
			{ match: "/guilds/g1/channels", json: CHANNELS },
			{ match: "before=", json: [] },
			{ match: "/channels/100/messages", json: GENERAL_MESSAGES },
			{ match: "/channels/300/messages", json: [] },
		]);
		const skill = new DiscordSkill({
			instanceId: "guild-1",
			connectorOptions: { token: "secret-token", guildId: "g1", fetchImpl: mock.fetchImpl },
		});
		expect(skill.describe()).toMatchObject({ name: "discord", datasourceId: "discord" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("maintenance tonight", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/discord\/guild-1\/chunks\//);
		expect(JSON.stringify(hits)).not.toContain("secret-token");
	});
});
