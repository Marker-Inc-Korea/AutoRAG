import { describe, expect, it } from "vitest";
import { SlackConnector } from "../../../src/datasource/skills/slack/connector.ts";
import { SlackSkill } from "../../../src/datasource/skills/slack/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const CHANNELS = {
	ok: true,
	channels: [
		{ id: "C01", name: "general" },
		{ id: "C02", name: "random" },
	],
	response_metadata: { next_cursor: "" },
};

const GENERAL_HISTORY = {
	ok: true,
	messages: [
		{ ts: "1700000001.000100", user: "U1", text: "Deploy freeze starts Friday" },
		{ ts: "1700000002.000200", user: "U2", text: "", subtype: "channel_join" },
	],
	response_metadata: { next_cursor: "" },
};

describe("SlackConnector", () => {
	it("returns not-configured without a token", async () => {
		const connector = new SlackConnector({ tokenEnv: "SLACK_TEST_TOKEN_UNSET" });
		expect(await connector.fetch()).toMatchObject({ ok: false, reason: "not-configured" });
	});

	it("fetches channel history into documents with channel hierarchy", async () => {
		const mock = createMockFetch([
			{ match: "conversations.list", json: CHANNELS },
			{ match: "conversations.history?channel=C01", json: GENERAL_HISTORY },
			{ match: "conversations.history?channel=C02", json: { ok: true, messages: [] } },
		]);
		const connector = new SlackConnector({ token: "xoxb-test", fetchImpl: mock.fetchImpl });

		const result = await connector.fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(1);
			expect(result.documents[0]).toMatchObject({
				docId: "C01-1700000001-000100",
				hierarchy: ["channels", "general"],
				content: "U1: Deploy freeze starts Friday",
			});
		}
		expect(mock.requests.some((r) => r.includes("xoxb-test"))).toBe(false);
	});

	it("maps invalid_auth to auth and ratelimited to rate-limited", async () => {
		const authMock = createMockFetch([{ match: "conversations.list", json: { ok: false, error: "invalid_auth" } }]);
		expect(await new SlackConnector({ token: "t", fetchImpl: authMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "auth",
		});
		const rateMock = createMockFetch([{ match: "conversations.list", json: { ok: false, error: "ratelimited" } }]);
		expect(await new SlackConnector({ token: "t", fetchImpl: rateMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "rate-limited",
		});
	});

	it("degrades per-channel scope failures to warnings and keeps other channels", async () => {
		const mock = createMockFetch([
			{ match: "conversations.list", json: CHANNELS },
			{ match: "conversations.history?channel=C01", json: { ok: false, error: "missing_scope" } },
			{ match: "conversations.history?channel=C02", json: GENERAL_HISTORY },
		]);
		const result = await new SlackConnector({ token: "t", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(1);
			expect(result.warnings?.some((w) => w.includes("missing_scope"))).toBe(true);
		}
	});

	it("honors the channels filter", async () => {
		const mock = createMockFetch([
			{ match: "conversations.list", json: CHANNELS },
			{ match: "conversations.history?channel=C01", json: GENERAL_HISTORY },
		]);
		const result = await new SlackConnector({
			token: "t",
			fetchImpl: mock.fetchImpl,
			channels: ["general"],
		}).fetch();
		expect(result.ok).toBe(true);
		expect(mock.requests.some((r) => r.includes("channel=C02"))).toBe(false);
	});

	it("maps network failure to unavailable", async () => {
		const mock = createMockFetch([{ match: "conversations.list", networkError: true }]);
		expect(await new SlackConnector({ token: "t", fetchImpl: mock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "unavailable",
		});
	});
});

describe("SlackSkill", () => {
	it("indexes and searches through the skill surface with opaque sources", async () => {
		const mock = createMockFetch([
			{ match: "conversations.list", json: CHANNELS },
			{ match: "conversations.history?channel=C01", json: GENERAL_HISTORY },
			{ match: "conversations.history?channel=C02", json: { ok: true, messages: [] } },
		]);
		const skill = new SlackSkill({
			instanceId: "ws-1",
			connectorOptions: { token: "xoxb-test", fetchImpl: mock.fetchImpl },
		});

		expect(skill.describe()).toMatchObject({ name: "slack", datasourceId: "slack", type: "slack-workspace" });
		const indexResult = await skill.index();
		expect(indexResult).toMatchObject({ ok: true, chunkCount: 1 });

		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("deploy freeze", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/slack\/ws-1\/chunks\//);
		expect(JSON.stringify(hits)).not.toContain("xoxb-test");

		const manifest = skill.skillManifest();
		expect(manifest.name).toBe("datasource-slack");
		expect(manifest.content).toContain("/slack/ws-1");
	});
});
