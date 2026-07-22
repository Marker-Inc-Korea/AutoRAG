import { describe, expect, it } from "vitest";
import { NotionConnector } from "../../../src/datasource/skills/notion/connector.ts";
import { NotionSkill } from "../../../src/datasource/skills/notion/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const SEARCH_RESULTS = {
	results: [
		{
			object: "page",
			id: "page-1",
			last_edited_time: "2024-02-01T00:00:00.000Z",
			parent: { type: "database_id", database_id: "db-9" },
			properties: { Name: { type: "title", title: [{ plain_text: "Onboarding guide" }] } },
		},
		{
			object: "database",
			id: "db-9",
			last_edited_time: "2024-02-02T00:00:00.000Z",
			title: [{ plain_text: "Team wiki" }],
		},
	],
	has_more: false,
};

const PAGE_BLOCKS = {
	results: [
		{ type: "heading_1", heading_1: { rich_text: [{ plain_text: "Welcome" }] } },
		{ type: "paragraph", paragraph: { rich_text: [{ plain_text: "New hires meet the buddy on day one." }] } },
		{ type: "unsupported_widget", unsupported_widget: {} },
	],
};

describe("NotionConnector", () => {
	it("returns not-configured without a token", async () => {
		expect(await new NotionConnector({ tokenEnv: "NOTION_TEST_UNSET" }).fetch()).toMatchObject({
			ok: false,
			reason: "not-configured",
		});
	});

	it("builds documents from search results and block children", async () => {
		const mock = createMockFetch([
			{ match: "/search", json: SEARCH_RESULTS },
			{ match: "/blocks/page-1/children", json: PAGE_BLOCKS },
		]);
		const result = await new NotionConnector({ token: "secret", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2);
			const page = result.documents.find((d) => d.docId === "page-1");
			expect(page).toMatchObject({ title: "Onboarding guide", hierarchy: ["databases", "db-9"] });
			expect(page?.content).toContain("New hires meet the buddy");
			const database = result.documents.find((d) => d.docId === "db-9");
			expect(database).toMatchObject({ title: "Team wiki", hierarchy: ["databases", "db-9"] });
		}
	});

	it("degrades per-page block failures to warnings but keeps the title document", async () => {
		const mock = createMockFetch([
			{ match: "/search", json: SEARCH_RESULTS },
			{ match: "/blocks/page-1/children", status: 500 },
		]);
		const result = await new NotionConnector({ token: "secret", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents.map((d) => d.docId)).toContain("page-1");
			expect(result.warnings?.length).toBeGreaterThan(0);
		}
	});

	it("maps 401 to auth and 429 to rate-limited", async () => {
		const authMock = createMockFetch([{ match: "/search", status: 401 }]);
		expect(await new NotionConnector({ token: "t", fetchImpl: authMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "auth",
		});
		const rateMock = createMockFetch([{ match: "/search", status: 429 }]);
		expect(await new NotionConnector({ token: "t", fetchImpl: rateMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "rate-limited",
		});
	});
});

describe("NotionSkill", () => {
	it("indexes and searches with opaque /notion sources", async () => {
		const mock = createMockFetch([
			{ match: "/search", json: SEARCH_RESULTS },
			{ match: "/blocks/page-1/children", json: PAGE_BLOCKS },
		]);
		const skill = new NotionSkill({
			instanceId: "ws-1",
			connectorOptions: { token: "secret-token", fetchImpl: mock.fetchImpl },
		});
		expect(skill.describe()).toMatchObject({ name: "notion", datasourceId: "notion" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("onboarding buddy", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/notion\/ws-1\/chunks\//);
		expect(JSON.stringify(hits)).not.toContain("secret-token");
	});
});
