import { describe, expect, it } from "vitest";
import { GDriveConnector } from "../../../src/datasource/skills/gdrive/connector.ts";
import { GDriveSkill } from "../../../src/datasource/skills/gdrive/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

const FILE_LIST = {
	files: [
		{
			id: "doc-1",
			name: "Q3 plan",
			mimeType: "application/vnd.google-apps.document",
			modifiedTime: "2024-04-01T00:00:00.000Z",
		},
		{
			id: "sheet-1",
			name: "Budget",
			mimeType: "application/vnd.google-apps.spreadsheet",
			modifiedTime: "2024-04-02T00:00:00.000Z",
		},
		{ id: "img-1", name: "logo.png", mimeType: "image/png" },
		{ id: "txt-1", name: "notes.txt", mimeType: "text/plain" },
	],
};

describe("GDriveConnector", () => {
	it("returns not-configured without a token", async () => {
		expect(await new GDriveConnector({ tokenEnv: "GDRIVE_TEST_UNSET" }).fetch()).toMatchObject({
			ok: false,
			reason: "not-configured",
		});
	});

	it("exports Docs/Sheets and downloads text files, skipping binaries", async () => {
		const mock = createMockFetch([
			{ match: "/files?", json: FILE_LIST },
			{ match: "/files/doc-1/export", text: "Q3 plan: expand to EU market." },
			{ match: "/files/sheet-1/export", text: "item,cost\nservers,1000" },
			{ match: "/files/txt-1?alt=media", text: "Meeting notes from Monday." },
		]);
		const result = await new GDriveConnector({ token: "ya29.secret", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents.map((d) => d.docId).sort()).toEqual(["doc-1", "sheet-1", "txt-1"]);
			expect(result.documents.find((d) => d.docId === "doc-1")?.content).toContain("EU market");
		}
		expect(mock.requests.some((r) => r.includes("/files/img-1"))).toBe(false);
	});

	it("degrades per-file export failures to warnings", async () => {
		const mock = createMockFetch([
			{ match: "/files?", json: FILE_LIST },
			{ match: "/files/doc-1/export", status: 500 },
			{ match: "/files/sheet-1/export", text: "a,b" },
			{ match: "/files/txt-1?alt=media", text: "notes" },
		]);
		const result = await new GDriveConnector({ token: "t", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents.map((d) => d.docId)).not.toContain("doc-1");
			expect(result.warnings?.some((w) => w.includes("http-500"))).toBe(true);
		}
	});

	it("restricts the listing to a configured folder", async () => {
		const mock = createMockFetch([{ match: "/files?", json: { files: [] } }]);
		await new GDriveConnector({ token: "t", folderId: "folder-9", fetchImpl: mock.fetchImpl }).fetch();
		expect(mock.requests[0]).toContain("folder-9");
		expect(mock.requests[0]).toContain("in+parents");
	});

	it("maps 401 to auth", async () => {
		const mock = createMockFetch([{ match: "/files?", status: 401 }]);
		expect(await new GDriveConnector({ token: "t", fetchImpl: mock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "auth",
		});
	});
});

describe("GDriveSkill", () => {
	it("indexes and searches with opaque /gdrive sources", async () => {
		const mock = createMockFetch([
			{ match: "/files?", json: FILE_LIST },
			{ match: "/files/doc-1/export", text: "Q3 plan: expand to EU market." },
			{ match: "/files/sheet-1/export", text: "item,cost" },
			{ match: "/files/txt-1?alt=media", text: "Meeting notes." },
		]);
		const skill = new GDriveSkill({
			instanceId: "acct-1",
			connectorOptions: { token: "ya29.secret", fetchImpl: mock.fetchImpl },
		});
		expect(skill.describe()).toMatchObject({ name: "gdrive", datasourceId: "gdrive" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 3 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("EU market expansion", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/gdrive\/acct-1\/chunks\//);
		expect(JSON.stringify(hits)).not.toContain("ya29.secret");
	});
});
