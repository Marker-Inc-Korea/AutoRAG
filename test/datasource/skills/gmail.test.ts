import { describe, expect, it } from "vitest";
import { GmailConnector } from "../../../src/datasource/skills/gmail/connector.ts";
import { GmailSkill } from "../../../src/datasource/skills/gmail/skill.ts";
import { createMockFetch } from "../../fixtures/mock-fetch.ts";

function b64url(text: string): string {
	return Buffer.from(text, "utf8").toString("base64url");
}

const MESSAGE_LIST = { messages: [{ id: "m1" }, { id: "m2" }] };

const MESSAGE_1 = {
	id: "m1",
	threadId: "t1",
	labelIds: ["INBOX"],
	internalDate: "1710000000000",
	payload: {
		mimeType: "multipart/alternative",
		headers: [
			{ name: "Subject", value: "Contract renewal" },
			{ name: "From", value: "legal@example.com" },
			{ name: "To", value: "me@example.com" },
			{ name: "Date", value: "Mon, 11 Mar 2024 10:00:00 +0000" },
		],
		parts: [{ mimeType: "text/plain", body: { data: b64url("The contract renews on April 1st.") } }],
	},
};

const MESSAGE_2 = {
	id: "m2",
	threadId: "t2",
	labelIds: ["INBOX"],
	internalDate: "1710000100000",
	snippet: "Lunch on Thursday?",
	payload: { mimeType: "text/plain", headers: [{ name: "Subject", value: "Lunch" }], body: {} },
};

describe("GmailConnector", () => {
	it("returns not-configured without a token", async () => {
		expect(await new GmailConnector({ tokenEnv: "GMAIL_TEST_UNSET" }).fetch()).toMatchObject({
			ok: false,
			reason: "not-configured",
		});
	});

	it("fetches full messages with decoded bodies and header blocks", async () => {
		const mock = createMockFetch([
			{ match: "/users/me/messages?", json: MESSAGE_LIST },
			{ match: "/users/me/messages/m1", json: MESSAGE_1 },
			{ match: "/users/me/messages/m2", json: MESSAGE_2 },
		]);
		const result = await new GmailConnector({ token: "ya29.secret", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2);
			const contract = result.documents.find((d) => d.docId === "m1");
			expect(contract?.title).toBe("Contract renewal");
			expect(contract?.content).toContain("Subject: Contract renewal");
			expect(contract?.content).toContain("The contract renews on April 1st.");
			expect(contract?.publishedAt).toBe(1710000000000);
			const lunch = result.documents.find((d) => d.docId === "m2");
			expect(lunch?.content).toContain("Lunch on Thursday?");
		}
	});

	it("degrades per-message failures to an aggregate count warning", async () => {
		const mock = createMockFetch([
			{ match: "/users/me/messages?", json: MESSAGE_LIST },
			{ match: "/users/me/messages/m1", status: 500 },
			{ match: "/users/me/messages/m2", json: MESSAGE_2 },
		]);
		const result = await new GmailConnector({ token: "t", fetchImpl: mock.fetchImpl }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(1);
			expect(result.warnings).toEqual(["1 message(s) failed to fetch"]);
		}
	});

	it("maps list-level 401 to auth and forwards labelIds and query", async () => {
		const authMock = createMockFetch([{ match: "/users/me/messages?", status: 401 }]);
		expect(await new GmailConnector({ token: "t", fetchImpl: authMock.fetchImpl }).fetch()).toMatchObject({
			ok: false,
			reason: "auth",
		});
		const paramMock = createMockFetch([{ match: "/users/me/messages?", json: { messages: [] } }]);
		await new GmailConnector({
			token: "t",
			labelIds: ["INBOX"],
			query: "from:legal",
			fetchImpl: paramMock.fetchImpl,
		}).fetch();
		expect(paramMock.requests[0]).toContain("labelIds=INBOX");
		expect(paramMock.requests[0]).toContain("q=from%3Alegal");
	});
});

describe("GmailSkill", () => {
	it("indexes and searches with opaque /gmail sources and no leaked addresses in diagnostics", async () => {
		const mock = createMockFetch([
			{ match: "/users/me/messages?", json: MESSAGE_LIST },
			{ match: "/users/me/messages/m1", json: MESSAGE_1 },
			{ match: "/users/me/messages/m2", json: MESSAGE_2 },
		]);
		const skill = new GmailSkill({
			instanceId: "acct-1",
			connectorOptions: { token: "ya29.secret", fetchImpl: mock.fetchImpl },
		});
		expect(skill.describe()).toMatchObject({ name: "gmail", datasourceId: "gmail" });
		const indexResult = await skill.index();
		expect(indexResult).toMatchObject({ ok: true, chunkCount: 2 });
		expect(JSON.stringify(indexResult.diagnostics)).not.toContain("@example.com");
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("contract renews April", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/gmail\/acct-1\/chunks\//);
	});
});
