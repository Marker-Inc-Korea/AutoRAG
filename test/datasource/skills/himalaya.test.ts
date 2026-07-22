import { describe, expect, it } from "vitest";
import { HimalayaConnector, type HimalayaRunResult } from "../../../src/datasource/skills/gmail/himalaya-connector.ts";
import { GmailSkill } from "../../../src/datasource/skills/gmail/skill.ts";

const ENVELOPES = JSON.stringify([
	{
		id: "101",
		flags: ["Seen"],
		subject: "Quarterly report ready",
		from: { name: "Finance", addr: "finance@example.com" },
		date: "2026-07-20 23:27-07:00",
	},
	{ id: "102", flags: [], subject: "Team lunch", from: { addr: "hr@example.com" }, date: "2026-07-21 10:00+09:00" },
]);

function runnerFrom(map: Record<string, HimalayaRunResult>): (args: readonly string[]) => Promise<HimalayaRunResult> {
	return async (args) => {
		const key = args.slice(0, 2).join(" ");
		return map[key] ?? { ok: false, stdout: "", stderr: "unexpected", code: 1 };
	};
}

const ok = (stdout: string): HimalayaRunResult => ({ ok: true, stdout, stderr: "", code: 0 });

describe("HimalayaConnector", () => {
	it("lists envelopes and reads bodies into documents with account/folder hierarchy", async () => {
		const runner = runnerFrom({
			"envelope list": ok(ENVELOPES),
			"message read": ok("Subject: x\n\nThe Q2 report is attached and revenue grew 12%."),
		});
		const connector = new HimalayaConnector({ account: "gmail", folder: "INBOX", runner });

		const result = await connector.fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2);
			expect(result.documents[0]).toMatchObject({
				docId: "gmail-INBOX-101",
				hierarchy: ["accounts", "gmail", "INBOX"],
				title: "Quarterly report ready",
			});
			expect(result.documents[0]?.content).toContain("revenue grew 12%");
			expect(result.documents[0]?.publishedAt).toEqual(expect.any(Number));
		}
	});

	it("degrades per-message read failures to an aggregate warning", async () => {
		const runner = runnerFrom({
			"envelope list": ok(ENVELOPES),
			"message read": { ok: false, stdout: "", stderr: "fetch failed", code: 1 },
		});
		const result = await new HimalayaConnector({ runner }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(2); // header-only documents survive
			expect(result.warnings).toEqual(["2 message(s) failed to read"]);
		}
	});

	it("maps listing failures without leaking stderr contents", async () => {
		const runner = runnerFrom({
			"envelope list": {
				ok: false,
				stdout: "",
				stderr: "cannot authenticate user bob@example.com via /Users/bob/.config",
				code: 1,
			},
		});
		const result = await new HimalayaConnector({ runner }).fetch();
		expect(result).toMatchObject({ ok: false, reason: "auth" });
		if (!result.ok) {
			expect(result.message).not.toContain("bob@example.com");
			expect(result.message).not.toContain("/Users/");
		}
	});

	it("returns unavailable when the binary cannot spawn", async () => {
		const connector = new HimalayaConnector({ binaryPath: "/nonexistent/himalaya-qa" });
		const result = await connector.fetch();
		expect(result).toMatchObject({ ok: false });
	});

	it("plugs into GmailSkill for indexing and opaque-source search", async () => {
		const runner = runnerFrom({
			"envelope list": ok(ENVELOPES),
			"message read": ok("Lunch is at the bibimbap place on Friday."),
		});
		const skill = new GmailSkill({ instanceId: "imap", connector: new HimalayaConnector({ runner }) });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("bibimbap lunch Friday", { topK: 3 });
		expect(hits?.[0]?.source).toMatch(/^\/gmail\/imap\/chunks\//);
	});
});
