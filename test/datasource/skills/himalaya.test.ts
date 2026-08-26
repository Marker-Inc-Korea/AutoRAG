import { chmodSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
		const commandIndex = args.findIndex((arg) => arg === "envelope" || arg === "message");
		const key = args.slice(commandIndex, commandIndex + 2).join(" ");
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

	it("uses Himalaya v2 mailbox/json flags and preserves bounded actionable diagnostics", async () => {
		const calls: string[][] = [];
		const runner = async (args: readonly string[]) => {
			calls.push([...args]);
			return {
				ok: false,
				stdout: "",
				stderr: "No backend matching auto is configured for this account.",
				code: 2,
			};
		};
		const result = await new HimalayaConnector({ account: "gmail", folder: "INBOX", runner }).fetch();
		expect(result).toMatchObject({ ok: false, reason: "api-error" });
		if (!result.ok) expect(result.message).toContain("No backend matching auto");
		expect(calls[0]).toEqual([
			"--account",
			"gmail",
			"--json",
			"envelope",
			"list",
			"--mailbox",
			"INBOX",
			"--page-size",
			"100",
		]);
	});

	it("fetches unchanged envelopes only once with workspace state", async () => {
		const workspaceRoot = mkdtempSync(join(tmpdir(), "autorag-himalaya-"));
		const calls: string[][] = [];
		const runner = async (args: readonly string[]) => {
			calls.push([...args]);
			const commandIndex = args.findIndex((arg) => arg === "envelope" || arg === "message");
			return commandIndex >= 0 && args[commandIndex] === "envelope"
				? ok(ENVELOPES)
				: ok("The Q2 report is attached and revenue grew 12%.");
		};
		const first = new HimalayaConnector({ account: "gmail", workspaceRoot, runner });
		expect((await first.fetch()).ok).toBe(true);
		const second = new HimalayaConnector({ account: "gmail", workspaceRoot, runner });
		const result = await second.fetch();
		expect(result).toMatchObject({ ok: true, changed: false });
		expect(calls.filter((args) => args.includes("message")).length).toBe(2);
		rmSync(workspaceRoot, { recursive: true, force: true });
	});

	it("returns unavailable when the binary cannot spawn", async () => {
		const connector = new HimalayaConnector({ binaryPath: "/nonexistent/himalaya-qa" });
		const result = await connector.fetch();
		expect(result).toMatchObject({ ok: false });
	});

	it("applies the managed HIMALAYA_CONFIG and workspace cwd to real CLI runs", async () => {
		const workspace = mkdtempSync(join(tmpdir(), "autorag-himalaya-managed-"));
		const binary = join(workspace, "himalaya");
		const envPath = join(workspace, "himalaya-env.json");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(envPath)}, JSON.stringify({
  config: process.env.HIMALAYA_CONFIG,
  cwd: process.cwd(),
}));
const args = process.argv.slice(2);
if (args.includes("envelope")) process.stdout.write('[{"id":"1","subject":"QA","date":"2026-07-20"}]');
else process.stdout.write("managed body");
`,
		);
		chmodSync(binary, 0o755);
		const result = await new HimalayaConnector({ binaryPath: binary, workspaceRoot: workspace }).fetch();
		expect(result).toMatchObject({ ok: true });
		expect(JSON.parse(readFileSync(envPath, "utf8"))).toEqual({
			config: join(workspace, ".autorag", "datasources", "himalaya", "default", "config.toml"),
			cwd: realpathSync(workspace),
		});
		rmSync(workspace, { recursive: true, force: true });
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
