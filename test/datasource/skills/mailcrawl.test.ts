import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MailcrawlClient, MailcrawlSkill } from "../../../src/datasource/skills/mailcrawl/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-mailcrawl-"));
	mkdirSync(join(root, "bin"), { recursive: true });
	binaryPath = join(root, "bin", "mailcrawl");
	logPath = join(root, "calls.jsonl");
});

afterEach(() => rmSync(root, { recursive: true, force: true }));

function writeFake(output: string): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  dataDir: process.env.MAILCRAWL_DATA_DIR ?? null,
  openai: process.env.OPENAI_API_KEY ?? null
}) + "\\n");
process.stdout.write(${JSON.stringify(output)});
`,
	);
	chmodSync(binaryPath, 0o755);
}

function calls(): readonly { readonly args: readonly string[]; readonly dataDir: string | null; readonly openai: string | null }[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8").trim().split("\n").map((line) => JSON.parse(line));
}

describe("MailcrawlClient", () => {
	it("runs sync and all search modes through the managed data directory", async () => {
		writeFake(JSON.stringify({ added: 2, updated: 1, unchanged: 4, chunksAdded: 3, archiveRevision: "r1" }));
		const client = new MailcrawlClient({
			binaryPath,
			workspacePath: root,
			account: "personal",
			mailbox: "INBOX",
			env: { OPENAI_API_KEY: "must-not-leak" },
		});

		expect(await client.sync()).toMatchObject({ ok: true, data: { messages: 2, chunksAdded: 3 } });

		writeFake(JSON.stringify([{
			chunkId: "msg-1:latest:0",
			messageId: "msg-1",
			threadId: "thread-1",
			accountId: "personal",
			mailbox: "INBOX",
			subject: "Contract renewal",
			from: "legal@example.com",
			to: ["me@example.com"],
			date: "2026-08-30T10:00:00Z",
			snippet: "Renewal is approved.",
			score: 0.9,
		}]));
		expect(await client.search("hybrid", "renewal", { topK: 5 })).toMatchObject({
			ok: true,
			hits: [{ chunkId: "msg-1:latest:0", messageId: "msg-1" }],
		});

		const recorded = calls();
		expect(recorded[0]?.args).toEqual(["sync", "--json", "--account", "personal", "--mailbox", "INBOX"]);
		expect(recorded[1]?.args).toEqual(["search", "--mode", "hybrid", "--limit", "5", "--json", "renewal"]);
		expect(recorded.every((call) => call.dataDir === join(root, ".autorag", "datasources", "mailcrawl", "default", "data"))).toBe(true);
		expect(recorded.every((call) => call.openai === null)).toBe(true);
	});

	it("returns bounded failure results without throwing", async () => {
		const client = new MailcrawlClient({ binaryPath: join(root, "missing") });
		expect(await client.search("bm25", "query")).toMatchObject({ ok: false, reason: "binary-missing" });
	});
});

describe("MailcrawlSkill", () => {
	it("indexes and exposes BM25, semantic, and hybrid opaque retrieval methods", async () => {
		const client = {
			async sync() { return { ok: true as const, data: { messages: 2, chunksAdded: 2 }, stdout: "", stderr: "", code: 0 }; },
			async index() { return { ok: true as const, data: { embedded: 2 }, stdout: "", stderr: "", code: 0 }; },
			async search() {
				const hits = [{
					chunkId: "m1:latest:0",
					messageId: "m1",
					threadId: "t1",
					accountId: "acct",
					mailbox: "INBOX",
					subject: "Policy",
					from: "a@example.com",
					to: [],
					date: "2026-08-30",
					snippet: "Director approval required.",
					score: 1,
					mode: "bm25" as const,
				}];
				return { ok: true as const, hits, stdout: "", stderr: "", code: 0 };
			},
		};
		const skill = new MailcrawlSkill({ client, instanceId: "personal" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual([
			"mailcrawl-bm25",
			"mailcrawl-semantic",
			"mailcrawl-hybrid",
		]);
		const result = await skill.retrievalMethods()[0]?.retrieve("approval", { topK: 5, allowedScopes: ["/mailcrawl/personal/**"] });
		expect(result?.[0]?.source).toBe("/mailcrawl/personal/chunks/m1:latest:0");
		expect(skill.skillManifest().content).toContain("mailcrawl");
	});
});
