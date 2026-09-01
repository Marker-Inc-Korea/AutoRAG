import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { TelecrawlClient, TelecrawlSkill } from "../../../src/datasource/skills/telecrawl/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-telecrawl-"));
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	binaryPath = join(binDir, "telecrawl");
	logPath = join(root, "calls.jsonl");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeTelecrawl(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  openai: process.env.OPENAI_API_KEY ?? null,
  updateCheck: process.env.CRAWLKIT_NO_UPDATE_CHECK ?? null,
}) + "\\n");
process.stdout.write(process.env.TELECRAWL_FAKE_OUTPUT ?? "{}");
`,
	);
	chmodSync(binaryPath, 0o755);
}

function calls(): readonly {
	readonly args: readonly string[];
	readonly openai: string | null;
	readonly updateCheck: string | null;
}[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => {
			const parsed: unknown = JSON.parse(line);
			if (!isCall(parsed)) throw new Error("unexpected fake telecrawl call");
			return parsed;
		});
}

describe("TelecrawlClient", () => {
	it("spawns import and search with bounded private environment", async () => {
		writeFakeTelecrawl();
		const client = new TelecrawlClient({
			binaryPath,
			databasePath: join(root, "archive.db"),
			sourcePath: join(root, "source"),
			env: {
				TELECRAWL_FAKE_OUTPUT: JSON.stringify({ imported: 4 }),
				OPENAI_API_KEY: "must-not-leak",
			},
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 4 });
		const searchClient = new TelecrawlClient({
			binaryPath,
			env: {
				TELECRAWL_FAKE_OUTPUT: JSON.stringify([
					{
						message_id: "tg-1",
						chat_jid: "engineering",
						chat_name: "Engineering",
						topic_name: "Deployments",
						sender_name: "Alice",
						timestamp: "2026-08-16T01:02:03Z",
						snippet: "Release starts at seven",
					},
				]),
			},
		});
		const search = await searchClient.search("release", { topK: 5 });
		expect(search).toMatchObject({
			ok: true,
			hits: [{ id: "tg-1", content: "Release starts at seven", title: "Engineering" }],
		});

		expect(calls()[0]?.args).toEqual([
			"--json",
			"--db",
			join(root, "archive.db"),
			"--source",
			join(root, "source"),
			"import",
		]);
		expect(calls()[1]?.args).toEqual(["--json", "search", "--limit", "5", "release"]);
		expect(calls().every((call) => call.openai === null)).toBe(true);
		expect(calls().every((call) => call.updateCheck === "1")).toBe(true);
	});

	it("runs telecrawl against its own default store without a managed --db injection", async () => {
		writeFakeTelecrawl();
		const client = new TelecrawlClient({
			binaryPath,
			workspacePath: root,
			env: { TELECRAWL_FAKE_OUTPUT: JSON.stringify({ imported: 1 }) },
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 1 });
		const args = calls()[0]?.args ?? [];
		expect(args).not.toContain("--db");
		expect(args.join(" ")).not.toContain(".autorag/datasources/telecrawl");
	});

	it("maps a missing binary and malformed output without throwing", async () => {
		const missing = new TelecrawlClient({ binaryPath: join(root, "missing") });
		expect(await missing.sync()).toMatchObject({ ok: false, reason: "binary-missing" });

		writeFakeTelecrawl();
		const malformed = new TelecrawlClient({
			binaryPath,
			env: { TELECRAWL_FAKE_OUTPUT: "not-json" },
		});
		expect(await malformed.search("query")).toMatchObject({ ok: false, reason: "invalid-output" });
	});
});

describe("TelecrawlSkill", () => {
	it("indexes and retrieves Telegram messages through the crawler surface", async () => {
		writeFakeTelecrawl();
		const syncClient = new TelecrawlClient({
			binaryPath,
			env: { TELECRAWL_FAKE_OUTPUT: JSON.stringify({ imported: 1 }) },
		});
		const skill = new TelecrawlSkill({ client: syncClient, instanceId: "personal" });
		expect(skill.describe()).toMatchObject({
			name: "telegram",
			type: "telegram-archive",
			requiresExternalCli: true,
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });

		const searchClient = new TelecrawlClient({
			binaryPath,
			env: {
				TELECRAWL_FAKE_OUTPUT: JSON.stringify([
					{ message_id: "tg-2", chat_name: "Ops", topic_name: "Releases", text: "Deploy freeze starts Friday" },
				]),
			},
		});
		const searchable = new TelecrawlSkill({ client: searchClient, instanceId: "personal" });
		const [method] = searchable.retrievalMethods();
		const results = await method?.retrieve("deploy freeze", { topK: 5 });
		expect(results?.[0]?.source).toBe("/telegram/personal/chunks/tg-2");
		expect(results?.[0]?.metadata).toMatchObject({ backend: "telecrawl", title: "Ops" });
	});
});

function isCall(
	value: unknown,
): value is { readonly args: readonly string[]; readonly openai: string | null; readonly updateCheck: string | null } {
	if (typeof value !== "object" || value === null) return false;
	if (!("args" in value) || !Array.isArray(value.args) || !value.args.every((arg) => typeof arg === "string"))
		return false;
	return (
		"openai" in value &&
		(value.openai === null || typeof value.openai === "string") &&
		"updateCheck" in value &&
		(value.updateCheck === null || typeof value.updateCheck === "string")
	);
}
