import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { WacrawlClient, WacrawlSkill } from "../../../src/datasource/skills/wacrawl/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-wacrawl-"));
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	binaryPath = join(binDir, "wacrawl");
	logPath = join(root, "calls.jsonl");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeWacrawl(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  openai: process.env.OPENAI_API_KEY ?? null,
  updateCheck: process.env.CRAWLKIT_NO_UPDATE_CHECK ?? null,
}) + "\\n");
process.stdout.write(process.env.WACRAWL_FAKE_OUTPUT ?? "{}");
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
			if (!isCall(parsed)) throw new Error("unexpected fake wacrawl call");
			return parsed;
		});
}

describe("WacrawlClient", () => {
	it("spawns sync and search with bounded private environment", async () => {
		writeFakeWacrawl();
		const client = new WacrawlClient({
			binaryPath,
			databasePath: join(root, "archive.db"),
			sourcePath: join(root, "source"),
			env: {
				WACRAWL_FAKE_OUTPUT: JSON.stringify({ messages: 3 }),
				OPENAI_API_KEY: "must-not-leak",
			},
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 3 });
		const searchClient = new WacrawlClient({
			binaryPath,
			env: {
				WACRAWL_FAKE_OUTPUT: JSON.stringify([
					{
						message_id: "m-1",
						chat_jid: "family",
						chat_name: "Family",
						sender_name: "Alice",
						timestamp: "2026-08-16T01:02:03Z",
						snippet: "Dinner is at seven",
					},
				]),
			},
		});
		const search = await searchClient.search("dinner", { topK: 5 });
		expect(search).toMatchObject({
			ok: true,
			hits: [{ id: "m-1", content: "Dinner is at seven", title: "Family" }],
		});

		expect(calls()[0]?.args).toEqual([
			"--json",
			"--db",
			join(root, "archive.db"),
			"--source",
			join(root, "source"),
			"sync",
		]);
		expect(calls()[1]?.args).toEqual(["--json", "--sync", "never", "search", "--limit", "5", "dinner"]);
		expect(calls().every((call) => call.openai === null)).toBe(true);
		expect(calls().every((call) => call.updateCheck === "1")).toBe(true);
	});

	it("runs wacrawl against its own default store without a managed --db injection", async () => {
		writeFakeWacrawl();
		const client = new WacrawlClient({
			binaryPath,
			workspacePath: root,
			env: { WACRAWL_FAKE_OUTPUT: JSON.stringify({ messages: 1 }) },
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 1 });
		const args = calls()[0]?.args ?? [];
		expect(args).not.toContain("--db");
		expect(args.join(" ")).not.toContain(".autorag/datasources/wacrawl");
	});

	it("maps a missing binary and malformed output without throwing", async () => {
		const missing = new WacrawlClient({ binaryPath: join(root, "missing") });
		expect(await missing.sync()).toMatchObject({ ok: false, reason: "binary-missing" });

		writeFakeWacrawl();
		const malformed = new WacrawlClient({
			binaryPath,
			env: { WACRAWL_FAKE_OUTPUT: "not-json" },
		});
		expect(await malformed.search("query")).toMatchObject({ ok: false, reason: "invalid-output" });
	});
});

describe("WacrawlSkill", () => {
	it("indexes and retrieves WhatsApp messages through the crawler surface", async () => {
		writeFakeWacrawl();
		const syncClient = new WacrawlClient({
			binaryPath,
			env: { WACRAWL_FAKE_OUTPUT: JSON.stringify({ messages: 1 }) },
		});
		const skill = new WacrawlSkill({ client: syncClient, instanceId: "personal" });
		expect(skill.describe()).toMatchObject({
			name: "whatsapp",
			type: "whatsapp-archive",
			requiresExternalCli: true,
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });

		const searchClient = new WacrawlClient({
			binaryPath,
			env: {
				WACRAWL_FAKE_OUTPUT: JSON.stringify([
					{ message_id: "m-2", chat_name: "Ops", text: "Deploy freeze starts Friday" },
				]),
			},
		});
		const searchable = new WacrawlSkill({ client: searchClient, instanceId: "personal" });
		const [method] = searchable.retrievalMethods();
		const results = await method?.retrieve("deploy freeze", { topK: 5 });
		expect(results?.[0]?.source).toBe("/whatsapp/personal/chunks/m-2");
		expect(results?.[0]?.metadata).toMatchObject({ backend: "wacrawl", title: "Ops" });
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
