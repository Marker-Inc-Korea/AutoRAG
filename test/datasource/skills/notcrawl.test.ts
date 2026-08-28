import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { NotcrawlClient, NotionSkill } from "../../../src/datasource/skills/notion/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-notcrawl-"));
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	binaryPath = join(binDir, process.platform === "win32" ? "notcrawl.mjs" : "notcrawl");
	logPath = join(root, "calls.jsonl");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeNotcrawl(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  openai: process.env.OPENAI_API_KEY ?? null,
  updateCheck: process.env.CRAWLKIT_NO_UPDATE_CHECK ?? null,
}) + "\\n");
process.stdout.write(process.env.NOTCRAWL_FAKE_OUTPUT ?? "{}");
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
			if (!isCall(parsed)) throw new Error("unexpected fake notcrawl call");
			return parsed;
		});
}

describe("NotcrawlClient", () => {
	it("spawns sync and search with bounded private environment", async () => {
		writeFakeNotcrawl();
		const client = new NotcrawlClient({
			binaryPath,
			configPath: join(root, "notcrawl.yaml"),
			env: {
				NOTCRAWL_FAKE_OUTPUT: JSON.stringify({ synced: 7 }),
				OPENAI_API_KEY: "must-not-leak",
			},
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 7 });
		const searchClient = new NotcrawlClient({
			binaryPath,
			env: {
				NOTCRAWL_FAKE_OUTPUT: JSON.stringify([
					{
						page_id: "page-1",
						title: "Onboarding",
						parent_title: "Team wiki",
						last_edited_time: "2026-08-16T01:02:03Z",
						snippet: "New hires meet their buddy on day one",
					},
				]),
			},
		});
		const search = await searchClient.search("onboarding", { topK: 5 });
		expect(search).toMatchObject({
			ok: true,
			hits: [{ id: "page-1", content: "New hires meet their buddy on day one", title: "Onboarding" }],
		});

		expect(calls()[0]?.args).toEqual(["--config", join(root, "notcrawl.yaml"), "sync"]);
		expect(calls()[1]?.args).toEqual(["search", "onboarding", "--limit", "5", "--json"]);
		expect(calls().every((call) => call.openai === null)).toBe(true);
		expect(calls().every((call) => call.updateCheck === "1")).toBe(true);
	});

	it("routes configured workspace execution through the managed launch context", async () => {
		writeFakeNotcrawl();
		const client = new NotcrawlClient({
			binaryPath,
			workspacePath: root,
			env: { NOTCRAWL_FAKE_OUTPUT: JSON.stringify({ synced: 2 }) },
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 2 });
		expect(calls()[0]?.args.slice(0, 2)).toEqual([
			"--db",
			join(root, ".autorag", "datasources", "notcrawl", "archive.db"),
		]);
		expect(calls()[0]?.args).toContain("sync");
	});

	it("maps a missing binary and malformed output without throwing", async () => {
		const missing = new NotcrawlClient({ binaryPath: join(root, "missing") });
		expect(await missing.sync()).toMatchObject({ ok: false, reason: "binary-missing" });

		writeFakeNotcrawl();
		const malformed = new NotcrawlClient({
			binaryPath,
			env: { NOTCRAWL_FAKE_OUTPUT: "not-json" },
		});
		expect(await malformed.search("query")).toMatchObject({ ok: false, reason: "invalid-output" });
	});
});

describe("NotionSkill", () => {
	it("indexes and retrieves Notion pages through the notcrawl surface", async () => {
		writeFakeNotcrawl();
		const syncClient = new NotcrawlClient({
			binaryPath,
			env: { NOTCRAWL_FAKE_OUTPUT: JSON.stringify({ synced: 1 }) },
		});
		const skill = new NotionSkill({ client: syncClient, instanceId: "workspace" });
		expect(skill.describe()).toMatchObject({
			name: "notion",
			type: "notion-archive",
			requiresExternalCli: true,
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });

		const searchClient = new NotcrawlClient({
			binaryPath,
			env: {
				NOTCRAWL_FAKE_OUTPUT: JSON.stringify([
					{ page_id: "page-2", title: "Runbook", parent_title: "Ops", content: "Restart cluster after drain" },
				]),
			},
		});
		const searchable = new NotionSkill({ client: searchClient, instanceId: "workspace" });
		const [method] = searchable.retrievalMethods();
		const results = await method?.retrieve("restart cluster", { topK: 5 });
		expect(results?.[0]?.source).toBe("/notion/workspace/chunks/page-2");
		expect(results?.[0]?.metadata).toMatchObject({ backend: "notcrawl", title: "Runbook" });
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
