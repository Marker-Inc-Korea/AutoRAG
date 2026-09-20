import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SlackSkill, SlacrawlClient } from "../../../src/datasource/skills/slack/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-slacrawl-"));
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	binaryPath = join(binDir, "slacrawl");
	logPath = join(root, "calls.jsonl");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeSlacrawl(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  openai: process.env.OPENAI_API_KEY ?? null,
  updateCheck: process.env.CRAWLKIT_NO_UPDATE_CHECK ?? null,
}) + "\\n");
process.stdout.write(process.env.SLACRAWL_FAKE_OUTPUT ?? "{}");
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
			if (!isCall(parsed)) throw new Error("unexpected fake slacrawl call");
			return parsed;
		});
}

describe("SlacrawlClient", () => {
	it("spawns sync and search with bounded private environment", async () => {
		writeFakeSlacrawl();
		const client = new SlacrawlClient({
			binaryPath,
			configPath: join(root, "slacrawl.yaml"),
			syncSource: "primary",
			env: {
				SLACRAWL_FAKE_OUTPUT: JSON.stringify({ synced: 5 }),
				OPENAI_API_KEY: "must-not-leak",
			},
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 5 });
		const searchClient = new SlacrawlClient({
			binaryPath,
			env: {
				SLACRAWL_FAKE_OUTPUT: JSON.stringify([
					{
						workspace_id: "W1",
						workspace_name: "Acme",
						channel_id: "C1",
						channel_name: "deployments",
						ts: "1700000001.000100",
						user_name: "Alice",
						normalized_text: "Release starts at seven",
					},
				]),
			},
		});
		const search = await searchClient.search("release", { topK: 5 });
		expect(search).toMatchObject({
			ok: true,
			hits: [{ id: "C1-1700000001-000100", content: "Release starts at seven", title: "#deployments" }],
		});

		expect(calls()[0]?.args).toEqual([
			"--config",
			join(root, "slacrawl.yaml"),
			"--json",
			"sync",
			"--source",
			"primary",
		]);
		expect(calls()[1]?.args).toEqual(["--json", "search", "--limit", "5", "release"]);
		expect(calls().every((call) => call.openai === null)).toBe(true);
		expect(calls().every((call) => call.updateCheck === "1")).toBe(true);
	});

	it("scopes search to the configured workspace so unscoped slacrawl reads cannot return nothing", async () => {
		writeFakeSlacrawl();
		const scoped = new SlacrawlClient({
			binaryPath,
			workspace: "T0435JHH63Z",
			env: { SLACRAWL_FAKE_OUTPUT: JSON.stringify([]) },
		});
		await scoped.search("release", { topK: 3 });
		expect(calls()[0]?.args).toEqual(["--json", "search", "-workspace", "T0435JHH63Z", "--limit", "3", "release"]);

		const unscoped = new SlacrawlClient({ binaryPath, env: { SLACRAWL_FAKE_OUTPUT: JSON.stringify([]) } });
		await unscoped.search("release", { topK: 3 });
		expect(calls()[1]?.args).not.toContain("-workspace");
	});

	it("runs slacrawl against its own default store without a managed --db injection", async () => {
		writeFakeSlacrawl();
		const client = new SlacrawlClient({
			binaryPath,
			workspacePath: root,
			env: { SLACRAWL_FAKE_OUTPUT: JSON.stringify({ synced: 2 }) },
		});

		expect(await client.sync()).toMatchObject({ ok: true, count: 2 });
		const args = calls()[0]?.args ?? [];
		expect(args).not.toContain("--db");
		expect(args.join(" ")).not.toContain(".autorag/datasources/slacrawl");
	});

	it("maps a missing binary and malformed output without throwing", async () => {
		const missing = new SlacrawlClient({ binaryPath: join(root, "missing") });
		expect(await missing.sync()).toMatchObject({ ok: false, reason: "binary-missing" });

		writeFakeSlacrawl();
		const malformed = new SlacrawlClient({
			binaryPath,
			env: { SLACRAWL_FAKE_OUTPUT: "not-json" },
		});
		expect(await malformed.search("query")).toMatchObject({ ok: false, reason: "invalid-output" });
	});

	it("treats JSON null search output as zero hits", async () => {
		writeFakeSlacrawl();
		const client = new SlacrawlClient({
			binaryPath,
			env: { SLACRAWL_FAKE_OUTPUT: "null\n" },
		});
		expect(await client.search("query")).toMatchObject({ ok: true, hits: [] });
	});
});

describe("SlackSkill", () => {
	it("indexes and retrieves Slack messages through the slacrawl surface", async () => {
		writeFakeSlacrawl();
		const syncClient = new SlacrawlClient({
			binaryPath,
			env: { SLACRAWL_FAKE_OUTPUT: JSON.stringify({ synced: 1 }) },
		});
		const skill = new SlackSkill({ client: syncClient, instanceId: "workspace" });
		expect(skill.describe()).toMatchObject({
			name: "slack",
			type: "slack-archive",
			requiresExternalCli: true,
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });

		const searchClient = new SlacrawlClient({
			binaryPath,
			env: {
				SLACRAWL_FAKE_OUTPUT: JSON.stringify([
					{ workspace_name: "Acme", channel_id: "C2", channel_name: "ops", ts: "2.5", text: "Freeze Friday" },
				]),
			},
		});
		const searchable = new SlackSkill({ client: searchClient, instanceId: "workspace" });
		const [method] = searchable.retrievalMethods();
		const results = await method?.retrieve("freeze", { topK: 5 });
		expect(results?.[0]?.source).toBe("/slack/workspace/chunks/C2-2-5");
		expect(results?.[0]?.metadata).toMatchObject({ backend: "slacrawl", title: "#ops" });
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
