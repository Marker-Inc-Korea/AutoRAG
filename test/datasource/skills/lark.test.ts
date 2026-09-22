import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DatasourceAccessContext } from "../../../src/datasource/access-context.ts";
import { DatasourceCliError } from "../../../src/datasource/errors.ts";
import { DatasourceResultFilter } from "../../../src/datasource/result-filter.ts";
import { buildDatasourceSkills } from "../../../src/datasource/skills/factory.ts";
import { LarkSkill } from "../../../src/datasource/skills/lark/index.ts";

let root: string;
let binaryPath: string;
let logPath: string;
let specPath: string;
let counterPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-lark-"));
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	binaryPath = join(binDir, "lark-cli");
	logPath = join(root, "calls.jsonl");
	specPath = join(root, "spec.json");
	counterPath = join(root, "counter.txt");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeSpec(spec: Record<string, unknown>): void {
	writeFileSync(specPath, JSON.stringify(spec));
}

function writeFakeLark(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
const spec = JSON.parse(readFileSync(${JSON.stringify(specPath)}, "utf8"));
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({
  args: process.argv.slice(2),
  openai: process.env.OPENAI_API_KEY ?? null,
  appSecret: process.env.LARK_APP_SECRET ?? null,
}) + "\\n");
const args = process.argv.slice(2);
if (args[0] === "--version") {
  process.stdout.write("lark-cli test\\n");
  process.exit(0);
}
if (args[0] === "auth" && args[1] === "status") {
  if (spec.auth === "fail") {
    process.stderr.write(String(spec.authStderr ?? "not logged in"));
    process.exit(1);
  }
  process.stdout.write(JSON.stringify({ ok: true, identity: "user" }));
  process.exit(0);
}
const rateLimitTimes = Number(spec.rateLimitTimes ?? 0);
if (rateLimitTimes > 0 && (args.includes("+messages-search") || args.includes("+search"))) {
  let seen = 0;
  try { seen = Number(readFileSync(${JSON.stringify(counterPath)}, "utf8")); } catch {}
  writeFileSync(${JSON.stringify(counterPath)}, String(seen + 1));
  if (seen < rateLimitTimes) {
    process.stderr.write(String(spec.rateLimitStderr ?? '{"code":99991400,"msg":"frequency limit"}'));
    process.exit(2);
  }
}
if (typeof spec.searchStderr === "string") {
  process.stderr.write(spec.searchStderr);
  process.exit(1);
}
if (args.includes("+messages-search")) {
  process.stdout.write(String(spec.messagesStdout ?? ""));
  process.exit(0);
}
if (args[0] === "drive" && args.includes("+search")) {
  process.stdout.write(String(spec.docsStdout ?? ""));
  process.exit(0);
}
process.stderr.write("unexpected lark-cli args");
process.exit(3);
`,
	);
	chmodSync(binaryPath, 0o755);
}

function calls(): readonly {
	readonly args: readonly string[];
	readonly openai: string | null;
	readonly appSecret: string | null;
}[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => {
			const parsed: unknown = JSON.parse(line);
			if (!isCall(parsed)) throw new Error("unexpected fake lark-cli call");
			return parsed;
		});
}

function isCall(value: unknown): value is { args: readonly string[]; openai: string | null; appSecret: string | null } {
	if (typeof value !== "object" || value === null) return false;
	const record = value as { args?: unknown; openai?: unknown; appSecret?: unknown };
	return (
		Array.isArray(record.args) &&
		record.args.every((arg) => typeof arg === "string") &&
		(record.openai === null || typeof record.openai === "string") &&
		(record.appSecret === null || typeof record.appSecret === "string")
	);
}

function messageEnvelope(messages: readonly Record<string, unknown>[]): string {
	return JSON.stringify({ ok: true, identity: "user", data: { messages, has_more: false } });
}

function docEnvelope(results: readonly Record<string, unknown>[]): string {
	return JSON.stringify({ ok: true, identity: "user", data: { results, has_more: false, total: results.length } });
}

const REFUND_MESSAGE = {
	message_id: "om_refund",
	chat_id: "oc_keep",
	content: "Refund exceptions require director approval",
	create_time: "1710000000000",
	chat_type: "group",
	chat_name: "finance",
};

const REFUND_DOC = {
	title_highlighted: "<h>Refund</h> policy",
	summary_highlighted: "Director approval is required before payout",
	entity_type: "DOC",
	result_meta: {
		token: "doxcnRefund1",
		url: "https://example.larksuite.com/docx/doxcnRefund1",
		update_time_iso: "2026-07-01T00:00:00Z",
	},
};

function skill(
	extra: {
		readonly chatIds?: readonly string[];
		readonly embedderBaseUrl?: string;
		readonly sleep?: (ms: number) => Promise<void>;
	} = {},
): LarkSkill {
	return new LarkSkill({
		binaryPath,
		instanceId: "default",
		env: { OPENAI_API_KEY: "must-not-leak", LARK_APP_SECRET: "app-secret-must-not-leak" },
		...(extra.chatIds !== undefined ? { chatIds: extra.chatIds } : {}),
		...(extra.embedderBaseUrl !== undefined ? { embedderBaseUrl: extra.embedderBaseUrl } : {}),
		...(extra.sleep !== undefined ? { sleep: extra.sleep } : {}),
	});
}

function method(target: LarkSkill, name: string) {
	const found = target.retrievalMethods().find((candidate) => candidate.describe().name === name);
	if (found === undefined) throw new Error(`missing method ${name}`);
	return found;
}

describe("LarkSkill remote search", () => {
	it("maps server search hits to opaque message and doc identities without a local archive", async () => {
		writeFakeLark();
		writeSpec({
			messagesStdout: messageEnvelope([
				REFUND_MESSAGE,
				{ message_id: "om_later", content: "Finance acknowledged the policy", create_time: "1710000001000" },
			]),
			docsStdout: docEnvelope([REFUND_DOC]),
		});
		const before = readdirSync(root).sort();
		const target = skill();

		const indexed = await target.index();
		const messages = await method(target, "lark-messages").retrieve("refund", { topK: 5 });
		const docs = await method(target, "lark-docs").retrieve("refund", { topK: 5 });
		const created = readdirSync(root)
			.sort()
			.filter((name) => !before.includes(name));

		expect(indexed).toMatchObject({ ok: true, chunkCount: 0 });
		expect(indexed.ok ? indexed.diagnostics.map((item) => item.message) : []).toContain(
			"remote-search datasource: no local mirror",
		);
		expect(target.polling()).toMatchObject({ mode: "none" });
		expect(target.polling().lastError).toBeUndefined();
		expect(messages.map((hit) => hit.source)).toEqual([
			"/lark/default/messages/om_refund",
			"/lark/default/messages/om_later",
		]);
		expect(messages.map((hit) => hit.content)).toEqual([
			"Refund exceptions require director approval",
			"Finance acknowledged the policy",
		]);
		expect(messages[0]?.metadata).toMatchObject({
			chatId: "oc_keep",
			observedAt: "1710000000000",
		});
		expect(messages[0]?.metadata.permalink).toBeUndefined();
		expect(docs).toMatchObject([
			{
				source: "/lark/default/docs/doxcnRefund1",
				content: "Director approval is required before payout",
				metadata: {
					permalink: "https://example.larksuite.com/docx/doxcnRefund1",
					observedAt: "2026-07-01T00:00:00Z",
				},
			},
		]);
		expect(created.every((name) => name === "calls.jsonl" || name === "counter.txt")).toBe(true);
		expect(calls().map((call) => call.args)).toEqual([
			["--version"],
			["auth", "status", "--json"],
			["im", "+messages-search", "--query", "refund", "--format", "json", "--page-size", "5"],
			["drive", "+search", "--query", "refund", "--format", "json", "--page-size", "5"],
		]);
		const manifest = target.skillManifest();
		expect(manifest.content).toContain("search_datasource_lark");
		expect(manifest.content).toContain("im +messages-mget --message-ids");
		expect(manifest.content).toContain("docs +fetch --doc");
		expect(manifest.content).toContain("--doc-format markdown");
		expect(manifest.content).not.toContain("search_datasource_documents");
		expect(manifest.content).not.toContain("LarkShell");
		expect(JSON.stringify(target.describe())).not.toContain("LarkShell");
		expect(calls().every((call) => call.openai === null && call.appSecret === null)).toBe(true);
	});

	it("is default-deny and splits chat and docs tags", () => {
		writeFakeLark();
		writeSpec({});
		const target = skill();
		const denied = new DatasourceAccessContext();
		const docsOnly = new DatasourceAccessContext({ allowedTags: ["lark:docs"] });
		const messages = method(target, "lark-messages").describe();
		const docs = method(target, "lark-docs").describe();

		expect(denied.isAccessible(target.describe())).toBe(false);
		expect(docsOnly.isAccessible(target.describe())).toBe(true);
		expect(docsOnly.isAccessible(messages)).toBe(false);
		expect(docsOnly.isAccessible(docs)).toBe(true);
		expect(messages.tags).toEqual(["lark:chat"]);
		expect(docs.tags).toEqual(["lark:docs"]);
		expect(messages.type).toBe("remote");
		expect(docs.type).toBe("remote");
	});

	it("narrows scope to one surface and passes configured chat ids only to message search", async () => {
		writeFakeLark();
		writeSpec({
			messagesStdout: messageEnvelope([REFUND_MESSAGE]),
			docsStdout: docEnvelope([REFUND_DOC]),
		});
		const target = skill({ chatIds: ["oc_keep", "oc_other"] });
		await target.index();

		const skippedMessages = await method(target, "lark-messages").retrieve("refund", {
			topK: 3,
			scope: "/lark/default/docs",
		});
		const skippedDocs = await method(target, "lark-docs").retrieve("refund", {
			topK: 3,
			scope: "/lark/default/messages",
		});
		const messages = await method(target, "lark-messages").retrieve("refund", { topK: 3 });
		const docs = await method(target, "lark-docs").retrieve("refund", { topK: 3 });

		expect(skippedMessages).toEqual([]);
		expect(skippedDocs).toEqual([]);
		expect(messages[0]?.source).toBe("/lark/default/messages/om_refund");
		expect(docs[0]?.source).toBe("/lark/default/docs/doxcnRefund1");
		const searchCalls = calls().filter((call) => call.args.includes("--query"));
		expect(searchCalls.map((call) => call.args)).toEqual([
			[
				"im",
				"+messages-search",
				"--query",
				"refund",
				"--format",
				"json",
				"--page-size",
				"3",
				"--chat-id",
				"oc_keep,oc_other",
			],
			["drive", "+search", "--query", "refund", "--format", "json", "--page-size", "3"],
		]);
	});

	it("drops hash-fragment ids and keeps the other hit", async () => {
		writeFakeLark();
		writeSpec({
			messagesStdout: messageEnvelope([{ message_id: "om_bad#frag", content: "must not leak" }, REFUND_MESSAGE]),
			docsStdout: docEnvelope([
				{
					summary_highlighted: "fragment doc",
					result_meta: { token: "dox#bad" },
				},
				REFUND_DOC,
			]),
		});
		const target = skill();
		const messages = await method(target, "lark-messages").retrieve("refund", { topK: 5 });
		const docs = await method(target, "lark-docs").retrieve("refund", { topK: 5 });

		expect(messages.map((hit) => hit.source)).toEqual(["/lark/default/messages/om_refund"]);
		expect(docs.map((hit) => hit.source)).toEqual(["/lark/default/docs/doxcnRefund1"]);
		expect(JSON.stringify(messages)).not.toContain("#");
		expect(JSON.stringify(docs)).not.toContain("#");
	});

	it("returns an empty hit list when the server has no matches", async () => {
		writeFakeLark();
		writeSpec({ messagesStdout: messageEnvelope([]), docsStdout: docEnvelope([]) });
		const target = skill();

		await expect(method(target, "lark-messages").retrieve("nothing", { topK: 5 })).resolves.toEqual([]);
		await expect(method(target, "lark-docs").retrieve("nothing", { topK: 5 })).resolves.toEqual([]);
	});

	it("records a missing binary on the no-op index and throws that reason on search", async () => {
		const target = new LarkSkill({ binaryPath: join(root, "missing-lark-cli"), instanceId: "default" });

		const indexed = await target.index();

		expect(indexed).toMatchObject({ ok: true, chunkCount: 0 });
		expect(target.polling().mode).toBe("none");
		expect(target.polling().lastError).toContain("binary-missing");
		await expect(method(target, "lark-messages").retrieve("refund", { topK: 1 })).rejects.toBeInstanceOf(
			DatasourceCliError,
		);
		await expect(method(target, "lark-messages").retrieve("refund", { topK: 1 })).rejects.toThrow("binary-missing");
	});

	it("surfaces auth-status stderr verbatim when the CLI is not logged in", async () => {
		writeFakeLark();
		const stderr = "user_access_token expired: token is dead";
		writeSpec({ auth: "fail", authStderr: stderr, messagesStdout: messageEnvelope([REFUND_MESSAGE]) });
		const target = skill();

		const indexed = await target.index();

		expect(indexed.ok).toBe(true);
		expect(target.polling().lastError).toContain(stderr);
		expect(JSON.stringify(indexed)).toContain(stderr);
	});

	it("surfaces search stderr verbatim when a scope is missing", async () => {
		writeFakeLark();
		const stderr = "permission denied: missing scope search:message";
		writeSpec({ searchStderr: stderr });
		const target = skill();

		await expect(method(target, "lark-messages").retrieve("refund", { topK: 2 })).rejects.toThrow(stderr);
	});

	it("retries rate-limit 99991400 and then returns the hit", async () => {
		writeFakeLark();
		const sleeps: number[] = [];
		writeSpec({
			rateLimitTimes: 2,
			rateLimitStderr: '{"code":99991400,"msg":"too many requests"}',
			messagesStdout: messageEnvelope([REFUND_MESSAGE]),
		});
		const target = skill({
			sleep: async (ms) => {
				sleeps.push(ms);
			},
		});

		const messages = await method(target, "lark-messages").retrieve("refund", { topK: 4 });

		expect(messages.map((hit) => hit.source)).toEqual(["/lark/default/messages/om_refund"]);
		expect(sleeps).toEqual([200, 400]);
		expect(calls().filter((call) => call.args.includes("+messages-search"))).toHaveLength(3);
	});

	it("throws the upstream rate-limit text when retries are exhausted", async () => {
		writeFakeLark();
		writeSpec({
			rateLimitTimes: 20,
			rateLimitStderr: "request failed code=99991400",
			messagesStdout: messageEnvelope([REFUND_MESSAGE]),
		});
		const target = skill({ sleep: async () => {} });

		await expect(method(target, "lark-messages").retrieve("refund", { topK: 2 })).rejects.toThrow("99991400");
		expect(calls().filter((call) => call.args.includes("+messages-search"))).toHaveLength(7);
	});

	it("rejects a remote embedder before spawning the CLI", async () => {
		writeFakeLark();
		writeSpec({});
		const target = skill({ embedderBaseUrl: "https://api.openai.com/v1" });

		const indexed = await target.index();

		expect(indexed.ok).toBe(false);
		if (!indexed.ok) expect(indexed.code).toBe("datasource-embedding-egress-rejected");
		expect(existsSync(logPath)).toBe(false);
	});

	it("keeps only docs when the trusted scope intersects a docs user scope", async () => {
		writeFakeLark();
		writeSpec({
			messagesStdout: messageEnvelope([REFUND_MESSAGE]),
			docsStdout: docEnvelope([REFUND_DOC]),
		});
		const target = skill();
		const messages = await method(target, "lark-messages").retrieve("refund", { topK: 5 });
		const docs = await method(target, "lark-docs").retrieve("refund", { topK: 5 });
		const filter = new DatasourceResultFilter();
		const filtered = filter.filter(
			new Map([
				["lark-messages", messages],
				["lark-docs", docs],
			]),
			target.retrievalMethods(),
			new DatasourceAccessContext({
				allowedTags: ["lark:chat", "lark:docs"],
				allowedScopes: ["/lark/default/**"],
			}),
			"/lark/default/docs",
		);

		expect(filtered.get("lark-messages")).toEqual([]);
		expect(filtered.get("lark-docs")?.map((hit) => hit.source)).toEqual(["/lark/default/docs/doxcnRefund1"]);
	});

	it("builds a lark skill from trusted config and passes channel ids to the CLI", async () => {
		writeFakeLark();
		writeSpec({
			messagesStdout: messageEnvelope([REFUND_MESSAGE]),
			docsStdout: docEnvelope([REFUND_DOC]),
		});
		const { skills, unknown } = buildDatasourceSkills({
			lark: {
				instanceId: "default",
				channels: { ids: ["oc_keep"] },
				connector: { binaryPath },
			},
		});

		expect(unknown).toEqual([]);
		const built = skills[0];
		expect(built?.describe()).toMatchObject({
			name: "lark",
			type: "lark-remote",
			instanceId: "default",
			requiresExternalCli: true,
		});
		const builtMessages = built
			?.retrievalMethods()
			.find((candidate) => candidate.describe().name === "lark-messages");
		const hits = await builtMessages?.retrieve("refund", { topK: 2 });
		expect(hits?.map((hit) => hit.source)).toEqual(["/lark/default/messages/om_refund"]);
		expect(calls().some((call) => call.args.includes("--chat-id") && call.args.includes("oc_keep"))).toBe(true);
		expect(built?.skillManifest().content).toContain("search_datasource_lark");
		expect(built?.skillManifest().content).not.toContain("search_datasource_documents");
	});
});
