import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../../../src/datasource/skills/factory.ts";
import { LazykatokClient } from "../../../../src/datasource/skills/lazykatok/client.ts";
import { LazykatokBm25Method } from "../../../../src/datasource/skills/lazykatok/methods.ts";
import type {
	LazykatokHit,
	LazykatokSearchMode,
	LazykatokSearchOptions,
	LazykatokSearchResult,
} from "../../../../src/datasource/skills/lazykatok/types.ts";

let root: string;
let binDir: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-lazykatok-direct-"));
	binDir = join(root, "bin");
	binaryPath = join(binDir, "lazykatok");
	logPath = join(root, "lazykatok-calls.jsonl");
	mkdirSync(binDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeLazykatok(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2) }) + "\\n");
process.stdout.write(process.env.LAZYKATOK_FAKE_OUTPUT ?? "{}");
`,
	);
	chmodSync(binaryPath, 0o755);
}

function loggedArgs(): readonly (readonly string[])[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => (JSON.parse(line) as { args: string[] }).args);
}

class StubSearchClient {
	public hits: readonly LazykatokHit[] = [];
	async search(
		_mode: LazykatokSearchMode,
		_query: string,
		_options?: LazykatokSearchOptions,
	): Promise<LazykatokSearchResult> {
		return { ok: true, hits: this.hits, data: { hits: this.hits }, stdout: "", stderr: "", code: 0 };
	}
}

describe("LazykatokClient direct CLI execution", () => {
	it("invokes lazykatok without forcing an AutoRAG-managed --workspace", async () => {
		writeFakeLazykatok();
		const client = new LazykatokClient({
			binaryPath,
			env: {
				PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}`,
				LAZYKATOK_FAKE_OUTPUT: JSON.stringify({ ready: true }),
			},
		});

		const result = await client.doctor();

		expect(result.ok).toBe(true);
		const args = loggedArgs()[0] ?? [];
		expect(args).not.toContain("--workspace");
	});

	it("parses real lazykatok search arrays with chat identity fields", async () => {
		writeFakeLazykatok();
		const realHits = [
			{
				ranker: "keyword",
				unit: "micro_chunk",
				rank: 1,
				chunk_id: "chunk_58b3852eace05c64",
				chat_name: "오픈소스 개발과제",
				sender_nickname: "투이컨설팅이헤지",
				started_at: "2026-04-13T05:57:50+00:00",
				ended_at: "2026-04-13T05:57:50+00:00",
				snippet: "발표 작업 자료 위해 투이컨설팅 류동현 선임 초대합니다.",
				score: 1.0,
				parent_chunk_ids: [],
				child_chunk_ids: [],
			},
		];
		const client = new LazykatokClient({
			binaryPath,
			env: {
				PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}`,
				LAZYKATOK_FAKE_OUTPUT: JSON.stringify(realHits),
			},
		});

		const result = await client.search("keyword", "류동현", { topK: 1 });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.data.hits).toHaveLength(1);
		expect(result.data.hits[0]).toMatchObject({
			chunkId: "chunk_58b3852eace05c64",
			content: "발표 작업 자료 위해 투이컨설팅 류동현 선임 초대합니다.",
			score: 1.0,
		});
		expect(result.data.hits[0]?.metadata).toMatchObject({
			chatName: "오픈소스 개발과제",
			senderNickname: "투이컨설팅이헤지",
			startedAt: "2026-04-13T05:57:50+00:00",
		});
	});
});

describe("KakaoTalk datasource through the config factory", () => {
	it("spawns the default lazykatok binary from PATH and maps real CLI payloads to kakao sources", async () => {
		const doctor = {
			archive: { status: "present" },
			command: "lazykatok",
			freshness: {
				last_sync: { chunks: 33272, total_messages: 50377 },
				recommendation: { sync_before_search: false },
			},
			name: "lazykatok",
			source_adapter: { configured: "fixture", fixture: "ok" },
		};
		const sync = {
			inserted_messages: 3,
			updated_messages: 0,
			total_messages: 50377,
			chunks: 33272,
			rebuilt_chats: 1,
		};
		const index = {
			full: false,
			dry_run: false,
			candidate_chunks: 33272,
			written_documents: 33272,
			embedder: "embeddinggemma-300m-q4",
		};
		const hits = [
			{
				ranker: "keyword",
				unit: "micro_chunk",
				rank: 1,
				chunk_id: "chunk_58b3852eace05c64",
				chat_id: "348487216782557",
				chat_name: "오픈소스 개발과제",
				sender_nickname: "정철현 박사님",
				started_at: "2026-04-13T05:57:50+00:00",
				ended_at: "2026-04-13T05:57:50+00:00",
				snippet: "발표 작업 자료 위해 초대합니다.",
				score: 1.0,
			},
		];
		// The shim answers only the real lazykatok JSON shapes and is reachable
		// solely under the name `lazykatok`, so PATH resolution proves the default
		// binary name the product spawns.
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
const reply = (value) => { process.stdout.write(JSON.stringify(value)); process.exit(0); };
if (args[0] === "doctor") reply(${JSON.stringify(doctor)});
if (args[0] === "sync") reply(${JSON.stringify(sync)});
if (args[0] === "index") reply(${JSON.stringify(index)});
if (args[0] === "search") reply(${JSON.stringify(hits)});
process.exit(1);
`,
		);
		chmodSync(binaryPath, 0o755);

		const built = buildDatasourceSkills({
			kakao: { connector: { env: { PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}` } } },
		});
		const skill = built.skills.find((candidate) => candidate.describe().datasourceId === "kakao");
		expect(skill).toBeDefined();
		if (skill === undefined) return;

		const indexed = await skill.index();

		expect(indexed).toMatchObject({ ok: true, chunkCount: 33272 });

		const bm25 = skill.retrievalMethods().find((method) => method.describe().type === "bm25");
		expect(bm25).toBeDefined();
		if (bm25 === undefined) return;
		const results = await bm25.retrieve("회의", { topK: 5 });

		expect(results.map((result) => result.source)).toEqual(["/kakao/default/chunks/chunk_58b3852eace05c64"]);
		expect(loggedArgs().map((args) => args[0])).toEqual(["doctor", "sync", "index", "search"]);
		// The product names the live macOS adapter on macOS; elsewhere the CLI's own
		// config adapter decides, so no --source flag is added.
		expect(loggedArgs()[1]).toEqual(
			process.platform === "darwin" ? ["sync", "--source", "macos", "--json"] : ["sync", "--json"],
		);
	});
});

describe("Lazykatok retrieval source identity", () => {
	it("labels kakao hits with the canonical slash datasource source", async () => {
		const client = new StubSearchClient();
		client.hits = [
			{
				chunkId: "chunk-001",
				content: "류동현 yoopro@2e.co.kr",
				score: 1.0,
				metadata: {
					chatName: "오픈소스 개발과제",
					senderNickname: "류동현투이컨설팅",
					startedAt: "2026-05-11T04:24:36+00:00",
				},
			},
		];
		const method = new LazykatokBm25Method({ client, instanceId: "default" });

		const results = await method.retrieve("류동현", { topK: 5 });

		expect(results).toHaveLength(1);
		const source = results[0]?.source ?? "";
		expect(source).toBe("/kakao/default/chunks/chunk-001");
		expect(results[0]?.metadata).toMatchObject({ datasourceId: "kakao", chatName: "오픈소스 개발과제" });
	});
});
