import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
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
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: JSON.stringify({ ready: true }) },
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
			env: { PATH: `${binDir}:${process.env.PATH ?? ""}`, LAZYKATOK_FAKE_OUTPUT: JSON.stringify(realHits) },
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
