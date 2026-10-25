/**
 * Live E2E: real Jev routing plus a real model carry out file organization and
 * a non-search task through AutoRAGAgent, then the files on disk are checked.
 * Every organize/extra query also records the agent's tool calls and asserts no
 * retrieval or web tool ran: no route changes the active tool set any more, so
 * the turn prompt is what keeps a file-organization turn from searching.
 *
 * Gated on AUTORAG_ORGANIZE_LIVE=1 plus OPENROUTER_API_KEY. The agent model id
 * comes from AUTORAG_ORGANIZE_MODEL (an OpenRouter id); pick one released in 2026 or later.
 * Jev routes through its default OpenRouter backend.
 *
 *   AUTORAG_ORGANIZE_LIVE=1 AUTORAG_ORGANIZE_MODEL=<id> OPENROUTER_API_KEY=... \
 *     bun run vitest run test/live-e2e/organize-route.test.ts
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-all-tool.ts";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-minsync-tool.ts";
import { WEB_FETCH_TOOL_NAME } from "../../src/agent/web-fetch-tool.ts";
import { WEB_SEARCH_TOOL_NAME } from "../../src/agent/web-search-tool.ts";
import { resolveModel } from "../../src/cli/config.ts";

/** Tools a file-organization or other-action turn must never call. */
const RETRIEVAL_TOOL_NAMES = [
	SEARCH_ALL_DOCUMENTS_TOOL_NAME,
	SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
	WEB_SEARCH_TOOL_NAME,
	WEB_FETCH_TOOL_NAME,
] as const;

const MODEL_ID = process.env.AUTORAG_ORGANIZE_MODEL ?? "";
const LIVE =
	process.env.AUTORAG_ORGANIZE_LIVE === "1" &&
	MODEL_ID.length > 0 &&
	(process.env.OPENROUTER_API_KEY ?? "").length > 0;

function routeOf(response: SearchDocumentsResponse): string | undefined {
	const routed = response.diagnostics?.find((diagnostic) => diagnostic.code === "query-routed");
	return /routed the question to (\w+)/u.exec(routed?.message ?? "")?.[1];
}

describe.skipIf(!LIVE)("live: Jev routes file organization and other actions, and the agent really does them", () => {
	let root: string;
	let agent: AutoRAGAgent;

	beforeAll(async () => {
		root = mkdtempSync(join(tmpdir(), "autorag-organize-live-"));
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		const model = await resolveModel({
			searchPaths: [docs],
			model: { provider: "openrouter", id: MODEL_ID },
		} as never);
		agent = new AutoRAGAgent({
			model,
			apiKey: process.env.OPENROUTER_API_KEY,
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			piAgentDir: join(root, "pi-agent"),
			jev: { backend: "openrouter" },
			everything: false,
			fsearch: false,
			jikji: false,
			webSearch: false,
		} as never);
	});
	afterAll(() => rmSync(root, { recursive: true, force: true }));

	/**
	 * Runs one live query with the agent's tool calls recorded, and fails when a
	 * retrieval or web tool ran: an `organize` or `extra` turn may not search.
	 */
	async function searchAction(query: string): Promise<SearchDocumentsResponse> {
		const called: string[] = [];
		const unsubscribe = agent.subscribe((event) => {
			if (event.type === "tool_execution_start") called.push(event.toolName);
		});
		try {
			const response = await agent.searchDocuments(query);
			expect(called.length, "the turn called no tool at all").toBeGreaterThan(0);
			const offending = called.filter(
				(name) =>
					(RETRIEVAL_TOOL_NAMES as readonly string[]).includes(name) || name.startsWith("search_datasource_"),
			);
			expect(offending, `retrieval or web tools ran on this turn: ${offending.join(", ")}`).toEqual([]);
			return response;
		} finally {
			unsubscribe();
		}
	}

	it("sorts a messy folder into per-type folders (organize)", async () => {
		const messy = join(root, "messy");
		mkdirSync(messy, { recursive: true });
		for (const name of ["a.pdf", "b.pdf", "c.png", "d.png", "e.txt"]) writeFileSync(join(messy, name), name);

		const response = await searchAction(
			`${messy} 폴더에 파일이 뒤죽박죽이야. 확장자별로 하위 폴더(pdf, png, txt)를 만들어서 정리해줘.`,
		);

		expect(routeOf(response)).toBe("organize");
		expect(existsSync(join(messy, "pdf", "a.pdf"))).toBe(true);
		expect(existsSync(join(messy, "pdf", "b.pdf"))).toBe(true);
		expect(existsSync(join(messy, "png", "c.png"))).toBe(true);
		expect(existsSync(join(messy, "png", "d.png"))).toBe(true);
		expect(existsSync(join(messy, "txt", "e.txt"))).toBe(true);
		expect(readdirSync(messy).filter((name) => name.includes("."))).toEqual([]);
		expect(response.results).toEqual([]);
	}, 240_000);

	it("renames files in bulk by pattern (organize)", async () => {
		const folder = join(root, "photos");
		mkdirSync(folder, { recursive: true });
		for (const name of ["IMG_001.jpg", "IMG_002.jpg", "IMG_003.jpg"]) writeFileSync(join(folder, name), name);

		const response = await searchAction(
			`${folder} 안의 IMG_ 로 시작하는 파일 이름을 전부 trip_ 으로 시작하게 일괄 변경해줘. (IMG_001.jpg -> trip_001.jpg)`,
		);

		expect(routeOf(response)).toBe("organize");
		expect(readdirSync(folder).sort()).toEqual(["trip_001.jpg", "trip_002.jpg", "trip_003.jpg"]);
		expect(readFileSync(join(folder, "trip_002.jpg"), "utf8")).toBe("IMG_002.jpg");
	}, 240_000);

	it("renames files according to what their content says (organize)", async () => {
		const folder = join(root, "scans");
		mkdirSync(folder, { recursive: true });
		writeFileSync(join(folder, "scan001.txt"), "INVOICE\nVendor: Acme Corp\nAmount due: 1,200 USD\nDue 2026-11-30\n");
		writeFileSync(
			join(folder, "scan002.txt"),
			"MEETING MINUTES\nTopic: Q4 roadmap review\nAttendees: Mina, Joon\nDate 2026-10-02\n",
		);

		const response = await searchAction(
			`${folder} 안의 scan001.txt, scan002.txt 는 이름이 의미가 없어. 각 파일의 내용을 읽고 invoice 인지 meeting 인지 알 수 있게 이름을 바꿔줘. 확장자는 유지하고 이름에 invoice 또는 meeting 이라는 단어를 넣어줘.`,
		);

		expect(routeOf(response)).toBe("organize");
		const names = readdirSync(folder).map((name) => name.toLowerCase());
		expect(names).toHaveLength(2);
		expect(names.some((name) => name.includes("invoice") && name.endsWith(".txt"))).toBe(true);
		expect(names.some((name) => name.includes("meeting") && name.endsWith(".txt"))).toBe(true);
		const invoice = readdirSync(folder).find((name) => name.toLowerCase().includes("invoice"));
		expect(readFileSync(join(folder, invoice ?? ""), "utf8")).toContain("Acme Corp");
	}, 240_000);

	it("converts files to another format (organize)", async () => {
		const folder = join(root, "convert");
		mkdirSync(folder, { recursive: true });
		writeFileSync(join(folder, "notes.md"), "# Notes\n\nHello from the live conversion test.\n");

		const response = await searchAction(
			`${folder}/notes.md 파일을 html 로 변환해서 같은 폴더에 notes.html 로 저장해줘.`,
		);

		expect(routeOf(response)).toBe("organize");
		const html = readdirSync(folder).find((name) => name.endsWith(".html"));
		expect(html).toBeDefined();
		expect(readFileSync(join(folder, html ?? ""), "utf8")).toContain("Hello from the live conversion test");
		expect(existsSync(join(folder, "notes.md"))).toBe(true);
	}, 360_000);

	it("commits in a git repository without searching (extra)", async () => {
		const repo = join(root, "repo");
		mkdirSync(repo, { recursive: true });
		const git = (...args: string[]) =>
			execFileSync("git", ["-C", repo, ...args], {
				encoding: "utf8",
				env: { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null" },
			});
		git("init", "-q", "-b", "main");
		git("config", "user.email", "qa@example.com");
		git("config", "user.name", "QA");
		writeFileSync(join(repo, "hello.txt"), "hello\n");

		const response = await searchAction(
			`${repo} git 저장소에 hello.txt 가 새로 생겼어. 이 변경을 커밋해줘. 커밋 메시지는 "add hello"로 해줘.`,
		);

		expect(routeOf(response)).toBe("extra");
		expect(git("log", "--format=%s").trim()).toBe("add hello");
		expect(response.results).toEqual([]);
	}, 240_000);

	it("answers a general-knowledge question in one extra turn (extra)", async () => {
		const response = await agent.searchDocuments("프랑스의 수도는 어디야?");

		expect(routeOf(response)).toBe("extra");
		// One turn: no retrieval prefetch, no fast-phase preliminary, no verification.
		expect(response.results).toEqual([]);
		expect(response.answer).toMatch(/파리|Paris/u);
	}, 240_000);

	it("answers a general-knowledge question without a preliminary in the stream (extra)", async () => {
		const events: string[] = [];
		let answer = "";
		for await (const event of agent.searchDocumentsStream("What is the capital of Australia?")) {
			events.push(event.type);
			if (event.type === "complete") answer = event.response.answer;
		}

		expect(events).toContain("complete");
		expect(events).not.toContain("preliminary");
		expect(answer).toMatch(/Canberra/u);
	}, 240_000);

	it("keeps a personal-records question on local search (local)", async () => {
		writeFileSync(join(root, "docs", "budget.txt"), "The Q3 budget was approved by Mina Park.\n");
		const response = await agent.searchDocuments("내 문서에서 Q3 예산을 누가 승인했는지 찾아줘");

		expect(routeOf(response)).toBe("local");
	}, 360_000);
});
