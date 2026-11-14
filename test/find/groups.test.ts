import { describe, expect, it } from "vitest";
import { buildGroupRunners } from "../../src/find/groups.ts";
import { fsearchRegexQuery } from "../../src/find/text.ts";
import { answer, fakeRuntime, judgeFrom, result } from "./fakes.ts";

function runner(key: string, deps: Parameters<typeof buildGroupRunners>[0]) {
	const found = buildGroupRunners(deps).find((entry) => entry.key === key);
	if (found === undefined) throw new Error(`no runner for ${key}`);
	return found.run();
}

describe("buildGroupRunners", () => {
	it("lists files, content, then datasources in config order", () => {
		const runners = buildGroupRunners({
			query: "q",
			topK: 5,
			runtime: fakeRuntime({ datasources: ["kakao", "slack"] }),
			platform: "darwin",
			datasources: [
				{ key: "source:kakao", datasourceId: "kakao", datasourceType: "kakao" },
				{ key: "source:slack", datasourceId: "slack", datasourceType: "slack" },
			],
		});
		expect(runners.map((entry) => entry.key)).toEqual(["files", "content", "source:kakao", "source:slack"]);
	});

	it("searches files through the NFC+NFD regex and maps entries to file hits", async () => {
		const calls: string[] = [];
		const runtime = fakeRuntime({
			searchFsearch: async (request) => {
				calls.push(String(request.query));
				return {
					ok: true,
					backend: "fsearch-cli",
					results: [{ path: "/tmp/보고서.md", name: "보고서.md", type: "file", size: 1, dateModified: undefined }],
				};
			},
		});
		const outcome = await runner("files", { query: "보고서", topK: 3, runtime, platform: "darwin" });
		expect(calls).toEqual([fsearchRegexQuery("보고서")]);
		expect(outcome.status).toBe("ready");
		expect(outcome.hits).toEqual([
			{
				id: "/tmp/보고서.md",
				kind: "file",
				title: "보고서.md",
				author: null,
				snippet: "/tmp/보고서.md",
				content: "",
				path: "/tmp/보고서.md",
				line: null,
				timestamp: null,
				url: null,
				score: null,
			},
		]);
	});

	it("marks the files group off when the filename index is disabled", async () => {
		const outcome = await runner("files", {
			query: "q",
			topK: 3,
			runtime: fakeRuntime(),
			platform: "darwin",
			fileSearchEnabled: false,
		});
		expect(outcome).toEqual({ status: "off", hits: [], error: null });
	});

	it("marks the files group unavailable when the index binary is missing", async () => {
		const runtime = fakeRuntime({
			searchFsearch: async () => ({ ok: false, reason: "binary-missing", message: "fsearch-cli not installed" }),
		});
		const outcome = await runner("files", { query: "q", topK: 3, runtime, platform: "darwin" });
		expect(outcome.status).toBe("unavailable");
		expect(outcome.error).toBe("fsearch-cli not installed");
	});

	it("marks the files group error when the search itself fails", async () => {
		const runtime = fakeRuntime({
			searchFsearch: async () => ({ ok: false, reason: "search-failed", message: "fsearch-cli exited 2" }),
		});
		const outcome = await runner("files", { query: "q", topK: 3, runtime, platform: "darwin" });
		expect(outcome.status).toBe("error");
		expect(outcome.error).toBe("fsearch-cli exited 2");
	});

	it("uses the Everything backend on win32", async () => {
		const runtime = fakeRuntime({
			searchEverything: async () => ({
				ok: true,
				results: [{ path: "C:\\doc.md", type: "file", size: undefined, dateModified: undefined }],
			}),
		});
		const outcome = await runner("files", { query: "doc", topK: 3, runtime, platform: "win32" });
		expect(outcome.status).toBe("ready");
		expect(outcome.hits[0]?.path).toBe("C:\\doc.md");
	});

	it("dedupes content hits per file, keeping the best chunk", async () => {
		const runtime = fakeRuntime({
			searchSelected: async () =>
				answer({
					results: [
						result({ source: "/tmp/a.md", content: "low", score: 0.2 }),
						result({ source: "/tmp/a.md", content: "high", score: 0.9 }),
						result({ source: "/tmp/b.md", content: "other", score: 0.5 }),
					],
				}),
		});
		const outcome = await runner("content", { query: "needle", topK: 5, runtime, platform: "darwin" });
		expect(outcome.hits.map((hit) => hit.path)).toEqual(["/tmp/a.md", "/tmp/b.md"]);
		expect(outcome.hits[0]?.content).toBe("high");
	});

	it("marks content unavailable when the local index is not built", async () => {
		const runtime = fakeRuntime({
			searchSelected: async () =>
				answer({ unsearched: [{ surface: "minsync", methods: ["hybrid"], reason: "index missing" }] }),
		});
		const outcome = await runner("content", { query: "needle", topK: 5, runtime, platform: "darwin" });
		expect(outcome).toEqual({ status: "unavailable", hits: [], error: null });
	});

	it("marks content off when semantic search is disabled", async () => {
		const outcome = await runner("content", {
			query: "needle",
			topK: 5,
			runtime: fakeRuntime(),
			platform: "darwin",
			semanticEnabled: false,
		});
		expect(outcome).toEqual({ status: "off", hits: [], error: null });
	});

	it("derives a source hit from result metadata", async () => {
		const runtime = fakeRuntime({
			searchSelected: async () =>
				answer({
					results: [
						result({
							id: "m1",
							source: "kakao",
							content: "예약 확인 부탁드립니다",
							score: 0.8,
							metadata: {
								chatName: "여행방",
								senderNickname: "연규",
								publishedAt: 1_700_000_000_000,
								url: "https://example.com/x",
							},
						}),
					],
				}),
		});
		const outcome = await runner("source:kakao", {
			query: "예약",
			topK: 5,
			runtime,
			platform: "darwin",
			datasources: [{ key: "source:kakao", datasourceId: "kakao", datasourceType: "kakao" }],
		});
		expect(outcome.status).toBe("ready");
		expect(outcome.hits[0]).toMatchObject({
			id: "m1",
			kind: "message",
			title: "여행방",
			author: "연규",
			path: null,
			line: null,
			timestamp: 1_700_000_000_000,
			url: "https://example.com/x",
			score: 0.8,
		});
	});

	it("isolates one datasource's failure from another", async () => {
		const runtime = fakeRuntime({
			datasources: ["bad", "good"],
			searchSelected: async (_query, selection) => {
				if (selection?.datasourceIds?.[0] === "bad") throw new Error("kakao CLI missing");
				return answer({ results: [result({ id: "ok", source: "good", content: "hit" })] });
			},
		});
		const deps = {
			query: "hit",
			topK: 5,
			runtime,
			platform: "darwin" as const,
			datasources: [
				{ key: "source:bad", datasourceId: "bad", datasourceType: "bad" },
				{ key: "source:good", datasourceId: "good", datasourceType: "good" },
			],
		};
		const bad = await runner("source:bad", deps);
		const good = await runner("source:good", deps);
		expect(bad.status).toBe("error");
		expect(bad.error).toContain("kakao CLI missing");
		expect(good.status).toBe("ready");
		expect(good.hits).toHaveLength(1);
	});

	it("applies the relevance gate to source hits", async () => {
		const runtime = fakeRuntime({
			searchSelected: async () =>
				answer({
					results: [
						result({ id: "a", source: "kakao", content: "alpha" }),
						result({ id: "b", source: "kakao", content: "beta" }),
					],
				}),
		});
		const outcome = await runner("source:kakao", {
			query: "needle",
			topK: 5,
			runtime,
			platform: "darwin",
			judge: judgeFrom([0.1, 0.9]),
			datasources: [{ key: "source:kakao", datasourceId: "kakao", datasourceType: "kakao" }],
		});
		expect(outcome.hits.map((hit) => hit.id)).toEqual(["b"]);
	});
});
