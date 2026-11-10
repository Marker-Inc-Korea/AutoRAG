import { describe, expect, it, vi } from "vitest";
import { runFind } from "../../src/find/run.ts";
import type { FindLine, FindSearchAnswer } from "../../src/find/types.ts";
import { answer, fakeRuntime, result } from "./fakes.ts";

function collect(): { readonly lines: FindLine[]; readonly emit: (line: FindLine) => void } {
	const lines: FindLine[] = [];
	return { lines, emit: (line) => lines.push(line) };
}

function deferred<T>(): { readonly promise: Promise<T>; readonly resolve: (value: T) => void } {
	let resolveFn!: (value: T) => void;
	const promise = new Promise<T>((resolve) => {
		resolveFn = resolve;
	});
	return { promise, resolve: resolveFn };
}

describe("runFind", () => {
	it("emits the groups line first and done last, every line JSON-serializable", async () => {
		const { lines, emit } = collect();
		await runFind({
			query: "q",
			topK: 5,
			runtime: fakeRuntime(),
			platform: "darwin",
			datasources: [],
			emit,
		});
		expect(lines[0]?.type).toBe("groups");
		expect(lines.at(-1)?.type).toBe("done");
		for (const line of lines) expect(() => JSON.parse(JSON.stringify(line))).not.toThrow();
		expect(lines.at(0)).toMatchObject({ type: "groups", groups: [{ key: "files" }, { key: "content" }] });
	});

	it("emits each group line in completion order, not plan order", async () => {
		const slow = deferred<FindSearchAnswer>();
		const runtime = fakeRuntime({
			searchSelected: async (_query, selection) => {
				if (selection?.datasourceIds?.[0] === "slow") return slow.promise;
				return answer({ results: [result({ id: "fast", source: "fast" })] });
			},
		});
		const { lines, emit } = collect();
		const run = runFind({
			query: "q",
			topK: 5,
			runtime,
			platform: "darwin",
			datasources: [
				{ key: "source:slow", datasourceId: "slow", datasourceType: "slow" },
				{ key: "source:fast", datasourceId: "fast", datasourceType: "fast" },
			],
			emit,
		});
		await vi.waitFor(() => {
			// The fast datasource resolves first; the slow one has not yet.
			expect(lines.some((line) => line.type === "group" && line.key === "source:fast")).toBe(true);
		});
		expect(lines.some((line) => line.type === "group" && line.key === "source:slow")).toBe(false);
		slow.resolve(answer({ results: [] }));
		await run;
		const keys = lines.filter((line) => line.type === "group").map((line) => (line.type === "group" ? line.key : ""));
		expect(keys.indexOf("source:fast")).toBeLessThan(keys.indexOf("source:slow"));
	});

	it("isolates a throwing group so the others still report", async () => {
		const runtime = fakeRuntime({
			searchSelected: async (_query, selection) => {
				if (selection?.datasourceIds?.[0] === "bad") throw new Error("boom");
				return answer({ results: [result({ id: "ok", source: "good" })] });
			},
		});
		const { lines, emit } = collect();
		await runFind({
			query: "q",
			topK: 5,
			runtime,
			platform: "darwin",
			datasources: [
				{ key: "source:bad", datasourceId: "bad", datasourceType: "bad" },
				{ key: "source:good", datasourceId: "good", datasourceType: "good" },
			],
			emit,
		});
		const bad = lines.find((line) => line.type === "group" && line.key === "source:bad");
		const good = lines.find((line) => line.type === "group" && line.key === "source:good");
		expect(bad).toMatchObject({ status: "error", error: "boom" });
		expect(good).toMatchObject({ status: "ready" });
		expect(lines.at(-1)?.type).toBe("done");
	});

	it("narrows to the requested groups", async () => {
		const { lines, emit } = collect();
		await runFind({
			query: "q",
			topK: 5,
			runtime: fakeRuntime(),
			platform: "darwin",
			datasources: [{ key: "source:kakao", datasourceId: "kakao", datasourceType: "kakao" }],
			groups: ["files", "source:kakao"],
			emit,
		});
		expect(lines[0]).toMatchObject({ groups: [{ key: "files" }, { key: "source:kakao" }] });
		const keys = lines.filter((line) => line.type === "group").map((line) => (line.type === "group" ? line.key : ""));
		expect(keys.sort()).toEqual(["files", "source:kakao"]);
	});

	it("sets the off status for a disabled group", async () => {
		const { lines, emit } = collect();
		await runFind({
			query: "q",
			topK: 5,
			runtime: fakeRuntime(),
			platform: "darwin",
			datasources: [],
			fileSearchEnabled: false,
			semanticEnabled: false,
			emit,
		});
		const files = lines.find((line) => line.type === "group" && line.key === "files");
		const content = lines.find((line) => line.type === "group" && line.key === "content");
		expect(files).toMatchObject({ status: "off", hits: [], error: null });
		expect(content).toMatchObject({ status: "off", hits: [], error: null });
	});

	it("skips the groups line when announce is false", async () => {
		const { lines, emit } = collect();
		await runFind({
			query: "q",
			topK: 5,
			runtime: fakeRuntime(),
			platform: "darwin",
			datasources: [],
			announce: false,
			emit,
		});
		expect(lines.some((line) => line.type === "groups")).toBe(false);
		expect(lines.at(-1)?.type).toBe("done");
	});
});
