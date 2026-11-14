import { describe, expect, it, type Mock, vi } from "vitest";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { ParallelRetriever } from "../../src/retrieval/merger.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";

const SEP = Date.UTC(2026, 8, 15);
const OCT = Date.UTC(2026, 9, 15);
const NOV = Date.UTC(2026, 10, 15);
const WINDOW_OCT = { since: Date.UTC(2026, 9, 1), until: Date.UTC(2026, 10, 1) };

function hit(id: string, source: string, metadata: Record<string, unknown>, score = 1): RetrievalResult {
	return { id, content: id, source, score, metadata };
}

interface SpiedMethod extends RetrievalMethod {
	retrieve: Mock<(query: string, options: RetrievalOptions) => Promise<RetrievalResult[]>>;
}

function method(
	name: string,
	results: readonly RetrievalResult[],
	descriptor: Partial<RetrievalMethodDescriptor> = {},
): SpiedMethod {
	return {
		describe: () => ({
			name,
			type: "bm25",
			description: name,
			status: "active",
			capabilities: [],
			...descriptor,
		}),
		retrieve: vi.fn(async () => [...results]),
	};
}

describe("time window through the retrieval pipeline", () => {
	it("keeps only results inside [since, until) and leaves queries without a window untouched", async () => {
		const m = method("chat", [
			hit("sep", "/chat/a/chunks/1", { publishedAt: SEP }),
			hit("oct", "/chat/a/chunks/2", { publishedAt: OCT }),
			hit("nov", "/chat/a/chunks/3", { publishedAt: NOV }),
			hit("edge-until", "/chat/a/chunks/4", { publishedAt: WINDOW_OCT.until }),
			hit("edge-since", "/chat/a/chunks/5", { publishedAt: WINDOW_OCT.since }),
		]);
		const retriever = new ParallelRetriever();

		const windowed = await retriever.retrieveWithDiagnostics([m], "q", { topK: 10, ...WINDOW_OCT });
		expect(windowed.results.get("chat")?.map((r) => r.id)).toEqual(["oct", "edge-since"]);

		const plain = await retriever.retrieveWithDiagnostics([m], "q", { topK: 10 });
		expect(plain.results.get("chat")).toHaveLength(5);
		expect(plain.window).toBeUndefined();
	});

	it("over-fetches when a window is set, so a thin top-K does not hide in-window evidence", async () => {
		const m = method("chat", []);
		await new ParallelRetriever().retrieveWithDiagnostics([m], "q", { topK: 10, ...WINDOW_OCT });
		expect(m.retrieve.mock.calls[0]?.[1]).toMatchObject({ topK: 50, ...WINDOW_OCT });

		const plain = method("plain", []);
		await new ParallelRetriever().retrieveWithDiagnostics([plain], "q", { topK: 10 });
		expect(plain.retrieve.mock.calls[0]?.[1].topK).toBe(10);
	});

	it("falls back to the local source mtime when a result has no publishedAt", async () => {
		const m = method("minsync", [
			hit("old", "/abs/old.md", { sourceMtimeMs: SEP }),
			hit("new", "/abs/new.md", { sourceMtimeMs: OCT }),
			hit("both", "/abs/both.md", { publishedAt: OCT, sourceMtimeMs: SEP }),
		]);
		const { results } = await new ParallelRetriever().retrieveWithDiagnostics([m], "q", { ...WINDOW_OCT });
		expect(results.get("minsync")?.map((r) => r.id)).toEqual(["new", "both"]);
	});

	it("excludes results with no timestamp and reports the method with the reason, never passing them as in-range", async () => {
		const timed = method("chat", [hit("t", "/chat/a/chunks/1", { publishedAt: OCT })], { datasourceId: "chat" });
		const untimed = method(
			"gallery",
			[
				hit("u1", "/gallery/a/1", {}),
				hit("u2", "/gallery/a/2", {}),
				hit("t2", "/gallery/a/3", { publishedAt: OCT }),
			],
			{ datasourceId: "gallery" },
		);

		const { results, diagnostics, unsearched, window } = await new ParallelRetriever().retrieveWithDiagnostics(
			[timed, untimed],
			"q",
			{ ...WINDOW_OCT },
		);

		expect(results.get("gallery")?.map((r) => r.id)).toEqual(["t2"]);
		expect(unsearched).toEqual([
			{
				surface: "gallery",
				methods: ["gallery"],
				reason: expect.stringMatching(
					/2 of 3 result\(s\) from "gallery" .*neither metadata\.publishedAt nor a source mtime/,
				),
			},
		]);
		expect(diagnostics).toContainEqual(
			expect.objectContaining({ code: "time-window-untimed", source: "gallery", severity: "warning" }),
		);
		expect(diagnostics.some((d) => d.source === "chat")).toBe(false);
		expect(window).toMatchObject({ examined: 4, matched: 2, untimed: 2 });
	});

	it("reports when the over-fetch ceiling was hit so a thin result differs from an empty range", async () => {
		// topK 2 -> depth 22; a method that returns exactly 22 candidates was cut by that depth.
		const full = method(
			"deep",
			Array.from({ length: 22 }, (_, i) => hit(`d${i}`, `/deep/${i}`, { publishedAt: SEP })),
		);
		const hitCeiling = await new ParallelRetriever().retrieveWithDiagnostics([full], "q", { topK: 2, ...WINDOW_OCT });
		expect(hitCeiling.window).toMatchObject({ examined: 22, matched: 0, ceilingHit: true, fetchDepth: 22 });

		const shallow = method("shallow", [hit("s", "/s/1", { publishedAt: SEP })]);
		const exhausted = await new ParallelRetriever().retrieveWithDiagnostics([shallow], "q", {
			topK: 2,
			...WINDOW_OCT,
		});
		expect(exhausted.window).toMatchObject({ examined: 1, matched: 0, ceilingHit: false });
		expect(hitCeiling.diagnostics).toContainEqual(expect.objectContaining({ code: "time-window-ceiling" }));
		expect(exhausted.diagnostics.some((d) => d.code === "time-window-ceiling")).toBe(false);
	});

	it("carries the window bounds on the report", async () => {
		const { window } = await new ParallelRetriever().retrieveWithDiagnostics([method("m", [])], "q", {
			since: 5,
		});
		expect(window).toMatchObject({ since: 5, examined: 0, matched: 0, untimed: 0, ceilingHit: false });
		expect(window?.until).toBeUndefined();
	});
});

describe("time window through RetrievalEngine", () => {
	function engineWith(...methods: RetrievalMethod[]): RetrievalEngine {
		const engine = new RetrievalEngine();
		for (const m of methods) engine.register(m);
		return engine;
	}

	it("composes with scope: both must match, and a window never surfaces a result scope excludes", async () => {
		const ds = method(
			"chat",
			[
				hit("in-both", "/chat/work/chunks/1", { publishedAt: OCT }),
				hit("wrong-scope", "/chat/home/chunks/2", { publishedAt: OCT }),
				hit("wrong-time", "/chat/work/chunks/3", { publishedAt: SEP }),
			],
			{ datasourceId: "chat", capabilities: ["scoped"] },
		);
		const { results } = await engineWith(ds).retrieve("q", { scope: "/chat/work/**", ...WINDOW_OCT });
		expect(results.map((r) => r.id)).toEqual(["in-both"]);
	});

	it("returns the merged top-K the caller asked for, not the over-fetch depth", async () => {
		const many = method(
			"minsync",
			Array.from({ length: 30 }, (_, i) => hit(`h${i}`, `/abs/${i}.md`, { sourceMtimeMs: OCT }, 30 - i)),
		);
		const { results, window } = await engineWith(many).retrieve("q", { topK: 3, ...WINDOW_OCT });
		expect(results).toHaveLength(3);
		expect(window).toMatchObject({ examined: 30, matched: 30 });
	});

	it("exposes the window report on retrieveSelected and retrieveByMethod and filters per method", async () => {
		const m = method("minsync", [
			hit("a", "/abs/a.md", { sourceMtimeMs: OCT }),
			hit("b", "/abs/b.md", { sourceMtimeMs: SEP }),
		]);
		const engine = engineWith(m);
		const selected = await engine.retrieveSelected("q", {}, { ...WINDOW_OCT });
		expect(selected.results.map((r) => r.id)).toEqual(["a"]);
		expect(selected.window).toMatchObject({ examined: 2, matched: 1 });
		const byMethod = await engine.retrieveByMethod("q", { ...WINDOW_OCT });
		expect(byMethod.byMethod.get("minsync")?.map((r) => r.id)).toEqual(["a"]);
		expect(byMethod.window).toMatchObject({ matched: 1 });
	});

	it("an empty range yields an explicit window report, not only an empty list", async () => {
		const m = method("minsync", [hit("b", "/abs/b.md", { sourceMtimeMs: SEP })]);
		const { results, window } = await engineWith(m).retrieve("q", { ...WINDOW_OCT });
		expect(results).toEqual([]);
		expect(window).toMatchObject({ examined: 1, matched: 0, untimed: 0, ceilingHit: false });
	});
});
