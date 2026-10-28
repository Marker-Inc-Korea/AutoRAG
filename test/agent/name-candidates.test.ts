import { describe, expect, it } from "vitest";
import {
	collectNameCandidates,
	MAX_NAME_CANDIDATES,
	mergeNameHits,
	searchNameCandidates,
} from "../../src/agent/name-candidates.ts";

interface Entry {
	readonly path: string;
}
type Outcome =
	| { readonly ok: true; readonly results: readonly Entry[] }
	| { readonly ok: false; readonly reason: string; readonly message: string };

describe("collectNameCandidates", () => {
	it("puts the query first, trims, and drops empty and repeated alternatives", () => {
		expect(collectNameCandidates("  invoice ", ["", "인보이스", "invoice", "  인보이스  "])).toEqual({
			candidates: ["invoice", "인보이스"],
			dropped: 0,
		});
	});

	it("treats case variants as the same candidate unless matching is case-sensitive", () => {
		expect(collectNameCandidates("Report", ["report"]).candidates).toEqual(["Report"]);
		expect(collectNameCandidates("Report", ["report"], { matchCase: true }).candidates).toEqual(["Report", "report"]);
	});

	it("keeps the first candidates up to the cap and reports how many were dropped", () => {
		const alternatives = Array.from({ length: MAX_NAME_CANDIDATES + 3 }, (_, index) => `name-${index}`);
		const { candidates, dropped } = collectNameCandidates("first", alternatives);
		expect(candidates).toHaveLength(MAX_NAME_CANDIDATES);
		expect(candidates[0]).toBe("first");
		expect(dropped).toBe(4);
	});

	it("returns nothing when every candidate is blank", () => {
		expect(collectNameCandidates("   ", [" ", ""])).toEqual({ candidates: [], dropped: 0 });
	});
});

describe("searchNameCandidates", () => {
	it("starts every candidate before any finishes", async () => {
		let started = 0;
		// tsconfig lib is ES2022, which predates Promise.withResolvers.
		let release!: () => void;
		const allStarted = new Promise<void>((resolve) => {
			release = resolve;
		});
		const outcomes = await searchNameCandidates(["a", "b", "c"], async (candidate): Promise<Outcome> => {
			started += 1;
			if (started === 3) release();
			await allStarted;
			return { ok: true, results: [{ path: `/${candidate}` }] };
		});
		expect(outcomes.map((outcome) => outcome.candidate)).toEqual(["a", "b", "c"]);
		expect(started).toBe(3);
	});

	it("turns a thrown search into a failure for that candidate only", async () => {
		const outcomes = await searchNameCandidates(["ok", "bad"], async (candidate): Promise<Outcome> => {
			if (candidate === "bad") throw new Error("socket closed");
			return { ok: true, results: [{ path: "/ok" }] };
		});
		expect(outcomes[0]?.result).toEqual({ ok: true, results: [{ path: "/ok" }] });
		expect(outcomes[1]?.result).toEqual({ ok: false, reason: "search-failed", message: "socket closed" });
	});
});

describe("mergeNameHits", () => {
	const ok = (candidate: string, ...paths: string[]) => ({
		candidate,
		result: { ok: true as const, results: paths.map((path) => ({ path })) },
	});

	it("interleaves the candidates' lists position by position so one candidate cannot fill the cut", () => {
		const hits = mergeNameHits([ok("a", "/a1", "/a2", "/a3"), ok("b", "/b1", "/b2")], 10);
		expect(hits.map((hit) => hit.entry.path)).toEqual(["/a1", "/b1", "/a2", "/b2", "/a3"]);
		const cut = mergeNameHits([ok("a", "/a1", "/a2", "/a3", "/a4"), ok("b", "/b1", "/b2", "/b3", "/b4")], 4);
		expect(cut.map((hit) => hit.entry.path)).toEqual(["/a1", "/b1", "/a2", "/b2"]);
	});

	it("puts a path found by more names ahead of paths found by one, even when it sits later in a list", () => {
		const hits = mergeNameHits([ok("invoice", "/x.pdf", "/y.pdf", "/z.pdf"), ok("인보이스", "/w.pdf", "/z.pdf")], 10);
		expect(hits.map((hit) => hit.entry.path)).toEqual(["/z.pdf", "/x.pdf", "/w.pdf", "/y.pdf"]);
		expect(hits[0]?.candidates).toEqual(["invoice", "인보이스"]);
	});

	it("lists a path found by several candidates once, with every candidate that found it", () => {
		const hits = mergeNameHits([ok("invoice", "/x.pdf", "/y.pdf"), ok("인보이스", "/x.pdf")], 10);
		expect(hits.map((hit) => hit.entry.path)).toEqual(["/x.pdf", "/y.pdf"]);
		expect(hits[0]?.candidates).toEqual(["invoice", "인보이스"]);
		expect(hits[1]?.candidates).toEqual(["invoice"]);
	});

	it("cuts the merged list at the limit and skips failed candidates", () => {
		const hits = mergeNameHits(
			[
				{ candidate: "bad", result: { ok: false as const, reason: "search-failed", message: "x" } },
				ok("a", "/a1", "/a2", "/a3"),
			],
			2,
		);
		expect(hits.map((hit) => hit.entry.path)).toEqual(["/a1", "/a2"]);
	});
});
