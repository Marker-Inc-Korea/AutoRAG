import { describe, expect, it, vi } from "vitest";
import { createRelevanceGate, JEV_RELEVANCE_THRESHOLD, type RelevanceCandidate } from "../../src/find/relevance.ts";
import type { FindJudge } from "../../src/find/types.ts";
import { judgeFrom } from "./fakes.ts";

function candidate(id: string, content: string): RelevanceCandidate {
	return { id, sourceLabel: "content", title: id, subtitle: null, content };
}

const judgeThrowing: FindJudge = async () => {
	throw new Error("no decision model");
};

describe("createRelevanceGate", () => {
	it("exposes the 0.5 default probability threshold", () => {
		expect(JEV_RELEVANCE_THRESHOLD).toBe(0.5);
	});

	it("auto-keeps a lexically matching candidate without calling the judge", async () => {
		const judge = vi.fn(judgeFrom([0]));
		const gate = createRelevanceGate({ judge });
		const kept = await gate.filter("widget", [
			candidate("a", "the widget handbook"),
			candidate("b", "unrelated noise"),
		]);
		expect(kept.map((entry) => entry.id)).toEqual(["a"]);
		// The one non-lexical candidate still goes to the judge.
		expect(judge).toHaveBeenCalledTimes(1);
	});

	it("keeps only judged candidates at or above the threshold, ordered by probability", async () => {
		const gate = createRelevanceGate({ judge: judgeFrom([0.9, 0.4, 0.6]) });
		const kept = await gate.filter("needle", [
			candidate("a", "alpha"),
			candidate("b", "beta"),
			candidate("c", "gamma"),
		]);
		expect(kept.map((entry) => entry.id)).toEqual(["a", "c"]);
	});

	it("respects a configured threshold", async () => {
		const gate = createRelevanceGate({ judge: judgeFrom([0.6]), threshold: 0.7 });
		const kept = await gate.filter("needle", [candidate("a", "alpha")]);
		expect(kept).toEqual([]);
	});

	it("fails open when the judge throws", async () => {
		const gate = createRelevanceGate({ judge: judgeThrowing });
		const kept = await gate.filter("needle", [candidate("a", "alpha"), candidate("b", "beta")]);
		expect(kept.map((entry) => entry.id)).toEqual(["a", "b"]);
	});

	it("fails open when the judge returns no numeric verdicts", async () => {
		const gate = createRelevanceGate({ judge: judgeFrom([null, null]) });
		const kept = await gate.filter("needle", [candidate("a", "alpha"), candidate("b", "beta")]);
		expect(kept.map((entry) => entry.id)).toEqual(["a", "b"]);
	});

	it("returns every candidate for an empty query", async () => {
		const gate = createRelevanceGate({ judge: judgeThrowing });
		const candidates = [candidate("a", "alpha")];
		expect(await gate.filter("   ", candidates)).toEqual(candidates);
	});
});
