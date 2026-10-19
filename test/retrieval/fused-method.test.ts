import { describe, expect, it, vi } from "vitest";
import { FusedRetrievalMethod } from "../../src/retrieval/fused-method.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

function result(id: string, score = 1, source = "/github/work/repository"): RetrievalResult {
	return { id, source, score, content: `content-${id}`, metadata: { method: "bm25", datasourceId: "work" } };
}

function makeMethod(name: string, results: RetrievalResult[] = []) {
	const descriptor: RetrievalMethodDescriptor = {
		name,
		type: "bm25",
		description: "Search the configured datasource",
		status: "active",
		capabilities: ["lexical", "scope"],
		datasourceId: "work",
		tags: ["team"],
	};
	return {
		describe: () => descriptor,
		retrieve: vi.fn<RetrievalMethod["retrieve"]>().mockResolvedValue(results),
	};
}

function makeFused(lexical: RetrievalMethod, semantic: RetrievalMethod) {
	return new FusedRetrievalMethod({ name: "work-hybrid", lexical, semantic });
}

describe("FusedRetrievalMethod", () => {
	it("describes hybrid retrieval with the lexical datasource identity and capabilities", () => {
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");

		expect(makeFused(lexical, semantic).describe()).toMatchObject({
			name: "work-hybrid",
			type: "hybrid",
			status: "active",
			datasourceId: "work",
			tags: ["team"],
			capabilities: expect.arrayContaining(["lexical", "scope", "hybrid"]),
		});
		expect(lexical.describe().capabilities).toEqual(["lexical", "scope"]);
	});

	it("promotes consensus below each head without comparing incompatible raw scores", async () => {
		const shared = {
			...result("shared", 10),
			metadata: { method: "bm25", datasourceId: "work", title: "Policy" },
		};
		const lexical = makeMethod("lexical", [result("lexical-first", 1000), shared]);
		const semantic = makeMethod("semantic", [
			result("semantic-first", 0.9),
			{ ...shared, score: 0.7, content: "semantic content", metadata: { method: "semantic" } },
		]);

		const fused = await makeFused(lexical, semantic).retrieve("approval policy", { topK: 1 });

		expect(fused).toHaveLength(1);
		expect(fused[0]).toEqual({
			...shared,
			score: 2 / 62,
			metadata: { ...shared.metadata, method: "work-hybrid", mode: "hybrid" },
		});
		expect(shared.score).toBe(10);
		expect(shared.metadata.method).toBe("bm25");
	});

	it("keeps lexical first-seen order when fusion scores tie", async () => {
		const lexical = makeMethod("lexical", [result("z"), result("b")]);
		const semantic = makeMethod("semantic", [result("a"), result("c")]);

		const fused = await makeFused(lexical, semantic).retrieve("policy", { topK: 4 });

		expect(fused.map((hit) => hit.id)).toEqual(["z", "a", "b", "c"]);
	});

	it("fuses only matching source and chunk ids, preserving distinct evidence", async () => {
		const lexical = makeMethod("lexical", [
			result("shared", 10, "/github/work/a"),
			result("other", 9, "/github/work/a"),
		]);
		const semantic = makeMethod("semantic", [
			result("shared", 0.9, "/github/work/b"),
			result("shared", 0.8, "/github/work/a"),
		]);

		const fused = await makeFused(lexical, semantic).retrieve("policy", {});

		expect(fused.map(({ id, source }) => ({ id, source }))).toEqual([
			{ id: "shared", source: "/github/work/a" },
			{ id: "shared", source: "/github/work/b" },
			{ id: "other", source: "/github/work/a" },
		]);
		expect(fused[0]?.score).toBeCloseTo(1 / 61 + 1 / 62);
	});

	it("does not inflate evidence repeated by one method", async () => {
		const lexical = makeMethod("lexical", [result("a"), result("a"), result("b")]);
		const semantic = makeMethod("semantic", [result("b")]);

		const fused = await makeFused(lexical, semantic).retrieve("policy", {});

		expect(fused.map((hit) => hit.id)).toEqual(["b", "a"]);
		expect(fused[0]?.score).toBeCloseTo(1 / 63 + 1 / 61);
		expect(fused[1]?.score).toBeCloseTo(1 / 61);
	});

	it("overfetches both rankings while forwarding query scope, filters, and cancellation", async () => {
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");
		const options = {
			topK: 5,
			scope: "/github/work",
			filters: { author: "alice" },
			signal: new AbortController().signal,
		};

		await makeFused(lexical, semantic).retrieve("approval policy", options);

		for (const method of [lexical, semantic]) {
			expect(method.retrieve).toHaveBeenCalledTimes(1);
			expect(method.retrieve).toHaveBeenCalledWith("approval policy", { ...options, topK: 15 });
		}
		expect(options.topK).toBe(5);
	});

	it("defaults to twenty results and overfetches sixty candidates per method", async () => {
		const hits = Array.from({ length: 60 }, (_, index) => result(`chunk-${index}`));
		const lexical = makeMethod("lexical", hits);
		const semantic = makeMethod("semantic", hits);

		const fused = await makeFused(lexical, semantic).retrieve("policy", {});

		expect(fused).toHaveLength(20);
		for (const method of [lexical, semantic]) {
			expect(method.retrieve).toHaveBeenCalledTimes(1);
			expect(method.retrieve).toHaveBeenCalledWith("policy", { topK: 60 });
		}
	});

	it("starts both retrieval methods before waiting for either result", async () => {
		let releaseLexical!: (results: RetrievalResult[]) => void;
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");
		lexical.retrieve.mockImplementation(
			() =>
				new Promise((resolve) => {
					releaseLexical = resolve;
				}),
		);
		const pending = makeFused(lexical, semantic).retrieve("policy", {});

		await vi.waitFor(() => expect(semantic.retrieve).toHaveBeenCalledOnce());
		releaseLexical([]);
		await expect(pending).resolves.toEqual([]);
	});

	it.each(["", " \t\n "])("skips both methods for an empty query %j", async (query) => {
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");

		await expect(makeFused(lexical, semantic).retrieve(query, {})).resolves.toEqual([]);

		expect(lexical.retrieve).not.toHaveBeenCalled();
		expect(semantic.retrieve).not.toHaveBeenCalled();
	});

	it.each([0, -1])("skips both methods for topK=%i", async (topK) => {
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");

		await expect(makeFused(lexical, semantic).retrieve("policy", { topK })).resolves.toEqual([]);

		expect(lexical.retrieve).not.toHaveBeenCalled();
		expect(semantic.retrieve).not.toHaveBeenCalled();
	});

	for (const unavailable of ["lexical", "semantic"] as const) {
		it.each(["empty", "rejected", "synchronous throw"])(
			`preserves the other ranking and scores when ${unavailable} is %s`,
			async (failure) => {
				const hits = [result("first", 0.2), result("second", 0.9), result("third", 0.5)];
				const lexical = makeMethod("lexical", hits);
				const semantic = makeMethod("semantic", hits);
				const method = unavailable === "lexical" ? lexical : semantic;
				if (failure === "empty") method.retrieve.mockResolvedValue([]);
				else if (failure === "rejected") method.retrieve.mockRejectedValue(new Error("native search failed"));
				else {
					method.retrieve.mockImplementation(() => {
						throw new Error("native search failed");
					});
				}

				const fused = await makeFused(lexical, semantic).retrieve("policy", { topK: 2 });

				expect(fused).toEqual(
					hits
						.slice(0, 2)
						.map((hit) => ({ ...hit, metadata: { ...hit.metadata, method: "work-hybrid", mode: "hybrid" } })),
				);
			},
		);

		it(`propagates ${unavailable} cancellation instead of falling back`, async () => {
			const lexical = makeMethod("lexical", [result("a")]);
			const semantic = makeMethod("semantic", [result("b")]);
			const cancellation = new DOMException("query cancelled", "AbortError");
			(unavailable === "lexical" ? lexical : semantic).retrieve.mockRejectedValue(cancellation);

			await expect(makeFused(lexical, semantic).retrieve("policy", {})).rejects.toBe(cancellation);
		});
	}

	it("returns no evidence when both rankings are empty", async () => {
		await expect(makeFused(makeMethod("lexical"), makeMethod("semantic")).retrieve("policy", {})).resolves.toEqual(
			[],
		);
	});

	it.each(["lexical", "semantic"] as const)(
		"surfaces the %s failure instead of an empty result when the other ranking is empty",
		async (unavailable) => {
			// Hybrid-only selection (`methods: ["work-hybrid"]`) has no sibling method to
			// report the failure, so the fused method must not swallow it as "no matches".
			const failure = new Error("native search failed");
			const lexical = makeMethod("lexical");
			const semantic = makeMethod("semantic");
			(unavailable === "lexical" ? lexical : semantic).retrieve.mockRejectedValue(failure);

			const error = await makeFused(lexical, semantic)
				.retrieve("policy", {})
				.catch((reason: unknown) => reason);

			expect(error).toBeInstanceOf(AggregateError);
			if (!(error instanceof AggregateError)) throw new Error("Expected the surfaced retrieval failure");
			expect(error.errors).toEqual([failure]);
			expect(error.message).toContain(failure.message);
		},
	);

	it("keeps the surviving ranking when one input fails and the other has matches", async () => {
		const hits = [result("first"), result("second")];
		const lexical = makeMethod("lexical", hits);
		const semantic = makeMethod("semantic");
		semantic.retrieve.mockRejectedValue(new Error("native search failed"));

		const fused = await makeFused(lexical, semantic).retrieve("policy", { topK: 2 });

		expect(fused.map((hit) => hit.id)).toEqual(["first", "second"]);
	});

	it("reports both original failures when neither method can search", async () => {
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");
		const lexicalError = new Error("crawler exited 7: cannot open /archive/team.sqlite");
		const semanticError = new Error("embedding service returned 503: model is unavailable");
		lexical.retrieve.mockImplementation(() => {
			throw lexicalError;
		});
		semantic.retrieve.mockRejectedValue(semanticError);

		const error = await makeFused(lexical, semantic)
			.retrieve("policy", {})
			.catch((reason: unknown) => reason);

		expect(error).toBeInstanceOf(AggregateError);
		if (!(error instanceof AggregateError)) throw new Error("Expected both retrieval failures");
		expect(error.errors).toEqual([lexicalError, semanticError]);
		expect(error.message).toContain(lexicalError.message);
		expect(error.message).toContain(semanticError.message);
	});

	it("does not start retrieval for an already aborted signal", async () => {
		const lexical = makeMethod("lexical");
		const semantic = makeMethod("semantic");
		const controller = new AbortController();
		const reason = new Error("user stopped the query");
		controller.abort(reason);

		await expect(makeFused(lexical, semantic).retrieve("policy", { signal: controller.signal })).rejects.toBe(reason);

		expect(lexical.retrieve).not.toHaveBeenCalled();
		expect(semantic.retrieve).not.toHaveBeenCalled();
	});

	it("discards completed results if the caller cancelled during retrieval", async () => {
		const lexical = makeMethod("lexical", [result("a")]);
		const semantic = makeMethod("semantic");
		const controller = new AbortController();
		const reason = new Error("user stopped the query");
		semantic.retrieve.mockImplementation(async () => {
			controller.abort(reason);
			return [result("b")];
		});

		await expect(makeFused(lexical, semantic).retrieve("policy", { signal: controller.signal })).rejects.toBe(reason);
	});
});
