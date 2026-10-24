import { describe, expect, it, type Mock, vi } from "vitest";
import { MinSyncRequiredError } from "../../src/minsync/errors.ts";
import { FallbackRetrievalMethod } from "../../src/retrieval/fallback.ts";
import { ParallelRetriever, ResultMerger } from "../../src/retrieval/merger.ts";
import type {
	RetrievalFallbackEvent,
	RetrievalMethod,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";

/** A scripted method whose `retrieve` call log the tests assert on. */
interface SpiedMethod extends RetrievalMethod {
	retrieve: Mock<(query: string, options: RetrievalOptions) => Promise<RetrievalResult[]>>;
}

function hit(id: string, score = 1): RetrievalResult {
	return { id, content: id, source: `/synthetic/${id}`, score, metadata: {} };
}

function method(
	name: string,
	outcome: RetrievalResult[] | Error,
	type: "hybrid" | "bm25" | "vector" = "hybrid",
): SpiedMethod {
	return {
		describe: () => ({
			name,
			type,
			description: `${name} method`,
			status: "active" as const,
			capabilities: [name],
			datasourceId: "ds",
		}),
		retrieve: vi.fn(async () => {
			if (outcome instanceof Error) throw outcome;
			return outcome;
		}),
	};
}

describe("FallbackRetrievalMethod", () => {
	it("answers from the primary alone and never touches a fallback that is not needed", async () => {
		const primary = method("ds-hybrid", [hit("a")]);
		const fallback = method("ds-bm25", [hit("b")], "bm25");
		const chain = new FallbackRetrievalMethod(primary, [fallback]);

		const results = await chain.retrieve("q", {});

		expect(results.map((r) => r.id)).toEqual(["a"]);
		expect(results[0]?.metadata).not.toHaveProperty("fallbackFrom");
		expect(fallback.retrieve).not.toHaveBeenCalled();
	});

	it("keeps the primary's identity so selection and diagnostics see one method", () => {
		const primary = method("ds-hybrid", []);
		const chain = new FallbackRetrievalMethod(primary, [method("ds-bm25", [], "bm25")]);

		expect(chain.describe()).toEqual(primary.describe());
	});

	it("does not fall back when the primary legitimately finds nothing", async () => {
		const fallback = method("ds-bm25", [hit("b")], "bm25");
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", []), [fallback]);

		await expect(chain.retrieve("q", {})).resolves.toEqual([]);
		expect(fallback.retrieve).not.toHaveBeenCalled();
	});

	it("falls back when the primary fails, and records why", async () => {
		const primary = method("ds-hybrid", new Error("embeddings unavailable"));
		const fallback = method("ds-bm25", [hit("b")], "bm25");
		const chain = new FallbackRetrievalMethod(primary, [fallback]);

		const results = await chain.retrieve("q", { topK: 3 });

		expect(results.map((r) => r.id)).toEqual(["b"]);
		expect(results[0]?.metadata).toMatchObject({
			fallbackFrom: "ds-hybrid",
			fallbackReason: "Error: embeddings unavailable",
		});
		expect(fallback.retrieve).toHaveBeenCalledWith("q", { topK: 3 });
	});

	it("tries fallbacks in order and stops at the first that answers", async () => {
		const first = method("ds-bm25", new Error("fts offline"), "bm25");
		const second = method("ds-cli", [hit("c")], "bm25");
		const third = method("ds-extra", [hit("d")], "bm25");
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", new Error("no embedder")), [first, second, third]);

		const results = await chain.retrieve("q", {});

		expect(results.map((r) => r.id)).toEqual(["c"]);
		expect(third.retrieve).not.toHaveBeenCalled();
	});

	it("rethrows the primary's error verbatim when every link fails", async () => {
		const primaryError = new Error("primary exploded");
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", primaryError), [
			method("ds-bm25", new Error("fallback exploded too"), "bm25"),
		]);

		await expect(chain.retrieve("q", {})).rejects.toBe(primaryError);
	});

	it("does not fall back after the caller aborted", async () => {
		const controller = new AbortController();
		controller.abort();
		const abortError = new Error("aborted");
		const fallback = method("ds-bm25", [hit("b")], "bm25");
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", abortError), [fallback]);

		await expect(chain.retrieve("q", { signal: controller.signal })).rejects.toBe(abortError);
		expect(fallback.retrieve).not.toHaveBeenCalled();
	});

	it("does not mask a required-MinSync failure with a fallback", async () => {
		const required = new MinSyncRequiredError("minsync missing");
		const fallback = method("ds-bm25", [hit("b")], "bm25");
		const chain = new FallbackRetrievalMethod(method("hybrid", required), [fallback]);

		await expect(chain.retrieve("q", {})).rejects.toBe(required);
		expect(fallback.retrieve).not.toHaveBeenCalled();
	});

	it("reports the fallback to the pipeline hook and keeps that hook away from the methods it wraps", async () => {
		const events: RetrievalFallbackEvent[] = [];
		const primary = method("ds-hybrid", new Error("embeddings unavailable"));
		const fallback = method("ds-bm25", [hit("b")], "bm25");
		const chain = new FallbackRetrievalMethod(primary, [fallback]);

		await chain.retrieve("q", { topK: 2, onFallback: (event) => events.push(event) });

		expect(events).toEqual([{ method: "ds-hybrid", fallback: "ds-bm25", reason: "Error: embeddings unavailable" }]);
		expect(primary.retrieve).toHaveBeenCalledWith("q", { topK: 2 });
		expect(fallback.retrieve).toHaveBeenCalledWith("q", { topK: 2 });
	});

	it("reports nothing when the primary answers, even with no hits", async () => {
		const events: RetrievalFallbackEvent[] = [];
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", []), [method("ds-bm25", [hit("b")], "bm25")]);

		await chain.retrieve("q", { onFallback: (event) => events.push(event) });

		expect(events).toEqual([]);
	});
});

describe("hybrid-first retrieval through the pipeline", () => {
	it("one hybrid chain gives the merger a single method vote, so a fallback never inflates corroboration", async () => {
		const retriever = new ParallelRetriever();
		const merger = new ResultMerger();
		const independent = method("independent", [hit("independent"), hit("floor", 0)], "bm25");
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", new Error("no embedder")), [
			method("ds-bm25", [hit("lexical-only")], "bm25"),
		]);

		const { results } = await retriever.retrieveWithDiagnostics([independent, chain], "q", {});
		const merged = merger.merge(results, { topK: 3, dedup: true });

		expect([...results.keys()]).toEqual(["independent", "ds-hybrid"]);
		// Same fixture as issue #1990: one vote each, so the tie keeps first-seen order.
		expect(merged[0]?.id).toBe("independent");
		expect(merged.find((r) => r.id === "lexical-only")?.metadata.retrievalMethods).toEqual(["ds-hybrid"]);
	});

	it("still answers from the fallback but reports the degraded surface with the primary's real error", async () => {
		const retriever = new ParallelRetriever();
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", new Error("semantic index is stale at /tmp/idx")), [
			method("ds-bm25", [hit("lexical-only")], "bm25"),
		]);

		const { results, diagnostics, unsearched } = await retriever.retrieveWithDiagnostics([chain], "q", {});

		expect(results.get("ds-hybrid")?.map((r) => r.id)).toEqual(["lexical-only"]);
		expect(unsearched).toEqual([
			{ surface: "ds", methods: ["ds-hybrid"], reason: "Error: semantic index is stale at /tmp/idx" },
		]);
		expect(diagnostics).toEqual([
			expect.objectContaining({
				code: "retrieval-method-failed",
				severity: "warning",
				source: "ds-hybrid",
				reason: "Error: semantic index is stale at /tmp/idx",
				message: expect.stringContaining('answered by its fallback "ds-bm25"'),
			}),
		]);
	});

	it("reports nothing when the hybrid method answers", async () => {
		const retriever = new ParallelRetriever();
		const chain = new FallbackRetrievalMethod(method("ds-hybrid", [hit("a")]), [
			method("ds-bm25", [hit("b")], "bm25"),
		]);

		const { diagnostics, unsearched } = await retriever.retrieveWithDiagnostics([chain], "q", {});

		expect(diagnostics).toEqual([]);
		expect(unsearched).toEqual([]);
	});
});
