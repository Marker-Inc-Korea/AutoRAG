import { describe, expect, it } from "vitest";
import { EvidenceOriginIndex } from "../../src/agent/evidence-origins.ts";

/** A candidate printed with the given query/method for the same source. */
function candidate(source: string, content: string, query: string, method = "vector") {
	return { source, content, query, method };
}

describe("EvidenceOriginIndex", () => {
	it("resolves a ref whose text exactly matches the remembered content", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://lease", "The lease runs until December 2027.", "lease end date"));

		expect(
			index.resolve({ source: "doc://lease", method: "vector", excerpt: "The lease runs until December 2027." }),
		).toEqual({
			query: "lease end date",
			method: "vector",
			source: "doc://lease",
		});
	});

	it("resolves a ref that contains the remembered content, and the reverse", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://memo", "The budget was approved by the CFO.", "who approved the budget"));

		// Ref text is longer than the stored chunk.
		expect(
			index.resolve({
				source: "doc://memo",
				method: "vector",
				excerpt: "Section 4: The budget was approved by the CFO. See appendix.",
			}),
		).toEqual({ query: "who approved the budget", method: "vector", source: "doc://memo" });

		// Ref text is shorter than the stored chunk (still at least 12 chars).
		const short = new EvidenceOriginIndex();
		short.add(candidate("doc://memo", "Section 4: The budget was approved by the CFO. See appendix.", "approval"));
		expect(short.resolve({ source: "doc://memo", method: "vector", excerpt: "The budget was approved" })).toEqual({
			query: "approval",
			method: "vector",
			source: "doc://memo",
		});
	});

	it("maps two chunks of the same source to their own queries", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://handbook", "Chapter one covers onboarding new hires.", "onboarding process"));
		index.add(candidate("doc://handbook", "Chapter two covers the vacation policy rules.", "vacation policy"));

		expect(
			index.resolve({
				source: "doc://handbook",
				method: "vector",
				excerpt: "Chapter two covers the vacation policy rules.",
			}),
		).toEqual({ query: "vacation policy", method: "vector", source: "doc://handbook" });
		expect(
			index.resolve({
				source: "doc://handbook",
				method: "vector",
				content: "Chapter one covers onboarding new hires.",
			}),
		).toEqual({ query: "onboarding process", method: "vector", source: "doc://handbook" });
	});

	it("picks the chunk sharing the longest prefix when neither contains the ref text", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://story", "alpha beta gamma delta epsilon", "first query"));
		index.add(candidate("doc://story", "alpha beta gamma zeta eta theta", "second query"));

		expect(
			index.resolve({ source: "doc://story", method: "vector", excerpt: "alpha beta gamma zeta something" }),
		).toEqual({
			query: "second query",
			method: "vector",
			source: "doc://story",
		});
	});

	it("falls back to the only candidate when the source is unique but the text differs", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://single", "Completely unrelated remembered text.", "the only query"));

		expect(
			index.resolve({ source: "doc://single", method: "vector", excerpt: "a ref text that matches nothing at all" }),
		).toEqual({ query: "the only query", method: "vector", source: "doc://single" });
	});

	it("returns undefined for a source that was never remembered", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://a", "Some remembered evidence text.", "query a"));

		expect(
			index.resolve({ source: "doc://b", method: "vector", excerpt: "Some remembered evidence text." }),
		).toBeUndefined();
	});

	it("keeps the first write for an identical (source, normalized content) pair", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://dup", "Repeated evidence content.", "first query"));
		index.add(candidate("doc://dup", "Repeated evidence content.", "second query", "keyword"));

		expect(index.resolve({ source: "doc://dup", method: "keyword", excerpt: "Repeated evidence content." })).toEqual({
			query: "first query",
			method: "vector",
			source: "doc://dup",
		});
	});

	it("drops the oldest candidate once the bound is exceeded", () => {
		const index = new EvidenceOriginIndex();
		for (let i = 0; i < 2100; i += 1) {
			index.add(candidate(`doc://${i}`, `Chunk number ${i} with enough text to match.`, `query ${i}`));
		}

		expect(
			index.resolve({ source: "doc://0", method: "vector", excerpt: "Chunk number 0 with enough text to match." }),
		).toBeUndefined();
		expect(
			index.resolve({
				source: "doc://2099",
				method: "vector",
				excerpt: "Chunk number 2099 with enough text to match.",
			}),
		).toEqual({ query: "query 2099", method: "vector", source: "doc://2099" });
	});

	it("forgets everything on clear", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://a", "Some remembered evidence text.", "query a"));
		index.clear();

		expect(
			index.resolve({ source: "doc://a", method: "vector", excerpt: "Some remembered evidence text." }),
		).toBeUndefined();
	});

	it("matches ignoring whitespace, case, and Unicode normalization form", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://cafe", "Caf\u00e9\tAU   lait", "coffee"));

		expect(index.resolve({ source: "doc://cafe", method: "vector", excerpt: "cafe\u0301 au lait" })).toEqual({
			query: "coffee",
			method: "vector",
			source: "doc://cafe",
		});
	});

	it("matches a source-less ref against candidates of any source by containment", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://x", "the secret handshake is a firm grip", "handshake"));

		expect(
			index.resolve({
				source: "",
				method: "all",
				excerpt: "Remember: the secret handshake is a firm grip always.",
			}),
		).toEqual({ query: "handshake", method: "vector", source: "doc://x" });
	});

	it("matches a source-less ref line of a joined multi-line excerpt bundle", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://a", "about the budget approval process", "bud"));
		index.add(candidate("doc://b", "about vacation policy rules", "vac"));

		expect(
			index.resolve({
				source: "",
				method: "all",
				content: "First excerpt about the budget approval process.\nSecond excerpt about vacation policy rules.",
			}),
		).toEqual({ query: "bud", method: "vector", source: "doc://a" });
	});

	it("does not match a source-less ref when both texts are shorter than the minimum", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://s", "abcdefghij", "short"));

		expect(index.resolve({ source: "", method: "all", excerpt: "abcdefghij" })).toBeUndefined();
	});

	it("returns empty-content candidates by same-source fallback but never by text", () => {
		const index = new EvidenceOriginIndex();
		index.add(candidate("doc://empty", "", "empty query"));

		expect(
			index.resolve({ source: "doc://empty", method: "vector", excerpt: "any unrelated ref text at all" }),
		).toEqual({
			query: "empty query",
			method: "vector",
			source: "doc://empty",
		});
		expect(index.resolve({ source: "", method: "all", excerpt: "any unrelated ref text at all" })).toBeUndefined();
	});
});
