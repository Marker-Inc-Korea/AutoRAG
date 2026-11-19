import { describe, expect, it } from "vitest";
import { unionChannelResults } from "../../src/minsync/channel-union.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

function hit(id: string, source = `/docs/${id}.md`, score = 1): RetrievalResult {
	return { id, source, content: `content of ${id}`, score, metadata: { method: "minsync" } };
}

describe("unionChannelResults", () => {
	it("interleaves the channels by rank so neither channel's head is buried", () => {
		const union = unionChannelResults([
			{ channel: "vector", results: [hit("v1"), hit("v2"), hit("v3")] },
			{ channel: "sparse", results: [hit("s1"), hit("s2"), hit("s3")] },
		]);

		expect(union.map((result) => result.id)).toEqual(["v1", "s1", "v2", "s2", "v3", "s3"]);
	});

	it("keeps one copy of a chunk both channels found and records where each channel ranked it", () => {
		const union = unionChannelResults([
			{ channel: "vector", results: [hit("a"), hit("shared"), hit("b")] },
			{ channel: "sparse", results: [hit("shared"), hit("c")] },
		]);

		expect(union.map((result) => result.id)).toEqual(["a", "shared", "c", "b"]);
		const shared = union.find((result) => result.id === "shared");
		expect(shared?.metadata.minsyncChannels).toEqual(["sparse", "vector"]);
		expect(shared?.metadata.minsyncChannelRanks).toEqual({ vector: 2, sparse: 1 });
		const onlyVector = union.find((result) => result.id === "a");
		expect(onlyVector?.metadata.minsyncChannels).toEqual(["vector"]);
		expect(onlyVector?.metadata.minsyncChannelRanks).toEqual({ vector: 1 });
	});

	it("treats two chunks of one document as distinct and the same id under another source as distinct", () => {
		const union = unionChannelResults([
			{ channel: "vector", results: [hit("c1", "/docs/a.md"), hit("c2", "/docs/a.md")] },
			{ channel: "sparse", results: [hit("c1", "/docs/b.md"), hit("c1", "/docs/a.md")] },
		]);

		expect(union.map((result) => `${result.source}#${result.id}`)).toEqual([
			"/docs/a.md#c1",
			"/docs/b.md#c1",
			"/docs/a.md#c2",
		]);
	});

	it("returns the other channel unchanged when one channel found nothing", () => {
		const union = unionChannelResults([
			{ channel: "vector", results: [] },
			{ channel: "sparse", results: [hit("s1"), hit("s2")] },
		]);

		expect(union.map((result) => result.id)).toEqual(["s1", "s2"]);
		expect(union[0]?.metadata.minsyncChannels).toEqual(["sparse"]);
	});

	it("cuts the union at the limit after interleaving, not per channel", () => {
		const union = unionChannelResults(
			[
				{ channel: "vector", results: [hit("v1"), hit("v2"), hit("v3")] },
				{ channel: "sparse", results: [hit("s1"), hit("s2"), hit("s3")] },
			],
			{ limit: 4 },
		);

		expect(union.map((result) => result.id)).toEqual(["v1", "s1", "v2", "s2"]);
	});

	it("does not mutate the channel results it was given", () => {
		const original = hit("v1");
		unionChannelResults([{ channel: "vector", results: [original] }]);

		expect(original.metadata).toEqual({ method: "minsync" });
	});
});
