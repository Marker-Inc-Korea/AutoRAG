import { describe, expect, it } from "vitest";
import { QWEN3_QUERY_PREFIX, resolveEmbeddingPrefixes } from "../../src/embedding/prefixes.ts";

describe("QWEN3_QUERY_PREFIX", () => {
	it("is the model card's get_detailed_instruct header, with no space after Query:", () => {
		expect(QWEN3_QUERY_PREFIX).toBe(
			"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
		);
	});
});

describe("resolveEmbeddingPrefixes", () => {
	it("gives MinSync's default local Qwen3 the query instruction and raw passages", () => {
		const expected = { query: QWEN3_QUERY_PREFIX, passage: "" };
		expect(resolveEmbeddingPrefixes(undefined)).toEqual(expected);
		expect(resolveEmbeddingPrefixes({})).toEqual(expected);
		expect(resolveEmbeddingPrefixes({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 })).toEqual(expected);
	});

	it("lets an operator-set queryPrefix win, including the empty string", () => {
		expect(resolveEmbeddingPrefixes({ id: "native:Qwen/Qwen3-Embedding-0.6B", queryPrefix: "" }).query).toBe("");
		expect(resolveEmbeddingPrefixes({ queryPrefix: "Instruct: mine\nQuery:" }).query).toBe("Instruct: mine\nQuery:");
	});

	it("lets an operator-set passagePrefix win and leaves the query default alone", () => {
		expect(resolveEmbeddingPrefixes({ passagePrefix: "passage: " })).toEqual({
			query: QWEN3_QUERY_PREFIX,
			passage: "passage: ",
		});
	});

	it("adds no default to another native model or to any remote endpoint", () => {
		const none = { query: "", passage: "" };
		expect(resolveEmbeddingPrefixes({ id: "native:BAAI/bge-small-en-v1.5" })).toEqual(none);
		expect(resolveEmbeddingPrefixes({ id: "tei:Qwen/Qwen3-Embedding-0.6B", baseUrl: "http://gpu:8080" })).toEqual(
			none,
		);
		expect(resolveEmbeddingPrefixes({ id: "openai:text-embedding-3-small" })).toEqual(none);
		expect(resolveEmbeddingPrefixes({ baseUrl: "http://127.0.0.1:8089/v1" })).toEqual(none);
	});

	it("passes an operator prefix through to a remote endpoint", () => {
		expect(
			resolveEmbeddingPrefixes({
				id: "tei:m",
				baseUrl: "http://gpu:8080",
				queryPrefix: "q: ",
				passagePrefix: "p: ",
			}),
		).toEqual({
			query: "q: ",
			passage: "p: ",
		});
	});
});
