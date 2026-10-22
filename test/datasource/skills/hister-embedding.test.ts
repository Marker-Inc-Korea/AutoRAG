import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import type { ResolvedEmbeddingEndpoint } from "../../../src/datasource/embedding-env.ts";
import {
	applyHisterSemanticSettings,
	deriveHisterSemanticSettings,
	differingSemanticKeys,
	HISTER_SEMANTIC_DEFAULTS,
	type HisterSemanticSettings,
	readHisterSemanticSettings,
	reindexRequired,
} from "../../../src/datasource/skills/hister/embedding.ts";
import { QWEN3_QUERY_PREFIX } from "../../../src/embedding/prefixes.ts";

function daemon(overrides: Partial<ResolvedEmbeddingEndpoint> = {}): ResolvedEmbeddingEndpoint {
	return {
		baseUrl: "http://127.0.0.1:64917",
		protocol: "openai",
		model: "Qwen/Qwen3-Embedding-0.6B",
		dimension: 1024,
		queryPrefix: QWEN3_QUERY_PREFIX,
		passagePrefix: "",
		backend: "daemon",
		...overrides,
	};
}

const SETTINGS: HisterSemanticSettings = {
	enable: true,
	embedding_endpoint: "http://127.0.0.1:64917/v1/embeddings",
	embedding_model: "Qwen/Qwen3-Embedding-0.6B",
	dimensions: 1024,
	query_prefix: QWEN3_QUERY_PREFIX,
	document_prefix: "",
};

describe("deriveHisterSemanticSettings", () => {
	it("maps the shared daemon endpoint onto Hister's semantic_search keys, byte for byte", () => {
		const result = deriveHisterSemanticSettings(daemon());
		expect(result).toEqual({ ok: true, settings: SETTINGS });
	});

	it("builds Hister's full /v1/embeddings URL from a server root, with or without /v1 or a slash", () => {
		for (const baseUrl of [
			"http://127.0.0.1:1234",
			"http://127.0.0.1:1234/",
			"http://127.0.0.1:1234/v1",
			"http://127.0.0.1:1234/v1/",
		]) {
			const result = deriveHisterSemanticSettings(daemon({ baseUrl }));
			expect(result.ok && result.settings.embedding_endpoint).toBe("http://127.0.0.1:1234/v1/embeddings");
		}
	});

	it("keeps an empty prefix empty, not Hister's own `query: ` default", () => {
		const result = deriveHisterSemanticSettings(daemon({ queryPrefix: undefined, passagePrefix: undefined }));
		expect(result.ok && result.settings.query_prefix).toBe("");
		expect(result.ok && result.settings.document_prefix).toBe("");
	});

	it.each(["localhost", "[::1]", "127.0.0.1"])("accepts the loopback host %s", (host) => {
		expect(deriveHisterSemanticSettings(daemon({ baseUrl: `http://${host}:9` })).ok).toBe(true);
	});

	it("refuses a TEI endpoint: Hister only speaks the OpenAI embeddings route", () => {
		const result = deriveHisterSemanticSettings(daemon({ protocol: "tei" }));
		expect(result.ok).toBe(false);
		expect(!result.ok && result.reason).toMatch(/OpenAI/);
	});

	it("refuses a remote endpoint: browsing text would leave the machine and the key cannot live in Hister's config", () => {
		const result = deriveHisterSemanticSettings(
			daemon({ baseUrl: "https://embeddings.example.com", backend: "remote", apiKeyEnv: "EMBED_KEY" }),
		);
		expect(result.ok).toBe(false);
		expect(!result.ok && result.reason).toMatch(/leave this machine/);
	});
});

describe("readHisterSemanticSettings", () => {
	it("falls back to Hister's own defaults for keys the file does not set", () => {
		const result = readHisterSemanticSettings("app:\n    title: Hister\n");
		expect(result).toEqual({ ok: true, settings: HISTER_SEMANTIC_DEFAULTS, explicit: [] });
	});

	it("reads the same values a real YAML parser does, across scalar styles", () => {
		const text = [
			"app:",
			"    enable: false",
			"semantic_search:",
			"    enable: true",
			"    embedding_endpoint: http://127.0.0.1:64917/v1/embeddings # comment",
			"    embedding_model: 'it''s a model'",
			"    dimensions: 1024",
			'    query_prefix: "Instruct: task\\nQuery:"',
			"    document_prefix: |-",
			"        line one",
			"        line two",
			"extractors: {}",
			"",
		].join("\n");
		const expected = parse(text).semantic_search;
		const result = readHisterSemanticSettings(text);
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		expect(result.settings).toEqual({
			enable: expected.enable,
			embedding_endpoint: expected.embedding_endpoint,
			embedding_model: expected.embedding_model,
			dimensions: expected.dimensions,
			query_prefix: expected.query_prefix,
			document_prefix: expected.document_prefix,
		});
	});

	it("reads a single-quoted value that spans two lines the way YAML folds it", () => {
		const text = "semantic_search:\n    query_prefix: 'Instruct: task\n      Query: '\n";
		const folded = parse(text).semantic_search.query_prefix;
		expect(folded).toBe("Instruct: task Query: ");
		const result = readHisterSemanticSettings(text);
		expect(result.ok && result.settings.query_prefix).toBe(folded);
	});

	it("reads the lenient continuation Hister's own parser accepts (observed: `config show` folds both to `a b Query: `)", () => {
		// A strict YAML parser rejects these; Hister (Go) folds them, and a hand-edited real config had this shape.
		for (const continuation of ["    Query: '", "Query: '"]) {
			const result = readHisterSemanticSettings(`semantic_search:\n    query_prefix: 'a b\n${continuation}\n`);
			expect(result.ok && result.settings.query_prefix).toBe("a b Query: ");
		}
	});

	it("does not mistake same-named keys in other blocks for semantic_search's", () => {
		const text = "extractors:\n    enable: true\n    dimensions: 7\nsemantic_search:\n    enable: false\n";
		const result = readHisterSemanticSettings(text);
		expect(result.ok && result.settings.enable).toBe(false);
		expect(result.ok && result.settings.dimensions).toBe(HISTER_SEMANTIC_DEFAULTS.dimensions);
	});

	it("reports a flow-style block it cannot edit safely, instead of guessing", () => {
		const result = readHisterSemanticSettings("semantic_search: {enable: true, dimensions: 8}\n");
		expect(result.ok).toBe(false);
		expect(!result.ok && result.reason).toMatch(/flow style/);
	});
});

describe("differingSemanticKeys / reindexRequired", () => {
	it("lists exactly the keys that differ", () => {
		expect(differingSemanticKeys(SETTINGS, SETTINGS)).toEqual([]);
		expect(differingSemanticKeys(SETTINGS, { ...SETTINGS, query_prefix: "x", dimensions: 8 })).toEqual([
			"dimensions",
			"query_prefix",
		]);
	});

	it("requires a reindex for what Hister fingerprints, and not for the query prefix", () => {
		expect(reindexRequired(["query_prefix"])).toBe(false);
		for (const key of ["embedding_endpoint", "embedding_model", "dimensions", "document_prefix", "enable"] as const) {
			expect(reindexRequired([key])).toBe(true);
		}
	});
});

describe("applyHisterSemanticSettings", () => {
	const BEFORE = [
		"app:",
		"    title: Hister",
		"    log_level: info",
		"semantic_search:",
		"    enable: false",
		"    embedding_endpoint: http://localhost:11434/v1/embeddings",
		"    embedding_model: qwen3-embedding:8b",
		"    dimensions: 2000",
		"    query_prefix: 'Instruct: Given a web search query, retrieve relevant passages that answer the query",
		"Query: '",
		"    similarity_threshold: 0.1",
		"hotkeys:",
		"    web:",
		"        /: focus_search_input",
		"",
	].join("\n");

	it("sets every key, parses to the intended values, and leaves unrelated lines byte for byte", () => {
		const result = applyHisterSemanticSettings(BEFORE, SETTINGS);
		expect(result.ok).toBe(true);
		if (!result.ok) return;
		const parsed = parse(result.text);
		expect(parsed.semantic_search).toMatchObject(SETTINGS);
		expect(parsed.semantic_search.similarity_threshold).toBe(0.1);
		expect(parsed.app).toEqual({ title: "Hister", log_level: "info" });
		expect(parsed.hotkeys.web["/"]).toBe("focus_search_input");
		const kept = (text: string) =>
			text
				.split("\n")
				.filter((line) => /^(app:| {4}title| {4}log_level|hotkeys| {4}web| {8}\/| {4}similarity)/.test(line));
		expect(kept(result.text)).toEqual(kept(BEFORE));
	});

	it("replaces a multi-line value completely instead of leaving its continuation behind", () => {
		const result = applyHisterSemanticSettings(BEFORE, SETTINGS);
		expect(result.ok && result.text).not.toContain("Query: '");
		const readBack = result.ok ? readHisterSemanticSettings(result.text) : undefined;
		expect(readBack?.ok && readBack.settings.query_prefix).toBe(QWEN3_QUERY_PREFIX);
	});

	it("is idempotent: a second apply changes nothing", () => {
		const once = applyHisterSemanticSettings(BEFORE, SETTINGS);
		expect(once.ok).toBe(true);
		if (!once.ok) return;
		const twice = applyHisterSemanticSettings(once.text, SETTINGS);
		expect(twice).toEqual({ ok: true, text: once.text });
	});

	it("adds missing keys inside the block and creates the block when absent", () => {
		const partial = applyHisterSemanticSettings("semantic_search:\n    enable: true\napp:\n    title: x\n", SETTINGS);
		expect(partial.ok && parse(partial.text).semantic_search).toMatchObject(SETTINGS);
		expect(partial.ok && parse(partial.text).app).toEqual({ title: "x" });
		const absent = applyHisterSemanticSettings("app:\n    title: x\n", SETTINGS);
		expect(absent.ok && parse(absent.text).semantic_search).toMatchObject(SETTINGS);
		expect(absent.ok && parse(absent.text).app).toEqual({ title: "x" });
	});

	it("expands an empty flow block, and refuses a non-empty one", () => {
		const empty = applyHisterSemanticSettings("semantic_search: {}\napp:\n    title: x\n", SETTINGS);
		expect(empty.ok && parse(empty.text).semantic_search).toMatchObject(SETTINGS);
		const full = applyHisterSemanticSettings("semantic_search: {enable: true}\n", SETTINGS);
		expect(full.ok).toBe(false);
	});

	it("keeps CRLF line endings", () => {
		const result = applyHisterSemanticSettings(BEFORE.replace(/\n/g, "\r\n"), SETTINGS);
		expect(result.ok && result.text.split("\r\n").length).toBeGreaterThan(5);
		expect(result.ok && result.text.replace(/\r\n/g, "")).not.toContain("\n");
		expect(result.ok && parse(result.text).semantic_search).toMatchObject(SETTINGS);
	});
});
