import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	embeddingEndpointSource,
	lazykatokEmbeddingEnv,
	mailcrawlEmbeddingEnv,
	type ResolvedEmbeddingEndpoint,
} from "../../src/datasource/embedding-env.ts";
import { LazykatokClient } from "../../src/datasource/skills/lazykatok/index.ts";
import { MailcrawlClient } from "../../src/datasource/skills/mailcrawl/index.ts";
import type { ManagedEmbedder } from "../../src/embedding/embedder.ts";

const daemon: ResolvedEmbeddingEndpoint = {
	baseUrl: "http://127.0.0.1:18080",
	protocol: "openai",
	model: "Qwen/Qwen3-Embedding-0.6B",
	dimension: 1024,
	backend: "daemon",
};

describe("mailcrawlEmbeddingEnv", () => {
	it("maps a daemon to mailcrawl's openai provider without opening remote egress", () => {
		expect(mailcrawlEmbeddingEnv(daemon, {})).toEqual({
			MAILCRAWL_EMBEDDER_PROVIDER: "openai",
			MAILCRAWL_EMBED_URL: "http://127.0.0.1:18080",
			MAILCRAWL_EMBED_MODEL: "Qwen/Qwen3-Embedding-0.6B",
			MAILCRAWL_EMBED_DIM: "1024",
		});
	});

	it("opts a non-loopback remote in and hands over only the named key", () => {
		const env = mailcrawlEmbeddingEnv(
			{ ...daemon, baseUrl: "https://api.example.com", protocol: "tei", backend: "remote", apiKeyEnv: "EMB_KEY" },
			{ EMB_KEY: "secret", UNRELATED: "x" },
		);
		expect(env).toMatchObject({
			MAILCRAWL_EMBEDDER_PROVIDER: "tei",
			MAILCRAWL_EMBED_ALLOW_REMOTE: "1",
			MAILCRAWL_EMBED_API_KEY_ENV: "EMB_KEY",
			EMB_KEY: "secret",
		});
		expect(env).not.toHaveProperty("UNRELATED");
	});

	it("does not opt a loopback remote into remote egress", () => {
		const env = mailcrawlEmbeddingEnv({ ...daemon, backend: "remote", baseUrl: "http://localhost:9000" }, {});
		expect(env).not.toHaveProperty("MAILCRAWL_EMBED_ALLOW_REMOTE");
	});

	it("fails when the named key variable is empty instead of sending a keyless request", () => {
		expect(() =>
			mailcrawlEmbeddingEnv({ ...daemon, backend: "remote", apiKeyEnv: "EMB_KEY" }, { EMB_KEY: "" }),
		).toThrow(/EMB_KEY is empty or missing/);
	});
});

describe("lazykatokEmbeddingEnv", () => {
	it("maps an OpenAI-compatible endpoint to lazykatok's embedder variables", () => {
		expect(lazykatokEmbeddingEnv(daemon, {})).toEqual({
			KATOK_EMBEDDER: "openai-compatible",
			KATOK_EMBED_URL: "http://127.0.0.1:18080",
			KATOK_EMBED_MODEL: "Qwen/Qwen3-Embedding-0.6B",
			KATOK_EMBED_DIM: "1024",
		});
	});

	it("passes a remote key as the bearer value", () => {
		const env = lazykatokEmbeddingEnv({ ...daemon, backend: "remote", apiKeyEnv: "EMB_KEY" }, { EMB_KEY: "secret" });
		expect(env?.KATOK_EMBED_API_KEY).toBe("secret");
	});

	it("offers nothing for a TEI endpoint, which lazykatok cannot speak", () => {
		expect(lazykatokEmbeddingEnv({ ...daemon, protocol: "tei" }, {})).toBeUndefined();
	});
});

describe("embeddingEndpointSource", () => {
	function fake(endpoint: Awaited<ReturnType<ManagedEmbedder["describeEndpoint"]>>) {
		const calls: string[] = [];
		const embedder: ManagedEmbedder = {
			identity: async () => ({ provider: "minsync", model: "m", dimension: 7 }),
			embed: async () => [],
			health: async () => ({ backend: "daemon", alive: true }),
			describeEndpoint: async () => {
				calls.push("describeEndpoint");
				return endpoint;
			},
			close: async () => undefined,
		};
		return { embedder, calls };
	}

	it("only probes, learning the vector length from the embedder when the endpoint omits it", async () => {
		const { embedder, calls } = fake({ ...daemon, dimension: undefined });
		expect(await embeddingEndpointSource(embedder)()).toMatchObject({ dimension: 7 });
		expect(calls).toEqual(["describeEndpoint"]);
	});

	it("reports no endpoint while the shared embedder would run in-process", async () => {
		const { embedder } = fake(undefined);
		expect(await embeddingEndpointSource(embedder)()).toBeUndefined();
	});
});

describe("CLI children use the shared endpoint", () => {
	let root: string;
	let binaryPath: string;
	let logPath: string;

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-cli-embedding-"));
		mkdirSync(join(root, "bin"), { recursive: true });
		binaryPath = join(root, "bin", "fake-cli");
		logPath = join(root, "calls.jsonl");
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => /^(MAILCRAWL|KATOK|LAZYKATOK)_/.test(k) || k === "EMB_KEY"));
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2), env }) + "\\n");
process.stdout.write(process.argv.includes("index") ? '{"embedded":1,"reused":0}' : "[]");
`,
		);
		chmodSync(binaryPath, 0o755);
	});
	afterEach(() => rmSync(root, { recursive: true, force: true }));

	const calls = (): { args: string[]; env: Record<string, string> }[] =>
		readFileSync(logPath, "utf8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line));

	it("mailcrawl indexes and searches semantically through the endpoint, replacing its native settings", async () => {
		let endpointCalls = 0;
		const client = new MailcrawlClient({
			binaryPath,
			env: { MAILCRAWL_NATIVE_MODEL: "native:other", MAILCRAWL_EMBEDDER_PROVIDER: "native" },
			embeddingEndpoint: async () => {
				endpointCalls += 1;
				return daemon;
			},
		});
		expect(await client.index()).toMatchObject({ ok: true });
		await client.search("semantic", "refund");
		await client.search("bm25", "refund");
		const [index, semantic, bm25] = calls();
		expect(index?.env).toMatchObject({ MAILCRAWL_EMBEDDER_PROVIDER: "openai", MAILCRAWL_EMBED_URL: daemon.baseUrl });
		expect(index?.env).not.toHaveProperty("MAILCRAWL_NATIVE_MODEL");
		expect(semantic?.env.MAILCRAWL_EMBEDDER_PROVIDER).toBe("openai");
		// BM25 never embeds, so it neither asks for an endpoint nor receives one.
		expect(bm25?.env).not.toHaveProperty("MAILCRAWL_EMBED_URL");
		// index and the semantic search ask once each; BM25 never embeds, so it never asks.
		expect(endpointCalls).toBe(2);
	});

	it("mailcrawl refuses to fall back to its native model when no endpoint is available", async () => {
		const client = new MailcrawlClient({ binaryPath, embeddingEndpoint: async () => undefined });
		const result = await client.index();
		expect(result).toMatchObject({ ok: false, reason: "embedding-unavailable" });
		expect(() => readFileSync(logPath, "utf8")).toThrow();
	});

	it("lazykatok runs its OpenAI-compatible embedder against the endpoint for index and semantic search", async () => {
		const client = new LazykatokClient({
			binaryPath,
			env: { KATOK_EMBEDDER: "local" },
			embeddingEndpoint: async () => daemon,
		});
		await client.index();
		await client.search("semantic", "hello");
		await client.search("keyword", "hello");
		const [index, semantic, keyword] = calls();
		expect(index?.env).toMatchObject({ KATOK_EMBEDDER: "openai-compatible", KATOK_EMBED_URL: daemon.baseUrl });
		expect(semantic?.env.KATOK_EMBED_MODEL).toBe(daemon.model);
		expect(keyword?.env).not.toHaveProperty("KATOK_EMBED_URL");
	});

	it("lazykatok reports a TEI endpoint instead of silently using fastembed", async () => {
		const client = new LazykatokClient({
			binaryPath,
			embeddingEndpoint: async () => ({ ...daemon, protocol: "tei" }),
		});
		const result = await client.index();
		expect(result).toMatchObject({ ok: false, reason: "embedding-unavailable" });
		expect(result.ok ? "" : result.stderr).toMatch(/TEI/);
	});

	it("keeps a CLI on its own embedder when no shared endpoint source is given", async () => {
		const client = new LazykatokClient({ binaryPath, env: { KATOK_EMBEDDER: "local" } });
		await client.index();
		expect(calls()[0]?.env).toEqual({ KATOK_EMBEDDER: "local" });
	});
});
