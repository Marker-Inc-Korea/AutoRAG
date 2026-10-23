import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DiscrawlClient } from "../../src/datasource/skills/discrawl/client.ts";
import {
	discrawlEmbeddingRuntimeFor,
	LOCAL_EMBEDDING_KEY_ENV,
	LOCAL_EMBEDDING_KEY_PLACEHOLDER,
} from "../../src/datasource/skills/discrawl/embedding.ts";
import type { EmbedderEndpoint } from "../../src/embedding/embedder.ts";

const daemon: EmbedderEndpoint = {
	baseUrl: "http://127.0.0.1:52982",
	protocol: "openai",
	model: "Qwen/Qwen3-Embedding-0.6B",
	dimension: 1024,
	backend: "daemon",
};

describe("discrawlEmbeddingRuntimeFor", () => {
	it("points discrawl at the daemon's OpenAI route with the underscore provider name discrawl requires", () => {
		expect(discrawlEmbeddingRuntimeFor(daemon)).toMatchObject({
			provider: "openai_compatible",
			model: "Qwen/Qwen3-Embedding-0.6B",
			baseUrl: "http://127.0.0.1:52982/v1",
			dimensions: 1024,
		});
	});

	it("names a placeholder key variable for the keyless daemon, because discrawl refuses an empty one", () => {
		// discrawl leaves `openai_compatible` embeddings off ("requires API key env ...") unless its key
		// variable is non-empty, and `minsync serve` ignores the value. AutoRAG supplies a fixed
		// non-secret placeholder under a variable name of its own, so no real credential is involved.
		const runtime = discrawlEmbeddingRuntimeFor(daemon);
		expect(runtime?.apiKeyEnv).toBe(LOCAL_EMBEDDING_KEY_ENV);
		expect(runtime?.apiKeyValue).toBe(LOCAL_EMBEDDING_KEY_PLACEHOLDER);
	});

	it("never invents a key value for a remote endpoint: the operator's variable carries it", () => {
		const runtime = discrawlEmbeddingRuntimeFor({
			baseUrl: "https://api.openai.com",
			protocol: "openai",
			model: "text-embedding-3-small",
			dimension: 1536,
			apiKeyEnv: "MY_EMBED_KEY",
			backend: "remote",
		});
		expect(runtime?.apiKeyValue).toBeUndefined();
	});

	it("passes a remote OpenAI-compatible endpoint and the NAME of its key variable, never the key", () => {
		const runtime = discrawlEmbeddingRuntimeFor({
			baseUrl: "https://api.openai.com",
			protocol: "openai",
			model: "text-embedding-3-small",
			dimension: 1536,
			apiKeyEnv: "MY_EMBED_KEY",
			backend: "remote",
		});
		expect(runtime).toMatchObject({
			provider: "openai_compatible",
			baseUrl: "https://api.openai.com/v1",
			dimensions: 1536,
			apiKeyEnv: "MY_EMBED_KEY",
		});
		expect(JSON.stringify(runtime)).not.toMatch(/sk-/);
	});

	it("does not double a /v1 the operator already wrote", () => {
		expect(discrawlEmbeddingRuntimeFor({ ...daemon, baseUrl: "http://gateway:8089/v1/" })?.baseUrl).toBe(
			"http://gateway:8089/v1",
		);
	});

	it("offers nothing for a TEI endpoint because discrawl cannot speak TEI", () => {
		expect(discrawlEmbeddingRuntimeFor({ ...daemon, protocol: "tei" })).toBeUndefined();
	});

	it("offers nothing without an endpoint, so discrawl keeps its own embedding config", () => {
		expect(discrawlEmbeddingRuntimeFor(undefined)).toBeUndefined();
	});

	it("omits the dimension when it is unknown instead of sending a guess", () => {
		expect(discrawlEmbeddingRuntimeFor({ ...daemon, dimension: undefined })).not.toHaveProperty("dimensions");
	});
});

describe("managed discrawl config file", () => {
	const roots: string[] = [];
	afterEach(() => {
		for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
	});

	it("hands the child the placeholder for a daemon and nothing else from the parent's secrets", async () => {
		const { DiscrawlClient } = await import("../../src/datasource/skills/discrawl/client.ts");
		const root = mkdtempSync(join(tmpdir(), "discrawl-env-"));
		roots.push(root);
		const bin = join(root, "discrawl");
		writeFileSync(bin, `#!/bin/sh\nprintf '%s|%s' "$AUTORAG_LOCAL_EMBEDDING_KEY" "$SOME_OTHER_SECRET"\n`);
		chmodSync(bin, 0o755);
		process.env.SOME_OTHER_SECRET = "leak";
		try {
			const client = new DiscrawlClient({
				root,
				binaryPath: bin,
				embeddingRuntime: {
					provider: "openai_compatible",
					model: "Qwen/Qwen3-Embedding-0.6B",
					baseUrl: "http://127.0.0.1:1/v1",
					dimensions: 1024,
					apiKeyEnv: LOCAL_EMBEDDING_KEY_ENV,
					apiKeyValue: LOCAL_EMBEDDING_KEY_PLACEHOLDER,
				},
			});
			await client.prepareEmbeddings();
			const result = await client.embed();
			expect(result.ok === false ? result.stdout : "").toBe(`${LOCAL_EMBEDDING_KEY_PLACEHOLDER}|`);
		} finally {
			delete process.env.SOME_OTHER_SECRET;
		}
	});

	it("writes api_key_env into the managed config and never an api key", async () => {
		const { DiscrawlClient } = await import("../../src/datasource/skills/discrawl/client.ts");
		const root = mkdtempSync(join(tmpdir(), "discrawl-remote-"));
		roots.push(root);
		const client = new DiscrawlClient({ root });
		await client.configureEmbeddings({
			provider: "openai_compatible",
			model: "text-embedding-3-small",
			baseUrl: "https://api.openai.com/v1",
			dimensions: 1536,
			apiKeyEnv: "MY_EMBED_KEY",
		});
		const written = readFileSync(join(root, ".autorag", "datasources", "discrawl", "config.toml"), "utf8");
		expect(written).toContain('api_key_env = "MY_EMBED_KEY"');
		expect(written).toContain('base_url = "https://api.openai.com/v1"');
		expect(written).not.toMatch(/sk-/);
	});

	it("leaves a config the operator wrote untouched even when a daemon is available", async () => {
		const { DiscrawlClient } = await import("../../src/datasource/skills/discrawl/client.ts");
		const root = mkdtempSync(join(tmpdir(), "discrawl-operator-"));
		roots.push(root);
		const configPath = join(root, "operator.toml");
		const original = '[search.embeddings]\nprovider = "ollama"\nmodel = "bge-m3"\n';
		writeFileSync(configPath, original);
		const client = new DiscrawlClient({ root, configPath });
		const result = await client.configureEmbeddings({
			provider: "openai_compatible",
			model: "Qwen/Qwen3-Embedding-0.6B",
			baseUrl: "http://127.0.0.1:52982/v1",
			dimensions: 1024,
		});
		expect(result.configured).toBe(false);
		expect(readFileSync(configPath, "utf8")).toBe(original);
	});

	it("resolves the daemon runtime and writes the managed config on search() even without prepareEmbeddings()", async () => {
		const root = mkdtempSync(join(tmpdir(), "discrawl-search-runtime-"));
		roots.push(root);
		const bin = join(root, "discrawl");
		writeFileSync(bin, `#!/bin/sh\nprintf '%s|%s\\n' "$*" "$AUTORAG_LOCAL_EMBEDDING_KEY" >&2\nprintf '[]'\n`);
		chmodSync(bin, 0o755);
		const client = new DiscrawlClient({
			root,
			binaryPath: bin,
			embeddingRuntime: async () => discrawlEmbeddingRuntimeFor(daemon),
		});
		const result = await client.search("semantic", "회의");
		expect(result.ok).toBe(true);
		const configPath = join(root, ".autorag", "datasources", "discrawl", "config.toml");
		expect(readFileSync(configPath, "utf8")).toContain('base_url = "http://127.0.0.1:52982/v1"');
		expect(result.stderr).toContain(`--config ${configPath}`);
		expect(result.stderr).toContain(`|${LOCAL_EMBEDDING_KEY_PLACEHOLDER}`);
	});
});
