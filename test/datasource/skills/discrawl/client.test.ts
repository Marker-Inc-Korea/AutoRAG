import { chmodSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { DiscrawlClient, nativeGatewayBaseUrl } from "../../../../src/datasource/skills/discrawl/client.ts";
import type { DiscrawlOptions } from "../../../../src/datasource/skills/discrawl/types.ts";

function stubBinary(script: string): string {
	const dir = mkdtempSync(join(tmpdir(), "discrawl-stub-"));
	const path = join(dir, "discrawl");
	writeFileSync(path, `#!/bin/sh\n${script}\n`, "utf8");
	chmodSync(path, 0o755);
	return path;
}

const SEARCH_ROWS = JSON.stringify([
	{
		id: "1513467741523415161",
		content: "공금 장부 잔액",
		score: 0.87,
		channel_name: "general",
		guild_id: "1512354544628011068",
		author_name: "리누스형",
	},
]);

describe("DiscrawlClient search", () => {
	it("does not create or inject an AutoRAG-managed workspace config", async () => {
		const root = mkdtempSync(join(tmpdir(), "discrawl-root-"));
		const binaryPath = stubBinary(`echo "$@" >&2; echo '[]'`);
		const result = await new DiscrawlClient({ binaryPath, root }).search("hybrid", "q");
		const configPath = join(root, ".autorag", "datasources", "discrawl", "config.toml");

		expect(result.ok).toBe(true);
		expect(existsSync(configPath)).toBe(false);
		expect(result.stderr).not.toContain("--config");
	});

	it("does not mutate an explicitly supplied config path", async () => {
		const root = mkdtempSync(join(tmpdir(), "discrawl-root-"));
		const configPath = join(root, "operator-config.toml");
		const original = 'guild_id = "guild-1"\n';
		writeFileSync(configPath, original);
		const binaryPath = stubBinary(`echo "$@" >&2; echo '[]'`);
		const result = await new DiscrawlClient({
			binaryPath,
			configPath,
			root,
			embeddingModel: "custom-model",
		}).search("hybrid", "q");

		expect(result.ok).toBe(true);
		expect(readFileSync(configPath, "utf8")).toBe(original);
		expect(result.stderr).toContain(`--config ${configPath}`);
	});

	it("parses a bare JSON array of hits", async () => {
		const binaryPath = stubBinary(`cat <<'EOF'\n${SEARCH_ROWS}\nEOF`);
		const result = await new DiscrawlClient({ binaryPath }).search("hybrid", "공금");
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.hits).toHaveLength(1);
			expect(result.hits[0]).toMatchObject({
				messageId: "1513467741523415161",
				channelName: "general",
				guildId: "1512354544628011068",
				authorName: "리누스형",
			});
		}
	});

	it("parses hits wrapped under a results key", async () => {
		const binaryPath = stubBinary(`echo '{"results": ${SEARCH_ROWS}}'`);
		const result = await new DiscrawlClient({ binaryPath }).search("fts", "공금");
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.hits).toHaveLength(1);
	});

	it("treats empty stdout as zero hits rather than an error", async () => {
		const result = await new DiscrawlClient({ binaryPath: stubBinary("true") }).search("fts", "q");
		expect(result).toMatchObject({ ok: true });
		if (result.ok) expect(result.hits).toEqual([]);
	});

	it("passes --json before the subcommand and maps topK to --limit", async () => {
		const binaryPath = stubBinary(`echo "$@" >&2; echo '[]'`);
		const result = await new DiscrawlClient({ binaryPath }).search("semantic", "q", { topK: 7 });
		expect(result.ok).toBe(true);
		expect(result.stderr).toMatch(/^--json search --mode semantic --limit 7 q/);
	});

	it("reports invalid JSON as invalid-shape", async () => {
		const binaryPath = stubBinary("echo 'not json'");
		expect(await new DiscrawlClient({ binaryPath }).search("fts", "q")).toMatchObject({
			ok: false,
			reason: "invalid-shape",
		});
	});

	it("maps a nonzero exit to nonzero-exit", async () => {
		const binaryPath = stubBinary("exit 3");
		expect(await new DiscrawlClient({ binaryPath }).search("fts", "q")).toMatchObject({
			ok: false,
			reason: "nonzero-exit",
			code: 3,
		});
	});

	it("maps a missing binary to binary-missing without leaking the path", async () => {
		const result = await new DiscrawlClient({ binaryPath: "/nonexistent/discrawl-xyz" }).search("fts", "q");
		expect(result).toMatchObject({ ok: false, reason: "binary-missing" });
		expect(result.stderr).not.toContain("/nonexistent");
	});

	it("times out slow invocations", async () => {
		const binaryPath = stubBinary("sleep 5");
		expect(await new DiscrawlClient({ binaryPath, timeoutMs: 150 }).search("fts", "q")).toMatchObject({
			ok: false,
			reason: "timeout",
		});
	});

	it("reaps a descendant that keeps the inherited stdio open after the direct child exits", async () => {
		if (process.platform === "win32") return;
		const binaryPath = join(tmpdir(), `discrawl-descendant-${process.pid}-${Date.now()}`);
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { spawn } from "node:child_process";
const descendant = spawn(process.execPath, ["-e", "setTimeout(() => process.exit(0), 5000)"], { stdio: "inherit" });
process.stdout.write(JSON.stringify([{ id: "descendant", content: "descendant", score: 1, grandchildPid: descendant.pid }]));
process.exit(0);
`,
		);
		chmodSync(binaryPath, 0o755);
		const client = new DiscrawlClient({ binaryPath, timeoutMs: 30_000 });
		let bound: ReturnType<typeof setTimeout> | undefined;
		try {
			const result = await Promise.race([
				client.search("fts", "descendant"),
				new Promise<never>((_, reject) => {
					bound = setTimeout(
						() => reject(new Error("discrawl search never settled after the direct child exited")),
						2_000,
					);
				}),
			]);
			expect(result.ok).toBe(true);
			if (!result.ok) return;
			const pidMatch = /"grandchildPid":(\d+)/.exec(result.stdout);
			if (pidMatch?.[1] === undefined) throw new Error("discrawl fixture did not report its descendant PID");
			try {
				process.kill(Number(pidMatch[1]), 0);
			} catch (error) {
				if (error instanceof Error && "code" in error && error.code === "ESRCH") return;
				throw error;
			}
			throw new Error(`discrawl descendant ${pidMatch[1]} survived after the client settled`);
		} finally {
			clearTimeout(bound);
		}
	}, 10_000);

	it("honors an abort signal", async () => {
		const controller = new AbortController();
		const binaryPath = stubBinary("sleep 5");
		const pending = new DiscrawlClient({ binaryPath }).search("fts", "q", { signal: controller.signal });
		controller.abort();
		expect(await pending).toMatchObject({ ok: false, reason: "aborted" });
	});

	it("suppresses paths in stderr diagnostics", async () => {
		const binaryPath = stubBinary("echo '/Users/secret/archive.db not found' >&2; exit 1");
		const result = await new DiscrawlClient({ binaryPath }).search("fts", "q");
		expect(result.ok).toBe(false);
		expect(result.stderr).not.toContain("/Users/secret");
	});
});

describe("DiscrawlClient managed embeddings", () => {
	it("emits the native OpenAI-compatible config and rebuilds on managed identity change", async () => {
		const root = mkdtempSync(join(tmpdir(), "discrawl-managed-"));
		const client = new DiscrawlClient({
			root,
			embeddingRuntime: {
				provider: "openai_compatible",
				model: "Qwen3",
				baseUrl: "http://127.0.0.1:18080/v1",
				dimensions: 1024,
			},
		});
		const first = await client.configureEmbeddings({
			provider: "openai_compatible",
			model: "Qwen3",
			baseUrl: "http://127.0.0.1:18080/v1",
			dimensions: 1024,
		});
		const path = join(root, ".autorag", "datasources", "discrawl", "config.toml");
		expect(first).toMatchObject({ configured: true, rebuildRequired: false });
		expect(readFileSync(path, "utf8")).toContain("[search.embeddings]");
		expect(readFileSync(path, "utf8")).toContain('provider = "openai_compatible"');
		const second = await client.configureEmbeddings({
			provider: "openai_compatible",
			model: "Qwen3-v2",
			baseUrl: "http://127.0.0.1:18080/v1",
			dimensions: 1024,
		});
		expect(second.rebuildRequired).toBe(true);
	});

	it("leaves explicit and malformed operator config byte-identical", async () => {
		const root = mkdtempSync(join(tmpdir(), "discrawl-explicit-"));
		const configPath = join(root, "operator.toml");
		const original = "garbage = [\n";
		writeFileSync(configPath, original);
		const client = new DiscrawlClient({
			root,
			configPath,
			embeddingRuntime: {
				provider: "openai_compatible",
				model: "Qwen3",
				baseUrl: "http://127.0.0.1:18080/v1",
				dimensions: 1024,
			},
		});
		expect(
			await client.configureEmbeddings({
				provider: "openai_compatible",
				model: "Qwen3",
				baseUrl: "http://127.0.0.1:18080/v1",
				dimensions: 1024,
			}),
		).toMatchObject({ configured: false });
		expect(readFileSync(configPath, "utf8")).toBe(original);
	});
});

describe("DiscrawlClient user-token gate", () => {
	it("refuses to spawn when a Discord user token is present", async () => {
		const binaryPath = stubBinary("echo '[]'");
		const result = await new DiscrawlClient({
			binaryPath,
			env: { DISCORD_USER_TOKEN: "mfa.some-user-token" },
		}).search("fts", "q");
		expect(result).toMatchObject({
			ok: false,
			reason: "user-token-rejected",
			violatingKey: "DISCORD_USER_TOKEN",
		});
	});

	it("never forwards a Discord bot token to the child process", async () => {
		const binaryPath = stubBinary("env >&2; echo '[]'");
		const result = await new DiscrawlClient({
			binaryPath,
			env: { DISCORD_BOT_TOKEN: "bot-token" },
		}).search("fts", "q");
		expect(result.ok).toBe(true);
		expect(result.stderr).not.toContain("DISCORD_BOT_TOKEN");
	});

	it("never forwards the user token value in diagnostics", async () => {
		const binaryPath = stubBinary("echo '[]'");
		const result = await new DiscrawlClient({
			binaryPath,
			env: { DISCORD_USER_TOKEN: "super-secret-value" },
		}).search("fts", "q");
		expect(JSON.stringify(result)).not.toContain("super-secret-value");
	});
});

describe("DiscrawlClient sync", () => {
	it("syncs the local archive through the wiretap subcommand", async () => {
		const binaryPath = stubBinary(`echo "$@" >&2; echo '{"messages": 12}'`);
		const result = await new DiscrawlClient({ binaryPath }).sync();
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.data.messages).toBe(12);
		expect(result.stderr.trim()).toBe("--json wiretap");
	});

	it("ignores a legacy bot source selector and never reaches the Discord API path", async () => {
		const binaryPath = stubBinary(`echo "$@" >&2; echo '{"messages": 3}'`);
		const legacy = { binaryPath, source: "discord", guildId: "g1" } as unknown as DiscrawlOptions;
		const result = await new DiscrawlClient(legacy).sync();
		expect(result.ok).toBe(true);
		expect(result.stderr.trim()).toBe("--json wiretap");
	});
});

describe("DiscrawlClient doctor and embed", () => {
	it("parses key=value doctor output", async () => {
		const binaryPath = stubBinary(
			"printf 'config=ok\\ndatabase=ok\\nfts=ok\\nembeddings=ok\\nembeddings_model=bge-m3\\n'",
		);
		const result = await new DiscrawlClient({ binaryPath }).doctor();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.data).toMatchObject({
				ready: true,
				databaseOk: true,
				embeddingsOk: true,
				embeddingModel: "bge-m3",
			});
		}
	});

	it("reports embeddings as not ok when the probe errored", async () => {
		const binaryPath = stubBinary(
			"printf 'config=ok\\ndatabase=ok\\nfts=ok\\nembeddings=ok\\nembeddings_probe=error\\n'",
		);
		const result = await new DiscrawlClient({ binaryPath }).doctor();
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.data.embeddingsOk).toBe(false);
	});

	it("parses key=value embed output", async () => {
		const binaryPath = stubBinary(
			"printf 'processed=1276\\nsucceeded=1271\\nfailed=0\\nremaining_backlog=0\\nmodel=bge-m3\\n'",
		);
		const result = await new DiscrawlClient({ binaryPath }).embed(1500);
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.data).toMatchObject({ processed: 1276, succeeded: 1271, failed: 0, remainingBacklog: 0 });
		}
	});
});

describe("nativeGatewayBaseUrl", () => {
	it("appends /v1 to a bare base URL", () => {
		expect(nativeGatewayBaseUrl("http://localhost:8080")).toBe("http://localhost:8080/v1");
	});

	it("strips trailing slashes before appending /v1", () => {
		expect(nativeGatewayBaseUrl("http://localhost:8080/gateway///")).toBe("http://localhost:8080/gateway/v1");
	});

	it("passes through a URL already ending with /v1", () => {
		expect(nativeGatewayBaseUrl("http://localhost:8080/v1")).toBe("http://localhost:8080/v1");
	});

	it("handles a URL with a single trailing slash", () => {
		expect(nativeGatewayBaseUrl("http://localhost:8080/")).toBe("http://localhost:8080/v1");
	});

	it("handles many repeated trailing slashes", () => {
		expect(nativeGatewayBaseUrl(`http://localhost:8080/path/${"/".repeat(100)}`)).toBe(
			"http://localhost:8080/path/v1",
		);
	});
});
