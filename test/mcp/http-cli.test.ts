import { type ChildProcess, spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client, StreamableHTTPClientTransport } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it } from "vitest";

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
	while (cleanups.length > 0) await cleanups.pop()?.();
});

interface Running {
	readonly child: ChildProcess;
	readonly url: string;
	readonly home: string;
	readonly stderr: () => string;
}

function fixture(): { root: string; home: string; config: string } {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-http-cli-"));
	cleanups.push(() => rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 }));
	const docs = join(root, "docs");
	const home = join(root, "home");
	mkdirSync(docs, { recursive: true });
	mkdirSync(home, { recursive: true });
	const config = join(root, "config.json");
	writeFileSync(
		config,
		JSON.stringify({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { workspacePath: join(root, ".autorag", "minsync"), autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: { watch: false },
		}),
	);
	return { root, home, config };
}

async function launch(args: string[], env: Record<string, string>): Promise<Running> {
	const { home, config } = fixture();
	const child = spawn(process.execPath, ["src/mcp/index.ts", ...args], {
		cwd: process.cwd(),
		env: { ...process.env, AUTORAG_HOME: home, AUTORAG_CONFIG: config, AUTORAG_MCP_TOKEN: "", ...env },
		stdio: ["ignore", "pipe", "pipe"],
	});
	let stderr = "";
	child.stderr?.on("data", (chunk: Buffer) => {
		stderr += chunk.toString("utf8");
	});
	cleanups.push(() => {
		child.kill("SIGTERM");
	});
	const url = await new Promise<string>((resolve, reject) => {
		const check = () => {
			const match = /listening on (http:\/\/\S+)/.exec(stderr);
			if (match?.[1] !== undefined) resolve(match[1]);
		};
		child.stderr?.on("data", check);
		child.once("exit", (code) => {
			reject(new Error(`exited ${code} before listening:\n${stderr}`));
		});
		check();
	});
	return { child, url, home, stderr: () => stderr };
}

const TOKEN = "t".repeat(40);

async function toolNames(url: string, token: string): Promise<string[]> {
	const client = new Client({ name: "autorag-http-cli-test", version: "1.0.0" });
	cleanups.push(() => client.close());
	await client.connect(
		new StreamableHTTPClientTransport(new URL(url), {
			requestInit: { headers: { authorization: `Bearer ${token}` } },
		}),
	);
	return (await client.listTools()).tools.map((tool) => tool.name);
}

describe("autorag-mcp --http", () => {
	it("serves read-only tools over HTTP with the token from the environment", async () => {
		const running = await launch(["--http", "--port", "0"], { AUTORAG_MCP_TOKEN: TOKEN });
		const names = await toolNames(running.url, TOKEN);
		expect(names).toContain("autorag.search");
		expect(names).not.toContain("autorag.refresh");
		expect(running.stderr()).not.toContain(TOKEN);
	});

	it("--allow-write offers the write tools", async () => {
		const running = await launch(["--http", "--port", "0", "--allow-write"], { AUTORAG_MCP_TOKEN: TOKEN });
		expect(await toolNames(running.url, TOKEN)).toContain("autorag.refresh");
	});

	it("honors AUTORAG_MCP_TOOLS", async () => {
		const running = await launch(["--http", "--port", "0"], {
			AUTORAG_MCP_TOKEN: TOKEN,
			AUTORAG_MCP_TOOLS: "autorag.status",
		});
		expect(await toolNames(running.url, TOKEN)).toEqual(["autorag.status"]);
	});

	it("without a token in the environment, creates one owner-only file under AUTORAG_HOME and reuses it", async () => {
		const first = await launch(["--http", "--port", "0"], {});
		const file = join(first.home, "mcp-http-token");
		expect(existsSync(file)).toBe(true);
		if (process.platform !== "win32") expect(statSync(file).mode & 0o777).toBe(0o600);
		const token = readFileSync(file, "utf8").trim();
		expect(token.length).toBeGreaterThanOrEqual(32);
		expect(first.stderr()).toContain(file);
		expect(first.stderr()).not.toContain(token);
		expect(await toolNames(first.url, token)).toContain("autorag.search");
	});

	it("refuses a short environment token", async () => {
		await expect(launch(["--http", "--port", "0"], { AUTORAG_MCP_TOKEN: "short" })).rejects.toThrow(/at least 32/);
	});

	it("refuses to bind beyond loopback without a public URL", async () => {
		await expect(
			launch(["--http", "--host", "0.0.0.0", "--port", "0"], { AUTORAG_MCP_TOKEN: TOKEN }),
		).rejects.toThrow(/public url/i);
	});

	it("rejects an unknown flag instead of silently serving stdio", async () => {
		await expect(launch(["--http", "--prot", "0"], { AUTORAG_MCP_TOKEN: TOKEN })).rejects.toThrow(/unknown/i);
	});

	it("--check passes against a server started with --public-url and prints no token", async () => {
		const port = 18000 + Math.floor(Math.random() * 1000);
		const publicUrl = `http://127.0.0.1:${port}`;
		const running = await launch(["--http", "--port", String(port), "--public-url", publicUrl], {
			AUTORAG_MCP_TOKEN: TOKEN,
		});
		const check = spawnSync(process.execPath, ["src/mcp/index.ts", "--check", `${publicUrl}/mcp`], {
			encoding: "utf8",
			env: { ...process.env, AUTORAG_MCP_TOKEN: TOKEN },
		});
		expect(check.status, check.stdout + check.stderr).toBe(0);
		expect(check.stdout).toContain("tools");
		expect(check.stdout).toContain("pass");
		expect(check.stdout + check.stderr).not.toContain(TOKEN);
		expect(running.child.exitCode).toBeNull();
	});

	it("--check exits non-zero and says why when the server has no OAuth surface", async () => {
		const running = await launch(["--http", "--port", "0"], { AUTORAG_MCP_TOKEN: TOKEN });
		const check = spawnSync(process.execPath, ["src/mcp/index.ts", "--check", running.url], {
			encoding: "utf8",
			env: { ...process.env, AUTORAG_MCP_TOKEN: TOKEN },
		});
		expect(check.status).toBe(1);
		expect(check.stdout).toContain("fail");
		expect(check.stdout).toContain("--public-url");
	});

	it("--check with --json prints one machine-readable document", async () => {
		const running = await launch(["--http", "--port", "0"], { AUTORAG_MCP_TOKEN: TOKEN });
		const check = spawnSync(process.execPath, ["src/mcp/index.ts", "--check", running.url, "--json"], {
			encoding: "utf8",
			env: { ...process.env, AUTORAG_MCP_TOKEN: TOKEN },
		});
		const parsed: unknown = JSON.parse(check.stdout);
		expect(parsed).toMatchObject({ ok: false, steps: expect.any(Array) });
	});

	it("--check refuses to run beside --http", async () => {
		await expect(
			launch(["--http", "--check", "http://127.0.0.1:1/mcp"], { AUTORAG_MCP_TOKEN: TOKEN }),
		).rejects.toThrow(/--check/);
	});
});
