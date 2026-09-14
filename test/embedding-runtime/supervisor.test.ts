import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { EmbeddingRuntimeSupervisor, SupervisorError } from "../../src/embedding-runtime/supervisor.ts";

const node = process.execPath;
const readyScript = `
const http = require("node:http");
const port = Number(process.argv[1]);
const child = http.createServer((req, res) => { if (req.url === "/health") { res.writeHead(200); res.end("OK"); } else { res.writeHead(404); res.end(); } });
child.listen(port, "127.0.0.1");
`;
const neverReadyScript = `
const http = require("node:http");
const port = Number(process.argv[1]);
http.createServer((_req, res) => { res.writeHead(503); res.end("token=secret api_key=hidden"); }).listen(port, "127.0.0.1");
`;

async function root(): Promise<string> {
	return mkdtemp(join(tmpdir(), "autorag-supervisor-"));
}

function supervisor(cacheRoot: string, script = readyScript, overrides: Record<string, unknown> = {}) {
	return new EmbeddingRuntimeSupervisor({
		cacheRoot,
		profileId: "qwen3-embedding-0.6b",
		executablePath: node,
		modelPath: join(cacheRoot, "model.gguf"),
		spawn: (_command, args, options) =>
			require("node:child_process").spawn(node, ["-e", script, args[args.length - 1]], options),
		readinessTimeoutMs: 300,
		readinessIntervalMs: 15,
		...overrides,
	});
}

describe("embedding runtime supervisor", () => {
	it("starts a stub, acquires lock and pid, and becomes ready", async () => {
		const cacheRoot = await root();
		const runtime = supervisor(cacheRoot);
		const status = await runtime.ensureRunning();
		expect(status.state).toBe("ready");
		expect(status.pid).toBeTypeOf("number");
		expect(status.port).toBeGreaterThan(0);
		expect(await readFile(join(cacheRoot, "embedding-runtime.pid"), "utf8")).toContain(String(status.pid));
		await runtime.shutdown();
	});

	it("reclaims stale pid state", async () => {
		const cacheRoot = await root();
		await writeFile(join(cacheRoot, "embedding-runtime.pid"), "999999\n");
		await writeFile(join(cacheRoot, "embedding-runtime.lock"), "stale\n");
		const runtime = supervisor(cacheRoot);
		expect((await runtime.ensureRunning()).state).toBe("ready");
		await runtime.shutdown();
	});

	it("rejects a live foreign pid", async () => {
		const cacheRoot = await root();
		await writeFile(join(cacheRoot, "embedding-runtime.pid"), String(process.pid));
		await writeFile(join(cacheRoot, "embedding-runtime.lock"), "foreign\n");
		await expect(supervisor(cacheRoot).ensureRunning()).rejects.toMatchObject({ code: "lock-conflict" });
	});

	it("surfaces sanitized readiness failure", async () => {
		const cacheRoot = await root();
		const runtime = supervisor(cacheRoot, neverReadyScript);
		await expect(runtime.ensureRunning()).rejects.toMatchObject({ code: "readiness-timeout" });
		await runtime.shutdown();
		expect(runtime.status()).toMatchObject({ state: "stopped", lastError: expect.stringContaining("timed out") });
	});

	it("does not retry a crashing child more than once", async () => {
		const cacheRoot = await root();
		const crashScript = `process.stderr.write("token=secret\n"); setTimeout(() => process.exit(9), 20);`;
		const runtime = supervisor(cacheRoot, crashScript, { readinessTimeoutMs: 180 });
		await expect(runtime.ensureRunning()).rejects.toBeInstanceOf(SupervisorError);
		await runtime.shutdown();
	});
});
