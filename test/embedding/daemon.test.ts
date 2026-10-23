import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ensureMinSyncDaemon, MinSyncDaemonError, probeMinSyncDaemon } from "../../src/embedding/daemon.ts";
import { writeFakeMinSyncServe } from "../helpers/fake-minsync-serve.ts";

let home: string;
let bin: string;
let servers: Server[] = [];

beforeEach(() => {
	home = mkdtempSync(join(tmpdir(), "autorag-daemon-home-"));
	bin = mkdtempSync(join(tmpdir(), "autorag-daemon-bin-"));
});
afterEach(() => {
	for (const server of servers) server.close();
	servers = [];
	rmSync(home, { recursive: true, force: true });
	rmSync(bin, { recursive: true, force: true });
});

const stateRoot = (): string => join(home, ".cache", "minsync", "serve");

/** Publishes the discovery files a running `minsync serve` leaves behind. */
async function publishRunningServer(instance: string, health: Record<string, unknown>): Promise<void> {
	const server = createServer((request, response) => {
		response.setHeader("content-type", "application/json");
		response.end(request.url === "/health" ? JSON.stringify({ ...health, instance, status: "ok" }) : "{}");
	});
	servers.push(server);
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const dir = join(stateRoot(), instance);
	mkdirSync(dir, { recursive: true });
	writeFileSync(join(dir, "host"), "127.0.0.1");
	writeFileSync(join(dir, "port"), String((server.address() as AddressInfo).port));
}

const qwenHealth = { model_id: "Qwen/Qwen3-Embedding-0.6B", dimension: 1024 };

describe("probeMinSyncDaemon", () => {
	it("returns nothing when no server has published discovery files", async () => {
		expect(await probeMinSyncDaemon({ home })).toBeUndefined();
	});

	it("returns the url, model and dimension of a server that answers /health", async () => {
		await publishRunningServer("Qwen-Qwen3-Embedding-0.6B-aaaa", qwenHealth);
		const found = await probeMinSyncDaemon({ home });
		expect(found).toMatchObject({ model: "Qwen/Qwen3-Embedding-0.6B", dimension: 1024 });
		expect(found?.baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
	});

	it("ignores a stale port file whose server is gone", async () => {
		const dir = join(stateRoot(), "Qwen-Qwen3-Embedding-0.6B-dead");
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "host"), "127.0.0.1");
		writeFileSync(join(dir, "port"), "1");
		expect(await probeMinSyncDaemon({ home })).toBeUndefined();
	});

	it("only accepts the requested model, so another model's server is never mistaken for it", async () => {
		await publishRunningServer("other-bbbb", { model_id: "intfloat/multilingual-e5-small", dimension: 384 });
		expect(await probeMinSyncDaemon({ home, model: "Qwen/Qwen3-Embedding-0.6B" })).toBeUndefined();
	});

	it("refuses a published non-loopback host", async () => {
		await publishRunningServer("Qwen-Qwen3-Embedding-0.6B-cccc", qwenHealth);
		writeFileSync(join(stateRoot(), "Qwen-Qwen3-Embedding-0.6B-cccc", "host"), "10.0.0.5");
		expect(await probeMinSyncDaemon({ home })).toBeUndefined();
	});
});

describe("ensureMinSyncDaemon", () => {
	it("reuses a running daemon without spawning anything", async () => {
		await publishRunningServer("Qwen-Qwen3-Embedding-0.6B-aaaa", qwenHealth);
		// The binary does not exist, so any attempt to start a second server would reject.
		const daemon = await ensureMinSyncDaemon({ home, binaryPath: join(bin, "does-not-exist") });
		expect(daemon.started).toBe(false);
		expect(daemon.dimension).toBe(1024);
	});

	it("starts `minsync serve` detached when none runs, then returns the daemon it published", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { dimension: 4 });
		const daemon = await ensureMinSyncDaemon({
			home,
			binaryPath,
			env: { ...process.env, HOME: home },
			readyTimeoutMs: 15_000,
		});
		try {
			expect(daemon.started).toBe(true);
			expect(daemon.dimension).toBe(4);
			const again = await probeMinSyncDaemon({ home });
			expect(again?.baseUrl).toBe(daemon.baseUrl);
		} finally {
			if (daemon.pid !== undefined) process.kill(daemon.pid, "SIGTERM");
		}
	}, 30_000);

	it("surfaces the server's own stderr when it exits before publishing", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, {
			failWith: "cannot load model: no network and no cached weights",
		});
		await expect(
			ensureMinSyncDaemon({ home, binaryPath, env: { ...process.env, HOME: home }, readyTimeoutMs: 15_000 }),
		).rejects.toThrow(/no network and no cached weights/);
	}, 30_000);

	it("fails fast when `minsync` exits 0 without a server being up, instead of waiting out the timeout", async () => {
		// An inert minsync (or one that does not know `serve`) prints something and exits 0.
		const binaryPath = join(bin, "minsync");
		writeFileSync(binaryPath, "#!/bin/sh\necho '{}'\n");
		chmodSync(binaryPath, 0o755);
		const started = Date.now();
		await expect(
			ensureMinSyncDaemon({ home, binaryPath, env: { ...process.env, HOME: home }, readyTimeoutMs: 60_000 }),
		).rejects.toThrow(/exited.*without/is);
		expect(Date.now() - started).toBeLessThan(10_000);
	}, 30_000);

	it("fails with a MinSyncDaemonError when the binary does not exist", async () => {
		await expect(
			ensureMinSyncDaemon({ home, binaryPath: join(bin, "missing"), env: { ...process.env, HOME: home } }),
		).rejects.toBeInstanceOf(MinSyncDaemonError);
	});
});
