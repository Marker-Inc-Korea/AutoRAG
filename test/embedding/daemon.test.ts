import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ensureInlDaemon, InlDaemonError, probeInlDaemon } from "../../src/embedding/daemon.ts";
import { writeFakeInlServe } from "../helpers/fake-inl-serve.ts";

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

const stateRoot = (): string => join(home, ".cache", "im-not-llm", "serve");

/** Publishes the discovery files a running `inl serve` leaves behind. */
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

const qwenHealth = { kind: "dense", model_id: "Qwen/Qwen3-Embedding-0.6B", dimension: 1024 };

describe("probeInlDaemon", () => {
	it("returns nothing when no server has published discovery files", async () => {
		expect(await probeInlDaemon({ home })).toBeUndefined();
	});

	it("returns the url, kind, model and dimension of a server that answers /health", async () => {
		await publishRunningServer("Qwen-Qwen3-Embedding-0.6B-aaaa", qwenHealth);
		const found = await probeInlDaemon({ home });
		expect(found).toMatchObject({ kind: "dense", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 1024 });
		expect(found?.baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
	});

	it("ignores a stale port file whose server is gone", async () => {
		const dir = join(stateRoot(), "Qwen-Qwen3-Embedding-0.6B-dead");
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "host"), "127.0.0.1");
		writeFileSync(join(dir, "port"), "1");
		expect(await probeInlDaemon({ home })).toBeUndefined();
	});

	it("only accepts the requested model, so another model's server is never mistaken for it", async () => {
		await publishRunningServer("other-bbbb", {
			kind: "dense",
			model_id: "intfloat/multilingual-e5-small",
			dimension: 384,
		});
		expect(await probeInlDaemon({ home, model: "Qwen/Qwen3-Embedding-0.6B" })).toBeUndefined();
	});

	it("matches the model case-insensitively", async () => {
		await publishRunningServer("qwen-lower", {
			kind: "dense",
			model_id: "qwen/qwen3-embedding-0.6b",
			dimension: 1024,
		});
		const found = await probeInlDaemon({ home, model: "Qwen/Qwen3-Embedding-0.6B" });
		expect(found?.model).toBe("qwen/qwen3-embedding-0.6b");
	});

	it("never takes a sparse server for a dense one, and finds it only when asked by kind", async () => {
		const sparse = { kind: "sparse", model_id: "nlpai-lab/KURE-Sparse-nano", dimension: 36282 };
		await publishRunningServer("kure-sparse-aaaa", sparse);
		expect(await probeInlDaemon({ home })).toBeUndefined();
		expect(await probeInlDaemon({ home, model: "nlpai-lab/KURE-Sparse-nano" })).toBeUndefined();
		const found = await probeInlDaemon({ home, kind: "sparse", model: "nlpai-lab/KURE-Sparse-nano" });
		expect(found).toMatchObject({ kind: "sparse", instance: "kure-sparse-aaaa" });
	});

	it("refuses a published non-loopback host", async () => {
		await publishRunningServer("Qwen-Qwen3-Embedding-0.6B-cccc", qwenHealth);
		writeFileSync(join(stateRoot(), "Qwen-Qwen3-Embedding-0.6B-cccc", "host"), "10.0.0.5");
		expect(await probeInlDaemon({ home })).toBeUndefined();
	});
});

/** PATH with only temp `bin`, so the real system `inl`/`minsync` are never found. */
const isolatedEnv = (): NodeJS.ProcessEnv => ({ HOME: home, PATH: bin });

async function killDaemonChild(pid: number | undefined): Promise<void> {
	if (pid !== undefined) process.kill(pid, "SIGTERM");
}

describe("ensureInlDaemon", () => {
	it("reuses a running daemon without spawning anything", async () => {
		await publishRunningServer("Qwen-Qwen3-Embedding-0.6B-aaaa", qwenHealth);
		// The launcher does not exist, so any attempt to start a second server would reject.
		const daemon = await ensureInlDaemon({ home, launcher: { path: join(bin, "does-not-exist") } });
		expect(daemon.started).toBe(false);
		expect(daemon.dimension).toBe(1024);
	});

	it("prefers `inl` on PATH over `minsync`", async () => {
		writeFakeInlServe(bin, { name: "inl", dimension: 7 });
		writeFakeInlServe(bin, { name: "minsync", dimension: 4 });
		const daemon = await ensureInlDaemon({
			home,
			env: isolatedEnv(),
			// Never consulted: `inl` answers first.
			resolveMinSyncBinary: async () => {
				throw new Error("minsync fallback must not be resolved when inl is on PATH");
			},
			readyTimeoutMs: 15_000,
		});
		try {
			expect(daemon.launcher).toBe("inl");
			expect(daemon.dimension).toBe(7);
		} finally {
			await killDaemonChild(daemon.pid);
		}
	}, 30_000);

	it("falls back to `minsync` on PATH when `inl` is absent", async () => {
		writeFakeInlServe(bin, { name: "minsync", dimension: 4 });
		const daemon = await ensureInlDaemon({ home, env: isolatedEnv(), readyTimeoutMs: 15_000 });
		try {
			expect(daemon.launcher).toBe("minsync");
			expect(daemon.dimension).toBe(4);
		} finally {
			await killDaemonChild(daemon.pid);
		}
	}, 30_000);

	it("starts the launcher detached when none runs, then returns the daemon it published", async () => {
		const launcherPath = writeFakeInlServe(bin, { dimension: 4 });
		const daemon = await ensureInlDaemon({
			home,
			launcher: { path: launcherPath },
			env: { ...process.env, HOME: home },
			readyTimeoutMs: 15_000,
		});
		try {
			expect(daemon.started).toBe(true);
			expect(daemon.dimension).toBe(4);
			const again = await probeInlDaemon({ home });
			expect(again?.baseUrl).toBe(daemon.baseUrl);
		} finally {
			await killDaemonChild(daemon.pid);
		}
	}, 30_000);

	it("surfaces the server's own stderr when it exits before publishing", async () => {
		const launcherPath = writeFakeInlServe(bin, {
			failWith: "cannot load model: no network and no cached weights",
		});
		await expect(
			ensureInlDaemon({
				home,
				launcher: { path: launcherPath },
				env: { ...process.env, HOME: home },
				readyTimeoutMs: 15_000,
			}),
		).rejects.toThrow(/no network and no cached weights/);
	}, 30_000);

	it("fails fast when the launcher exits 0 without a server being up, instead of waiting out the timeout", async () => {
		// An inert `inl` (or a `minsync` that does not know `serve`) prints something and exits 0.
		const launcherPath = join(bin, "inl");
		writeFileSync(launcherPath, "#!/bin/sh\necho '{}'\n");
		chmodSync(launcherPath, 0o755);
		const started = Date.now();
		await expect(
			ensureInlDaemon({
				home,
				launcher: { path: launcherPath },
				env: { ...process.env, HOME: home },
				readyTimeoutMs: 60_000,
			}),
		).rejects.toThrow(/exited.*without/is);
		expect(Date.now() - started).toBeLessThan(10_000);
	}, 30_000);

	it("fails with an InlDaemonError when neither `inl` nor `minsync` is on PATH", async () => {
		await expect(ensureInlDaemon({ home, env: isolatedEnv() })).rejects.toBeInstanceOf(InlDaemonError);
	});

	it("fails with an InlDaemonError when the launcher does not exist", async () => {
		await expect(
			ensureInlDaemon({ home, launcher: { path: join(bin, "missing") }, env: { ...process.env, HOME: home } }),
		).rejects.toBeInstanceOf(InlDaemonError);
	});
});
