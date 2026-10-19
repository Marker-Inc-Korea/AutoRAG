import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { InProcessEmbedder } from "../../src/embedding/in-process.ts";
import { createManagedEmbedder } from "../../src/embedding/managed.ts";
import { writeFakeMinSyncServe } from "../helpers/fake-minsync-serve.ts";

let home: string;
let bin: string;
beforeEach(() => {
	home = mkdtempSync(join(tmpdir(), "autorag-managed-home-"));
	bin = mkdtempSync(join(tmpdir(), "autorag-managed-bin-"));
});
afterEach(() => {
	rmSync(home, { recursive: true, force: true });
	rmSync(bin, { recursive: true, force: true });
});

function killDaemon(pid: number | undefined): void {
	if (pid !== undefined) process.kill(pid, "SIGTERM");
}

const inProcess = (calls: string[]): InProcessEmbedder => ({
	close: async () => undefined,
	identity: async () => ({ provider: "minsync", model: "Qwen/Qwen3-Embedding-0.6B", dimension: 4 }),
	embed: async (texts) => {
		calls.push(...texts);
		return texts.map(() => [9, 9, 9, 9]);
	},
});

describe("createManagedEmbedder (local)", () => {
	it("starts the daemon on ensureReady when none runs, then embeds through it with the shared identity", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { dimension: 4 });
		const inProcessCalls: string[] = [];
		const embedder = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => binaryPath,
			inProcess: async () => inProcess(inProcessCalls),
			readyTimeoutMs: 15_000,
		});
		const ready = await embedder.ensureReady();
		try {
			expect(ready).toMatchObject({ backend: "daemon", daemonStarted: true });
			expect(await embedder.embed(["abc"])).toEqual([[0.03, 0.04, 0.05, 0.06]]);
			expect(inProcessCalls).toEqual([]);
			// Daemon and in-process vectors are identical, so both must report one identity or every
			// store keyed on it would re-embed whenever the daemon appears or disappears.
			expect(await embedder.identity()).toEqual({
				provider: "minsync",
				model: "Qwen/Qwen3-Embedding-0.6B",
				dimension: 4,
			});
		} finally {
			killDaemon(ready.daemonPid);
			await embedder.close();
		}
	}, 40_000);

	it("never starts a daemon while answering: embed alone uses the in-process engine", async () => {
		let resolved = 0;
		const inProcessCalls: string[] = [];
		const embedder = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => {
				resolved += 1;
				return "unused";
			},
			inProcess: async () => inProcess(inProcessCalls),
		});
		expect(await embedder.embed(["abc"])).toEqual([[9, 9, 9, 9]]);
		expect(inProcessCalls).toEqual(["abc"]);
		expect(resolved).toBe(0);
	});

	it("prefers a daemon that is already running over the in-process engine, without starting another", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { dimension: 4 });
		const starter = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => binaryPath,
			inProcess: async () => inProcess([]),
			readyTimeoutMs: 15_000,
		});
		const startedReady = await starter.ensureReady();
		try {
			let resolved = 0;
			const inProcessCalls: string[] = [];
			const consumer = createManagedEmbedder({
				home,
				env: { ...process.env, HOME: home },
				resolveBinary: async () => {
					resolved += 1;
					return binaryPath;
				},
				inProcess: async () => inProcess(inProcessCalls),
			});
			expect(await consumer.embed(["abc"])).toEqual([[0.03, 0.04, 0.05, 0.06]]);
			expect((await consumer.ensureReady()).daemonStarted).toBe(false);
			expect(inProcessCalls).toEqual([]);
			expect(resolved).toBe(0);
		} finally {
			killDaemon(startedReady.daemonPid);
			await starter.close();
		}
	}, 40_000);

	it("starts a new daemon when the one it used has since exited (idle shutdown)", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { dimension: 4 });
		const embedder = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => binaryPath,
			inProcess: async () => inProcess([]),
			readyTimeoutMs: 15_000,
		});
		const first = await embedder.ensureReady();
		expect(first.daemonStarted).toBe(true);
		killDaemon(first.daemonPid);
		// The fake removes its discovery files on SIGTERM; wait until that has happened.
		const deadline = Date.now() + 10_000;
		while (Date.now() < deadline && (await embedder.describeEndpoint()) !== undefined) {
			await new Promise((resolve) => setTimeout(resolve, 100));
		}
		const second = await embedder.ensureReady();
		try {
			expect(second).toMatchObject({ backend: "daemon", daemonStarted: true });
			expect(second.daemonPid).not.toBe(first.daemonPid);
		} finally {
			killDaemon(second.daemonPid);
			await embedder.close();
		}
	}, 60_000);

	it("falls back to the in-process engine when the daemon cannot start and reports why", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { failWith: "port bind failed" });
		const inProcessCalls: string[] = [];
		const embedder = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => binaryPath,
			inProcess: async () => inProcess(inProcessCalls),
			readyTimeoutMs: 15_000,
		});
		const ready = await embedder.ensureReady();
		expect(ready.backend).toBe("in-process");
		expect(ready.daemonError).toMatch(/port bind failed/);
		expect(await embedder.embed(["abc"])).toEqual([[9, 9, 9, 9]]);
		expect(inProcessCalls).toEqual(["abc"]);
	}, 40_000);

	it("reports both failures when neither the daemon nor the in-process engine is available", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { failWith: "port bind failed" });
		const embedder = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => binaryPath,
			inProcess: async () => {
				throw new Error("addon @nomadamas/minsync-native is not installed");
			},
			readyTimeoutMs: 15_000,
		});
		await embedder.ensureReady();
		await expect(embedder.embed(["abc"])).rejects.toThrow(/port bind failed[\s\S]*not installed/);
	}, 40_000);

	it("does not retry a failed daemon start on every ensureReady", async () => {
		const binaryPath = writeFakeMinSyncServe(bin, { failWith: "port bind failed" });
		let resolved = 0;
		const embedder = createManagedEmbedder({
			home,
			env: { ...process.env, HOME: home },
			resolveBinary: async () => {
				resolved += 1;
				return binaryPath;
			},
			inProcess: async () => inProcess([]),
			readyTimeoutMs: 15_000,
		});
		await embedder.ensureReady();
		await embedder.ensureReady();
		expect(resolved).toBe(1);
	}, 40_000);
});

describe("createManagedEmbedder (remote)", () => {
	it("never starts a daemon or loads the in-process engine when a remote endpoint is configured", async () => {
		let started = false;
		let loaded = false;
		const embedder = createManagedEmbedder({
			home,
			remote: { baseUrl: "http://127.0.0.1:9", model: "m", dimension: 3, protocol: "openai" },
			resolveBinary: async () => {
				started = true;
				return "unused";
			},
			inProcess: async () => {
				loaded = true;
				return inProcess([]);
			},
			fetchImpl: (async () =>
				new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 2, 3] }] }), {
					status: 200,
				})) as typeof fetch,
		});
		expect((await embedder.ensureReady()).backend).toBe("remote");
		expect(await embedder.embed(["a"])).toEqual([[1, 2, 3]]);
		expect(started).toBe(false);
		expect(loaded).toBe(false);
	});
});
