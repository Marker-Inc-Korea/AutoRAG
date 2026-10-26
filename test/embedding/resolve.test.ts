import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ensureMinSyncDaemon, stopMinSyncDaemon } from "../../src/embedding/daemon.ts";
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

/** Starts the fake daemon the way an operator's launchd service would: AutoRAG is not involved. */
async function startOperatorDaemon(): Promise<number | undefined> {
	const daemon = await ensureMinSyncDaemon({
		home,
		binaryPath: writeFakeMinSyncServe(bin, { dimension: 4 }),
		env: { ...process.env, HOME: home },
		readyTimeoutMs: 15_000,
	});
	return daemon.pid;
}

describe("createManagedEmbedder (local)", () => {
	it("embeds through a daemon the operator started and reports the shared identity", async () => {
		const pid = await startOperatorDaemon();
		try {
			const embedder = createManagedEmbedder({ home });
			expect(await embedder.health()).toMatchObject({ backend: "daemon", alive: true });
			expect(await embedder.identity()).toEqual({
				provider: "minsync",
				model: "Qwen/Qwen3-Embedding-0.6B",
				dimension: 4,
			});
			expect(await embedder.embed(["abc"])).toEqual([[0.03, 0.04, 0.05, 0.06]]);
		} finally {
			killDaemon(pid);
		}
	}, 40_000);

	it("never starts a daemon: with none running it uses the in-process engine", async () => {
		const calls: string[] = [];
		const embedder = createManagedEmbedder({ home, inProcess: async () => inProcess(calls) });
		expect(await embedder.embed(["hello"])).toEqual([[9, 9, 9, 9]]);
		expect(await embedder.describeEndpoint()).toBeUndefined();
		expect(calls).toEqual(["hello"]);
	});

	it("notices a daemon that appears while it runs, and stops using one that went away", async () => {
		const calls: string[] = [];
		const embedder = createManagedEmbedder({ home, inProcess: async () => inProcess(calls) });
		expect(await embedder.health()).toMatchObject({ backend: "in-process" });
		await startOperatorDaemon();
		expect(await embedder.health()).toMatchObject({ backend: "daemon", alive: true });
		// stopMinSyncDaemon resolves only once the server no longer answers, so the next probe is deterministic.
		expect(await stopMinSyncDaemon({ home })).toBeDefined();
		expect((await embedder.health()).backend).toBe("in-process");
		expect(await embedder.embed(["after"])).toEqual([[9, 9, 9, 9]]);
		expect(calls).toEqual(["after"]);
	}, 40_000);

	it("reports nothing alive, with the reason, when neither a daemon nor the in-process engine exists", async () => {
		const embedder = createManagedEmbedder({
			home,
			inProcess: async () => {
				throw new Error("addon not installed");
			},
		});
		expect(await embedder.health()).toMatchObject({ backend: "none", alive: false });
		expect((await embedder.health()).detail).toMatch(/addon not installed/);
		await expect(embedder.embed(["x"])).rejects.toThrow(/Start `minsync serve`/);
	});
});

describe("createManagedEmbedder roles", () => {
	const prefixes = { query: "Q:", passage: "P:" };
	// The fake daemon's vector for a text is [(len+0)/100, (len+1)/100, ...], so the vector shows what it was sent.
	const lengthOf = (vector: readonly number[]): number => Math.round((vector[0] ?? 0) * 100);

	it("prefixes queries, not documents, on the way to a daemon, and never asks it to prefix again", async () => {
		const pid = await startOperatorDaemon();
		try {
			const embedder = createManagedEmbedder({ home, prefixes });
			const [asQuery] = await embedder.embed(["abc"], { role: "query" });
			const [asDocument] = await embedder.embed(["abc"], { role: "document" });
			const [byDefault] = await embedder.embed(["abc"]);
			expect(lengthOf(asQuery as number[])).toBe("Q:abc".length);
			expect(lengthOf(asDocument as number[])).toBe("P:abc".length);
			expect(byDefault).toEqual(asDocument);
		} finally {
			killDaemon(pid);
		}
	}, 40_000);

	it("prefixes for the in-process engine", async () => {
		const calls: string[] = [];
		const embedder = createManagedEmbedder({ home, prefixes, inProcess: async () => inProcess(calls) });
		await embedder.embed(["a"], { role: "query" });
		await embedder.embed(["a"], { role: "document" });
		expect(calls).toEqual(["Q:a", "P:a"]);
	});

	it("prefixes for a remote endpoint, per input of a batch", async () => {
		const sent: string[][] = [];
		const embedder = createManagedEmbedder({
			prefixes,
			remote: { baseUrl: "http://127.0.0.1:9", model: "m", dimension: 1, protocol: "tei" },
			fetchImpl: (async (_url: string, init?: RequestInit) => {
				const inputs = (JSON.parse(String(init?.body)) as { inputs: string[] }).inputs;
				sent.push(inputs);
				return new Response(JSON.stringify(inputs.map(() => [1])), { status: 200 });
			}) as unknown as typeof fetch,
		});
		await embedder.embed(["a", "b"], { role: "query" });
		await embedder.embed(["a", "b"]);
		expect(sent).toEqual([
			["Q:a", "Q:b"],
			["P:a", "P:b"],
		]);
	});

	it("sends text unchanged when no prefixes are configured", async () => {
		const calls: string[] = [];
		const embedder = createManagedEmbedder({ home, inProcess: async () => inProcess(calls) });
		await embedder.embed(["a"], { role: "query" });
		expect(calls).toEqual(["a"]);
	});

	it("reports the prefixes of the endpoint it embeds through, for external CLIs", async () => {
		const pid = await startOperatorDaemon();
		try {
			const endpoint = await createManagedEmbedder({ home, prefixes }).describeEndpoint();
			expect(endpoint).toMatchObject({ queryPrefix: "Q:", passagePrefix: "P:" });
		} finally {
			killDaemon(pid);
		}
	}, 40_000);
});

describe("createManagedEmbedder (remote)", () => {
	it("never looks for a daemon or loads the in-process engine when a remote endpoint is configured", async () => {
		let loaded = false;
		const embedder = createManagedEmbedder({
			home,
			remote: { baseUrl: "http://127.0.0.1:9", model: "m", dimension: 3, protocol: "openai" },
			inProcess: async () => {
				loaded = true;
				return inProcess([]);
			},
			fetchImpl: (async () =>
				new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 2, 3] }] }), {
					status: 200,
				})) as typeof fetch,
		});
		expect(await embedder.embed(["a"])).toEqual([[1, 2, 3]]);
		expect((await embedder.describeEndpoint())?.backend).toBe("remote");
		expect(loaded).toBe(false);
	});

	it("probes the remote server root's /health, ignoring a trailing /v1", async () => {
		const urls: string[] = [];
		const embedder = createManagedEmbedder({
			remote: { baseUrl: "https://emb.example.com/v1", model: "m", dimension: 3, protocol: "openai" },
			fetchImpl: (async (url: string) => {
				urls.push(url);
				return new Response("{}", { status: 200 });
			}) as unknown as typeof fetch,
		});
		expect(await embedder.health()).toMatchObject({ backend: "remote", alive: true });
		expect(urls).toEqual(["https://emb.example.com/health"]);
	});

	it("counts a 404 from /health as alive (hosted APIs have no such route) but a 5xx or a network error as down", async () => {
		const respond = (status: number | "throw") =>
			createManagedEmbedder({
				remote: { baseUrl: "https://emb.example.com", model: "m", dimension: 3, protocol: "openai" },
				fetchImpl: (async () => {
					if (status === "throw") throw new Error("connect ECONNREFUSED");
					return new Response("", { status });
				}) as unknown as typeof fetch,
			}).health();
		expect(await respond(404)).toMatchObject({ alive: true });
		expect(await respond(503)).toMatchObject({ alive: false, detail: "HTTP 503" });
		expect(await respond("throw")).toMatchObject({ alive: false, detail: expect.stringContaining("ECONNREFUSED") });
	});
});
