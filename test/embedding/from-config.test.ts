import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createEmbedderFromConfig, ensureMinSyncDaemon, releaseEmbedders } from "../../src/embedding/index.ts";
import { QWEN3_QUERY_PREFIX } from "../../src/embedding/prefixes.ts";
import { writeFakeMinSyncServe } from "../helpers/fake-minsync-serve.ts";

let servers: Server[] = [];
afterEach(async () => {
	for (const server of servers) server.close();
	servers = [];
	await releaseEmbedders();
});

async function teiServer(): Promise<{ readonly baseUrl: string; readonly inputs: string[][] }> {
	const inputs: string[][] = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk) => {
			body += chunk;
		});
		request.on("end", () => {
			const parsed = JSON.parse(body) as { inputs: string[] };
			inputs.push(parsed.inputs);
			response.setHeader("content-type", "application/json");
			response.end(JSON.stringify(parsed.inputs.map(() => [1, 2])));
		});
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	servers.push(server);
	return { baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, inputs };
}

describe("createEmbedderFromConfig prefixes", () => {
	it("applies the operator's queryPrefix and passagePrefix to a remote endpoint", async () => {
		const remote = await teiServer();
		const embedder = createEmbedderFromConfig({
			embedder: {
				id: "tei:some/model",
				baseUrl: remote.baseUrl,
				dimension: 2,
				queryPrefix: "q: ",
				passagePrefix: "p: ",
			},
		});
		await embedder.embed(["x"], { role: "query" });
		await embedder.embed(["x"], { role: "document" });
		expect(remote.inputs).toEqual([["q: x"], ["p: x"]]);
	});

	it("sends a remote endpoint's text unchanged when the operator set nothing", async () => {
		const remote = await teiServer();
		const embedder = createEmbedderFromConfig({
			embedder: { id: "tei:Qwen/Qwen3-Embedding-0.6B", baseUrl: remote.baseUrl, dimension: 2 },
		});
		await embedder.embed(["x"], { role: "query" });
		expect(remote.inputs).toEqual([["x"]]);
	});

	describe("with MinSync's local Qwen3 behind a daemon", () => {
		let home: string;
		let bin: string;
		let pid: number | undefined;
		beforeEach(async () => {
			home = mkdtempSync(join(tmpdir(), "autorag-from-config-home-"));
			bin = mkdtempSync(join(tmpdir(), "autorag-from-config-bin-"));
			pid = (
				await ensureMinSyncDaemon({
					home,
					binaryPath: writeFakeMinSyncServe(bin, { dimension: 4 }),
					env: { ...process.env, HOME: home },
					readyTimeoutMs: 15_000,
				})
			).pid;
		}, 30_000);
		afterEach(() => {
			if (pid !== undefined) process.kill(pid, "SIGTERM");
			rmSync(home, { recursive: true, force: true });
			rmSync(bin, { recursive: true, force: true });
		});
		const lengthOf = (vector: readonly number[] | undefined): number => Math.round((vector?.[0] ?? 0) * 100);

		it("defaults to the Qwen3 instruction for queries and sends documents raw", async () => {
			const embedder = createEmbedderFromConfig({ home });
			expect(lengthOf((await embedder.embed(["abc"], { role: "query" }))[0])).toBe(QWEN3_QUERY_PREFIX.length + 3);
			expect(lengthOf((await embedder.embed(["abc"], { role: "document" }))[0])).toBe(3);
		});

		it("lets an operator-set empty queryPrefix win", async () => {
			const embedder = createEmbedderFromConfig({ home, embedder: { queryPrefix: "" } });
			expect(lengthOf((await embedder.embed(["abc"], { role: "query" }))[0])).toBe(3);
		});
	});
});
