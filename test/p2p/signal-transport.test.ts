import { mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	type SignalIncomingMessage,
	type SignalTransport,
	type SpawnedProcess,
	startSignalDaemon,
} from "../../src/p2p/signal-transport.ts";

const servers: Server[] = [];
const transports: SignalTransport[] = [];
const roots: string[] = [];

function workspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-signal-transport-"));
	roots.push(root);
	return root;
}

afterEach(async () => {
	for (const transport of transports.splice(0)) await transport.close();
	for (const server of servers.splice(0)) await new Promise<void>((resolve) => server.close(() => resolve()));
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

interface FakeDaemon {
	readonly server: Server;
	readonly port: number;
	readonly rpcCalls: { method: string; params: unknown }[];
	readonly sseClients: import("node:http").ServerResponse[];
	pushSseEvent(payload: unknown): void;
}

async function startFakeDaemon(): Promise<FakeDaemon> {
	const rpcCalls: { method: string; params: unknown }[] = [];
	const sseClients: import("node:http").ServerResponse[] = [];
	const server = createServer((req, res) => {
		if (req.method === "GET" && req.url === "/api/v1/events") {
			res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
			res.write("retry: 1000\n\n");
			sseClients.push(res);
			return;
		}
		if (req.method === "POST" && req.url === "/api/v1/rpc") {
			const chunks: Buffer[] = [];
			req.on("data", (chunk: Buffer) => chunks.push(chunk));
			req.on("end", () => {
				const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
					id: number;
					method: string;
					params?: unknown;
				};
				rpcCalls.push({ method: body.method, params: body.params });
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { timestamp: 1700000000000 } }));
			});
			return;
		}
		res.writeHead(404);
		res.end();
	});
	servers.push(server);
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	if (address === null || typeof address === "string") throw new Error("fake daemon failed to bind");
	return {
		server,
		port: address.port,
		rpcCalls,
		sseClients,
		pushSseEvent(payload: unknown) {
			for (const client of sseClients) client.write(`data: ${JSON.stringify(payload)}\n\n`);
		},
	};
}

/** A spawn seam that never launches a real process; the fake daemon stands in. */
function fakeSpawn(): { calls: string[][]; spawn: (args: readonly string[]) => SpawnedProcess } {
	const calls: string[][] = [];
	return {
		calls,
		spawn: (args: readonly string[]) => {
			calls.push([...args]);
			return {
				pid: 4242,
				kill: () => {},
				onExit: () => Promise.resolve(0),
				stderrText: () => "",
			};
		},
	};
}

describe("startSignalDaemon", () => {
	it("spawns signal-cli daemon with the account, data dir, and http endpoint", async () => {
		const daemon = await startFakeDaemon();
		const { calls, spawn } = fakeSpawn();
		const dataDir = workspace();
		const transport = await startSignalDaemon({
			account: "+821012345678",
			dataDir,
			host: "127.0.0.1",
			port: daemon.port,
			spawnProcess: spawn,
		});
		transports.push(transport);
		expect(calls).toHaveLength(1);
		const argv = calls[0]!;
		expect(argv).toContain("daemon");
		expect(argv).toContain("-a");
		expect(argv).toContain("+821012345678");
		expect(argv).toContain("--data-dir");
		expect(argv).toContain(dataDir);
		expect(argv).toContain("--http");
		expect(argv).toContain(`127.0.0.1:${daemon.port}`);
		expect(argv).toContain("--no-receive-stdout");
	});

	it("sends a message through JSON-RPC with recipient and message params", async () => {
		const daemon = await startFakeDaemon();
		const transport = await startSignalDaemon({
			account: "+821012345678",
			dataDir: workspace(),
			host: "127.0.0.1",
			port: daemon.port,
			spawnProcess: fakeSpawn().spawn,
		});
		transports.push(transport);
		await transport.sendMessage("+821087654321", "hello peer");
		const send = daemon.rpcCalls.find((call) => call.method === "send");
		expect(send).toBeDefined();
		expect(send!.params).toMatchObject({
			recipient: ["+821087654321"],
			message: "hello peer",
		});
	});

	it("delivers incoming data messages from the event stream to subscribers", async () => {
		const daemon = await startFakeDaemon();
		const transport = await startSignalDaemon({
			account: "+821012345678",
			dataDir: workspace(),
			host: "127.0.0.1",
			port: daemon.port,
			spawnProcess: fakeSpawn().spawn,
		});
		transports.push(transport);
		const received: SignalIncomingMessage[] = [];
		transport.onMessage((message) => received.push(message));
		// allow the events subscription to connect
		await new Promise((resolve) => setTimeout(resolve, 50));
		daemon.pushSseEvent({
			jsonrpc: "2.0",
			method: "receive",
			params: {
				envelope: {
					source: "+821087654321",
					sourceNumber: "+821087654321",
					sourceUuid: "11111111-2222-3333-4444-555555555555",
					timestamp: 1700000000000,
					dataMessage: { timestamp: 1700000000000, message: "ping from peer", expiresInSeconds: 0 },
				},
				account: "+821012345678",
			},
		});
		await expect.poll(() => received.length, { timeout: 2000, interval: 10 }).toBe(1);
		expect(received[0]).toMatchObject({
			source: "+821087654321",
			sourceUuid: "11111111-2222-3333-4444-555555555555",
			message: "ping from peer",
			timestamp: 1700000000000,
		});
	});

	it("ignores non-data envelopes such as receipts and typing indicators", async () => {
		const daemon = await startFakeDaemon();
		const transport = await startSignalDaemon({
			account: "+821012345678",
			dataDir: workspace(),
			host: "127.0.0.1",
			port: daemon.port,
			spawnProcess: fakeSpawn().spawn,
		});
		transports.push(transport);
		const received: SignalIncomingMessage[] = [];
		transport.onMessage((message) => received.push(message));
		await new Promise((resolve) => setTimeout(resolve, 50));
		daemon.pushSseEvent({
			jsonrpc: "2.0",
			method: "receive",
			params: {
				envelope: {
					source: "+821087654321",
					timestamp: 1700000000000,
					typingMessage: { action: "STARTED", timestamp: 1700000000000 },
				},
				account: "+821012345678",
			},
		});
		await new Promise((resolve) => setTimeout(resolve, 100));
		expect(received).toHaveLength(0);
	});

	it("surfaces JSON-RPC errors as rejected promises", async () => {
		const server = createServer((req, res) => {
			if (req.method === "GET" && req.url === "/api/v1/events") {
				res.writeHead(200, { "content-type": "text/event-stream" });
				return;
			}
			const chunks: Buffer[] = [];
			req.on("data", (chunk: Buffer) => chunks.push(chunk));
			req.on("end", () => {
				const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as { id: number };
				res.writeHead(200, { "content-type": "application/json" });
				res.end(
					JSON.stringify({
						jsonrpc: "2.0",
						id: body.id,
						error: { code: -32602, message: "Unregistered user" },
					}),
				);
			});
		});
		servers.push(server);
		await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
		const address = server.address();
		if (address === null || typeof address === "string") throw new Error("error daemon failed to bind");
		const transport = await startSignalDaemon({
			account: "+821012345678",
			dataDir: workspace(),
			host: "127.0.0.1",
			port: address.port,
			spawnProcess: fakeSpawn().spawn,
		});
		transports.push(transport);
		await expect(transport.sendMessage("+821087654321", "hi")).rejects.toThrow(/Unregistered user/);
	});
});

describe("registerAccount helpers", () => {
	it("spawns a one-shot register command with the phone number", async () => {
		const { registerAccount } = await import("../../src/p2p/signal-transport.ts");
		const { calls, spawn } = fakeSpawn();
		await registerAccount({ number: "+821012345678", dataDir: workspace(), spawnProcess: spawn });
		expect(calls).toHaveLength(1);
		const argv = calls[0]!;
		expect(argv).toContain("register");
		expect(argv).toContain("+821012345678");
	});

	it("spawns verify with the SMS/voice code", async () => {
		const { verifyAccount } = await import("../../src/p2p/signal-transport.ts");
		const { calls, spawn } = fakeSpawn();
		await verifyAccount({
			number: "+821012345678",
			code: "123-456",
			dataDir: workspace(),
			spawnProcess: spawn,
		});
		const argv = calls[0]!;
		expect(argv).toContain("verify");
		expect(argv).toContain("123-456");
	});
});
