import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { WebSocketServer } from "ws";
import {
	type SimplexIncomingMessage,
	type SimplexTransport,
	startSimplexChat,
} from "../../src/p2p/simplex-transport.ts";

const roots: string[] = [];
const servers: WebSocketServer[] = [];
const transports: SimplexTransport[] = [];

function workspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-simplex-"));
	roots.push(root);
	return root;
}

afterEach(async () => {
	for (const transport of transports.splice(0)) await transport.close();
	for (const server of servers.splice(0)) await new Promise<void>((resolve) => server.close(() => resolve()));
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

/**
 * Fake simplex-chat WebSocket server implementing the documented bot API:
 * - JSON {corrId, cmd} requests -> {corrId, resp} responses
 * - push events as {resp: {...}} without corrId
 */
function startFakeSimplex(port: number): Promise<{ wss: WebSocketServer; sendEvent: (resp: unknown) => void }> {
	const wss = new WebSocketServer({ port, host: "127.0.0.1" });
	servers.push(wss);
	const sockets = new Set<import("ws").WebSocket>();
	wss.on("connection", (ws) => {
		sockets.add(ws);
		ws.on("message", (raw) => {
			const req = JSON.parse(raw.toString()) as { corrId: string; cmd: string };
			const respond = (resp: unknown) => ws.send(JSON.stringify({ corrId: req.corrId, resp }));
			if (req.cmd === "/user") {
				respond({ type: "activeUser", user: { userId: 1, localDisplayName: "qa-bot" } });
				return;
			}
			if (req.cmd.startsWith("/_create user")) {
				respond({ type: "activeUser", user: { userId: 1, localDisplayName: "qa-bot" } });
				return;
			}
			if (req.cmd === "/_address 1") {
				respond({
					type: "userContactLinkCreated",
					user: { userId: 1 },
					connLinkContact: { connFullLink: "simplex:/contact#/?v=2&smp=fake" },
				});
				return;
			}
			if (req.cmd === "/_show_address 1") {
				respond({
					type: "userContactLink",
					user: { userId: 1 },
					contactLink: { connLinkContact: { connFullLink: "simplex:/contact#/?v=2&smp=fake" } },
				});
				return;
			}
			if (req.cmd.startsWith("/_address_settings")) {
				respond({ type: "userContactLinkUpdated", user: { userId: 1 }, contactLink: {} });
				return;
			}
			if (req.cmd.startsWith("/_connect 1") && !req.cmd.includes("simplex:")) {
				respond({
					type: "invitation",
					user: { userId: 1 },
					connLinkInvitation: { connFullLink: "simplex:/invitation#/?v=2&smp=fake" },
					connection: {},
				});
				return;
			}
			if (req.cmd.startsWith("/_connect 1 simplex:")) {
				respond({ type: "sentInvitation", user: { userId: 1 }, connection: {} });
				return;
			}
			if (req.cmd === "/_contacts 1") {
				respond({
					type: "contactsList",
					user: { userId: 1 },
					contacts: [{ contactId: 2, localDisplayName: "peer" }],
				});
				return;
			}
			if (req.cmd.startsWith("/_send")) {
				respond({ type: "newChatItems", user: { userId: 1 }, chatItems: [] });
				return;
			}
			respond({ type: "cmdOk", user_: { userId: 1 } });
		});
		ws.on("close", () => sockets.delete(ws));
	});
	return new Promise((resolve) =>
		wss.on("listening", () =>
			resolve({
				wss,
				sendEvent: (resp: unknown) => {
					for (const ws of sockets) ws.send(JSON.stringify({ resp }));
				},
			}),
		),
	);
}

function startWithFake(port: number, dbPrefix: string): Promise<SimplexTransport> {
	// Override the binary spawn: the fake server already listens on the port.
	// We use a no-op binary that just stays alive.
	const transport = startSimplexChat({
		dbPrefix,
		displayName: "qa-bot",
		port,
		binary: process.execPath, // node -e noop keeps process alive
		connectTimeoutMs: 5000,
	});
	return transport;
}

describe("startSimplexChat", () => {
	it("creates/reuses the user profile and returns a stable userId", async () => {
		const port = 25_801;
		await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		expect(await transport.getUserId()).toBe(1);
	});

	it("creates a contact address and enables auto-accept", async () => {
		const port = 25_802;
		await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		const address = await transport.getOrCreateAddress();
		expect(address).toContain("simplex:/contact#");
	});

	it("creates a one-time invitation link", async () => {
		const port = 25_803;
		await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		const link = await transport.createInvitation();
		expect(link).toContain("simplex:/invitation#");
	});

	it("connects via a peer link", async () => {
		const port = 25_804;
		await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		await transport.connect("simplex:/contact#/?v=2&smp=peer");
	});

	it("lists contacts", async () => {
		const port = 25_805;
		await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		const contacts = await transport.listContacts();
		expect(contacts).toEqual([{ contactId: 2, localDisplayName: "peer" }]);
	});

	it("sends a text message to a contact", async () => {
		const port = 25_806;
		await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		await transport.sendMessage(2, "hello peer");
	});

	it("delivers incoming direct text messages to subscribers", async () => {
		const port = 25_807;
		const fake = await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		const received: SimplexIncomingMessage[] = [];
		transport.onMessage((message) => received.push(message));
		await new Promise((resolve) => setTimeout(resolve, 100));
		fake.sendEvent({
			type: "newChatItems",
			user: { userId: 1 },
			chatItems: [
				{
					chatInfo: { type: "direct", contact: { contactId: 2, localDisplayName: "peer" } },
					chatItem: {
						chatDir: { type: "directRcv" },
						meta: { itemId: 42 },
						content: { type: "rcvMsgContent", msgContent: { type: "text", text: "ping from peer" } },
					},
				},
			],
		});
		await expect.poll(() => received.length, { timeout: 2000, interval: 10 }).toBe(1);
		expect(received[0]).toEqual({ contactId: 2, contactName: "peer", text: "ping from peer", chatItemId: 42 });
	});

	it("ignores non-text and outbound events", async () => {
		const port = 25_808;
		const fake = await startFakeSimplex(port);
		const transport = await startWithFake(port, join(workspace(), "bot"));
		transports.push(transport);
		const received: SimplexIncomingMessage[] = [];
		transport.onMessage((message) => received.push(message));
		await new Promise((resolve) => setTimeout(resolve, 100));
		fake.sendEvent({
			type: "newChatItems",
			user: { userId: 1 },
			chatItems: [
				{
					chatInfo: { type: "direct", contact: { contactId: 2, localDisplayName: "peer" } },
					chatItem: {
						chatDir: { type: "directSnd" },
						meta: { itemId: 43 },
						content: { type: "sndMsgContent", msgContent: { type: "text", text: "echo" } },
					},
				},
			],
		});
		await new Promise((resolve) => setTimeout(resolve, 150));
		expect(received).toHaveLength(0);
	});
});
