import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { createRequire } from "node:module";
import type { AddressInfo } from "node:net";
import { type WebSocket, WebSocketServer } from "ws";

// discord.js is CJS and spreads the CJS build of @discordjs/rest's defaults; the ESM build is a different object.
const { DefaultRestOptions } = createRequire(import.meta.url)("@discordjs/rest") as {
	DefaultRestOptions: { api: string };
};

/** A message the bot posted to a channel of the fake guild. */
export interface FakePostedMessage {
	readonly channelId: string;
	readonly content: string;
}

export interface FakeDiscordOptions {
	readonly guildId: string;
	readonly botToken: string;
}

/**
 * A local stand-in for Discord's REST API and gateway, just deep enough for
 * discord.js (as used by `pi-discord-remote`) to log in, create and delete a
 * session channel, receive user messages, and post replies. No network egress.
 */
export class FakeDiscord {
	readonly posted: FakePostedMessage[] = [];
	readonly reactions: { channelId: string; messageId: string; emoji: string }[] = [];
	readonly channels = new Map<string, { id: string; name: string; topic?: string }>();
	readonly deletedChannelIds: string[] = [];
	identifiedWithToken: string | undefined;
	private readonly http: Server;
	private readonly wss = new WebSocketServer({ noServer: true });
	private readonly sockets = new Set<WebSocket>();
	private sequence = 0;
	private nextId = 1000;
	private previousApi: string | undefined;

	private readonly options: FakeDiscordOptions;

	private constructor(options: FakeDiscordOptions) {
		this.options = options;
		this.http = createServer((req, res) => void this.handle(req, res));
		this.http.on("upgrade", (req, socket, head) => {
			this.wss.handleUpgrade(req, socket, head, (ws) => this.onSocket(ws));
		});
	}

	static async start(options: FakeDiscordOptions): Promise<FakeDiscord> {
		const fake = new FakeDiscord(options);
		await new Promise<void>((resolve) => fake.http.listen(0, "127.0.0.1", resolve));
		// discord.js spreads DefaultRestOptions at Client construction: point it at the fake.
		fake.previousApi = DefaultRestOptions.api;
		DefaultRestOptions.api = `http://127.0.0.1:${fake.port}/api`;
		return fake;
	}

	get port(): number {
		return (this.http.address() as AddressInfo).port;
	}

	/** The id of the channel the extension created for the session, once it exists. */
	get sessionChannelId(): string | undefined {
		return [...this.channels.keys()].find((id) => !this.deletedChannelIds.includes(id));
	}

	/** Delivers a message from a human user to a channel, as the gateway would. */
	sendUserMessage(channelId: string, userId: string, content: string): string {
		const id = String(this.nextId++);
		this.dispatch("MESSAGE_CREATE", {
			id,
			channel_id: channelId,
			guild_id: this.options.guildId,
			content,
			author: { id: userId, username: `user${userId}`, discriminator: "0", bot: false },
			timestamp: new Date().toISOString(),
			type: 0,
			attachments: [],
			embeds: [],
			mentions: [],
			mention_roles: [],
			pinned: false,
			tts: false,
			mention_everyone: false,
			edited_timestamp: null,
			flags: 0,
			components: [],
		});
		return id;
	}

	/** Resolves once the bot has posted a message matching `predicate`. */
	async waitForPost(
		predicate: (message: FakePostedMessage) => boolean,
		timeoutMs = 20_000,
	): Promise<FakePostedMessage> {
		const deadline = Date.now() + timeoutMs;
		while (Date.now() < deadline) {
			const hit = this.posted.find(predicate);
			if (hit !== undefined) return hit;
			await new Promise((resolve) => setTimeout(resolve, 25));
		}
		throw new Error(`fake discord: no matching post within ${timeoutMs}ms; saw ${JSON.stringify(this.posted)}`);
	}

	async waitFor(condition: () => boolean, label: string, timeoutMs = 20_000): Promise<void> {
		const deadline = Date.now() + timeoutMs;
		while (Date.now() < deadline) {
			if (condition()) return;
			await new Promise((resolve) => setTimeout(resolve, 25));
		}
		throw new Error(`fake discord: timed out waiting for ${label}`);
	}

	async close(): Promise<void> {
		if (this.previousApi !== undefined) DefaultRestOptions.api = this.previousApi;
		for (const socket of this.sockets) socket.terminate();
		this.wss.close();
		await new Promise<void>((resolve) => this.http.close(() => resolve()));
	}

	private send(socket: WebSocket, payload: Record<string, unknown>): void {
		socket.send(JSON.stringify(payload));
	}

	private dispatch(type: string, data: unknown): void {
		this.sequence += 1;
		for (const socket of this.sockets) this.send(socket, { op: 0, t: type, s: this.sequence, d: data });
	}

	private onSocket(socket: WebSocket): void {
		this.sockets.add(socket);
		socket.on("close", () => this.sockets.delete(socket));
		this.send(socket, { op: 10, d: { heartbeat_interval: 30_000 } });
		socket.on("message", (raw) => {
			const frame = JSON.parse(String(raw)) as { op: number; d?: { token?: string } };
			if (frame.op === 1) this.send(socket, { op: 11 });
			if (frame.op !== 2) return;
			this.identifiedWithToken = frame.d?.token;
			this.sequence += 1;
			this.send(socket, {
				op: 0,
				t: "READY",
				s: this.sequence,
				d: {
					v: 10,
					user: { id: "900", username: "fakebot", discriminator: "0", bot: true },
					guilds: [],
					session_id: "fake-session",
					resume_gateway_url: `ws://127.0.0.1:${this.port}`,
					application: { id: "900", flags: 0 },
					shard: [0, 1],
				},
			});
		});
	}

	private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
		const url = new URL(req.url ?? "/", "http://fake");
		const path = url.pathname.replace(/^\/api\/v\d+/, "").replace(/^\/api/, "");
		const body = await readBody(req);
		const json = (value: unknown, status = 200): void => {
			res.writeHead(status, { "content-type": "application/json" });
			res.end(JSON.stringify(value));
		};
		if (req.headers.authorization !== `Bot ${this.options.botToken}` && path !== "/gateway") {
			json({ message: "401: Unauthorized", code: 0 }, 401);
			return;
		}
		if (req.method === "GET" && path === "/gateway/bot") {
			json({
				url: `ws://127.0.0.1:${this.port}`,
				shards: 1,
				session_start_limit: { total: 1000, remaining: 1000, reset_after: 0, max_concurrency: 1 },
			});
			return;
		}
		if (req.method === "GET" && path === `/guilds/${this.options.guildId}`) {
			json({ id: this.options.guildId, name: "fake-guild", channels: [], roles: [], emojis: [], features: [] });
			return;
		}
		if (req.method === "POST" && path === `/guilds/${this.options.guildId}/channels`) {
			const input = JSON.parse(body) as { name: string; topic?: string };
			const id = String(this.nextId++);
			this.channels.set(id, { id, name: input.name, ...(input.topic === undefined ? {} : { topic: input.topic }) });
			json({
				id,
				type: 0,
				guild_id: this.options.guildId,
				name: input.name,
				topic: input.topic ?? null,
				position: 0,
				permission_overwrites: [],
			});
			return;
		}
		const channelMatch = /^\/channels\/(\d+)$/.exec(path);
		if (channelMatch !== null && req.method === "GET") {
			const channel = this.channels.get(channelMatch[1] ?? "");
			if (channel === undefined || this.deletedChannelIds.includes(channel.id)) {
				json({ message: "Unknown Channel", code: 10003 }, 404);
				return;
			}
			json({
				id: channel.id,
				type: 0,
				guild_id: this.options.guildId,
				name: channel.name,
				position: 0,
				permission_overwrites: [],
			});
			return;
		}
		if (channelMatch !== null && req.method === "DELETE") {
			const id = channelMatch[1] ?? "";
			this.deletedChannelIds.push(id);
			json({ id, type: 0, guild_id: this.options.guildId, name: this.channels.get(id)?.name ?? "" });
			return;
		}
		const postMatch = /^\/channels\/(\d+)\/messages$/.exec(path);
		if (postMatch !== null && req.method === "POST") {
			const input = JSON.parse(body) as { content?: string };
			const channelId = postMatch[1] ?? "";
			this.posted.push({ channelId, content: input.content ?? "" });
			json({
				id: String(this.nextId++),
				channel_id: channelId,
				content: input.content ?? "",
				author: { id: "900", username: "fakebot", discriminator: "0", bot: true },
				timestamp: new Date().toISOString(),
				type: 0,
				attachments: [],
				embeds: [],
				mentions: [],
				mention_roles: [],
				pinned: false,
				tts: false,
				mention_everyone: false,
				edited_timestamp: null,
				flags: 0,
				components: [],
			});
			return;
		}
		const reactionMatch = /^\/channels\/(\d+)\/messages\/(\d+)\/reactions\/([^/]+)\/@me$/.exec(path);
		if (reactionMatch !== null && req.method === "PUT") {
			this.reactions.push({
				channelId: reactionMatch[1] ?? "",
				messageId: reactionMatch[2] ?? "",
				emoji: decodeURIComponent(reactionMatch[3] ?? ""),
			});
			res.writeHead(204);
			res.end();
			return;
		}
		json({ message: `fake discord: unhandled ${req.method} ${path}`, code: 0 }, 404);
	}
}

function readBody(req: IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		req.on("data", (chunk: Buffer) => chunks.push(chunk));
		req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
		req.on("error", reject);
	});
}
