import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";

/** One Bot API call the bridge made, in arrival order. */
export interface FakeTelegramCall {
	readonly method: string;
	readonly body: Record<string, unknown>;
}

interface PendingPoll {
	readonly resolve: (updates: unknown[]) => void;
	readonly timer: NodeJS.Timeout;
}

/** The owner account every fake private chat belongs to. */
export const FAKE_TELEGRAM_USER = { id: 424242, is_bot: false, first_name: "Owner", username: "owner" } as const;
export const FAKE_TELEGRAM_BOT = {
	id: 777000,
	is_bot: true,
	first_name: "AutoRAG QA",
	username: "autorag_qa_bot",
} as const;

/** Methods the bridge may call that need no state; they answer `{ok:true,result:true}`. */
const ACK_METHODS = new Set([
	"setMyCommands",
	"deleteMyCommands",
	"sendChatAction",
	"setMessageReaction",
	"answerCallbackQuery",
	"deleteWebhook",
	"setChatMenuButton",
	"deleteMessage",
	"deleteMessages",
	"pinChatMessage",
	"unpinChatMessage",
]);

/**
 * A local Telegram Bot API. Tests act as the phone: {@link say} sends a message from the
 * paired owner and {@link replies} reads what the bot sent back. No network, no real bot.
 */
export class FakeTelegramApi {
	readonly calls: FakeTelegramCall[] = [];
	private readonly server: Server;
	private updateId = 1000;
	private messageId = 1;
	/** Updates not yet acknowledged by a `getUpdates` offset, as Telegram keeps them. */
	private readonly queued: { update_id: number }[] = [];
	private poll: PendingPoll | undefined;
	private readonly sent: { chat_id: unknown; message_id: number; text: string; edited: boolean }[] = [];

	private constructor(server: Server) {
		this.server = server;
	}

	static async start(): Promise<FakeTelegramApi> {
		const holder: { api?: FakeTelegramApi } = {};
		const server = createServer((req, res) => {
			void holder.api?.handle(req).then(
				(payload) => {
					res.setHeader("content-type", "application/json");
					res.end(JSON.stringify(payload));
				},
				(error: unknown) => {
					res.statusCode = 500;
					res.end(String(error));
				},
			);
		});
		await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
		holder.api = new FakeTelegramApi(server);
		return holder.api;
	}

	get origin(): string {
		return `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
	}

	/** Messages and edits the bot sent, collapsed to the latest text of each message. */
	get replies(): readonly string[] {
		return this.sent.map((message) => message.text);
	}

	/** The owner types `text` in the bot's private chat. */
	say(text: string): void {
		this.push({
			message: {
				message_id: this.messageId++,
				date: Math.floor(Date.now() / 1000),
				chat: { id: FAKE_TELEGRAM_USER.id, type: "private", first_name: "Owner" },
				from: FAKE_TELEGRAM_USER,
				text,
				...(text.startsWith("/")
					? { entities: [{ type: "bot_command", offset: 0, length: text.split(/\s/u)[0]?.length ?? text.length }] }
					: {}),
			},
		});
	}

	methodsCalled(): readonly string[] {
		return this.calls.map((call) => call.method);
	}

	async stop(): Promise<void> {
		if (this.poll !== undefined) {
			clearTimeout(this.poll.timer);
			this.poll.resolve([]);
			this.poll = undefined;
		}
		this.server.closeAllConnections();
		await new Promise<void>((resolve) => this.server.close(() => resolve()));
	}

	private push(update: Record<string, unknown>): void {
		this.queued.push({ update_id: this.updateId++, ...update });
		if (this.poll !== undefined) {
			const { resolve, timer } = this.poll;
			this.poll = undefined;
			clearTimeout(timer);
			resolve(this.queued.slice());
		}
	}

	private async handle(req: IncomingMessage): Promise<unknown> {
		const chunks: Buffer[] = [];
		for await (const chunk of req) chunks.push(chunk as Buffer);
		const raw = Buffer.concat(chunks).toString("utf8");
		const method = (req.url ?? "").split("/").pop() ?? "";
		let body: Record<string, unknown> = {};
		try {
			body = raw.length > 0 ? (JSON.parse(raw) as Record<string, unknown>) : {};
		} catch {
			// multipart uploads are recorded by method only
		}
		this.calls.push({ method, body });
		switch (method) {
			case "getMe":
				return { ok: true, result: { ...FAKE_TELEGRAM_BOT, can_join_groups: true, has_topics_enabled: false } };
			case "getUpdates":
				return {
					ok: true,
					result: await this.nextUpdates(Number(body.offset ?? 0), Number(body.timeout ?? 0)),
				};
			case "sendMessage":
			case "sendMessageDraft":
			case "sendRichMessage": {
				const id = this.messageId++;
				const rich = body.rich_message as { markdown?: string } | undefined;
				const text = String(rich?.markdown ?? body.text ?? "");
				this.sent.push({ chat_id: body.chat_id, message_id: id, text, edited: false });
				return {
					ok: true,
					result: {
						message_id: id,
						date: Math.floor(Date.now() / 1000),
						chat: { id: body.chat_id, type: "private" },
						from: FAKE_TELEGRAM_BOT,
						text,
					},
				};
			}
			case "editMessageText":
			case "editRichMessage": {
				const target = this.sent.find((message) => message.message_id === Number(body.message_id));
				if (target !== undefined) {
					const rich = body.rich_message as { markdown?: string } | undefined;
					target.text = String(rich?.markdown ?? body.text ?? "");
					target.edited = true;
				}
				return {
					ok: true,
					result: {
						message_id: body.message_id,
						date: Math.floor(Date.now() / 1000),
						chat: { id: body.chat_id, type: "private" },
						from: FAKE_TELEGRAM_BOT,
						text: body.text,
					},
				};
			}
			default:
				if (ACK_METHODS.has(method)) return { ok: true, result: true };
				return { ok: true, result: true };
		}
	}

	/** Telegram's `getUpdates`: a negative offset peeks the newest update; a positive one acknowledges everything below it. */
	private nextUpdates(offset: number, timeoutSeconds: number): Promise<unknown[]> {
		if (offset < 0) return Promise.resolve(this.queued.slice(offset));
		while (this.queued.length > 0 && (this.queued[0]?.update_id ?? offset) < offset) this.queued.shift();
		if (this.queued.length > 0) return Promise.resolve(this.queued.slice());
		if (timeoutSeconds <= 0) return Promise.resolve([]);
		return new Promise<unknown[]>((resolve) => {
			const timer = setTimeout(
				() => {
					this.poll = undefined;
					resolve([]);
				},
				Math.min(Math.max(timeoutSeconds, 1), 3) * 1000,
			);
			this.poll = { resolve, timer };
		});
	}
}
