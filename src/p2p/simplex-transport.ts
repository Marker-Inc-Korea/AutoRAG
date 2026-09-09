import { type ChildProcess, spawn } from "node:child_process";

/**
 * Thin adapter over the simplex-chat CLI (AGPLv3), spawned as a subprocess
 * and driven through its documented WebSocket bot API (`simplex-chat -p`).
 * The CLI is never linked; all interaction crosses the WebSocket boundary.
 */

export interface SimplexIncomingMessage {
	readonly contactId: number;
	readonly contactName: string;
	readonly text: string;
	readonly chatItemId: number;
}

export interface SimplexTransport {
	readonly dbPrefix: string;
	readonly displayName: string;
	getUserId(): Promise<number>;
	getOrCreateAddress(): Promise<string>;
	createInvitation(): Promise<string>;
	connect(link: string): Promise<void>;
	listContacts(): Promise<{ contactId: number; localDisplayName: string }[]>;
	sendMessage(contactId: number, text: string): Promise<void>;
	onMessage(handler: (message: SimplexIncomingMessage) => void): () => void;
	close(): Promise<void>;
}

export interface StartSimplexOptions {
	readonly dbPrefix: string;
	readonly displayName: string;
	readonly binary?: string;
	readonly port?: number;
	readonly connectTimeoutMs?: number;
}

interface WsRequest {
	readonly corrId: string;
	readonly cmd: string;
}

interface WsResponse {
	readonly corrId: string;
	readonly resp: unknown;
}

interface WsEvent {
	readonly resp: unknown;
}

class SimplexError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "SimplexError";
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function responseType(payload: unknown): string | undefined {
	if (!isRecord(payload)) return undefined;
	return typeof payload.type === "string" ? payload.type : undefined;
}

/** Minimal shape extraction from the auto-generated API types. */
function contactOf(payload: unknown): { contactId: number; localDisplayName: string } | undefined {
	if (!isRecord(payload)) return undefined;
	const contact = payload.contact;
	if (!isRecord(contact)) return undefined;
	if (typeof contact.contactId !== "number" || typeof contact.localDisplayName !== "string") return undefined;
	return { contactId: contact.contactId, localDisplayName: contact.localDisplayName };
}

function userIdOf(payload: unknown): number | undefined {
	if (!isRecord(payload)) return undefined;
	const user = payload.user;
	if (!isRecord(user) || typeof user.userId !== "number") return undefined;
	return user.userId;
}

function extractTextFromChatItem(item: unknown): { chatItemId: number; text: string } | undefined {
	if (!isRecord(item)) return undefined;
	const chatItem = item.chatItem;
	if (!isRecord(chatItem)) return undefined;
	const meta = chatItem.meta;
	const content = chatItem.content;
	if (!isRecord(meta) || !isRecord(content)) return undefined;
	if (typeof meta.itemId !== "number") return undefined;
	// CIContent discriminated union: { type: "rcvMsgContent", msgContent: { type: "text", text } }
	const msgContent = content.msgContent;
	if (!isRecord(msgContent)) return undefined;
	if (msgContent.type !== "text" || typeof msgContent.text !== "string") return undefined;
	return { chatItemId: meta.itemId, text: msgContent.text };
}

async function waitForPort(port: number, proc: ChildProcess, timeoutMs: number): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	for (;;) {
		if (proc.exitCode !== null) {
			throw new SimplexError(`simplex-chat exited with code ${proc.exitCode} before the WebSocket server started`);
		}
		try {
			const ws = new WebSocket(`ws://127.0.0.1:${port}`);
			await new Promise<void>((resolve, reject) => {
				ws.addEventListener("open", () => {
					ws.close();
					resolve();
				});
				ws.addEventListener("error", () => reject(new Error("connect failed")));
			});
			return;
		} catch {
			if (Date.now() >= deadline) {
				throw new SimplexError(`simplex-chat WebSocket server did not start within ${timeoutMs}ms`);
			}
			await new Promise((resolve) => setTimeout(resolve, 150));
		}
	}
}

class SimplexClient implements SimplexTransport {
	readonly dbPrefix: string;
	readonly displayName: string;
	private readonly proc: ChildProcess;
	private readonly ws: WebSocket;
	private nextCorrId = 1;
	private readonly pending = new Map<string, { resolve: (resp: unknown) => void; reject: (error: Error) => void }>();
	private readonly messageHandlers: ((message: SimplexIncomingMessage) => void)[] = [];
	private closed = false;
	private userId: number | undefined;

	constructor(dbPrefix: string, displayName: string, proc: ChildProcess, ws: WebSocket) {
		this.dbPrefix = dbPrefix;
		this.displayName = displayName;
		this.proc = proc;
		this.ws = ws;
		ws.addEventListener("message", (event: MessageEvent) => {
			const data = typeof event.data === "string" ? event.data : Buffer.from(event.data as ArrayBuffer).toString();
			this.dispatch(data);
		});
		ws.addEventListener("close", () => this.failAllPending(new SimplexError("simplex-chat WebSocket closed")));
		ws.addEventListener("error", () => {
			/* close handler covers it */
		});
	}

	private dispatch(raw: string): void {
		let parsed: unknown;
		try {
			parsed = JSON.parse(raw);
		} catch {
			return;
		}
		if (!isRecord(parsed) || !isRecord(parsed.resp)) return;
		const resp = parsed.resp;
		if (typeof parsed.corrId === "string") {
			const entry = this.pending.get(parsed.corrId);
			if (entry !== undefined) {
				this.pending.delete(parsed.corrId);
				const type = responseType(resp);
				if (type === "chatCmdError" || type === "chatError") {
					entry.reject(new SimplexError(`simplex-chat command error: ${JSON.stringify(resp).slice(0, 300)}`));
				} else {
					entry.resolve(resp);
				}
			}
			return;
		}
		this.handleEvent(resp);
	}

	private handleEvent(resp: unknown): void {
		if (responseType(resp) !== "newChatItems" || !isRecord(resp)) return;
		const chatItems = resp.chatItems;
		if (!Array.isArray(chatItems)) return;
		for (const item of chatItems) {
			if (!isRecord(item)) continue;
			const chatInfo = item.chatInfo;
			if (!isRecord(chatInfo) || chatInfo.type !== "direct") continue;
			const contact = contactOf(chatInfo);
			const extracted = extractTextFromChatItem(item);
			if (contact === undefined || extracted === undefined) continue;
			// Only inbound messages (rcv direction)
			const chatItem = item.chatItem;
			if (!isRecord(chatItem)) continue;
			const chatDir = chatItem.chatDir;
			if (!isRecord(chatDir) || chatDir.type !== "directRcv") continue;
			const message: SimplexIncomingMessage = {
				contactId: contact.contactId,
				contactName: contact.localDisplayName,
				text: extracted.text,
				chatItemId: extracted.chatItemId,
			};
			for (const handler of this.messageHandlers) {
				try {
					handler(message);
				} catch {
					// subscriber errors must not break the event loop
				}
			}
		}
	}

	private cmd(command: string): Promise<unknown> {
		if (this.closed) return Promise.reject(new SimplexError("transport is closed"));
		const corrId = String(this.nextCorrId++);
		return new Promise((resolve, reject) => {
			this.pending.set(corrId, { resolve, reject });
			const request: WsRequest = { corrId, cmd: command };
			try {
				this.ws.send(JSON.stringify(request));
			} catch (error) {
				this.pending.delete(corrId);
				reject(
					new SimplexError(`WebSocket send failed: ${error instanceof Error ? error.message : String(error)}`),
				);
			}
		});
	}

	private failAllPending(error: Error): void {
		for (const entry of this.pending.values()) entry.reject(error);
		this.pending.clear();
	}

	private async requireUserId(): Promise<number> {
		if (this.userId !== undefined) return this.userId;
		// Try existing active user first.
		const active = await this.cmd("/user");
		if (responseType(active) === "activeUser") {
			const id = userIdOf(active);
			if (id !== undefined) {
				this.userId = id;
				return id;
			}
		}
		// Create the profile.
		const created = await this.cmd(
			`/_create user ${JSON.stringify({
				profile: { displayName: this.displayName, fullName: "" },
				pastTimestamp: false,
				userChatRelay: false,
				clientService: false,
			})}`,
		);
		if (responseType(created) !== "activeUser") {
			throw new SimplexError(`failed to create SimpleX user profile: ${JSON.stringify(created).slice(0, 300)}`);
		}
		const id = userIdOf(created);
		if (id === undefined) throw new SimplexError("simplex-chat did not return a user id");
		this.userId = id;
		return id;
	}

	async getUserId(): Promise<number> {
		return this.requireUserId();
	}

	async getOrCreateAddress(): Promise<string> {
		const userId = await this.requireUserId();
		try {
			const existing = await this.cmd(`/_show_address ${userId}`);
			if (responseType(existing) === "userContactLink" && isRecord(existing)) {
				const link = existing.contactLink;
				if (isRecord(link) && isRecord(link.connLinkContact)) {
					const full = link.connLinkContact.connFullLink;
					if (typeof full === "string") return full;
				}
			}
		} catch {
			// No address yet — create one below.
		}
		const created = await this.cmd(`/_address ${userId}`);
		if (responseType(created) !== "userContactLinkCreated" || !isRecord(created)) {
			throw new SimplexError(`failed to create SimpleX address: ${JSON.stringify(created).slice(0, 300)}`);
		}
		const link = created.connLinkContact;
		if (!isRecord(link) || typeof link.connFullLink !== "string") {
			throw new SimplexError("simplex-chat did not return a contact address");
		}
		// Enable auto-accept so peers can connect without manual approval.
		await this.cmd(
			`/_address_settings ${userId} ${JSON.stringify({
				businessAddress: false,
				autoAccept: { acceptIncognito: false },
			})}`,
		);
		return link.connFullLink;
	}

	async createInvitation(): Promise<string> {
		const userId = await this.requireUserId();
		const resp = await this.cmd(`/_connect ${userId}`);
		if (responseType(resp) !== "invitation" || !isRecord(resp)) {
			throw new SimplexError(`failed to create invitation: ${JSON.stringify(resp).slice(0, 300)}`);
		}
		const link = resp.connLinkInvitation;
		if (!isRecord(link) || typeof link.connFullLink !== "string") {
			throw new SimplexError("simplex-chat did not return an invitation link");
		}
		return link.connFullLink;
	}

	async connect(link: string): Promise<void> {
		const userId = await this.requireUserId();
		const resp = await this.cmd(`/_connect ${userId} ${link}`);
		const type = responseType(resp);
		if (type === "sentInvitation" || type === "sentConfirmation" || type === "contactAlreadyExists") return;
		throw new SimplexError(`failed to connect via link: ${JSON.stringify(resp).slice(0, 300)}`);
	}

	async listContacts(): Promise<{ contactId: number; localDisplayName: string }[]> {
		const userId = await this.requireUserId();
		const resp = await this.cmd(`/_contacts ${userId}`);
		if (responseType(resp) !== "contactsList" || !isRecord(resp) || !Array.isArray(resp.contacts)) {
			throw new SimplexError(`failed to list contacts: ${JSON.stringify(resp).slice(0, 300)}`);
		}
		const contacts: { contactId: number; localDisplayName: string }[] = [];
		for (const entry of resp.contacts) {
			const contact = contactOf({ contact: entry });
			if (contact !== undefined) contacts.push(contact);
		}
		return contacts;
	}

	async sendMessage(contactId: number, text: string): Promise<void> {
		const resp = await this.cmd(
			`/_send @${contactId} json ${JSON.stringify([{ msgContent: { type: "text", text }, mentions: {} }])}`,
		);
		if (responseType(resp) !== "newChatItems") {
			throw new SimplexError(`failed to send message: ${JSON.stringify(resp).slice(0, 300)}`);
		}
	}

	onMessage(handler: (message: SimplexIncomingMessage) => void): () => void {
		this.messageHandlers.push(handler);
		return () => {
			const index = this.messageHandlers.indexOf(handler);
			if (index !== -1) this.messageHandlers.splice(index, 1);
		};
	}

	async close(): Promise<void> {
		if (this.closed) return;
		this.closed = true;
		this.failAllPending(new SimplexError("transport closed"));
		try {
			this.ws.close();
		} catch {
			/* already closed */
		}
		this.proc.kill("SIGTERM");
		await new Promise<void>((resolve) => {
			const timer = setTimeout(() => {
				this.proc.kill("SIGKILL");
				resolve();
			}, 3000);
			this.proc.once("exit", () => {
				clearTimeout(timer);
				resolve();
			});
		});
	}
}

/**
 * Spawn `simplex-chat -p <port> -d <dbPrefix>` and connect to its WebSocket
 * bot API. Creates the local user profile if it does not exist yet.
 */
export async function startSimplexChat(options: StartSimplexOptions): Promise<SimplexTransport> {
	const port = options.port ?? 5225;
	const binary = options.binary ?? "simplex-chat";
	const args = ["-p", String(port), "-d", options.dbPrefix];
	if (options.displayName !== undefined) args.push("--create-bot-display-name", options.displayName);
	const proc = spawn(binary, args, {
		stdio: ["ignore", "pipe", "pipe"],
	});
	await waitForPort(port, proc, options.connectTimeoutMs ?? 15_000);
	const ws = await new Promise<WebSocket>((resolve, reject) => {
		const sock = new WebSocket(`ws://127.0.0.1:${port}`);
		sock.addEventListener("open", () => resolve(sock));
		sock.addEventListener("error", () => reject(new SimplexError(`WebSocket connect failed on port ${port}`)));
	});
	const client = new SimplexClient(options.dbPrefix, options.displayName, proc, ws);
	await client.getUserId();
	return client;
}
