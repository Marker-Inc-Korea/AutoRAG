import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ChatMessage, ChatRecord, ChatSummary } from "../shared/search-contract";
import { hasErrorCode } from "./fs-entry";

export interface ChatStore {
	/** Insert or replace one record; summary fields are derived from its messages. */
	save(record: ChatRecord): Promise<void>;
	list(): Promise<readonly ChatSummary[]>;
	get(chatId: string): Promise<ChatRecord | null>;
	search(query: string): Promise<readonly ChatSummary[]>;
	clear(): Promise<void>;
}

export interface ChatStoreDeps {
	readonly directory: string;
	readonly now?: () => Date;
}

const STORE_FILENAME = "chats.json";
const TITLE_MAX_LENGTH = 80;

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isChatRecord(value: unknown): value is ChatRecord {
	if (!isObject(value)) return false;
	return (
		typeof value.id === "string" &&
		typeof value.title === "string" &&
		typeof value.snippet === "string" &&
		typeof value.updatedAt === "string" &&
		Array.isArray(value.messages)
	);
}

function truncateTitle(text: string): string {
	const normalized = text.replace(/\s+/gu, " ").trim();
	return normalized.length <= TITLE_MAX_LENGTH ? normalized : `${normalized.slice(0, TITLE_MAX_LENGTH - 1)}…`;
}

function stripAnswerMarkup(answer: string): string {
	return answer
		.replace(/\*\*/gu, "")
		.replace(/\[\d+\]/gu, "")
		.replace(/\s+([.,!?;:])/gu, "$1")
		.replace(/\s+/gu, " ")
		.trim();
}

function deriveTitle(messages: readonly ChatMessage[], fallback: string): string {
	const firstUser = messages.find((message) => message.role === "user");
	return truncateTitle(firstUser?.text ?? fallback);
}

function deriveSnippet(messages: readonly ChatMessage[], fallback: string): string {
	const firstAssistant = messages.find((message) => message.role === "assistant" && message.quick !== null);
	if (firstAssistant?.role !== "assistant" || firstAssistant.quick === null) return fallback;
	return stripAnswerMarkup(firstAssistant.quick.answer);
}

function toSummary(record: ChatRecord): ChatSummary {
	return { id: record.id, title: record.title, snippet: record.snippet, updatedAt: record.updatedAt };
}

/** JSON persistence for chat history. Writes are serialized and atomic. */
export function createChatStore(deps: ChatStoreDeps): ChatStore {
	const filePath = join(deps.directory, STORE_FILENAME);
	const now = deps.now ?? (() => new Date());
	let records: Map<string, ChatRecord> | undefined;
	let queue: Promise<void> = Promise.resolve();

	async function loadRecords(): Promise<Map<string, ChatRecord>> {
		if (records !== undefined) return records;
		try {
			const parsed: unknown = JSON.parse(await readFile(filePath, "utf8"));
			records = new Map((Array.isArray(parsed) ? parsed.filter(isChatRecord) : []).map((record) => [record.id, record]));
		} catch (error) {
			if (error instanceof SyntaxError || hasErrorCode(error, "ENOENT") || hasErrorCode(error, "ENOTDIR")) {
				records = new Map();
			} else {
				throw error;
			}
		}
		return records;
	}

	async function writeRecords(nextRecords: Map<string, ChatRecord>): Promise<void> {
		await mkdir(deps.directory, { recursive: true });
		const temporaryPath = join(deps.directory, `.${STORE_FILENAME}.${process.pid}.${randomUUID()}.tmp`);
		try {
			await writeFile(temporaryPath, `${JSON.stringify([...nextRecords.values()], null, 2)}\n`, {
				encoding: "utf8",
				mode: 0o600,
			});
			await rename(temporaryPath, filePath);
		} finally {
			await rm(temporaryPath, { force: true });
		}
	}

	function enqueue<T>(operation: () => Promise<T>): Promise<T> {
		const result = queue.then(operation);
		queue = result.then(
			() => undefined,
			() => undefined,
		);
		return result;
	}

	function normalizeRecord(record: ChatRecord): ChatRecord {
		const updatedAt = Number.isNaN(Date.parse(record.updatedAt)) ? now().toISOString() : record.updatedAt;
		const withTimestamp = { ...record, updatedAt };
		return {
			...withTimestamp,
			title: deriveTitle(record.messages, record.title),
			snippet: deriveSnippet(record.messages, record.snippet),
		};
	}

	return {
		save: (record) =>
			enqueue(async () => {
				const loaded = await loadRecords();
				loaded.set(record.id, normalizeRecord(record));
				await writeRecords(loaded);
			}),
		list: () =>
			enqueue(async () =>
				[...(await loadRecords()).values()]
					.map(toSummary)
					.sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt)),
			),
		get: (chatId) => enqueue(async () => (await loadRecords()).get(chatId) ?? null),
		search: (query) =>
			enqueue(async () => {
				const needle = query.trim().toLowerCase();
				const summaries = [...(await loadRecords()).values()]
					.map(toSummary)
					.sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
				if (needle.length === 0) return summaries;
				return summaries.filter((record) => `${record.title}\n${record.snippet}`.toLowerCase().includes(needle));
			}),
		clear: () =>
			enqueue(async () => {
				const loaded = await loadRecords();
				loaded.clear();
				await writeRecords(loaded);
			}),
	};
}
