import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createChatStore, type ChatStore } from "../src/main/chat-store";
import type { ChatRecord } from "../src/shared/search-contract";

const tempDirs: string[] = [];

async function createStore(now: () => Date = () => new Date("2026-09-28T12:00:00.000Z")): Promise<ChatStore> {
	const directory = await mkdtemp(join(tmpdir(), "autorag-chat-store-"));
	tempDirs.push(directory);
	return createChatStore({ directory, now });
}

afterEach(async () => {
	await Promise.all(tempDirs.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

function createRecord(id: string, userText: string, quickAnswer: string, updatedAt: string): ChatRecord {
	return {
		id,
		title: "placeholder title",
		snippet: "placeholder snippet",
		updatedAt,
		messages: [
			{
				role: "user",
				text: userText,
				attachments: [{ kind: "file", id: "/Users/jeff/Documents/policy.md", name: "policy.md" }],
				at: updatedAt,
			},
			{
				role: "assistant",
				sessionId: `${id}-session`,
				quick: { answer: quickAnswer, evidence: [], meta: "0.9s · 1 sources" },
				deep: null,
				stopped: false,
				at: updatedAt,
			},
		],
	};
}

describe("createChatStore", () => {
	it("round-trips records and derives title, snippet, and summary metadata", async () => {
		const store = await createStore();
		const question =
			"Where is the refund approval policy for enterprise contracts and what changed in the July finance review process?";
		const record = createRecord(
			"chat-1",
			question,
			"**Refund exceptions** need director approval [1] before payout [2].",
			"2026-09-28T10:00:00.000Z",
		);

		await store.save(record);

		expect(await store.list()).toEqual([
			{
				id: "chat-1",
				title: `${question.slice(0, 79)}…`,
				snippet: "Refund exceptions need director approval before payout.",
				updatedAt: "2026-09-28T10:00:00.000Z",
			},
		]);
		const stored = await store.get("chat-1");
		expect(stored?.messages).toEqual(record.messages);
		expect(stored?.title).toBe(`${question.slice(0, 79)}…`);
		expect(stored?.snippet).toBe("Refund exceptions need director approval before payout.");
		expect(await readdir(join(tempDirs[0] ?? ""))).toEqual(["chats.json"]);
	});

	it("searches titles and snippets case-insensitively", async () => {
		const store = await createStore();
		await store.save(createRecord("chat-1", "Refund approval", "Director approval is required.", "2026-09-28T10:00:00.000Z"));
		await store.save(createRecord("chat-2", "Lunch places", "The **best** option is documented [4].", "2026-09-28T11:00:00.000Z"));

		expect((await store.search("REFUND")).map((record) => record.id)).toEqual(["chat-1"]);
		expect((await store.search("director")).map((record) => record.id)).toEqual(["chat-1"]);
		expect((await store.search("missing"))).toEqual([]);
	});

	it("clears every persisted chat", async () => {
		const store = await createStore();
		await store.save(createRecord("chat-1", "Refund approval", "Answer", "2026-09-28T10:00:00.000Z"));
		await store.save(createRecord("chat-2", "Lunch places", "Answer", "2026-09-28T11:00:00.000Z"));

		await store.clear();

		expect(await store.list()).toEqual([]);
		expect(await store.get("chat-1")).toBeNull();
	});

	it("treats corrupt JSON as an empty store and recovers on the next write", async () => {
		const directory = await mkdtemp(join(tmpdir(), "autorag-chat-store-corrupt-"));
		tempDirs.push(directory);
		await writeFile(join(directory, "chats.json"), "{not json", "utf8");
		const store = createChatStore({ directory, now: () => new Date("2026-09-28T12:00:00.000Z") });

		expect(await store.list()).toEqual([]);
		expect(await store.get("chat-1")).toBeNull();

		await store.save(createRecord("chat-1", "Refund approval", "Answer", "2026-09-28T10:00:00.000Z"));
		expect((await store.list()).map((record) => record.id)).toEqual(["chat-1"]);
	});
});
