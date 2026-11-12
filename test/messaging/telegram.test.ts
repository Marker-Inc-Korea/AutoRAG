import { describe, expect, it } from "vitest";
import { telegramKind } from "../../src/messaging/channels/telegram.ts";
import { MessagingConfigError, MessagingError } from "../../src/messaging/types.ts";

const valid = { kind: "telegram", tokenEnv: "TELEGRAM_BOT_TOKEN", chatId: "-1001234567890" };
const TOKEN = "123456:ABC-secretTOKEN";

interface Seen {
	url: string;
	method?: string;
	body: Record<string, unknown>;
}

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function stub(reply: (n: number) => Response = (n) => json({ ok: true, result: { message_id: 100 + n } })) {
	const seen: Seen[] = [];
	const fetchImpl = (async (url: string | URL | Request, init?: RequestInit) => {
		seen.push({
			url: String(url),
			...(init?.method === undefined ? {} : { method: init.method }),
			body: JSON.parse(String(init?.body ?? "{}")),
		});
		return reply(seen.length);
	}) as typeof fetch;
	return { seen, fetchImpl };
}

function channel(
	fetchImpl: typeof fetch,
	config: Record<string, unknown> = valid,
	env: Record<string, string | undefined> = { TELEGRAM_BOT_TOKEN: TOKEN },
) {
	return telegramKind.create("tg", telegramKind.validate(config, "p"), { env, fetch: fetchImpl });
}

describe("telegram config", () => {
	it("accepts a numeric id, a negative group id, and an @channel name, and keeps ids as strings", () => {
		expect(telegramKind.validate(valid, "p")).toEqual({
			kind: "telegram",
			tokenEnv: "TELEGRAM_BOT_TOKEN",
			chatId: "-1001234567890",
		});
		expect(telegramKind.validate({ ...valid, chatId: 42 }, "p")).toMatchObject({ chatId: "42" });
		expect(telegramKind.validate({ ...valid, chatId: "@my_channel" }, "p")).toMatchObject({ chatId: "@my_channel" });
		expect(telegramKind.validate({ ...valid, threadId: 7 }, "p")).toMatchObject({ threadId: 7 });
	});

	it("refuses a token in place of a variable name, a bad chat id, and unknown fields", () => {
		const pasted = () => telegramKind.validate({ ...valid, tokenEnv: TOKEN }, "p");
		expect(pasted).toThrow(MessagingConfigError);
		expect(pasted).not.toThrow(/secretTOKEN/);
		expect(() => telegramKind.validate({ ...valid, chatId: undefined }, "p")).toThrow(/p\.chatId/);
		expect(() => telegramKind.validate({ ...valid, chatId: "my chat" }, "p")).toThrow(/p\.chatId/);
		expect(() => telegramKind.validate({ ...valid, chatId: 1.5 }, "p")).toThrow(/p\.chatId/);
		expect(() => telegramKind.validate({ ...valid, threadId: 0 }, "p")).toThrow(/p\.threadId/);
		expect(() => telegramKind.validate({ ...valid, apiBase: "https://evil.test" }, "p")).toThrow(
			/p\.apiBase is not a recognized field/,
		);
	});
});

describe("telegram send", () => {
	it("is a fixed channel that names its chat but never the token", () => {
		const tg = channel(stub().fetchImpl);
		expect(tg.addressing).toBe("fixed");
		expect(tg.describe()).toContain("-1001234567890");
		expect(tg.describe()).not.toContain(TOKEN);
	});

	it("calls sendMessage on the bot API with the chat and the text, and returns the message id", async () => {
		const { seen, fetchImpl } = stub();
		const receipt = await channel(fetchImpl).send({ text: "build is green" });
		expect(seen).toHaveLength(1);
		expect(seen[0]?.url).toBe(`https://api.telegram.org/bot${TOKEN}/sendMessage`);
		expect(seen[0]?.method).toBe("POST");
		expect(seen[0]?.body).toEqual({
			chat_id: "-1001234567890",
			text: "build is green",
			link_preview_options: { is_disabled: true },
		});
		expect(receipt).toEqual({ destination: "Telegram chat -1001234567890", ids: ["101"] });
	});

	it("sends plain text: no parse mode, so model output can never break or be reinterpreted", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ text: "*bold* _x_ <b>y</b> [a](b)" });
		expect(seen[0]?.body).not.toHaveProperty("parse_mode");
	});

	it("posts into a forum topic when a thread is configured", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl, { ...valid, threadId: 9 }).send({ text: "x" });
		expect(seen[0]?.body.message_thread_id).toBe(9);
	});

	it("prefixes the subject when one is given, since a chat has no subject line", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ subject: "Deploy", text: "done" });
		expect(seen[0]?.body.text).toBe("Deploy\n\ndone");
	});

	it("splits a message over 4096 characters into ordered parts and returns every id", async () => {
		const { seen, fetchImpl } = stub();
		const text = Array.from({ length: 900 }, (_, i) => `line ${i} ${"x".repeat(5)}`).join("\n");
		const receipt = await channel(fetchImpl).send({ text });
		expect(seen.length).toBeGreaterThan(1);
		for (const call of seen) expect(String(call.body.text).length).toBeLessThanOrEqual(4096);
		expect(
			seen
				.map((c) => c.body.text)
				.join("\n")
				.replace(/\s+/g, ""),
		).toBe(text.replace(/\s+/g, ""));
		expect(receipt.ids).toHaveLength(seen.length);
	});

	it("stops at the first failed part and says how many went out, so nothing is sent twice", async () => {
		const { seen, fetchImpl } = stub((n) =>
			n === 2
				? json({ ok: false, error_code: 429, description: "Too Many Requests" }, 429)
				: json({ ok: true, result: { message_id: n } }),
		);
		const text = `${"a".repeat(4000)}\n${"b".repeat(4000)}\n${"c".repeat(4000)}`;
		const error = await channel(fetchImpl)
			.send({ text })
			.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toMatch(/1 of 3 parts/);
		expect(seen).toHaveLength(2);
	});

	it("reads the token at send time and says which variable is missing", async () => {
		const { seen, fetchImpl } = stub();
		await expect(channel(fetchImpl, valid, {}).send({ text: "x" })).rejects.toThrow(/TELEGRAM_BOT_TOKEN/);
		expect(seen).toHaveLength(0);
	});

	it("scrubs the token from a provider error, including the one in the URL", async () => {
		const { fetchImpl } = stub(() => json({ ok: false, error_code: 401, description: `Unauthorized ${TOKEN}` }, 401));
		const error = await channel(fetchImpl)
			.send({ text: "x" })
			.catch((e: unknown) => e);
		expect((error as Error).message).toContain("401");
		expect((error as Error).message).not.toContain("secretTOKEN");
	});

	it("treats ok:false on a 200 as a failure and an answer without a message id as unconfirmed", async () => {
		await expect(
			channel(stub(() => json({ ok: false, description: "chat not found" })).fetchImpl).send({ text: "x" }),
		).rejects.toThrow(/chat not found/);
		await expect(channel(stub(() => json({ ok: true, result: {} })).fetchImpl).send({ text: "x" })).rejects.toThrow(
			/message id/i,
		);
	});

	it("refuses an empty text after the subject is added", async () => {
		const { seen, fetchImpl } = stub();
		await expect(channel(fetchImpl).send({ text: "" })).rejects.toThrow(MessagingError);
		expect(seen).toHaveLength(0);
	});
});
