import { describe, expect, it } from "vitest";
import { discordKind } from "../../src/messaging/channels/discord.ts";
import { MessagingConfigError, MessagingError } from "../../src/messaging/types.ts";

const valid = { kind: "discord", tokenEnv: "DISCORD_BOT_TOKEN", channelId: "1234567890123456789" };
const TOKEN = "MTIz.secretDiscordToken";

interface Seen {
	url: string;
	method?: string;
	headers: Headers;
	body: Record<string, unknown>;
}

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function stub(
	reply: (n: number) => Response = (n) => json({ id: `90000000000000000${n}`, channel_id: "1234567890123456789" }),
) {
	const seen: Seen[] = [];
	const fetchImpl = (async (url: string | URL | Request, init?: RequestInit) => {
		seen.push({
			url: String(url),
			...(init?.method === undefined ? {} : { method: init.method }),
			headers: new Headers(init?.headers),
			body: JSON.parse(String(init?.body ?? "{}")),
		});
		return reply(seen.length);
	}) as typeof fetch;
	return { seen, fetchImpl };
}

function channel(
	fetchImpl: typeof fetch,
	config: Record<string, unknown> = valid,
	env: Record<string, string | undefined> = { DISCORD_BOT_TOKEN: TOKEN },
) {
	return discordKind.create("dc", discordKind.validate(config, "p"), { env, fetch: fetchImpl });
}

describe("discord config", () => {
	it("accepts a snowflake channel id as a string or a safe number", () => {
		expect(discordKind.validate(valid, "p")).toEqual({
			kind: "discord",
			tokenEnv: "DISCORD_BOT_TOKEN",
			channelId: "1234567890123456789",
		});
		expect(discordKind.validate({ ...valid, channelId: 123456789012345 }, "p")).toMatchObject({
			channelId: "123456789012345",
		});
	});

	it("refuses a token in place of a variable name, a non-snowflake channel, and unknown fields", () => {
		const pasted = () => discordKind.validate({ ...valid, tokenEnv: TOKEN }, "p");
		expect(pasted).toThrow(MessagingConfigError);
		expect(pasted).not.toThrow(/secretDiscordToken/);
		expect(() => discordKind.validate({ ...valid, channelId: undefined }, "p")).toThrow(/p\.channelId/);
		expect(() => discordKind.validate({ ...valid, channelId: "#general" }, "p")).toThrow(/p\.channelId/);
		expect(() => discordKind.validate({ ...valid, channelId: "../guilds/1" }, "p")).toThrow(/p\.channelId/);
		expect(() => discordKind.validate({ ...valid, apiBase: "https://evil.test" }, "p")).toThrow(
			/p\.apiBase is not a recognized field/,
		);
	});
});

describe("discord send", () => {
	it("is a fixed channel that names its channel id but never the token", () => {
		const dc = channel(stub().fetchImpl);
		expect(dc.addressing).toBe("fixed");
		expect(dc.describe()).toContain("1234567890123456789");
		expect(dc.describe()).not.toContain(TOKEN);
	});

	it("posts to the channel's messages endpoint as the bot and returns the message id", async () => {
		const { seen, fetchImpl } = stub();
		const receipt = await channel(fetchImpl).send({ text: "deploy finished" });
		expect(seen).toHaveLength(1);
		expect(seen[0]?.url).toBe("https://discord.com/api/v10/channels/1234567890123456789/messages");
		expect(seen[0]?.method).toBe("POST");
		expect(seen[0]?.headers.get("authorization")).toBe(`Bot ${TOKEN}`);
		expect(seen[0]?.body.content).toBe("deploy finished");
		expect(receipt).toEqual({ destination: "Discord channel 1234567890123456789", ids: ["900000000000000001"] });
	});

	it("can never ping anyone: model text with @everyone, a role, or a user mention notifies nobody", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ text: "@everyone <@&123> <@456> hello" });
		expect(seen[0]?.body.allowed_mentions).toEqual({ parse: [] });
	});

	it("suppresses link embeds so a pasted URL does not expand", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ text: "see https://example.org" });
		expect(seen[0]?.body.flags).toBe(4);
	});

	it("puts the subject on a bold first line", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ subject: "Deploy", text: "done" });
		expect(seen[0]?.body.content).toBe("**Deploy**\n\ndone");
	});

	it("splits over 2000 characters into ordered parts and returns every id", async () => {
		const { seen, fetchImpl } = stub();
		const text = Array.from({ length: 400 }, (_, i) => `line ${i} xxxxx`).join("\n");
		const receipt = await channel(fetchImpl).send({ text });
		expect(seen.length).toBeGreaterThan(1);
		for (const call of seen) expect(String(call.body.content).length).toBeLessThanOrEqual(2000);
		expect(
			seen
				.map((c) => c.body.content)
				.join("\n")
				.replace(/\s+/g, ""),
		).toBe(text.replace(/\s+/g, ""));
		expect(receipt.ids).toHaveLength(seen.length);
	});

	it("stops at the first failed part and says how many went out", async () => {
		const { seen, fetchImpl } = stub((n) =>
			n === 2 ? json({ message: "You are being rate limited.", code: 0 }, 429) : json({ id: String(n) }),
		);
		const text = `${"a".repeat(1900)}\n${"b".repeat(1900)}\n${"c".repeat(1900)}`;
		const error = await channel(fetchImpl)
			.send({ text })
			.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toMatch(/1 of 3 parts/);
		expect(seen).toHaveLength(2);
	});

	it("reads the token at send time and says which variable is missing", async () => {
		const { seen, fetchImpl } = stub();
		await expect(channel(fetchImpl, valid, {}).send({ text: "x" })).rejects.toThrow(/DISCORD_BOT_TOKEN/);
		expect(seen).toHaveLength(0);
	});

	it("scrubs the token from a provider error and keeps Discord's reason and status", async () => {
		const { fetchImpl } = stub(() => json({ message: `Missing Access for ${TOKEN}`, code: 50001 }, 403));
		const error = await channel(fetchImpl)
			.send({ text: "x" })
			.catch((e: unknown) => e);
		expect((error as Error).message).toContain("403");
		expect((error as Error).message).toContain("Missing Access");
		expect((error as Error).message).not.toContain("secretDiscordToken");
	});

	it("treats an answer without a message id as unconfirmed, and an empty message as a refusal", async () => {
		await expect(channel(stub(() => json({})).fetchImpl).send({ text: "x" })).rejects.toThrow(/message id/i);
		const { seen, fetchImpl } = stub();
		await expect(channel(fetchImpl).send({ text: " " })).rejects.toThrow(MessagingError);
		expect(seen).toHaveLength(0);
	});
});
