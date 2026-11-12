import { describe, expect, it } from "vitest";
import { agentmailKind } from "../../src/messaging/channels/agentmail.ts";
import { MessagingConfigError, MessagingError } from "../../src/messaging/types.ts";

const valid = { kind: "agentmail", apiKeyEnv: "AGENTMAIL_API_KEY", inboxId: "my-agent@agentmail.to" };

interface Seen {
	url: string;
	method?: string;
	headers: Headers;
	body: Record<string, unknown>;
}

function stub(reply: () => Response = () => json({ message_id: "<m1@agentmail.to>", thread_id: "t1" })) {
	const seen: Seen[] = [];
	const fetchImpl = (async (url: string | URL | Request, init?: RequestInit) => {
		seen.push({
			url: String(url),
			...(init?.method === undefined ? {} : { method: init.method }),
			headers: new Headers(init?.headers),
			body: JSON.parse(String(init?.body ?? "{}")),
		});
		return reply();
	}) as typeof fetch;
	return { seen, fetchImpl };
}

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function channel(env: Record<string, string | undefined>, fetchImpl: typeof fetch, config = valid) {
	return agentmailKind.create("mail", agentmailKind.validate(config, "messaging.channels.mail"), {
		env,
		fetch: fetchImpl,
	});
}

describe("agentmail config", () => {
	it("accepts the key's variable name and the agent's inbox", () => {
		expect(agentmailKind.validate(valid, "p")).toEqual({
			kind: "agentmail",
			apiKeyEnv: "AGENTMAIL_API_KEY",
			inboxId: "my-agent@agentmail.to",
			region: "us",
		});
		expect(agentmailKind.validate({ ...valid, region: "eu" }, "p")).toMatchObject({ region: "eu" });
	});

	it("refuses a pasted key in place of a variable name, without echoing it", () => {
		const attempt = () => agentmailKind.validate({ ...valid, apiKeyEnv: "am_us_SECRET123" }, "p");
		expect(attempt).toThrow(MessagingConfigError);
		expect(attempt).not.toThrow(/SECRET123/);
	});

	it("requires an inbox that is an email address, a known region, and nothing else", () => {
		expect(() => agentmailKind.validate({ ...valid, inboxId: undefined }, "p")).toThrow(/p\.inboxId/);
		expect(() => agentmailKind.validate({ ...valid, inboxId: "not-an-address" }, "p")).toThrow(/p\.inboxId/);
		expect(() => agentmailKind.validate({ ...valid, region: "mars" }, "p")).toThrow(/p\.region/);
		expect(() => agentmailKind.validate({ ...valid, baseUrl: "https://evil.test" }, "p")).toThrow(
			/p\.baseUrl is not a recognized field/,
		);
		expect(() => agentmailKind.validate({ ...valid, apiKeyEnv: undefined }, "p")).toThrow(/p\.apiKeyEnv/);
	});
});

describe("agentmail send", () => {
	const env = { AGENTMAIL_API_KEY: "am_us_testkey" };

	it("is a recipient channel that says which inbox it sends from", () => {
		const mail = channel(env, stub().fetchImpl);
		expect(mail.addressing).toBe("recipient");
		expect(mail.describe()).toContain("my-agent@agentmail.to");
		expect(mail.describe()).not.toContain("am_us_testkey");
	});

	it("posts to the inbox's send endpoint with the key as a bearer token", async () => {
		const { seen, fetchImpl } = stub();
		const receipt = await channel(env, fetchImpl).send({
			to: ["minji@example.org"],
			subject: "Summary",
			text: "Body text",
		});
		expect(seen).toHaveLength(1);
		expect(seen[0]?.url).toBe("https://api.agentmail.to/v0/inboxes/my-agent%40agentmail.to/messages/send");
		expect(seen[0]?.method).toBe("POST");
		expect(seen[0]?.headers.get("authorization")).toBe("Bearer am_us_testkey");
		expect(seen[0]?.body).toEqual({ to: ["minji@example.org"], subject: "Summary", text: "Body text" });
		expect(receipt).toEqual({ destination: "minji@example.org", ids: ["<m1@agentmail.to>"] });
	});

	it("uses the EU endpoint for the eu region", async () => {
		const { seen, fetchImpl } = stub();
		await channel(env, fetchImpl, { ...valid, region: "eu" } as typeof valid).send({
			to: ["a@example.org"],
			subject: "s",
			text: "t",
		});
		expect(seen[0]?.url.startsWith("https://api.agentmail.eu/v0/inboxes/")).toBe(true);
	});

	it("reads the key at send time, so a variable set after startup works", async () => {
		const live: Record<string, string | undefined> = {};
		const { fetchImpl } = stub();
		const mail = channel(live, fetchImpl);
		await expect(mail.send({ to: ["a@example.org"], subject: "s", text: "t" })).rejects.toThrow(/AGENTMAIL_API_KEY/);
		live.AGENTMAIL_API_KEY = "am_us_late";
		await expect(mail.send({ to: ["a@example.org"], subject: "s", text: "t" })).resolves.toBeDefined();
	});

	it("sends nothing when recipients or subject are bad", async () => {
		const { seen, fetchImpl } = stub();
		const mail = channel(env, fetchImpl);
		const bad = [
			{ subject: "s", text: "t" },
			{ to: [], subject: "s", text: "t" },
			{ to: ["not-an-address"], subject: "s", text: "t" },
			{ to: ["a@example.org\r\nBcc: x@evil.test"], subject: "s", text: "t" },
			{ to: ["a@example.org"], text: "t" },
			{ to: ["a@example.org"], subject: "  ", text: "t" },
			{ to: ["a@example.org"], subject: "line\r\nBcc: x@evil.test", text: "t" },
			{ to: Array.from({ length: 11 }, (_, i) => `u${i}@example.org`), subject: "s", text: "t" },
		];
		for (const message of bad) await expect(mail.send(message)).rejects.toThrow(MessagingError);
		expect(seen).toHaveLength(0);
	});

	it("scrubs the key from a provider error and names the HTTP status", async () => {
		const { fetchImpl } = stub(() =>
			json(
				{ name: "MessageRejectedError", message: "key am_us_testkey cannot send", code: "message_rejected" },
				403,
			),
		);
		const error = await channel(env, fetchImpl)
			.send({ to: ["a@example.org"], subject: "s", text: "t" })
			.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toContain("403");
		expect((error as Error).message).toContain("message_rejected");
		expect((error as Error).message).not.toContain("am_us_testkey");
	});

	it("fails when the provider answers success without a message id, instead of reporting a send it cannot prove", async () => {
		const { fetchImpl } = stub(() => json({ ok: true }));
		await expect(channel(env, fetchImpl).send({ to: ["a@example.org"], subject: "s", text: "t" })).rejects.toThrow(
			/message id/i,
		);
	});
});
