import { describe, expect, it } from "vitest";
import { slackKind } from "../../src/messaging/channels/slack.ts";
import { MessagingConfigError, MessagingError } from "../../src/messaging/types.ts";

const valid = { kind: "slack", webhookEnv: "SLACK_WEBHOOK_URL" };
const HOOK = "https://hooks.slack.com/services/T000/B000/secretWEBHOOKpath";

interface Seen {
	url: string;
	method?: string;
	body: Record<string, unknown>;
}

function stub(reply: (n: number) => Response = () => new Response("ok", { status: 200 })) {
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

function channel(fetchImpl: typeof fetch, env: Record<string, string | undefined> = { SLACK_WEBHOOK_URL: HOOK }) {
	return slackKind.create("sl", slackKind.validate(valid, "p"), { env, fetch: fetchImpl });
}

describe("slack config", () => {
	it("takes only the name of the variable that holds the webhook URL", () => {
		expect(slackKind.validate(valid, "p")).toEqual({ kind: "slack", webhookEnv: "SLACK_WEBHOOK_URL" });
	});

	it("refuses the webhook URL itself in the config, without echoing it, and unknown fields", () => {
		const pasted = () => slackKind.validate({ ...valid, webhookEnv: HOOK }, "p");
		expect(pasted).toThrow(MessagingConfigError);
		expect(pasted).not.toThrow(/secretWEBHOOKpath/);
		expect(() => slackKind.validate({ kind: "slack" }, "p")).toThrow(/p\.webhookEnv/);
		expect(() => slackKind.validate({ ...valid, channel: "#general" }, "p")).toThrow(
			/p\.channel is not a recognized field/,
		);
	});
});

describe("slack send", () => {
	it("is a fixed channel; it describes itself without the URL, which is the secret", () => {
		const sl = channel(stub().fetchImpl);
		expect(sl.addressing).toBe("fixed");
		expect(sl.describe()).toContain("Slack");
		expect(sl.describe()).not.toContain("hooks.slack.com");
	});

	it("posts the text to the webhook and reports the send", async () => {
		const { seen, fetchImpl } = stub();
		const receipt = await channel(fetchImpl).send({ text: "deploy finished" });
		expect(seen).toHaveLength(1);
		expect(seen[0]?.url).toBe(HOOK);
		expect(seen[0]?.method).toBe("POST");
		expect(seen[0]?.body.text).toBe("deploy finished");
		expect(receipt.destination).toBe("Slack incoming webhook");
		expect(receipt.ids).toEqual(["ok"]);
	});

	it("sends plain text with no link unfurling and no mrkdwn, so model text is never reinterpreted", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ text: "*bold* <!channel> <!here> <@U123> https://example.org" });
		expect(seen[0]?.body.mrkdwn).toBe(false);
		expect(seen[0]?.body.unfurl_links).toBe(false);
		expect(seen[0]?.body.unfurl_media).toBe(false);
	});

	it("defuses broadcast mentions: <!channel>, <!here>, <!everyone> reach nobody", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ text: "<!channel> <!here> <!everyone> <!subteam^S123> hi" });
		const text = String(seen[0]?.body.text);
		expect(text).not.toMatch(/<!/);
		expect(text).toContain("hi");
	});

	it("puts a subject on a first line", async () => {
		const { seen, fetchImpl } = stub();
		await channel(fetchImpl).send({ subject: "Deploy", text: "done" });
		expect(seen[0]?.body.text).toBe("Deploy\n\ndone");
	});

	it("splits a very long message into ordered parts", async () => {
		const { seen, fetchImpl } = stub();
		const text = Array.from({ length: 2000 }, (_, i) => `line ${i} xxxxx`).join("\n");
		const receipt = await channel(fetchImpl).send({ text });
		expect(seen.length).toBeGreaterThan(1);
		for (const call of seen) expect(String(call.body.text).length).toBeLessThanOrEqual(3000);
		expect(
			seen
				.map((c) => c.body.text)
				.join("\n")
				.replace(/\s+/g, ""),
		).toBe(text.replace(/\s+/g, ""));
		expect(receipt.ids).toHaveLength(seen.length);
	});

	it("stops at the first failed part and says how many went out", async () => {
		const { seen, fetchImpl } = stub((n) =>
			n === 2 ? new Response("rate_limited", { status: 429 }) : new Response("ok"),
		);
		const text = `${"a".repeat(2900)}\n${"b".repeat(2900)}\n${"c".repeat(2900)}`;
		const error = await channel(fetchImpl)
			.send({ text })
			.catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toMatch(/1 of 3 parts/);
		expect(seen).toHaveLength(2);
	});

	it("reads the URL at send time and names the variable that is missing", async () => {
		const { seen, fetchImpl } = stub();
		await expect(channel(fetchImpl, {}).send({ text: "x" })).rejects.toThrow(/SLACK_WEBHOOK_URL/);
		expect(seen).toHaveLength(0);
	});

	it("refuses a variable that does not hold a Slack webhook URL, so a stray value is never POSTed to", async () => {
		const { seen, fetchImpl } = stub();
		for (const bad of [
			"https://evil.test/services/T/B/x",
			"http://hooks.slack.com/services/T/B/x",
			"not a url",
			"https://hooks.slack.com.evil.test/services/T/B/x",
		]) {
			await expect(channel(fetchImpl, { SLACK_WEBHOOK_URL: bad }).send({ text: "x" })).rejects.toThrow(
				/Slack incoming webhook/i,
			);
		}
		expect(seen).toHaveLength(0);
	});

	it("scrubs the URL from an error and keeps Slack's reason and status", async () => {
		const { fetchImpl } = stub(() => new Response(`no_service for ${HOOK}`, { status: 404 }));
		const error = await channel(fetchImpl)
			.send({ text: "x" })
			.catch((e: unknown) => e);
		expect((error as Error).message).toContain("404");
		expect((error as Error).message).toContain("no_service");
		expect((error as Error).message).not.toContain("secretWEBHOOKpath");
	});

	it("treats any answer other than ok as unconfirmed, and refuses an empty message", async () => {
		await expect(
			channel(stub(() => new Response("<html>login</html>", { status: 200 })).fetchImpl).send({ text: "x" }),
		).rejects.toThrow(/did not confirm/i);
		const { seen, fetchImpl } = stub();
		await expect(channel(fetchImpl).send({ text: "  " })).rejects.toThrow(MessagingError);
		expect(seen).toHaveLength(0);
	});
});
