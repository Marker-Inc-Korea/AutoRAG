import { describe, expect, it } from "vitest";
import { createSendMessageTool, SEND_MESSAGE_TOOL_NAME } from "../../src/messaging/tool.ts";
import { MessagingError, type OutboundChannel, type OutboundMessage } from "../../src/messaging/types.ts";

interface Fake extends OutboundChannel {
	readonly calls: OutboundMessage[];
}

function fake(id: string, addressing: "fixed" | "recipient", behavior?: (m: OutboundMessage) => void): Fake {
	const calls: OutboundMessage[] = [];
	return {
		id,
		kind: "fake",
		addressing,
		calls,
		describe: () => `fake ${addressing} channel ${id}`,
		async send(message) {
			calls.push(message);
			behavior?.(message);
			return { destination: `dest-of-${id}`, ids: ["m1", "m2"] };
		},
	};
}

const text = (result: { content: unknown }) =>
	(result.content as { type: string; text: string }[]).map((part) => part.text).join("");

function toolWith(...channels: OutboundChannel[]) {
	return createSendMessageTool(new Map(channels.map((c) => [c.id, c])));
}

describe("send_message tool", () => {
	it("is named send_message and lists every configured channel in its description", () => {
		const tool = toolWith(fake("ops", "fixed"), fake("mail", "recipient"));
		expect(tool.name).toBe("send_message");
		expect(SEND_MESSAGE_TOOL_NAME).toBe("send_message");
		expect(tool.description).toContain("ops: fake fixed channel ops");
		expect(tool.description).toContain("mail: fake recipient channel mail");
	});

	it("sends to a fixed channel and reports where it went", async () => {
		const ops = fake("ops", "fixed");
		const result = await toolWith(ops).execute("c1", { channel: "ops", text: "build is green" });
		expect(result.isError).toBeUndefined();
		expect(ops.calls).toEqual([{ text: "build is green" }]);
		expect(text(result)).toContain("dest-of-ops");
		expect(text(result)).toContain("m1");
		expect(result.details).toMatchObject({ channel: "ops", destination: "dest-of-ops", ids: ["m1", "m2"] });
	});

	it("passes recipients and subject to a recipient channel", async () => {
		const mail = fake("mail", "recipient");
		await toolWith(mail).execute("c1", {
			channel: "mail",
			to: ["a@example.com", "b@example.com"],
			subject: "Report",
			text: "see attached",
		});
		expect(mail.calls).toEqual([{ text: "see attached", subject: "Report", to: ["a@example.com", "b@example.com"] }]);
	});

	it("refuses an unknown channel and names the ones that exist", async () => {
		const result = await toolWith(fake("ops", "fixed")).execute("c1", { channel: "nope", text: "x" });
		expect(result.isError).toBe(true);
		expect(text(result)).toMatch(/nope/);
		expect(text(result)).toMatch(/ops/);
	});

	it("refuses recipients on a fixed channel instead of silently dropping them", async () => {
		const ops = fake("ops", "fixed");
		const result = await toolWith(ops).execute("c1", { channel: "ops", text: "x", to: ["a@example.com"] });
		expect(result.isError).toBe(true);
		expect(text(result)).toMatch(/fixed destination/i);
		expect(ops.calls).toHaveLength(0);
	});

	it("requires recipients on a recipient channel", async () => {
		const mail = fake("mail", "recipient");
		for (const to of [undefined, []]) {
			const result = await toolWith(mail).execute("c1", {
				channel: "mail",
				text: "x",
				...(to === undefined ? {} : { to }),
			});
			expect(result.isError).toBe(true);
			expect(text(result)).toMatch(/"to"/);
		}
		expect(mail.calls).toHaveLength(0);
	});

	it("refuses an empty message", async () => {
		const ops = fake("ops", "fixed");
		const result = await toolWith(ops).execute("c1", { channel: "ops", text: "  \n " });
		expect(result.isError).toBe(true);
		expect(ops.calls).toHaveLength(0);
	});

	it("reports a channel failure as an error result the model can read, not as an exception", async () => {
		const broken = fake("ops", "fixed", () => {
			throw new MessagingError("Telegram rejected the message (HTTP 400): chat not found");
		});
		const result = await toolWith(broken).execute("c1", { channel: "ops", text: "x" });
		expect(result.isError).toBe(true);
		expect(text(result)).toContain("chat not found");
		expect(text(result)).toMatch(/not sent|was not delivered/i);
	});

	it("does not swallow an abort or an unexpected failure", async () => {
		const exploding = fake("ops", "fixed", () => {
			throw new Error("boom");
		});
		await expect(toolWith(exploding).execute("c1", { channel: "ops", text: "x" })).rejects.toThrow("boom");
	});

	it("never retries a send by itself", async () => {
		const ops = fake("ops", "fixed");
		await toolWith(ops).execute("c1", { channel: "ops", text: "once" });
		expect(ops.calls).toHaveLength(1);
	});
});
