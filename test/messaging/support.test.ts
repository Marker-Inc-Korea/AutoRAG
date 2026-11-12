import { describe, expect, it } from "vitest";
import { postJson, requireEnv, scrubSecrets, splitText } from "../../src/messaging/support.ts";
import { MessagingError } from "../../src/messaging/types.ts";

describe("splitText", () => {
	it("returns short text as a single part, unchanged", () => {
		expect(splitText("hello", 10)).toEqual(["hello"]);
	});

	it("never produces a part longer than the limit", () => {
		const text = `${"word ".repeat(500)}end`;
		for (const part of splitText(text, 100)) expect(part.length).toBeLessThanOrEqual(100);
	});

	it("breaks at a newline that leaves a reasonably full part, else a space, else mid-word", () => {
		expect(splitText("aaaaaa\nbb cc", 9)).toEqual(["aaaaaa", "bb cc"]);
		expect(splitText("aa\nbbbb cccc", 9)).toEqual(["aa\nbbbb", "cccc"]);
		expect(splitText("aaaa bbbb cccc", 9)).toEqual(["aaaa bbbb", "cccc"]);
		expect(splitText("abcdefghij", 4)).toEqual(["abcd", "efgh", "ij"]);
	});

	it("loses no characters other than the whitespace it breaks on", () => {
		const text = Array.from({ length: 300 }, (_, i) => `line ${i}`).join("\n");
		const parts = splitText(text, 120);
		expect(parts.join("\n").replace(/\s+/g, "")).toBe(text.replace(/\s+/g, ""));
	});

	it("does not cut an astral character (emoji, rare Hangul or CJK) in half", () => {
		const parts = splitText("😀".repeat(10), 5);
		for (const part of parts) expect(part).toMatch(/^(😀)+$/u);
		expect(parts.join("")).toBe("😀".repeat(10));
	});

	it("rejects a limit that cannot hold one character", () => {
		expect(() => splitText("x", 1)).toThrow();
	});
});

describe("scrubSecrets", () => {
	it("replaces every occurrence of every secret and ignores empty ones", () => {
		expect(scrubSecrets("a tok123 b tok123 c key", ["tok123", "", "key"])).toBe(
			"a [redacted] b [redacted] c [redacted]",
		);
	});
});

describe("requireEnv", () => {
	it("returns the trimmed value", () => {
		expect(requireEnv({ KEY: "  v  " }, "KEY", "Telegram bot token")).toBe("v");
	});

	it("names the variable and what it is for, never a value", () => {
		expect(() => requireEnv({}, "KEY", "Telegram bot token")).toThrow(MessagingError);
		expect(() => requireEnv({ KEY: "  " }, "KEY", "Telegram bot token")).toThrow(
			/KEY.*Telegram bot token|Telegram bot token.*KEY/,
		);
	});
});

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

describe("postJson", () => {
	it("posts the body as JSON with the given headers and returns the parsed reply", async () => {
		let seen: { url: string; init: RequestInit } | undefined;
		const fetchStub = (async (url: string | URL | Request, init?: RequestInit) => {
			seen = { url: String(url), init: init as RequestInit };
			return jsonResponse({ ok: true, id: 7 });
		}) as typeof fetch;
		const reply = await postJson(fetchStub, {
			label: "Test",
			url: "https://example.test/send",
			headers: { Authorization: "Bearer k" },
			body: { text: "hi" },
			secrets: ["k"],
		});
		expect(reply).toEqual({ ok: true, id: 7 });
		expect(seen?.url).toBe("https://example.test/send");
		expect(seen?.init.method).toBe("POST");
		expect(JSON.parse(String(seen?.init.body))).toEqual({ text: "hi" });
		expect(new Headers(seen?.init.headers).get("content-type")).toBe("application/json");
		expect(new Headers(seen?.init.headers).get("authorization")).toBe("Bearer k");
	});

	it("returns a non-JSON reply as text", async () => {
		const fetchStub = (async () => new Response("ok", { status: 200 })) as typeof fetch;
		expect(await postJson(fetchStub, { label: "T", url: "https://x.test", body: {}, secrets: [] })).toBe("ok");
	});

	it("turns an HTTP error into a MessagingError with the status and the scrubbed provider message", async () => {
		const fetchStub = (async () => jsonResponse({ message: "bad token SECRET99" }, 401)) as typeof fetch;
		const error = await postJson(fetchStub, {
			label: "Discord",
			url: "https://x.test",
			body: {},
			secrets: ["SECRET99"],
		}).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toContain("Discord");
		expect((error as Error).message).toContain("401");
		expect((error as Error).message).toContain("bad token [redacted]");
		expect((error as Error).message).not.toContain("SECRET99");
	});

	it("turns a network failure into a MessagingError without leaking the URL's secret", async () => {
		const fetchStub = (async () => {
			throw new TypeError("fetch failed for https://api.telegram.org/botSECRET99/sendMessage");
		}) as typeof fetch;
		const error = await postJson(fetchStub, {
			label: "Telegram",
			url: "https://api.telegram.org/botSECRET99/sendMessage",
			body: {},
			secrets: ["SECRET99"],
		}).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toContain("Telegram");
		expect((error as Error).message).not.toContain("SECRET99");
	});

	it("aborts with the caller's signal", async () => {
		const controller = new AbortController();
		const fetchStub = ((_url: unknown, init?: RequestInit) =>
			new Promise<Response>((_resolve, reject) => {
				init?.signal?.addEventListener("abort", () => reject(init.signal?.reason ?? new Error("aborted")));
			})) as typeof fetch;
		const pending = postJson(fetchStub, {
			label: "T",
			url: "https://x.test",
			body: {},
			secrets: [],
			signal: controller.signal,
		});
		controller.abort();
		await expect(pending).rejects.toThrow();
	});

	it("gives up after the timeout", async () => {
		const fetchStub = ((_url: unknown, init?: RequestInit) =>
			new Promise<Response>((_resolve, reject) => {
				init?.signal?.addEventListener("abort", () => reject(init.signal?.reason ?? new Error("aborted")));
			})) as typeof fetch;
		const error = await postJson(fetchStub, {
			label: "Slow",
			url: "https://x.test",
			body: {},
			secrets: [],
			timeoutMs: 20,
		}).catch((e: unknown) => e);
		expect(error).toBeInstanceOf(MessagingError);
		expect((error as Error).message).toMatch(/timed out/i);
	});
});
