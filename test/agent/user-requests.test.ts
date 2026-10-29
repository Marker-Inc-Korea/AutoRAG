import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import {
	createUserRequestPersistence,
	renderUserRequests,
	USER_REQUEST_ENTRY_TYPE,
	type UserRequest,
	UserRequestLog,
} from "../../src/agent/user-requests.ts";

type Handler = (event: unknown, ctx: unknown) => void;

/** A pi API that keeps appended entries, and replays them through the session manager the way pi does. */
function fakePi() {
	const handlers = new Map<string, Handler[]>();
	const appended: { customType: string; data: unknown }[] = [];
	const pi = {
		on: (event: string, handler: Handler) => void handlers.set(event, [...(handlers.get(event) ?? []), handler]),
		appendEntry: (customType: string, data: unknown) => void appended.push({ customType, data }),
	} as unknown as ExtensionAPI;
	const fire = (event: string, branch: SessionEntry[] = []) => {
		const ctx = { sessionManager: { getBranch: () => branch, getEntries: () => branch } };
		for (const handler of handlers.get(event) ?? []) handler({ type: event }, ctx);
	};
	return { pi, appended, fire };
}

function entry(id: string, text: string, kind: "request" | "steer" = "request"): SessionEntry {
	return {
		type: "custom",
		id,
		parentId: null,
		timestamp: "2026-10-23T00:00:00.000Z",
		customType: USER_REQUEST_ENTRY_TYPE,
		data: { text, kind },
	} as SessionEntry;
}

describe("UserRequestLog", () => {
	it("keeps every user message in order, with steers marked", () => {
		const log = new UserRequestLog();
		log.record("first question");
		log.record("follow-up", "request");
		log.record("not that one", "steer");
		expect(log.entries()).toEqual([
			{ text: "first question", kind: "request" },
			{ text: "follow-up", kind: "request" },
			{ text: "not that one", kind: "steer" },
		]);
	});

	it("ignores blank text and never stores a truncated message", () => {
		const log = new UserRequestLog();
		log.record("   ");
		log.record("x".repeat(50_000));
		expect(log.entries()).toHaveLength(1);
		expect(log.entries()[0]?.text).toHaveLength(50_000);
	});

	it("starts over on reset", () => {
		const log = new UserRequestLog();
		log.record("old");
		log.reset();
		expect(log.entries()).toEqual([]);
	});
});

describe("createUserRequestPersistence", () => {
	it("saves each message to the session as it is recorded", () => {
		const log = new UserRequestLog();
		const { pi, appended, fire } = fakePi();
		createUserRequestPersistence(log)(pi);
		fire("session_start");
		log.record("clean up my downloads");
		log.record("only the pdfs", "steer");
		expect(appended).toEqual([
			{ customType: USER_REQUEST_ENTRY_TYPE, data: { text: "clean up my downloads", kind: "request" } },
			{ customType: USER_REQUEST_ENTRY_TYPE, data: { text: "only the pdfs", kind: "steer" } },
		]);
	});

	it("rebuilds the log from the branch on session start, so /resume brings the conversation back", () => {
		const log = new UserRequestLog();
		log.record("stale from another session");
		const { pi, fire } = fakePi();
		createUserRequestPersistence(log)(pi);
		fire("session_start", [entry("a", "first"), entry("b", "second"), entry("c", "fix", "steer")]);
		expect(log.entries().map((item) => item.text)).toEqual(["first", "second", "fix"]);
		expect(log.entries().map((item) => item.kind)).toEqual(["request", "request", "steer"]);
	});

	it("starts empty in a fresh session (/new) and follows the branch on tree navigation", () => {
		const log = new UserRequestLog();
		log.record("before /new");
		const { pi, fire } = fakePi();
		createUserRequestPersistence(log)(pi);
		fire("session_start", []);
		expect(log.entries()).toEqual([]);
		fire("session_tree", [entry("a", "branch only")]);
		expect(log.entries().map((item) => item.text)).toEqual(["branch only"]);
	});

	it("skips malformed entries instead of failing the session", () => {
		const log = new UserRequestLog();
		const { pi, fire } = fakePi();
		createUserRequestPersistence(log)(pi);
		const broken = { ...entry("x", "ignored"), data: { text: 42 } } as SessionEntry;
		fire("session_start", [broken, entry("y", "kept")]);
		expect(log.entries().map((item) => item.text)).toEqual(["kept"]);
	});

	it("stops saving once the session shuts down", () => {
		const log = new UserRequestLog();
		const { pi, appended, fire } = fakePi();
		createUserRequestPersistence(log)(pi);
		fire("session_start");
		fire("session_shutdown");
		log.record("after shutdown");
		expect(appended).toEqual([]);
		expect(log.entries()).toHaveLength(1);
	});
});

describe("renderUserRequests", () => {
	const request = (text: string): UserRequest => ({ text, kind: "request" });

	it("lists every message in order, verbatim, marking corrections", () => {
		const text = renderUserRequests(
			[request("clean up my downloads"), request("also the pdfs"), { text: "not the contracts", kind: "steer" }],
			10_000,
		);
		expect(text).toBe(
			[
				"[1] clean up my downloads",
				"[2] also the pdfs",
				"[3] (correction while the agent was working) not the contracts",
			].join("\n"),
		);
	});

	it("says so when there is nothing to show", () => {
		expect(renderUserRequests([], 1000)).toBe("(no user message available)");
	});

	it("keeps a long conversation whole when it fits, however long a single message is", () => {
		const long = "a".repeat(40_000);
		const text = renderUserRequests([request(long), request("second")], 100_000);
		expect(text).toContain(long);
		expect(text).toContain("[2] second");
		expect(text).not.toMatch(/omitted|truncated/);
	});

	it("over the limit, keeps the opening request and the newest messages and drops the middle", () => {
		const messages = [
			"opening request",
			...Array.from({ length: 50 }, (_, i) => `middle message ${i} ${"x".repeat(40)}`),
			"latest steer",
		];
		const text = renderUserRequests(messages.map(request), 600);
		expect(text.length).toBeLessThanOrEqual(600);
		expect(text).toContain("[1] opening request");
		expect(text).toContain("latest steer");
		expect(text).toMatch(/\[\d+ earlier messages omitted\]/);
		expect(text).not.toContain("middle message 0 ");
	});

	it("numbers messages by their place in the conversation even after dropping some", () => {
		const text = renderUserRequests(
			[request("one"), request("y".repeat(300)), request("y".repeat(300)), request("last")],
			200,
		);
		expect(text).toContain("[1] one");
		expect(text).toContain("[4] last");
	});

	it("cuts the middle of a single message that alone exceeds the limit, keeping both ends", () => {
		const text = renderUserRequests([request(`START ${"m".repeat(5000)} END`)], 400);
		expect(text.length).toBeLessThanOrEqual(400);
		expect(text).toContain("START");
		expect(text).toContain("END");
		expect(text).toMatch(/characters omitted/);
	});
});

describe("UserRequestLog attach", () => {
	it("a stale detach does not unplug a newer session's saver", () => {
		const log = new UserRequestLog();
		const saved: string[] = [];
		const detachOld = log.attach(() => void saved.push("old"));
		log.attach(() => void saved.push("new"));
		detachOld();
		log.record("hello");
		expect(saved).toEqual(["new"]);
	});
});
