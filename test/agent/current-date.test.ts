import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, getSystemMessageText } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { currentDateAddendum } from "../../src/agent/current-date.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";

describe("currentDateAddendum", () => {
	it("states the local date, weekday, and UTC offset of the given instant", () => {
		const text = currentDateAddendum(new Date(2026, 9, 25, 19, 40, 30));
		expect(text).toContain("2026-10-25");
		expect(text).toContain("Sunday");
		expect(text).toMatch(/UTC[+-]\d{2}:\d{2}/);
		expect(text).toContain("<current_date>");
	});

	it("zero-pads month and day", () => {
		expect(currentDateAddendum(new Date(2026, 0, 3, 4, 5, 0))).toContain("2026-01-03");
	});

	it("is byte-identical for every instant of one local day, so the cache prefix survives the clock", () => {
		const morning = currentDateAddendum(new Date(2026, 9, 25, 0, 0, 0));
		expect(currentDateAddendum(new Date(2026, 9, 25, 8, 5, 59))).toBe(morning);
		expect(currentDateAddendum(new Date(2026, 9, 25, 19, 40, 30))).toBe(morning);
		expect(currentDateAddendum(new Date(2026, 9, 25, 23, 59, 59))).toBe(morning);
		expect(currentDateAddendum(new Date(2026, 9, 26, 0, 0, 0))).not.toBe(morning);
	});

	it("tells the model to resolve relative dates against that date", () => {
		expect(currentDateAddendum(new Date(2026, 9, 25, 19, 40))).toMatch(/relative/i);
	});
});

describe("the system prompt the model receives", () => {
	let root: string;
	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-date-"));
		mkdirSync(join(root, "agent"), { recursive: true });
	});
	afterEach(() => {
		vi.useRealTimers();
		rmSync(root, { recursive: true, force: true });
	});

	async function systemPromptsSeenAt(instants: readonly Date[]): Promise<string[]> {
		const registration = registerFauxProvider({ api: `faux-date-${Date.now()}`, models: [{ id: "date-model" }] });
		const seen: string[] = [];
		const step = () => (context: { messages: { role: string }[] }) => {
			const system = context.messages.filter((message) => message.role === "system");
			seen.push(system.map((message) => getSystemMessageText(message as never)).join("\n"));
			return fauxAssistantMessage("ok", { stopReason: "stop" });
		};
		registration.setResponses(instants.map(() => step() as never));
		try {
			vi.useFakeTimers({ toFake: ["Date"] });
			const [first] = instants;
			if (first !== undefined) vi.setSystemTime(first);
			const { session } = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
			});
			for (const [index, instant] of instants.entries()) {
				vi.setSystemTime(instant);
				await session.prompt(`prompt ${index}`);
			}
			session.dispose();
		} finally {
			registration.unregister();
		}
		return seen;
	}

	it("carries the current date, refreshed on every prompt instead of frozen at session start", async () => {
		const seen = await systemPromptsSeenAt([new Date(2026, 9, 25, 19, 40), new Date(2026, 9, 27, 8, 5)]);
		expect(seen).toHaveLength(2);
		expect(seen[0]).toContain("AutoRAG system prompt");
		expect(seen[0]).toContain("2026-10-25");
		expect(seen[1]).toContain("2026-10-27");
		expect(seen[1]).not.toContain("2026-10-25");
	});

	it("sends the same system prompt for prompts in different minutes of one day", async () => {
		const seen = await systemPromptsSeenAt([
			new Date(2026, 9, 25, 19, 40),
			new Date(2026, 9, 25, 19, 41),
			new Date(2026, 9, 25, 23, 58),
		]);
		expect(seen).toHaveLength(3);
		expect(seen[1]).toBe(seen[0]);
		expect(seen[2]).toBe(seen[0]);
	});
});
