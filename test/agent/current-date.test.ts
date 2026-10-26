import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, getSystemMessageText } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { currentDateAddendum } from "../../src/agent/current-date.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";

describe("currentDateAddendum", () => {
	it("states the local date, weekday, time, and UTC offset of the given instant", () => {
		const text = currentDateAddendum(new Date(2026, 9, 25, 19, 40, 30));
		expect(text).toContain("2026-10-25");
		expect(text).toContain("Sunday");
		expect(text).toContain("19:40");
		expect(text).toMatch(/UTC[+-]\d{2}:\d{2}/);
		expect(text).toContain("<current_date>");
	});

	it("zero-pads month, day, hour, and minute", () => {
		const text = currentDateAddendum(new Date(2026, 0, 3, 4, 5, 0));
		expect(text).toContain("2026-01-03");
		expect(text).toContain("04:05");
	});

	it("tells the model to resolve relative dates against that instant", () => {
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

	it("carries the current date, refreshed on every turn instead of frozen at session start", async () => {
		const registration = registerFauxProvider({ api: `faux-date-${Date.now()}`, models: [{ id: "date-model" }] });
		const seen: string[] = [];
		const step = () => (context: { messages: { role: string }[] }) => {
			const system = context.messages.filter((message) => message.role === "system");
			seen.push(system.map((message) => getSystemMessageText(message as never)).join("\n"));
			return fauxAssistantMessage("ok", { stopReason: "stop" });
		};
		registration.setResponses([step() as never, step() as never]);
		try {
			vi.useFakeTimers({ toFake: ["Date"] });
			vi.setSystemTime(new Date(2026, 9, 25, 19, 40));
			const { session } = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
			});
			await session.prompt("first");
			vi.setSystemTime(new Date(2026, 9, 27, 8, 5));
			await session.prompt("second");
			session.dispose();
		} finally {
			registration.unregister();
		}
		expect(seen).toHaveLength(2);
		expect(seen[0]).toContain("AutoRAG system prompt");
		expect(seen[0]).toContain("2026-10-25");
		expect(seen[0]).toContain("19:40");
		expect(seen[1]).toContain("2026-10-27");
		expect(seen[1]).toContain("08:05");
		expect(seen[1]).not.toContain("2026-10-25");
	});
});
