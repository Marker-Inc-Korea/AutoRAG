import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, getSystemMessageText } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { environmentAddendum, runtimeContextAddendum } from "../../src/agent/environment-prompt.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";

describe("environmentAddendum", () => {
	it("names the OS, architecture, the bash tool's shell, and the working directory", () => {
		const text = environmentAddendum({
			cwd: "/Users/me/docs",
			platform: "darwin",
			arch: "arm64",
			release: "25.6.0",
			shell: "/bin/bash",
		});
		expect(text).toContain("<environment>");
		expect(text).toContain("macOS");
		expect(text).toContain("arm64");
		expect(text).toContain("25.6.0");
		expect(text).toContain("/bin/bash");
		expect(text).toContain("/Users/me/docs");
	});

	it("names Windows and Linux by their product names", () => {
		const base = {
			cwd: "C:\\work",
			arch: "x64",
			release: "10.0.26100",
			shell: "C:\\Program Files\\Git\\bin\\bash.exe",
		};
		expect(environmentAddendum({ ...base, platform: "win32" })).toContain("Windows");
		expect(environmentAddendum({ ...base, platform: "linux" })).toContain("Linux");
	});

	it("says so when no shell could be resolved instead of inventing one", () => {
		const text = environmentAddendum({
			cwd: "/x",
			platform: "win32",
			arch: "x64",
			release: "10",
			shell: undefined,
		});
		expect(text).toMatch(/no shell/i);
	});
});

describe("runtimeContextAddendum", () => {
	it("puts the stable environment before the date, which changes every minute and so goes last", () => {
		const text = runtimeContextAddendum("/Users/me/docs", new Date(2026, 9, 25, 19, 40));
		expect(text.indexOf("<environment>")).toBeGreaterThanOrEqual(0);
		expect(text.indexOf("<current_date>")).toBeGreaterThan(text.indexOf("<environment>"));
	});
});

describe("the system prompt the model receives", () => {
	let root: string;
	beforeEach(() => {
		root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-env-")));
		mkdirSync(join(root, "agent"), { recursive: true });
	});
	afterEach(() => {
		vi.useRealTimers();
		rmSync(root, { recursive: true, force: true });
	});

	it("ends with the session's cwd, then the date: base prompt, environment, date, in that order", async () => {
		const registration = registerFauxProvider({ api: `faux-env-${Date.now()}`, models: [{ id: "env-model" }] });
		let seen = "";
		registration.setResponses([
			((context: { messages: { role: string }[] }) => {
				seen = context.messages
					.filter((message) => message.role === "system")
					.map((message) => getSystemMessageText(message as never))
					.join("\n");
				return fauxAssistantMessage("ok", { stopReason: "stop" });
			}) as never,
		]);
		try {
			const { session } = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
			});
			await session.prompt("hi");
			session.dispose();
		} finally {
			registration.unregister();
		}
		const base = seen.indexOf("AutoRAG system prompt");
		const environment = seen.indexOf("<environment>");
		const date = seen.indexOf("<current_date>");
		expect(base).toBeGreaterThanOrEqual(0);
		expect(environment).toBeGreaterThan(base);
		expect(date).toBeGreaterThan(environment);
		expect(seen).toContain(root);
	});
});
