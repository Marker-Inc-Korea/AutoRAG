import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";
import { FAKE_TELEGRAM_USER, FakeTelegramApi } from "../helpers/fake-telegram.ts";

const BOT_TOKEN = "777000:FAKE-TOKEN-FOR-LOCAL-QA";
const realFetch = globalThis.fetch;

let root: string;
let fake: FakeTelegramApi;
let previousAgentDir: string | undefined;
let previousFamily: string | undefined;

beforeEach(async () => {
	// The bridge binds unix sockets under the agent dir; macOS caps socket paths near 104 bytes,
	// which os.tmpdir() (/var/folders/...) already overshoots, so use /tmp directly.
	root = mkdtempSync(join(process.platform === "win32" ? tmpdir() : "/tmp", "arag-tg-"));
	fake = await FakeTelegramApi.start();
	previousAgentDir = process.env.PI_CODING_AGENT_DIR;
	previousFamily = process.env.PI_TELEGRAM_NETWORK_FAMILY;
	// "auto" sends bridge traffic through global fetch; the default opens https sockets itself.
	process.env.PI_TELEGRAM_NETWORK_FAMILY = "auto";
	// The bridge hard-codes api.telegram.org; point only that origin at the local fake.
	globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
		const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
		return realFetch(url.replace("https://api.telegram.org", fake.origin), init);
	}) as typeof fetch;
});

afterEach(async () => {
	globalThis.fetch = realFetch;
	if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
	else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
	if (previousFamily === undefined) delete process.env.PI_TELEGRAM_NETWORK_FAMILY;
	else process.env.PI_TELEGRAM_NETWORK_FAMILY = previousFamily;
	await fake.stop();
	rmSync(root, { recursive: true, force: true });
});

/** The slice of pi's TUI context the bridge touches; every call is a no-op. */
function quietUi(): ExtensionUIContext {
	return {
		notify: () => undefined,
		setStatus: () => undefined,
		setWidget: () => undefined,
		confirm: async () => true,
		select: async () => undefined,
		input: async () => undefined,
		onTerminalInput: () => () => undefined,
		setWorkingMessage: () => undefined,
		setWorkingVisible: () => undefined,
		setTitle: () => undefined,
		theme: undefined as never,
	} as unknown as ExtensionUIContext;
}

/** A TUI-hosted librarian with the Telegram bridge enabled, paired to the fake owner. */
async function telegramAgent(answer: string) {
	const registration = registerFauxProvider({ api: `faux-telegram-${randomUUID()}`, models: [{ id: "tg-model" }] });
	registration.setResponses([fauxAssistantMessage(answer, { stopReason: "stop" })]);
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	process.env.PI_CODING_AGENT_DIR = agentDir;
	const model = registration.getModel();
	writeFileSync(
		join(agentDir, "settings.json"),
		JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
	);
	writeFileSync(
		join(agentDir, "telegram.json"),
		JSON.stringify({ profiles: { default: { botToken: BOT_TOKEN, allowedUserId: FAKE_TELEGRAM_USER.id } } }),
	);
	const agent = new AutoRAGAgent({
		model,
		apiKey: "test-key",
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		telegram: true,
		jev: {
			backend: new MockBackend({
				[QUERY_ROUTE_QUESTION_ID]: {
					answer: "direct",
					distribution: { local: 0.1, web: 0.1, direct: 0.8 },
					confidence: 0.8,
				},
				[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
				[FOLLOW_UP_QUESTION_ID]: { answer: 0.1 },
			}),
		},
	});
	const hosted = await agent.createPiInteractiveRuntime();
	// What InteractiveMode binds: a long-lived "tui" session, which the bridge requires to poll.
	await hosted.runtime.session.bindExtensions({ mode: "tui", uiContext: quietUi() });
	return {
		agent,
		registration,
		session: hosted.runtime.session,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

describe("Telegram remote control of the librarian", () => {
	it("registers the bridge's commands on the AutoRAG session when enabled", async () => {
		const tg = await telegramAgent("unused");
		try {
			const commands = tg.session.extensionRunner.getRegisteredCommands().map((command) => command.name);
			expect(commands).toEqual(expect.arrayContaining(["telegram-setup", "telegram-connect", "telegram-status"]));
		} finally {
			await tg.dispose();
		}
	});

	it("answers a message sent from Telegram through the AutoRAG pipeline and replies in the chat", async () => {
		const tg = await telegramAgent("Paris is the capital of France.");
		try {
			await tg.session.prompt("/telegram-connect");
			await vi.waitFor(() => expect(fake.methodsCalled()).toContain("getUpdates"), { timeout: 15_000 });

			fake.say("What is the capital of France?");

			await vi.waitFor(() => expect(fake.replies.join("\n")).toContain("Paris is the capital of France."), {
				timeout: 30_000,
			});
			// One model call, issued by AutoRAG's direct route: not a raw pi turn.
			expect(tg.registration.state.callCount).toBe(1);
			const prompts = tg.session.messages.flatMap((message) =>
				message.role === "user" ? [JSON.stringify(message.content)] : [],
			);
			// Same prompt a terminal user gets: the bridge's envelope never reaches the model or the session name.
			expect(prompts.join("\n")).toContain(
				"Answer this query directly from your own general knowledge: What is the capital of France?",
			);
			expect(prompts.join("\n")).not.toMatch(/\[telegram\]|\[time\]/u);
			expect(tg.session.sessionName).toBe("What is the capital of France?");
		} finally {
			await tg.dispose();
		}
	});

	it("keeps the bridge off the session unless it is enabled", async () => {
		const registration = registerFauxProvider({ api: `faux-off-${randomUUID()}`, models: [{ id: "off" }] });
		const agentDir = join(root, "agent-off");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			apiKey: "k",
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory-off.json"),
			minSync: { autoInstall: false },
			jikji: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions-off"),
		});
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			await hosted.runtime.session.bindExtensions({});
			const commands = hosted.runtime.session.extensionRunner.getRegisteredCommands().map((command) => command.name);
			expect(commands).not.toContain("telegram-connect");
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	});
});
