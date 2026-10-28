import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeDiscord } from "../helpers/fake-discord.ts";

// pi-discord-remote resolves its config path from the home directory when it is
// imported, so HOME must point at a sandbox before any import runs.
const sandboxHome = vi.hoisted(() => {
	const { mkdtempSync: mk } = require("node:fs") as typeof import("node:fs");
	const { tmpdir: tmp } = require("node:os") as typeof import("node:os");
	const { join: j } = require("node:path") as typeof import("node:path");
	const home = mk(j(tmp(), "autorag-discord-home-"));
	process.env.HOME = home;
	return home;
});

import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FILE_SEARCH_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";

const GUILD = "7001";
const TOKEN = "fake-bot-token";
const ALLOWED = "42";
const STRANGER = "99";

let root: string;
let fake: FakeDiscord;

beforeEach(async () => {
	root = mkdtempSync(join(tmpdir(), "autorag-discord-remote-"));
	fake = await FakeDiscord.start({ guildId: GUILD, botToken: TOKEN });
});

afterEach(async () => {
	await fake.close();
	rmSync(root, { recursive: true, force: true });
	rmSync(join(sandboxHome, ".pi"), { recursive: true, force: true });
});

function writeDiscordConfig(allowedUserIds: string[]): void {
	const dir = join(sandboxHome, ".pi", "agent", "pi-discord-remote");
	mkdirSync(dir, { recursive: true });
	writeFileSync(
		join(dir, "config.json"),
		JSON.stringify({ token: TOKEN, guildId: GUILD, allowedUserIds, reactions: true, toolResponses: false }),
	);
}

/** A UI that ignores everything but records `notify` lines. pi copies the object, so it needs real own keys. */
function recordingUi(notices: string[]): ExtensionUIContext {
	const ignore = (): void => undefined;
	return {
		notify: (message: string) => notices.push(message),
		onTerminalInput: () => ignore,
		select: async () => undefined,
		confirm: async () => false,
		input: async () => undefined,
		editor: async () => undefined,
		setStatus: ignore,
		setWorkingMessage: ignore,
		setWorkingVisible: ignore,
		setWorkingIndicator: ignore,
		setHiddenThinkingLabel: ignore,
		setWidget: ignore,
		setFooter: ignore,
		setHeader: ignore,
		setTitle: ignore,
		pasteToEditor: ignore,
		setEditorText: ignore,
		getEditorText: () => "",
		getToolsExpanded: () => false,
		setToolsExpanded: ignore,
	} as unknown as ExtensionUIContext;
}

async function discordAgent(
	answers: string[],
	options: { discordRemote: boolean; route?: "extra" | "local"; followUp?: number } = { discordRemote: true },
) {
	const route = options.route ?? "extra";
	const registration = registerFauxProvider({
		api: `faux-discord-${randomUUID()}`,
		models: [{ id: "discord-model" }],
	});
	registration.setResponses(answers.map((answer) => fauxAssistantMessage(answer, { stopReason: "stop" })));
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	const model = registration.getModel();
	writeFileSync(
		join(agentDir, "settings.json"),
		JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
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
		jev: {
			backend: new MockBackend({
				[QUERY_ROUTE_QUESTION_ID]: {
					answer: route,
					distribution: { local: 0.1, organize: 0.1, extra: 0.1, [route]: 0.8 },
					confidence: 0.8,
				},
				[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
				[FOLLOW_UP_QUESTION_ID]: { answer: options.followUp ?? 0.1 },
				// The verification phase also asks whether a file-name search helps; these runs must not run one.
				[FILE_SEARCH_QUESTION_ID]: { answer: 0 },
			}),
		},
		discordRemote: options.discordRemote,
	});
	const hosted = await agent.createPiInteractiveRuntime();
	const notices: string[] = [];
	await hosted.runtime.session.bindExtensions({
		uiContext: recordingUi(notices),
		onError: (e) => notices.push(`EXT-ERROR ${e.error}`),
	});
	return {
		agent,
		notices,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

describe("Discord remote control", () => {
	it("starts a session channel on launch and answers a Discord question in it", async () => {
		writeDiscordConfig([ALLOWED]);
		const tui = await discordAgent(["The handbook lives in docs/."]);
		try {
			await fake.waitFor(() => fake.sessionChannelId !== undefined, "session channel");
			const channelId = fake.sessionChannelId as string;
			expect(fake.identifiedWithToken).toBe(TOKEN);

			const messageId = fake.sendUserMessage(channelId, ALLOWED, "Where is the handbook?");
			const reply = await fake.waitForPost((post) => post.content.includes("The handbook lives in docs/."));

			expect(reply.channelId).toBe(channelId);
			// The final answer pings the asker.
			expect(reply.content).toContain(`<@${ALLOWED}>`);
			expect(fake.reactions).toContainEqual({ channelId, messageId, emoji: "⏳" });
		} finally {
			await tui.dispose();
		}
	});

	it("shows the fast answer while verification runs, then the final answer pinging the asker", async () => {
		writeDiscordConfig([ALLOWED]);
		const tui = await discordAgent(["Fast answer: Lyon.", "Verified: Paris."], {
			discordRemote: true,
			route: "local",
			followUp: 0.9,
		});
		try {
			await fake.waitFor(() => fake.sessionChannelId !== undefined, "session channel");
			fake.sendUserMessage(fake.sessionChannelId as string, ALLOWED, "Capital of France?");
			await fake.waitForPost((post) => post.content.includes("Verified: Paris."));
			await tui.agent.waitForIdle();
			const answers = fake.posted.filter((post) => post.content.includes("Paris") || post.content.includes("Lyon"));
			expect(answers.map((post) => post.content)).toEqual(["Fast answer: Lyon.", `<@${ALLOWED}> Verified: Paris.`]);
		} finally {
			await tui.dispose();
		}
	});

	it("ignores users outside the allow-list", async () => {
		writeDiscordConfig([ALLOWED]);
		const tui = await discordAgent(["should never be asked"]);
		try {
			await fake.waitFor(() => fake.sessionChannelId !== undefined, "session channel");
			const channelId = fake.sessionChannelId as string;
			fake.sendUserMessage(channelId, STRANGER, "run rm -rf");
			await fake.waitForPost((post) => post.content.includes("not on the allow-list"));
			expect(fake.posted.some((post) => post.content.includes("should never be asked"))).toBe(false);
		} finally {
			await tui.dispose();
		}
	});

	it("does not touch Discord unless enabled", async () => {
		writeDiscordConfig([ALLOWED]);
		const tui = await discordAgent(["unused"], { discordRemote: false });
		try {
			// bindExtensions already ran session_start; a disabled agent never reached Discord.
			expect(tui.notices).toEqual([]);
			expect(fake.channels.size).toBe(0);
			expect(fake.identifiedWithToken).toBeUndefined();
		} finally {
			await tui.dispose();
		}
	});

	it("deletes the session channel on shutdown", async () => {
		writeDiscordConfig([ALLOWED]);
		const tui = await discordAgent(["unused"]);
		await fake.waitFor(() => fake.sessionChannelId !== undefined, "session channel");
		const channelId = fake.sessionChannelId as string;
		await tui.dispose();
		await fake.waitFor(() => fake.deletedChannelIds.includes(channelId), "channel delete");
	});
});
