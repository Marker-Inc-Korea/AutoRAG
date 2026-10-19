/**
 * Live E2E: a real model answers a question sent from a (local) Telegram chat.
 *
 * Gated on AUTORAG_TELEGRAM_LIVE=1 plus an OpenRouter credential; the offline suite never
 * calls a model. Telegram itself is the local fake Bot API, so no bot token is needed, but
 * the librarian, the pi session, the bridge, Jev routing, retrieval and the model are real.
 *
 *   AUTORAG_TELEGRAM_LIVE=1 AUTORAG_TELEGRAM_LIVE_MODEL=anthropic/claude-haiku-5.5 \
 *     bun run vitest run test/live-e2e/telegram-remote.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { FAKE_TELEGRAM_USER, FakeTelegramApi } from "../helpers/fake-telegram.ts";

const LIVE = process.env.AUTORAG_TELEGRAM_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_TELEGRAM_LIVE_MODEL ?? "anthropic/claude-haiku-5.5";
const realFetch = globalThis.fetch;

let root: string;
let fake: FakeTelegramApi;

beforeEach(async () => {
	root = mkdtempSync("/tmp/arag-tg-live-");
	fake = await FakeTelegramApi.start();
	process.env.PI_TELEGRAM_NETWORK_FAMILY = "auto";
	// Only the Telegram origin is redirected; model traffic goes to the real provider.
	globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
		const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
		return realFetch(
			url.startsWith("https://api.telegram.org") ? url.replace("https://api.telegram.org", fake.origin) : input,
			init,
		);
	}) as typeof fetch;
});

afterEach(async () => {
	globalThis.fetch = realFetch;
	delete process.env.PI_TELEGRAM_NETWORK_FAMILY;
	await fake.stop();
	rmSync(root, { recursive: true, force: true });
});

describe.skipIf(!LIVE)("Telegram remote control, live model", () => {
	it("answers a question from the corpus sent over Telegram, citing the file, and takes a second turn", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(
			join(docs, "lease.md"),
			"# Seoul office lease\n\nThe Seoul office lease renews on 2027-03-31. Monthly rent is 4,800,000 KRW.\n",
		);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		writeFileSync(
			join(agentDir, "telegram.json"),
			JSON.stringify({
				profiles: { default: { botToken: "777000:LIVE-QA", allowedUserId: FAKE_TELEGRAM_USER.id } },
			}),
		);
		// The model the way `autorag tui` resolves it from config/env, so catalog lookup and auth match.
		writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
		const config = resolveConfig({
			flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
			env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel(config);
		const agent = new AutoRAGAgent({
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			// Machine-wide finders would search the developer's real disk, not the fixture corpus.
			everything: false,
			fsearch: false,
			webSearch: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
			telegram: true,
		});
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			await hosted.runtime.session.bindExtensions({
				mode: "tui",
				uiContext: {
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
				} as unknown as ExtensionUIContext,
			});
			await hosted.runtime.session.prompt("/telegram-connect");
			await vi.waitFor(() => expect(fake.methodsCalled()).toContain("getUpdates"), { timeout: 15_000 });

			fake.say("When does the Seoul office lease renew? Answer only from my indexed docs.");
			await vi.waitFor(() => expect(fake.replies.join("\n")).toContain("2027"), { timeout: 120_000 });

			// A message sent while a search runs steers it (by design); wait for the answer to settle to ask a new one.
			await agent.waitForIdle();
			const toolCalls = hosted.runtime.session.messages.flatMap((message) =>
				message.role === "assistant"
					? message.content.flatMap((block) =>
							block.type === "toolCall"
								? [`${block.name} ${JSON.stringify(block.arguments).slice(0, 160)}`]
								: [],
						)
					: [],
			);
			console.log("TOOLCALLS_TURN1", JSON.stringify(toolCalls, null, 1));

			fake.say("And what is the monthly rent?");
			await vi.waitFor(() => expect(fake.replies.join("\n")).toMatch(/4[,.]?800[,.]?000/), { timeout: 120_000 });
		} finally {
			await hosted.dispose();
		}
	}, 300_000);
});
