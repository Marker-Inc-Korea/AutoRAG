/**
 * Live E2E: a real model is asked to message a real channel, through the real agent, the real `send_message`
 * tool, and the real provider API. Gated so the offline suite never calls a model or sends anything:
 *
 *   AUTORAG_MESSAGING_LIVE=1 OPENROUTER_API_KEY=... \
 *   TELEGRAM_BOT_TOKEN=... AUTORAG_MESSAGING_TELEGRAM_CHAT=<chat id> \
 *   DISCORD_BOT_TOKEN=... AUTORAG_MESSAGING_DISCORD_CHANNEL=<channel id> \
 *   SLACK_WEBHOOK_URL=... AGENTMAIL_API_KEY=... AUTORAG_MESSAGING_EMAIL_INBOX=<inbox> AUTORAG_MESSAGING_EMAIL_TO=<address> \
 *     bunx vitest run test/live-e2e/messaging.test.ts
 *
 * Each lane runs only when its variables are set and its channel kind is registered, so a lane is never
 * reported as passing without a real send. Every message carries a nonce; the acceptance bar is that the
 * model called `send_message` with the right channel, the provider returned its own message id, and (where the
 * provider allows reading it back) the delivered text holds the nonce. These messages go to the operator's own
 * chats: they are labelled as QA.
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { randomBytes } from "node:crypto";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import {
	channelKinds,
	normalizeMessagingConfig,
	registerBuiltinChannelKinds,
	SEND_MESSAGE_TOOL_NAME,
} from "../../src/messaging/index.ts";

const LIVE = process.env.AUTORAG_MESSAGING_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_MESSAGING_LIVE_MODEL ?? "anthropic/claude-haiku-5.5";
const env = process.env;

registerBuiltinChannelKinds();
const kinds = new Set(channelKinds());

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-messaging-live-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

interface Recorded {
	readonly args: Record<string, unknown>;
	readonly output: string;
	readonly isError: boolean;
	readonly details: { channel?: string; destination?: string; ids?: string[] };
}

interface Internals {
	readonly tools: {
		name: string;
		execute: (...args: never[]) => Promise<{ content: unknown; isError?: boolean; details?: unknown }>;
	}[];
}

const textOf = (content: unknown): string =>
	Array.isArray(content) ? content.map((part) => (typeof part?.text === "string" ? part.text : "")).join("") : "";

async function liveAgent(messaging: unknown, permissionMode: "full" | "ask" = "full") {
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	process.env.PI_CODING_AGENT_DIR = agentDir;
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "note.txt"), "Unrelated filler so the corpus is not empty.\n");
	writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
	const config = resolveConfig({
		flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
		env: { ...env, AUTORAG_CONFIG: join(root, "config.json") },
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
		everything: false,
		fsearch: false,
		webSearch: false,
		filee: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		jev: { backend: "openrouter" },
		permissionMode,
		messaging: normalizeMessagingConfig(messaging),
	});
	const calls: Recorded[] = [];
	const tool = (agent as unknown as Internals).tools.find((candidate) => candidate.name === SEND_MESSAGE_TOOL_NAME);
	if (tool === undefined) throw new Error("the agent registered no send_message tool");
	const real = tool.execute;
	tool.execute = (async (...args: never[]) => {
		const result = await real(...args);
		calls.push({
			args: args[1] as unknown as Record<string, unknown>,
			output: textOf(result.content),
			isError: result.isError === true,
			details: (result.details ?? {}) as Recorded["details"],
		});
		return result;
	}) as never;
	return { agent, calls };
}

const nonce = () => `qa-${randomBytes(4).toString("hex")}`;
const ask = (channel: string, text: string) =>
	`Send this exact message to the "${channel}" channel using the send_message tool: ${text}`;

describe.skipIf(!LIVE)("send_message with a live model", () => {
	it.skipIf(!kinds.has("telegram") || !env.TELEGRAM_BOT_TOKEN || !env.AUTORAG_MESSAGING_TELEGRAM_CHAT)(
		"delivers to a real Telegram chat and Telegram returns the message id",
		async () => {
			const { agent, calls } = await liveAgent({
				channels: {
					tg: { kind: "telegram", tokenEnv: "TELEGRAM_BOT_TOKEN", chatId: env.AUTORAG_MESSAGING_TELEGRAM_CHAT },
				},
			});
			const mark = nonce();
			await agent.searchDocuments(ask("tg", `AutoRAG messaging QA ${mark} (safe to ignore)`));
			const sent = calls.filter((call) => !call.isError);
			expect(sent, `calls: ${JSON.stringify(calls)}`).toHaveLength(1);
			expect(sent[0]?.args.channel).toBe("tg");
			expect(String(sent[0]?.args.text)).toContain(mark);
			expect(sent[0]?.details.ids?.[0]).toMatch(/^\d+$/);
		},
		300_000,
	);

	it.skipIf(!kinds.has("discord") || !env.DISCORD_BOT_TOKEN || !env.AUTORAG_MESSAGING_DISCORD_CHANNEL)(
		"delivers to a real Discord channel, reads it back, and a mention in the text pings nobody",
		async () => {
			const channelId = String(env.AUTORAG_MESSAGING_DISCORD_CHANNEL);
			const { agent, calls } = await liveAgent({
				channels: { dc: { kind: "discord", tokenEnv: "DISCORD_BOT_TOKEN", channelId } },
			});
			const mark = nonce();
			await agent.searchDocuments(ask("dc", `AutoRAG messaging QA ${mark} @everyone (safe to ignore)`));
			const sent = calls.filter((call) => !call.isError);
			expect(sent, `calls: ${JSON.stringify(calls)}`).toHaveLength(1);
			const id = sent[0]?.details.ids?.[0];
			expect(id).toMatch(/^\d+$/);
			const back = await fetch(`https://discord.com/api/v10/channels/${channelId}/messages/${id}`, {
				headers: { authorization: `Bot ${env.DISCORD_BOT_TOKEN}` },
			});
			expect(back.status).toBe(200);
			const message = (await back.json()) as { content: string; mention_everyone: boolean };
			expect(message.content).toContain(mark);
			expect(message.mention_everyone).toBe(false);
		},
		300_000,
	);

	it.skipIf(!kinds.has("slack") || !env.SLACK_WEBHOOK_URL)(
		"delivers through a real Slack webhook, which answers ok",
		async () => {
			const { agent, calls } = await liveAgent({
				channels: { sl: { kind: "slack", webhookEnv: "SLACK_WEBHOOK_URL" } },
			});
			const mark = nonce();
			await agent.searchDocuments(ask("sl", `AutoRAG messaging QA ${mark} (safe to ignore)`));
			const sent = calls.filter((call) => !call.isError);
			expect(sent, `calls: ${JSON.stringify(calls)}`).toHaveLength(1);
			expect(String(sent[0]?.args.text)).toContain(mark);
			expect(sent[0]?.details.ids).toEqual(["ok"]);
		},
		300_000,
	);

	it.skipIf(
		!kinds.has("agentmail") ||
			!env.AGENTMAIL_API_KEY ||
			!env.AUTORAG_MESSAGING_EMAIL_INBOX ||
			!env.AUTORAG_MESSAGING_EMAIL_TO,
	)(
		"sends a real email from the agent's own inbox",
		async () => {
			const { agent, calls } = await liveAgent({
				channels: {
					mail: { kind: "agentmail", apiKeyEnv: "AGENTMAIL_API_KEY", inboxId: env.AUTORAG_MESSAGING_EMAIL_INBOX },
				},
			});
			const mark = nonce();
			await agent.searchDocuments(
				`Email ${env.AUTORAG_MESSAGING_EMAIL_TO} through the "mail" channel with send_message. Subject: "AutoRAG QA ${mark}". Body: "AutoRAG messaging QA ${mark} (safe to ignore)".`,
			);
			const sent = calls.filter((call) => !call.isError);
			expect(sent, `calls: ${JSON.stringify(calls)}`).toHaveLength(1);
			expect(sent[0]?.args.to).toEqual([env.AUTORAG_MESSAGING_EMAIL_TO]);
			expect(String(sent[0]?.args.subject)).toContain(mark);
			expect(sent[0]?.details.ids?.[0]).toBeTruthy();
		},
		300_000,
	);

	it.skipIf(!kinds.has("slack") || !env.SLACK_WEBHOOK_URL)(
		"in ask mode with nobody to approve, the model's send is blocked and nothing leaves the machine",
		async () => {
			const { agent, calls } = await liveAgent(
				{ channels: { sl: { kind: "slack", webhookEnv: "SLACK_WEBHOOK_URL" } } },
				"ask",
			);
			const response = await agent.searchDocuments(ask("sl", `AutoRAG messaging QA ${nonce()} MUST-NOT-ARRIVE`));
			expect(
				calls.filter((call) => !call.isError),
				`the send must not reach the provider: ${JSON.stringify(calls)}`,
			).toHaveLength(0);
			expect(response.answer.length).toBeGreaterThan(0);
		},
		300_000,
	);
});
