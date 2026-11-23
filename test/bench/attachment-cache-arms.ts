import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall, type Message } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { estimateTextTokens } from "../../src/agent/token-count.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { AttachmentRootsSkill } from "../../src/datasource/attachments.ts";
import type { DatasourceAttachmentRoot, DatasourceSkill } from "../../src/datasource/types.ts";
import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

/**
 * Prompt-cache benchmark for the attachment-folder context.
 *
 * Question: does handing the agent a datasource's attachment folders disturb the part of the request a provider
 * caches (system prompt, tool declarations, earlier turns)? Two arms run the same scripted session on the same
 * synthetic datasource: `off` (no folders) and `on` (folders reported). The folder gains its first file between
 * searches, as a download would.
 *
 *   bun test/bench/attachment-cache-arms.ts                      # offline: byte-level prefix check, faux model
 *   bun test/bench/attachment-cache-arms.ts --live MODEL [--runs N]
 *                                                                # live: provider-reported cache reads per request
 *
 * Offline, per arm, over the requests the model receives:
 *   head     sha of the system message (prompt + tool declarations), workspace path normalized
 *   extends  how many request-to-request steps keep the earlier request as an exact prefix
 *   tokens   size of the last request, so the arms' difference is what the feature adds
 * Live sends only synthetic text (never real chats) to an OpenRouter model, with a per-session nonce at the head of
 * the system prompt so sessions and arms never share a cache entry, and reports cacheRead / (input + cacheRead +
 * cacheWrite) over requests 2.. of each session. MODEL must be a 2026+ release.
 */

const { values } = parseArgs({
	options: { live: { type: "string" }, runs: { type: "string", default: "3" } },
});

const CHATTER: readonly RetrievalResult[] = [1, 2, 3].map((n) => ({
	id: `c${n}`,
	source: `/telegram/default/chunks/c${n}`,
	content: `Gajae sent a photo of the Bali villa invoice, part ${n}. Total due 4,800 USD by Friday.`,
	score: 1 - n / 10,
	metadata: {},
}));

function stubSkill(): DatasourceSkill {
	const method: RetrievalMethod = {
		describe: () => ({
			name: "telegram.keyword",
			type: "bm25",
			description: "synthetic",
			status: "active",
			capabilities: ["keyword"],
			datasourceId: "telegram",
			tags: ["telegram"],
		}),
		retrieve: async (_query: string, options: RetrievalOptions) => CHATTER.slice(0, options.topK ?? CHATTER.length),
	};
	return {
		describe: () => ({
			name: "telegram",
			type: "chat",
			description: "Telegram chats",
			capabilities: ["keyword"],
			tags: ["telegram"],
			status: "active",
			datasourceId: "telegram",
			instanceId: "default",
		}),
		polling: () => ({ mode: "none" }),
		skillManifest: () => ({
			name: "datasource-telegram",
			description: "Search Telegram chats.",
			content: "# Telegram\nSearch with search_datasource_telegram.",
		}),
		index: async () => ({
			ok: true,
			instanceId: "default",
			skill: "telegram",
			chunkCount: 3,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [method],
		describeSources: () => [
			{
				source: "/telegram/default",
				datasourceId: "telegram",
				skill: "telegram",
				instanceId: "default",
				contentType: "chat",
				metadata: {},
			},
		],
	};
}

/** The one private member the bench needs: the exact tool list a real session declares, so the request head has its real size. */
interface SessionToolSource {
	steerableTools(): AgentTool[];
}

interface Fixture {
	readonly root: string;
	readonly mediaDir: string;
	readonly agent: AutoRAGAgent;
	readonly tools: AgentTool[];
}

function build(arm: "off" | "on", model: ConstructorParameters<typeof AutoRAGAgent>[0]["model"]): Fixture {
	const root = mkdtempSync(join(tmpdir(), `attach-cache-${arm}-`));
	const mediaDir = join(root, "media");
	const roots: DatasourceAttachmentRoot[] = [{ id: "media", path: mediaDir, description: "Cached media." }];
	const base = stubSkill();
	const agent = new AutoRAGAgent({
		model,
		searchPaths: ["test/fixtures/sample-project"],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		minSync: { autoInstall: false },
		datasourceSkills: [arm === "on" ? new AttachmentRootsSkill(base, roots) : base],
	});
	// steerableTools is private and has no public equivalent; the structural type names exactly what is read.
	const toolSource: SessionToolSource = agent as unknown as SessionToolSource;
	return { root, mediaDir, agent, tools: toolSource.steerableTools() };
}

/** Length of the longest common prefix of two strings. */
function commonPrefix(a: string, b: string): number {
	const n = Math.min(a.length, b.length);
	let i = 0;
	while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i += 1;
	return i;
}

interface Request {
	readonly system: string;
	readonly whole: string;
	/** Every tool result so far: which tool, and its text. */
	readonly toolResults: readonly { readonly tool: string; readonly text: string }[];
}

interface OfflineArm {
	readonly head: string;
	readonly requests: number;
	readonly extendingSteps: number;
	readonly divergedAt: number | undefined;
	readonly tokens: number;
	readonly manifestStates: readonly string[];
	readonly searchStates: readonly string[];
}

const STATE = /\((has files|empty|not created yet)\)/gu;
const statesIn = (text: string): string[] => [...text.matchAll(STATE)].map((match) => match[1] ?? "");

async function runOffline(arm: "off" | "on"): Promise<OfflineArm> {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "bench" }] });
	const fixture = build(arm, registration.getModel());
	const requests: Request[] = [];
	const call = (name: string, args: Parameters<typeof fauxToolCall>[1]) =>
		fauxAssistantMessage([fauxToolCall(name, args)], { stopReason: "toolUse" });
	const steps: FauxResponseStep[] = [
		(ctx) => {
			requests.push(capture(ctx.messages));
			return call("load_datasource_skill", { name: "datasource-telegram" });
		},
		(ctx) => {
			requests.push(capture(ctx.messages));
			return call("search_datasource_telegram", { query: "villa invoice", topK: 3 });
		},
		(ctx) => {
			requests.push(capture(ctx.messages));
			// A download lands between the two searches.
			mkdirSync(fixture.mediaDir, { recursive: true });
			writeFileSync(join(fixture.mediaDir, "invoice.png"), "x");
			return fauxAssistantMessage("The villa invoice is 4,800 USD.", { stopReason: "stop" });
		},
		(ctx) => {
			requests.push(capture(ctx.messages));
			return call("search_datasource_telegram", { query: "villa invoice again", topK: 3 });
		},
		(ctx) => {
			requests.push(capture(ctx.messages));
			return fauxAssistantMessage("Same: 4,800 USD due Friday.", { stopReason: "stop" });
		},
	];
	registration.setResponses(steps);
	const session = await createAutoRAGPiSession({
		cwd: fixture.root,
		model: registration.getModel(),
		getSystemPrompt: () => fixture.agent.getSystemPrompt(),
		customTools: fixture.tools,
		persistSession: false,
	});
	try {
		await session.session.prompt("What does the villa invoice say?");
		await session.session.prompt("Search once more for it.");
		let extendingSteps = 0;
		let divergedAt: number | undefined;
		for (let i = 0; i + 1 < requests.length; i += 1) {
			const a = requests[i]?.whole ?? "";
			const b = requests[i + 1]?.whole ?? "";
			if (b.startsWith(a)) extendingSteps += 1;
			else divergedAt ??= commonPrefix(a, b);
		}
		const last = requests.at(-1);
		// The workspace path is part of the system prompt and differs per temp dir; it is not part of the question.
		const normalized = (requests[0]?.system ?? "").split(fixture.root).join("<ROOT>");
		return {
			head: createHash("sha256").update(normalized).digest("hex").slice(0, 12),
			requests: requests.length,
			extendingSteps,
			divergedAt,
			tokens: estimateTextTokens(last?.whole ?? ""),
			manifestStates: (last?.toolResults ?? [])
				.filter((r) => r.tool === "load_datasource_skill")
				.flatMap((r) => statesIn(r.text)),
			searchStates: (last?.toolResults ?? [])
				.filter(
					(r) =>
						r.tool === "search_datasource_telegram" && r.text.includes("Attachment folders for this datasource"),
				)
				.flatMap((r) => statesIn(r.text)),
		};
	} finally {
		await session.dispose();
		registration.unregister();
		rmSync(fixture.root, { recursive: true, force: true });
	}
}

/** What the model is sent for one request: the system message, the whole transcript, and each tool result by tool. */
function capture(messages: readonly Message[]): Request {
	const toolResults = messages.flatMap((message) =>
		message.role === "toolResult"
			? [
					{
						tool: message.toolName,
						text: message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
					},
				]
			: [],
	);
	const first = messages[0];
	const system =
		first?.role === "system"
			? typeof first.content === "string"
				? first.content
				: first.content.map((part) => part.text).join("")
			: "";
	return { system, whole: messages.map((message) => JSON.stringify(message)).join("\n"), toolResults };
}

async function offline(): Promise<void> {
	const off = await runOffline("off");
	const on = await runOffline("on");
	const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;
	console.log(
		"| arm | requests | system+tools head (sha) | steps keeping the earlier request as exact prefix | last request tokens |",
	);
	console.log("|---|---|---|---|---|");
	for (const [name, arm] of [
		["off", off],
		["on", on],
	] as const) {
		console.log(
			`| ${name} | ${arm.requests} | ${arm.head} | ${arm.extendingSteps}/${arm.requests - 1} (${pct(arm.extendingSteps / (arm.requests - 1))}) | ${arm.tokens} |`,
		);
	}
	console.log(`\nhead identical across arms: ${off.head === on.head}`);
	console.log(`tokens the feature adds over the session: ${on.tokens - off.tokens}`);
	console.log(`folder state printed by load_datasource_skill: ${on.manifestStates.join(", ") || "(none)"}`);
	console.log(`folder state printed by each search, in order: ${on.searchStates.join(" -> ") || "(none)"}`);
	if (off.head !== on.head || off.divergedAt !== undefined || on.divergedAt !== undefined) process.exitCode = 1;
}

async function live(modelId: string): Promise<void> {
	const root = mkdtempSync(join(tmpdir(), "attach-cache-live-"));
	// An isolated agent dir: no real MCP servers, extensions, skills or credentials files are loaded.
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	process.env.PI_CODING_AGENT_DIR = agentDir;
	writeFileSync(
		join(root, "config.json"),
		JSON.stringify({ searchPaths: [join(process.cwd(), "test/fixtures/sample-project")] }),
	);
	const config = resolveConfig({
		flags: { "model-provider": "openrouter", "model-id": modelId },
		env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json"), PI_CODING_AGENT_DIR: agentDir },
		cwd: root,
		readOnly: true,
	});
	const resolved = await resolveAgentModel(config);
	const runs = Number(values.runs);
	const results: Record<
		"off" | "on",
		{ ratio: number; read: number; write: number; input: number; requests: number; notices: number }[]
	> = { off: [], on: [] };
	// Interleaved so provider-side drift (routing, load) hits both arms equally.
	for (let run = 0; run < runs; run += 1) {
		for (const arm of run % 2 === 0 ? (["off", "on"] as const) : (["on", "off"] as const)) {
			const fixture = build(arm, resolved.model);
			const nonce = `bench-nonce-${randomUUID()}`;
			const session = await createAutoRAGPiSession({
				cwd: fixture.root,
				agentDir,
				model: resolved.model,
				...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
				...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
				getSystemPrompt: () => `${nonce}\n${fixture.agent.getSystemPrompt()}`,
				customTools: fixture.tools,
				persistSession: false,
			});
			try {
				const prompts = [
					"Call load_datasource_skill with name datasource-telegram, then call search_datasource_telegram with query 'villa invoice' and topK 3. Reply with one short sentence.",
					"Call search_datasource_telegram again with query 'invoice due date' and topK 3. Reply with one short sentence.",
					"Call search_datasource_telegram once more with query 'Friday payment' and topK 3. Reply with one short sentence.",
				];
				for (const prompt of prompts) {
					await session.session.prompt(prompt);
					// A download lands after each turn.
					mkdirSync(fixture.mediaDir, { recursive: true });
					writeFileSync(join(fixture.mediaDir, `f${Date.now()}.png`), "x");
				}
				const messages = session.session.messages;
				const usages = messages.flatMap((m) => (m.role === "assistant" ? [m.usage] : []));
				// Request 1 can only write the cache; every later request should read it.
				const later = usages.slice(1);
				const read = later.reduce((sum, u) => sum + u.cacheRead, 0);
				const write = later.reduce((sum, u) => sum + u.cacheWrite, 0);
				const input = later.reduce((sum, u) => sum + u.input, 0);
				const ratio = read / Math.max(1, read + write + input);
				const notices = messages.filter(
					(m) =>
						m.role === "toolResult" &&
						m.content.some(
							(part) => part.type === "text" && part.text.includes("Attachment folders for this datasource"),
						),
				).length;
				results[arm].push({ ratio, read, write, input, requests: usages.length, notices });
				console.log(
					`run ${run + 1} ${arm}: requests=${usages.length} notices=${notices} hit=${pct(ratio)} read=${read} write=${write} uncached=${input}`,
				);
				console.log(
					`   per request, cacheRead/uncached: ${usages.map((u) => `${u.cacheRead}/${u.input}`).join("  ")}`,
				);
			} finally {
				await session.dispose();
				rmSync(fixture.root, { recursive: true, force: true });
			}
		}
	}
	rmSync(root, { recursive: true, force: true });
	const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
	console.log(`\nmodel: ${modelId}`);
	console.log(
		"| arm | sessions | mean hit ratio (requests 2..) | min | max | mean cacheRead | mean uncached input | mean requests | mean notices |",
	);
	console.log("|---|---|---|---|---|---|---|---|---|");
	for (const arm of ["off", "on"] as const) {
		const r = results[arm];
		const ratios = r.map((x) => x.ratio);
		console.log(
			`| ${arm} | ${r.length} | ${pct(mean(ratios))} | ${pct(Math.min(...ratios))} | ${pct(Math.max(...ratios))} | ${Math.round(mean(r.map((x) => x.read)))} | ${Math.round(mean(r.map((x) => x.input)))} | ${mean(r.map((x) => x.requests)).toFixed(1)} | ${mean(r.map((x) => x.notices)).toFixed(1)} |`,
		);
	}
}

function pct(x: number): string {
	return `${(x * 100).toFixed(1)}%`;
}

if (values.live !== undefined) await live(values.live);
else await offline();
