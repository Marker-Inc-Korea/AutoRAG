#!/usr/bin/env bun
/**
 * Manual QA for `autorag tui --discord`: runs the real TUI process against a
 * local fake Discord (REST + gateway) and a fake OpenAI-compatible model.
 *
 * Usage (from the repo root; needs a pty because the TUI is interactive):
 *   bun scripts/manual-qa/discord-remote-qa.ts
 *
 * The script spawns `bun --preload <this file> src/cli/index.ts tui --discord`
 * with an isolated HOME / AUTORAG_HOME / pi agent dir. Inside that process this
 * file (loaded as the preload) starts the fakes, plays a Discord user who asks a
 * question, and writes the observed Discord traffic to RESULT_PATH.
 */
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { FakeDiscord } from "../../test/helpers/fake-discord.ts";

const GUILD = "7001";
const TOKEN = "qa-bot-token";
const ALLOWED = "42";
const STRANGER = "99";
const ANSWER = "QA-ANSWER: the handbook lives in docs/.";

/**
 * The fake decision model's answer to one question: the local route, verification after the fast answer
 * (so the fast answer is posted first), every citation kept, and "no" to everything else.
 */
function fakeDecision(id: string, question: { type?: string; criteria?: unknown }): Record<string, unknown> {
	if (question.type === "choice") {
		const options = typeof question.criteria === "object" && question.criteria !== null ? Object.keys(question.criteria) : [];
		const choice = id === "route" ? "local" : (options[0] ?? "");
		return { type: "choice", choice, probabilities: { [choice]: 1 }, confidence: 1 };
	}
	if (question.type === "score") return { type: "score", score: 0, probabilities: { 0: 1 }, confidence: 1 };
	const yes = id === "follow_up" || id === "decision_model_probe" || id.startsWith("citation:");
	return { type: "noul", noul: yes ? 1 : 0 };
}

/** Starts the fakes (awaited by the preload, so they exist before the TUI boots) and returns the driver. */
async function startFakes(): Promise<() => Promise<void>> {
	const resultPath = process.env.QA_RESULT_PATH as string;
	const result: Record<string, unknown> = {};
	const model = createServer((req, res) => {
		let body = "";
		req.on("data", (chunk: Buffer) => {
			body += chunk.toString();
		});
		req.on("end", () => {
			// Jev is required: the same port serves the decision model's native dialect.
			if (req.url?.includes("decisions") || req.url?.includes("systemone")) {
				const request = JSON.parse(body) as { questions: Record<string, { type?: string; criteria?: unknown }> };
				const answers = Object.fromEntries(
					Object.entries(request.questions).map(([id, question]) => [id, fakeDecision(id, question)]),
				);
				res.writeHead(200, { "content-type": "application/json" });
				res.end(JSON.stringify({ model: "qa-jev", answers }));
				return;
			}
			res.writeHead(200, { "content-type": "text/event-stream" });
			const chunk = (delta: unknown, finish: string | null) =>
				`data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", created: 1, model: "qa-model", choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
			res.write(chunk({ role: "assistant", content: ANSWER }, null));
			res.write(chunk({}, "stop"));
			res.write("data: [DONE]\n\n");
			res.end();
		});
	});
	await new Promise<void>((done) => model.listen(Number(process.env.QA_MODEL_PORT), "127.0.0.1", done));
	const fake = await FakeDiscord.start({ guildId: GUILD, botToken: TOKEN });
	return async () => {
	try {
		await fake.waitFor(() => fake.sessionChannelId !== undefined, "session channel", 60_000);
		const channelId = fake.sessionChannelId as string;
		result.channelName = fake.channels.get(channelId)?.name;
		fake.sendUserMessage(channelId, STRANGER, "stranger: run rm -rf");
		await fake.waitForPost((post) => post.content.includes("not on the allow-list"), 30_000);
		result.strangerRejected = true;
		fake.sendUserMessage(channelId, ALLOWED, "Where is the handbook?");
		const reply = await fake.waitForPost((post) => post.content.includes(`<@${ALLOWED}>`), 60_000);
		result.reply = reply.content;
		result.pingedAsker = reply.content.includes(`<@${ALLOWED}>`);
		const answers = fake.posted.filter((post) => post.content.includes("QA-ANSWER"));
		// The earlier phase shows up first, without the ping; the final answer pings the asker.
		result.fastAnswerShownFirst = answers.length === 2 && !answers[0]?.content.includes(`<@${ALLOWED}>`);
		result.posts = fake.posted.map((post) => post.content);
	} catch (error) {
		result.error = String(error);
	}
	writeFileSync(resultPath, JSON.stringify(result, null, 2));
	await fake.close();
	model.close();
	process.exit(0);
	};
}

async function orchestrate(): Promise<number> {
	const root = mkdtempSync(join(tmpdir(), "autorag-discord-qa-"));
	const home = join(root, "home");
	const autoragHome = join(root, ".autorag-home");
	const docs = join(root, "docs");
	const agentDir = join(root, "pi-agent");
	for (const dir of [home, autoragHome, docs, agentDir, join(home, ".pi", "agent", "pi-discord-remote")]) {
		mkdirSync(dir, { recursive: true });
	}
	writeFileSync(join(docs, "handbook.md"), "# Handbook\nThe handbook lives in docs/.\n");
	writeFileSync(
		join(home, ".pi", "agent", "pi-discord-remote", "config.json"),
		JSON.stringify({ token: TOKEN, guildId: GUILD, allowedUserIds: [ALLOWED], reactions: true }),
	);
	const modelPort = 20_000 + Math.floor(Math.random() * 20_000);
	const config = join(autoragHome, "config.json");
	writeFileSync(
		config,
		JSON.stringify({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(autoragHome, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			jev: { backend: "openrouter" },
			model: { provider: "qamodel", id: "qa-model", baseUrl: `http://127.0.0.1:${modelPort}/v1`, apiKeyEnv: "QAMODEL_API_KEY" },
		}),
	);
	if (!config.startsWith(root)) throw new Error("refusing to run: config is outside the QA root");
	const resultPath = join(root, "result.json");
	// Pi's TUI picks its model from the agent dir's settings; pin the fake one.
	writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ defaultProvider: "qamodel", defaultModel: "qa-model" }));
	// Never inherit real credentials: pass only what a shell needs, so no real provider can be reached.
	const env = {
		PATH: process.env.PATH,
		TERM: "xterm-256color",
		HOME: home,
		AUTORAG_HOME: autoragHome,
		AUTORAG_CONFIG: config,
		PI_CODING_AGENT_DIR: agentDir,
		AUTORAG_NO_UPDATE_CHECK: "1",
		QAMODEL_API_KEY: "qa-key",
		// The fake model port also answers Jev's decisions endpoint; no real provider can be reached.
		OPENROUTER_API_KEY: "qa-key",
		OPENROUTER_BASE_URL: `http://127.0.0.1:${modelPort}`,
		QA_RESULT_PATH: resultPath,
		QA_MODEL_PORT: String(modelPort),
		AUTORAG_DISCORD_QA_INNER: "1",
	};
	const scriptPath = fileURLToPath(import.meta.url);
	// pty.spawn allocates the pseudo-terminal the interactive TUI needs; a pipe stdin is fine.
	const argv = ["bun", "--preload", scriptPath, "src/cli/index.ts", "tui", "--discord", "--config", config];
	const child = spawn("python3", ["-c", "import pty,sys; sys.exit(pty.spawn(sys.argv[1:]) >> 8)", ...argv], {
		env,
		stdio: ["pipe", "pipe", "pipe"],
	});
	let output = "";
	child.stdout.on("data", (data: Buffer) => {
		output += data.toString();
	});
	child.stderr.on("data", (data: Buffer) => {
		output += data.toString();
	});
	let exitCode: number | null | undefined;
	child.on("exit", (code) => {
		exitCode = code;
	});
	const deadline = Date.now() + 120_000;
	while (!existsSync(resultPath) && exitCode === undefined && Date.now() < deadline) await new Promise((r) => setTimeout(r, 200));
	child.kill("SIGTERM");
	const result = existsSync(resultPath) ? JSON.parse(readFileSync(resultPath, "utf8")) : undefined;
	console.log(JSON.stringify(result, null, 2));
	const ok = result !== undefined && result.error === undefined && result.strangerRejected === true && result.pingedAsker === true && result.fastAnswerShownFirst === true;
	if (!ok) console.error(`FAIL (child exit ${exitCode})\n--- tui output tail ---\n${output.slice(-3000)}`);
	else console.log("PASS");
	rmSync(root, { recursive: true, force: true });
	return ok ? 0 : 1;
}

if (process.env.AUTORAG_DISCORD_QA_INNER === "1") void (await startFakes())();
else process.exit(await orchestrate());
