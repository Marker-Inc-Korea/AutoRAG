/**
 * Live E2E: secret & OTP redaction across mirror indexing, decision models, and real LLM sessions.
 *
 * Gated so the offline suite never touches the network or spends credits:
 *
 *   AUTORAG_SECRET_REDACTION_LIVE=1 OPENROUTER_API_KEY=... \
 *     bunx vitest run test/live-e2e/secret-redaction.test.ts
 *
 * Uses Qwen3.7 Flash (2026-07) and TypeSafe Jev 1.13 on OpenRouter through a local loopback spy
 * proxy so the test proves at the HTTP wire level that no raw secret or OTP leaves the process.
 */

import { once } from "node:events";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { loadMirrorIndex, syncParsedMirrors } from "../../src/mirror/index.ts";
import { createDefaultParserRegistry } from "../../src/parser/index.ts";

const LIVE = process.env.AUTORAG_SECRET_REDACTION_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
/** Qwen3.7 Flash (2026-07) on OpenRouter. */
const OPENROUTER_MODEL = { provider: "openrouter", id: "qwen/qwen3.7-flash" };
const OPENROUTER_UPSTREAM = "https://openrouter.ai/api/v1";

// Synthesized at runtime so no literal credential sits in source.
const AWS_KEY = `AKIA${"Q7W8".repeat(4)}`;
const GITHUB_KEY = `ghp_${"kL9mN2pQ4r".repeat(3)}xyz123`;
const OTP_EN = "482913";
const OTP_KO = "558201";

interface SpyProxy {
	readonly baseUrl: string;
	readonly bodies: string[];
	close(): Promise<void>;
}

async function startSpyProxy(upstreamBase: string): Promise<SpyProxy> {
	const bodies: string[] = [];
	const server: Server = createServer(async (req, res) => {
		const chunks: Buffer[] = [];
		for await (const chunk of req) chunks.push(Buffer.from(chunk));
		const body = Buffer.concat(chunks);
		bodies.push(body.toString("utf8"));

		const headers = new Headers();
		for (const [key, value] of Object.entries(req.headers)) {
			if (value === undefined) continue;
			const lower = key.toLowerCase();
			if (lower === "host" || lower === "content-length" || lower === "connection") continue;
			headers.set(key, Array.isArray(value) ? value.join(", ") : value);
		}
		const target = `${upstreamBase}${req.url ?? ""}`;
		const upstream = await fetch(target, {
			method: req.method ?? "POST",
			headers,
			...(body.length > 0 ? { body } : {}),
		});
		res.statusCode = upstream.status;
		upstream.headers.forEach((value, key) => {
			if (key.toLowerCase() === "transfer-encoding" || key.toLowerCase() === "content-encoding") return;
			res.setHeader(key, value);
		});
		const responseBytes = Buffer.from(await upstream.arrayBuffer());
		res.end(responseBytes);
	});
	server.listen(0, "127.0.0.1");
	await once(server, "listening");
	const { port } = server.address() as AddressInfo;
	const closeServer = promisify(server.close.bind(server));
	return {
		baseUrl: `http://127.0.0.1:${port}`,
		bodies,
		close: () => closeServer(),
	};
}

describe.skipIf(!LIVE)("live E2E: secret and OTP redaction on OpenRouter", () => {
	let proxy: SpyProxy;
	const tempDirs: string[] = [];

	function tempDir(prefix: string): string {
		const dir = mkdtempSync(join(tmpdir(), prefix));
		tempDirs.push(dir);
		return dir;
	}

	beforeAll(async () => {
		proxy = await startSpyProxy(OPENROUTER_UPSTREAM);
	});

	afterAll(async () => {
		await proxy.close();
	});

	afterEach(() => {
		while (tempDirs.length > 0) {
			rmSync(tempDirs.pop()!, { recursive: true, force: true });
		}
	});

	it("redacts secrets in indexed mirror markdown and prevents wire leaks to the live LLM", async () => {
		const root = tempDir("autorag-live-redact-");
		const docs = join(root, "docs");
		const cfgPath = join(root, "config.json");
		mkdirSync(docs, { recursive: true });
		writeFileSync(
			join(docs, "runbook.txt"),
			[
				"Project Orion launch date: 2026-12-15",
				`Deploy AWS access key: ${AWS_KEY}`,
				`[Web발신] 인증번호 [${OTP_KO}] 를 입력해주세요`,
			].join("\n"),
		);
		writeFileSync(cfgPath, JSON.stringify({ searchPaths: [docs] }));

		// 1) Mirror indexing redacts secrets on disk.
		await syncParsedMirrors({ root, searchPaths: [docs], registry: createDefaultParserRegistry() });
		const entry = loadMirrorIndex(root).entries["/docs/runbook.txt"];
		expect(entry).toBeDefined();
		const mirrored = readFileSync(entry!.outputPath, "utf8");
		expect(mirrored).toContain("2026-12-15");
		expect(mirrored).not.toContain(AWS_KEY);
		expect(mirrored).not.toContain(OTP_KO);
		expect(mirrored).toContain("[REDACTED:aws-access-token]");
		expect(mirrored).toContain("[REDACTED:otp-code]");

		// 2) Live LLM session through loopback spy proxy: both user prompt and tool output carry secrets.
		const config = resolveConfig({
			flags: { config: cfgPath },
			env: process.env,
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel({ ...config, model: OPENROUTER_MODEL });
		const proxiedModel = { ...resolved.model, baseUrl: proxy.baseUrl };
		const apiKey = resolved.apiKey ?? resolved.providerApiKeys?.[resolved.model.provider];

		const readRawNote: AgentTool = {
			name: "read_runbook",
			label: "Read runbook",
			description: "Reads the local runbook note",
			parameters: Type.Object({}),
			async execute() {
				// Return raw (unredacted) content from disk as a tool result to prove model-input redaction catches it too.
				return {
					content: [{ type: "text", text: readFileSync(join(docs, "runbook.txt"), "utf8") }],
					details: {},
				};
			},
		};

		const beforeCount = proxy.bodies.length;
		const session = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: proxiedModel,
			...(apiKey !== undefined ? { apiKey } : {}),
			getSystemPrompt: () => "You are a concise assistant. Call read_runbook first, then answer in one sentence.",
			customTools: [readRawNote],
		});
		try {
			await session.session.prompt(
				`Call read_runbook. Tell me the Project Orion launch date and quote any token or code you see. (Context: ${GITHUB_KEY}, verification code ${OTP_EN})`,
			);
			const outbound = proxy.bodies.slice(beforeCount);
			expect(outbound.length).toBeGreaterThanOrEqual(1);
			const wire = outbound.join("\n");
			expect(wire).not.toContain(AWS_KEY);
			expect(wire).not.toContain(GITHUB_KEY);
			expect(wire).not.toContain(OTP_EN);
			expect(wire).not.toContain(OTP_KO);
			expect(wire).toContain("[REDACTED:github-token]");
			expect(wire).toContain("[REDACTED:otp-code]");

			const lastAssistant = [...session.session.messages].reverse().find((m) => m.role === "assistant");
			const reply = JSON.stringify(lastAssistant ?? "");
			expect(reply).not.toContain(AWS_KEY);
			expect(reply).not.toContain(GITHUB_KEY);
			expect(reply).not.toContain(OTP_EN);
			expect(reply).not.toContain(OTP_KO);
			expect(reply).toContain("2026-12-15");
		} finally {
			session.session.dispose();
		}
	}, 90_000);

	it("redacts secrets before calling live OpenRouter decision model", async () => {
		const judge = createJevJudge({ backend: "openrouter", model: "typesafe/jev-1.13" });
		const judgment = await judge(`User asked to delete old backup after pasting ${AWS_KEY} and 인증번호: ${OTP_KO}`, [
			{
				id: "has_redacted",
				type: "noul",
				question: "Does the state contain a redacted placeholder marker like [REDACTED:...]?",
			},
		]);
		expect(judgment.backend).toBe("openrouter");
		expect(judgment.answers.has_redacted?.answer as number).toBeGreaterThan(0.5);
	}, 60_000);
});
