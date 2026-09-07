import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runServe } from "../../src/cli/commands/serve.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { main, parseArgs } from "../../src/cli/index.ts";
import type { SimplexPeerServer } from "../../src/p2p/simplex-server.ts";
import type { SimplexIncomingMessage, SimplexTransport } from "../../src/p2p/simplex-transport.ts";

let root: string;
let configPath: string;
const noop = (): void => undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-serve-"));
	configPath = join(root, "config.json");
	mkdirSync(join(root, "docs"));
	writeFileSync(
		configPath,
		JSON.stringify({
			searchPaths: [join(root, "docs")],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			p2p: { injectionClassifier: false },
		}),
	);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: { config: configPath },
		json: false,
		debug: false,
		cwd: root,
		stdout: noop,
		stderr: noop,
		...overrides,
	};
}

function stubServer(): SimplexPeerServer {
	return { close: async () => undefined };
}

function stubTransport(): SimplexTransport & { closed: boolean } {
	const state = { closed: false };
	return {
		dbPrefix: "stub",
		displayName: "stub",
		get closed() {
			return state.closed;
		},
		getUserId: async () => 1,
		getOrCreateAddress: async () => "simplex:/contact#stub",
		createInvitation: async () => "simplex:/invitation#stub",
		connect: async () => undefined,
		listContacts: async () => [],
		sendMessage: async () => undefined,
		onMessage: (_handler: (message: SimplexIncomingMessage) => void) => () => {},
		close: async () => {
			state.closed = true;
		},
	};
}

describe("autorag serve", () => {
	it("parses serve flags and lists the command in help", async () => {
		const parsed = parseArgs(["serve", "--port", "25225"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["serve"]);
		expect(parsed.flags.port).toBe("25225");

		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			await main(["--help"]);
			const usage = out.mock.calls.map((call) => String(call[0] ?? "")).join("");
			expect(usage).toContain("serve");
		} finally {
			out.mockRestore();
		}
	});

	it("refuses to start with p2p disabled and no --force", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				p2p: { enabled: false, injectionClassifier: false },
			}),
		);
		const stderr: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath },
				stderr: (line) => stderr.push(line),
			}),
			{ startSimplexPeerServer: async () => stubServer(), startSimplexChat: async () => stubTransport() },
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/p2p.*disabled|p2p.*not.*enabled|enable.*p2p|p2p.*enable/i);
	});

	it("starts with --force even when p2p is disabled in config, closes transport cleanly", async () => {
		const stdout: string[] = [];
		const transport = stubTransport();
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true },
				stdout: (line) => stdout.push(line),
			}),
			{
				startSimplexChat: async () => transport,
				startSimplexPeerServer: async () => stubServer(),
				waitUntilStopped: async (server) => {
					await server.close();
				},
			},
		);
		expect(code).toBe(0);
		expect(transport.closed).toBe(true);
	});

	it("reports the SimpleX address and port, never private material", async () => {
		const stdout: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true },
				stdout: (line) => stdout.push(line),
				json: true,
			}),
			{
				startSimplexChat: async () => stubTransport(),
				startSimplexPeerServer: async () => stubServer(),
				waitUntilStopped: async (server) => {
					await server.close();
				},
			},
		);
		expect(code).toBe(0);
		const payload = JSON.parse(stdout[0] ?? "{}") as Record<string, unknown>;
		expect(payload.ok).toBe(true);
		expect(payload.address).toBe("simplex:/contact#stub");
		expect(payload.port).toBe(5225);
		const stdoutText = stdout.join(" ");
		expect(stdoutText).not.toContain("privateKey");
		expect(stdoutText).not.toContain("pubkey");
		expect(stdoutText).not.toContain("secret");
	});

	it("constructs the production search agent in remote-session mode", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				p2p: { enabled: true, injectionClassifier: false },
			}),
		);
		let receivedAgent: { remoteSession?: boolean; searchDocuments: unknown } | undefined;
		const code = await runServe(makeCtx({ flags: { config: configPath } }), {
			startSimplexChat: async () => stubTransport(),
			startSimplexPeerServer: async (options) => {
				receivedAgent = options.agent as typeof receivedAgent;
				return stubServer();
			},
			waitUntilStopped: async () => undefined,
		});
		expect(code).toBe(0);
		expect(receivedAgent?.remoteSession).toBe(true);
		expect(typeof receivedAgent?.searchDocuments).toBe("function");
	});

	it("surfaces ConfigError for invalid config", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				p2p: "not-an-object",
			}),
		);
		const stderr: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath },
				stderr: (line) => stderr.push(line),
			}),
			{ startSimplexPeerServer: async () => stubServer(), startSimplexChat: async () => stubTransport() },
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/ConfigError|config|invalid/i);
	});

	it("passes --port through to the simplex-chat WebSocket server", async () => {
		const stdout: string[] = [];
		let chatOptions: { port?: number } | undefined;
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true, port: "25299" },
				stdout: (line) => stdout.push(line),
				json: true,
			}),
			{
				startSimplexChat: async (options) => {
					chatOptions = options;
					return stubTransport();
				},
				startSimplexPeerServer: async () => stubServer(),
				waitUntilStopped: async () => undefined,
			},
		);
		expect(code).toBe(0);
		expect(chatOptions?.port).toBe(25299);
	});

	it("passes classifier model, search roots, and quota limits to the server", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				p2p: {
					enabled: true,
					injectionClassifier: true,
					piiNer: true,
					quotas: { queriesPerHour: 7, burst: 2 },
				},
			}),
		);
		let received: Record<string, unknown> | undefined;
		const code = await runServe(makeCtx({ flags: { config: configPath } }), {
			modelResolver: () => ({
				model: {
					id: "stub",
					name: "stub",
					api: "openai-completions",
					provider: "stub",
					baseUrl: "http://127.0.0.1:9",
					reasoning: false,
					input: ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: 128,
					maxTokens: 16,
				},
			}),
			startSimplexChat: async () => stubTransport(),
			startSimplexPeerServer: async (options) => {
				received = options as unknown as Record<string, unknown>;
				return stubServer();
			},
			waitUntilStopped: async () => undefined,
		});
		expect(code).toBe(0);
		expect(received?.injectionClassifier).toBe(true);
		expect(typeof received?.injectionClassifierModel).toBe("function");
		expect(received?.pseudonymize).toBe(true);
		expect(received?.quotas).toEqual({ queriesPerHour: 7, burst: 2 });
		expect(received?.workspaceRoots).toEqual([join(root, "docs")]);
		expect(received?.policyStore).toBeDefined();
	});

	it("fails closed at startup when the classifier is enabled without a model", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				p2p: { enabled: true, injectionClassifier: true },
			}),
		);
		const stderr: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath },
				stderr: (line) => stderr.push(line),
			}),
			{
				modelResolver: () => {
					throw new Error("no model");
				},
				startSimplexPeerServer: async () => stubServer(),
				startSimplexChat: async () => stubTransport(),
			},
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/classifier|model/i);
	});
});
