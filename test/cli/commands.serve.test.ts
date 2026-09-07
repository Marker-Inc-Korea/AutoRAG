import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runServe } from "../../src/cli/commands/serve.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { main, parseArgs } from "../../src/cli/index.ts";
import type { SignalPeerServer } from "../../src/p2p/signal-server.ts";
import type { SignalIncomingMessage, SignalTransport } from "../../src/p2p/signal-transport.ts";

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
			p2p: { injectionClassifier: false, account: "+821012345678" },
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

function stubServer(): SignalPeerServer {
	return { close: async () => undefined };
}

function stubTransport(account = "+821012345678"): SignalTransport & { closed: boolean } {
	const state = { closed: false };
	return {
		account,
		get closed() {
			return state.closed;
		},
		sendMessage: async () => undefined,
		onMessage: (_handler: (message: SignalIncomingMessage) => void) => () => {},
		close: async () => {
			state.closed = true;
		},
	};
}

describe("autorag serve", () => {
	it("parses serve flags and lists the command in help", async () => {
		const parsed = parseArgs(["serve", "--port", "17583", "--host", "127.0.0.1"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["serve"]);
		expect(parsed.flags.port).toBe("17583");
		expect(parsed.flags.host).toBe("127.0.0.1");

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
				p2p: { enabled: false, injectionClassifier: false, account: "+821012345678" },
			}),
		);
		const stderr: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath },
				stderr: (line) => stderr.push(line),
			}),
			{ startSignalPeerServer: async () => stubServer(), startSignalDaemon: async () => stubTransport() },
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/p2p.*disabled|p2p.*not.*enabled|enable.*p2p|p2p.*enable/i);
	});

	it("refuses to start without a Signal account", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				p2p: { enabled: true, injectionClassifier: false },
			}),
		);
		const stderr: string[] = [];
		const code = await runServe(makeCtx({ flags: { config: configPath }, stderr: (line) => stderr.push(line) }), {
			startSignalPeerServer: async () => stubServer(),
			startSignalDaemon: async () => stubTransport(),
		});
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/account/i);
	});

	it("starts with --force even when p2p is disabled in config, closes daemon cleanly", async () => {
		const stdout: string[] = [];
		const transport = stubTransport();
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true },
				stdout: (line) => stdout.push(line),
			}),
			{
				startSignalDaemon: async () => transport,
				startSignalPeerServer: async () => stubServer(),
				waitUntilStopped: async (server) => {
					await server.close();
				},
			},
		);
		expect(code).toBe(0);
		expect(transport.closed).toBe(true);
	});

	it("reports the account and daemon bind, never private material", async () => {
		const stdout: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true },
				stdout: (line) => stdout.push(line),
				json: true,
			}),
			{
				startSignalDaemon: async () => stubTransport(),
				startSignalPeerServer: async () => stubServer(),
				waitUntilStopped: async (server) => {
					await server.close();
				},
			},
		);
		expect(code).toBe(0);
		const payload = JSON.parse(stdout[0] ?? "{}") as Record<string, unknown>;
		expect(payload.ok).toBe(true);
		expect(payload.account).toBe("+821012345678");
		expect(payload.daemon).toEqual({ host: "127.0.0.1", port: 7583 });
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
				p2p: { enabled: true, injectionClassifier: false, account: "+821012345678" },
			}),
		);
		let receivedAgent: { remoteSession?: boolean; searchDocuments: unknown } | undefined;
		const code = await runServe(makeCtx({ flags: { config: configPath } }), {
			startSignalDaemon: async () => stubTransport(),
			startSignalPeerServer: async (options) => {
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
			{ startSignalPeerServer: async () => stubServer(), startSignalDaemon: async () => stubTransport() },
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/ConfigError|config|invalid/i);
	});

	it("passes --port and --host through to the signal-cli daemon bind", async () => {
		const stdout: string[] = [];
		let daemonOptions: { host?: string; port?: number } | undefined;
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true, port: "17590", host: "127.0.0.2" },
				stdout: (line) => stdout.push(line),
				json: true,
			}),
			{
				startSignalDaemon: async (options) => {
					daemonOptions = options;
					return stubTransport();
				},
				startSignalPeerServer: async () => stubServer(),
				waitUntilStopped: async () => undefined,
			},
		);
		expect(code).toBe(0);
		expect(daemonOptions?.host).toBe("127.0.0.2");
		expect(daemonOptions?.port).toBe(17590);
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
					account: "+821012345678",
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
			startSignalDaemon: async () => stubTransport(),
			startSignalPeerServer: async (options) => {
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
				p2p: { enabled: true, account: "+821012345678", injectionClassifier: true },
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
				startSignalPeerServer: async () => stubServer(),
				startSignalDaemon: async () => stubTransport(),
			},
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/classifier|model/i);
	});
});
