import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runServe } from "../../src/cli/commands/serve.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { main, parseArgs } from "../../src/cli/index.ts";
import type { P2pServer } from "../../src/p2p/server.ts";

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

function stubServer(overrides: Partial<P2pServer> = {}): P2pServer {
	return {
		url: "http://127.0.0.1:19470",
		origin: "http://127.0.0.1:19470",
		host: "127.0.0.1",
		port: 19470,
		queue: {} as never,
		close: async () => undefined,
		...overrides,
	};
}

describe("autorag serve", () => {
	it("parses serve flags and lists the command in help", async () => {
		const parsed = parseArgs(["serve", "--port", "19470", "--host", "127.0.0.1"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["serve"]);
		expect(parsed.flags.port).toBe("19470");
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
		const stderr: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath },
				stderr: (line) => stderr.push(line),
			}),
			{ startP2pServer: async () => stubServer() },
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/p2p.*disabled|p2p.*not.*enabled|enable.*p2p|p2p.*enable/i);
	});

	it("starts with --force even when p2p is disabled in config", async () => {
		const stdout: string[] = [];
		let closeCalled = false;
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true },
				stdout: (line) => stdout.push(line),
			}),
			{
				startP2pServer: async () =>
					stubServer({
						close: async () => {
							closeCalled = true;
						},
					}),
				waitUntilStopped: async (server) => {
					await server.close();
				},
			},
		);
		expect(code).toBe(0);
		expect(closeCalled).toBe(true);
	});

	it("starts on a free port with stub server, prints fingerprint (not full key), shuts down cleanly", async () => {
		const stdout: string[] = [];
		let closeCalled = false;
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true },
				stdout: (line) => stdout.push(line),
				json: true,
			}),
			{
				startP2pServer: async () =>
					stubServer({
						port: 19471,
						close: async () => {
							closeCalled = true;
						},
					}),
				waitUntilStopped: async (server) => {
					await server.close();
				},
				getFingerprint: async () => "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
			},
		);
		expect(code).toBe(0);
		expect(closeCalled).toBe(true);

		const payload = JSON.parse(stdout[0] ?? "{}") as Record<string, unknown>;
		expect(payload.ok).toBe(true);
		expect(payload.host).toBe("127.0.0.1");
		expect(payload.port).toBe(19471);
		expect(payload.fingerprint).toBe("abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890");
		// No full pubkey or private key in output
		const stdoutText = stdout.join(" ");
		expect(stdoutText).not.toContain("privateKey");
		expect(stdoutText).not.toContain("pubkey");
		expect(stdoutText).not.toContain("secret");
	});

	it("surfaces ConfigError for invalid config", async () => {
		// Write a config with invalid p2p value
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
			{ startP2pServer: async () => stubServer() },
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/ConfigError|config|invalid/i);
	});

	it("uses flags --port and --host", async () => {
		const stdout: string[] = [];
		const code = await runServe(
			makeCtx({
				flags: { config: configPath, force: true, port: "19472", host: "0.0.0.0" },
				stdout: (line) => stdout.push(line),
				json: true,
			}),
			{
				startP2pServer: async (opts) =>
					stubServer({
						host: opts.host ?? "127.0.0.1",
						port: opts.port ?? 19472,
						close: async () => undefined,
					}),
				waitUntilStopped: async () => undefined,
				getFingerprint: async () => "test-fp",
			},
		);
		expect(code).toBe(0);
		const payload = JSON.parse(stdout[0] ?? "{}") as Record<string, unknown>;
		expect(payload.host).toBe("0.0.0.0");
		expect(payload.port).toBe(19472);
	});
});
