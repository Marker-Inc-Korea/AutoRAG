import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { runUi } from "../../src/cli/commands/ui.ts";
import { main, parseArgs } from "../../src/cli/index.ts";

let root: string;
let configPath: string;
const noop = (): void => undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-ui-"));
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
		flags: { config: configPath, "no-open": true, port: "0" },
		json: false,
		debug: false,
		cwd: root,
		stdout: noop,
		stderr: noop,
		...overrides,
	};
}

describe("autorag ui", () => {
	it("parses ui flags and lists the command in help", async () => {
		const parsed = parseArgs(["ui", "--no-open", "--port", "0", "--host", "127.0.0.1"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["ui"]);
		expect(parsed.flags["no-open"]).toBe(true);
		expect(parsed.flags.port).toBe("0");
		expect(parsed.flags.host).toBe("127.0.0.1");

		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			await main(["--help"]);
			const usage = out.mock.calls.map((call) => String(call[0] ?? "")).join("");
			expect(usage).toContain("ui");
			expect(usage).toContain("--no-open");
		} finally {
			out.mockRestore();
		}
	});

	it("starts a loopback server, prints the URL, and skips the browser", async () => {
		const stdout: string[] = [];
		let opened = 0;
		const code = await runUi(
			makeCtx({
				json: true,
				flags: { config: configPath, "no-open": true, port: "0" },
				stdout: (line) => stdout.push(line),
			}),
			{
				openBrowser: async () => {
					opened += 1;
				},
				waitUntilStopped: async (server) => {
					await server.close();
				},
			},
		);
		expect(code).toBe(0);
		expect(opened).toBe(0);
		const payload = JSON.parse(stdout[0] ?? "{}") as { ok: boolean; url: string; host: string };
		expect(payload.ok).toBe(true);
		expect(payload.host).toBe("127.0.0.1");
		expect(payload.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/\?token=/);
	});

	it("rejects a non-loopback host", async () => {
		const stderr: string[] = [];
		const code = await runUi(
			makeCtx({
				flags: { config: configPath, host: "0.0.0.0", "no-open": true, port: "0" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/loopback/i);
	});

	it("uses configured deployment settings and requires the configured token", async () => {
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [join(root, "docs")],
				workspacePath: root,
				memoryPath: join(root, "memory.json"),
				ui: {
					host: "127.0.0.1",
					port: 0,
					allowRemote: true,
					publicOrigin: "https://admin.example.test",
					corsOrigins: ["https://admin.example.test"],
					tokenEnv: "TEST_AUTORAG_UI_TOKEN",
				},
			}),
		);
		vi.stubEnv("TEST_AUTORAG_UI_TOKEN", "deployment-token-123456");
		const started: Array<{
			host?: string;
			port?: number;
			allowRemote?: boolean;
			publicOrigin?: string;
			corsOrigins?: readonly string[];
			token: string;
		}> = [];

		const code = await runUi(
			makeCtx({
				json: true,
				flags: { config: configPath, "no-open": true },
				stdout: noop,
			}),
			{
				startServer: async (options) => {
					started.push(options);
					return {
						url: "https://admin.example.test/?token=deployment-token-123456",
						origin: "https://admin.example.test",
						host: "127.0.0.1",
						port: 0,
						token: options.token,
						close: async () => undefined,
					};
				},
				waitUntilStopped: async () => undefined,
			},
		);
		vi.unstubAllEnvs();

		expect(code).toBe(0);
		expect(started).toEqual([
			expect.objectContaining({
				host: "127.0.0.1",
				port: 0,
				allowRemote: true,
				publicOrigin: "https://admin.example.test",
				corsOrigins: ["https://admin.example.test"],
				token: "deployment-token-123456",
			}),
		]);
	});
});
