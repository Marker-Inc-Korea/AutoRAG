import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

let root: string;
const noop = (): void => undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-p2p-"));
	mkdirSync(join(root, ".autorag", "p2p"), { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// We test through main() which dispatches to the lazy-loaded command.
// We set up a config and identity so commands actually work.
// ---------------------------------------------------------------------------

async function _run(args: string[]): Promise<{ code: number; stdout: string[]; stderr: string[] }> {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const code = await (await import("../../src/cli/index.ts")).main([
		"p2p",
		"--config",
		join(root, "config.json"),
		...args,
	]);
	// We can't easily capture stdout from main — it writes directly to process.stdout.
	// For the CLI tests, we'll test the underlying command function directly.
	return { code, stdout, stderr };
}

import { runP2p } from "../../src/cli/commands/p2p.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { decodePairingCode, generateIdentity, loadPeerRegistry } from "../../src/p2p/identity.ts";

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: {},
		json: false,
		debug: false,
		cwd: root,
		stdout: noop,
		stderr: noop,
		...overrides,
	};
}

function setupIdentity(): void {
	generateIdentity(root);
}

describe("autorag p2p pair", () => {
	it("prints a decodable pairing code when called without flags", async () => {
		setupIdentity();
		const stdout: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["pair"],
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(code).toBe(0);
		expect(stdout.length).toBeGreaterThan(0);

		// The pairing code should be base64url that decodes to { endpoint, pubkey, alias }
		const _codeLine = stdout.find((l) => l.startsWith("code=") || l.includes("code"));
		const output = stdout.join("\n");
		// Extract the base64url value from output
		const match = output.match(/[A-Za-z0-9_-]{20,}/);
		expect(match).toBeTruthy();
		const decoded = decodePairingCode(match![0]);
		expect(decoded.endpoint).toBeTruthy();
		expect(decoded.pubkey).toBeTruthy();
		expect(decoded.alias).toBeTruthy();
	});

	it("prints a pairing code with provided endpoint and alias", async () => {
		setupIdentity();
		const stdout: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { endpoint: "127.0.0.1:9470", alias: "my-instance" },
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(code).toBe(0);
		const output = stdout.join("\n");
		const match = output.match(/[A-Za-z0-9_-]{20,}/);
		expect(match).toBeTruthy();
		const decoded = decodePairingCode(match![0]);
		expect(decoded.endpoint).toBe("127.0.0.1:9470");
		expect(decoded.alias).toBe("my-instance");
	});

	it("accepts a valid pairing code with --accept", async () => {
		setupIdentity();
		// Generate a code to accept
		const identity = generateIdentity(join(root, "remote-workspace"));
		const code = identity.pairingCode("192.168.1.100:9470", "friend-alice");
		const stdout: string[] = [];
		const exitCode = await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: code, alias: "alice" },
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(exitCode).toBe(0);

		// Verify it was added to the registry
		const registry = loadPeerRegistry(root);
		expect(registry.alice).toBeTruthy();
		expect(registry.alice.fingerprint).toBeTruthy();
		expect(registry.alice.endpoint).toBe("192.168.1.100:9470");
	});

	it("rejects a malformed pairing code with --accept", async () => {
		setupIdentity();
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: "not-a-valid-pairing-code!!!" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/(error|invalid|malformed)/i);
	});

	it("rejects --accept without a code value", async () => {
		setupIdentity();
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: true },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
	});

	it("rejects an empty --accept code", async () => {
		setupIdentity();
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: "" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
	});
});

describe("autorag p2p peers", () => {
	it("lists peers after accepting one", async () => {
		setupIdentity();
		// Accept a peer first
		const remoteIdentity = generateIdentity(join(root, "remote-ws"));
		const code = remoteIdentity.pairingCode("10.0.0.5:9470", "bob");
		await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: code, alias: "bob" },
			}),
		);

		// Now list peers
		const stdout: string[] = [];
		const exitCode = await runP2p(
			makeCtx({
				positionals: ["peers"],
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(exitCode).toBe(0);
		const output = stdout.join("\n");
		expect(output).toContain("bob");
		expect(output).toContain("10.0.0.5:9470");
		// Must NOT leak the full public key
		expect(output).not.toContain(remoteIdentity.pubkey);
		// Fingerprint should be visible though
		expect(output).toContain(remoteIdentity.fingerprint);
	});

	it("shows an empty list when no peers added", async () => {
		setupIdentity();
		const stdout: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["peers"],
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(code).toBe(0);
		// Should indicate no peers, not error
		const output = stdout.join("\n");
		expect(output).toBeTruthy();
	});

	it("removes a peer by alias with --remove", async () => {
		setupIdentity();
		const remoteIdentity = generateIdentity(join(root, "remote-ws"));
		const code = remoteIdentity.pairingCode("10.0.0.5:9470", "bob");
		await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: code, alias: "bob" },
			}),
		);

		// Verify it exists
		let registry = loadPeerRegistry(root);
		expect(registry.bob).toBeTruthy();

		// Remove it
		const stdout: string[] = [];
		const exitCode = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { remove: "bob" },
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(exitCode).toBe(0);

		// Verify it's gone
		registry = loadPeerRegistry(root);
		expect(registry.bob).toBeUndefined();
	});

	it("errors on --remove with non-existent alias", async () => {
		setupIdentity();
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { remove: "nonexistent" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
	});

	it("must NOT leak full pubkey or private key in output", async () => {
		setupIdentity();
		const remoteIdentity = generateIdentity(join(root, "remote-ws"));
		const pairingCode = remoteIdentity.pairingCode("10.0.0.5:9470", "bob");
		await runP2p(
			makeCtx({
				positionals: ["pair"],
				flags: { accept: pairingCode, alias: "bob" },
			}),
		);

		const stdout: string[] = [];
		await runP2p(
			makeCtx({
				positionals: ["peers"],
				stdout: (line) => stdout.push(line),
			}),
		);
		const output = stdout.join("\n");
		// Should NOT contain the raw base64 pubkey (it has +/ chars, not hex-only)
		expect(output).not.toContain(remoteIdentity.pubkey);
		// Fingerprint is hex-only; full pubkey includes + or / which should not appear
		expect(output).not.toMatch(/[A-Za-z0-9+/]{80,}/);
	});
});

describe("autorag p2p dispatch error handling", () => {
	it("prints help for 'p2p' with no subcommand", async () => {
		setupIdentity();
		const stdout: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: [],
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(code).toBe(0);
		const output = stdout.join("\n");
		expect(output).toContain("Usage");
	});

	it("errors for an unknown subcommand", async () => {
		setupIdentity();
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["unknown-cmd"],
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
	});
});
