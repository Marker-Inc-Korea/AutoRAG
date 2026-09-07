import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runP2p } from "../../src/cli/commands/p2p.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { loadSignalPeerRegistry } from "../../src/p2p/signal-server.ts";

let root: string;
const noop = (): void => undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-p2p-"));
	mkdirSync(join(root, ".autorag", "p2p"), { recursive: true });
	writeFileSync(
		join(root, "config.json"),
		JSON.stringify({ searchPaths: [root], workspacePath: root, memoryPath: join(root, "memory.json") }),
	);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: { config: join(root, "config.json") },
		json: false,
		debug: false,
		cwd: root,
		stdout: noop,
		stderr: noop,
		...overrides,
	};
}

describe("autorag p2p register/verify", () => {
	it("rejects register without an E.164 number", async () => {
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({ positionals: ["register", "not-a-number"], stderr: (line) => stderr.push(line) }),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/E\.164/i);
	});

	it("rejects verify without a code", async () => {
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({ positionals: ["verify", "+821012345678"], stderr: (line) => stderr.push(line) }),
		);
		expect(code).toBe(2);
	});
});

describe("autorag p2p peers", () => {
	it("adds, lists, and removes a peer by Signal id", async () => {
		const stdout: string[] = [];
		const add = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { config: join(root, "config.json"), add: "alice", "signal-id": "+821099998888" },
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(add).toBe(0);

		const registry = loadSignalPeerRegistry(root);
		expect(registry.alice?.signalId).toBe("+821099998888");

		const listOut: string[] = [];
		const list = await runP2p(makeCtx({ positionals: ["peers"], stdout: (line) => listOut.push(line) }));
		expect(list).toBe(0);
		expect(listOut.join("\n")).toContain("alice");
		expect(listOut.join("\n")).toContain("+821099998888");

		const remove = await runP2p(
			makeCtx({ positionals: ["peers"], flags: { config: join(root, "config.json"), remove: "alice" } }),
		);
		expect(remove).toBe(0);
		expect(loadSignalPeerRegistry(root).alice).toBeUndefined();
	});

	it("rejects --add without --signal-id", async () => {
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { config: join(root, "config.json"), add: "alice" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/signal-id/i);
	});

	it("reports an empty registry", async () => {
		const stdout: string[] = [];
		const code = await runP2p(makeCtx({ positionals: ["peers"], stdout: (line) => stdout.push(line) }));
		expect(code).toBe(0);
		expect(stdout.join("\n")).toMatch(/No peers registered/i);
	});

	it("rejects removing an unknown peer", async () => {
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { config: join(root, "config.json"), remove: "ghost" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
	});
});
