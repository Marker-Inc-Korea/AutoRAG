import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runP2p } from "../../src/cli/commands/p2p.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import {
	listPendingPeerRequests,
	loadPeerRequestDecision,
	savePendingPeerRequest,
} from "../../src/p2p/approval-store.ts";
import { loadSimplexPeerRegistry } from "../../src/p2p/simplex-server.ts";

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

describe("autorag p2p peers", () => {
	it("adds, lists, and removes a peer by SimpleX contact id", async () => {
		const stdout: string[] = [];
		const add = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { config: join(root, "config.json"), add: "alice", "contact-id": "42" },
				stdout: (line) => stdout.push(line),
			}),
		);
		expect(add).toBe(0);

		const registry = loadSimplexPeerRegistry(root);
		expect(registry.alice?.contactId).toBe(42);

		const listOut: string[] = [];
		const list = await runP2p(makeCtx({ positionals: ["peers"], stdout: (line) => listOut.push(line) }));
		expect(list).toBe(0);
		expect(listOut.join("\n")).toContain("alice");
		expect(listOut.join("\n")).toContain("42");

		const remove = await runP2p(
			makeCtx({ positionals: ["peers"], flags: { config: join(root, "config.json"), remove: "alice" } }),
		);
		expect(remove).toBe(0);
		expect(loadSimplexPeerRegistry(root).alice).toBeUndefined();
	});

	it("rejects --add without --contact-id", async () => {
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { config: join(root, "config.json"), add: "alice" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/contact-id/i);
	});

	it("rejects a non-integer --contact-id", async () => {
		const stderr: string[] = [];
		const code = await runP2p(
			makeCtx({
				positionals: ["peers"],
				flags: { config: join(root, "config.json"), add: "alice", "contact-id": "not-a-number" },
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toMatch(/contact-id/i);
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

describe("autorag p2p requests", () => {
	it("lists, approves, and denies pending peer requests", async () => {
		savePendingPeerRequest(root, {
			id: "req-1",
			contactId: 42,
			query: "refund policy",
			createdAt: new Date().toISOString(),
			sources: ["/docs/shared.md"],
			payload: { v: 1, status: "ok", answer: "ok", results: [], files: [], diagnostics: [] },
		});
		const listOut: string[] = [];
		const listed = await runP2p(
			makeCtx({ positionals: ["requests"], json: true, stdout: (line) => listOut.push(line) }),
		);
		expect(listed).toBe(0);
		const listedPayload = JSON.parse(listOut[0] ?? "{}") as { requests: Array<{ id: string }> };
		expect(listedPayload.requests.map((request) => request.id)).toEqual(["req-1"]);

		const approveOut: string[] = [];
		const approved = await runP2p(
			makeCtx({
				positionals: ["requests", "approve", "req-1"],
				json: true,
				stdout: (line) => approveOut.push(line),
			}),
		);
		expect(approved).toBe(0);
		expect(JSON.parse(approveOut[0] ?? "{}")).toMatchObject({ ok: true, id: "req-1", decision: "approve" });
		expect(loadPeerRequestDecision(root, "req-1")?.decision).toBe("approve");
		expect(listPendingPeerRequests(root)).toHaveLength(0);

		savePendingPeerRequest(root, {
			id: "req-2",
			contactId: 42,
			query: "secret payroll",
			createdAt: new Date().toISOString(),
			sources: ["/docs/secret.md"],
			payload: { v: 1, status: "ok", answer: "secret", results: [], files: [], diagnostics: [] },
		});
		const denyOut: string[] = [];
		const denied = await runP2p(
			makeCtx({
				positionals: ["requests", "deny", "req-2"],
				json: true,
				stdout: (line) => denyOut.push(line),
			}),
		);
		expect(denied).toBe(0);
		expect(JSON.parse(denyOut[0] ?? "{}")).toMatchObject({ ok: true, id: "req-2", decision: "deny" });
	});
});
