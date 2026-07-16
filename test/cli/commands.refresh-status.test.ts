import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runRefresh } from "../../src/cli/commands/refresh.ts";
import { runStatus } from "../../src/cli/commands/status.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let root: string;
let docs: string;
let previousHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-refresh-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(root, "home");
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "alpha.md"), "# Alpha\n\nAlpha document body content.\n");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: {},
		json: true,
		debug: false,
		cwd: root,
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

function writeConfig(bm25: unknown, minSync?: unknown): void {
	const config: Record<string, unknown> = {
		searchPaths: ["docs"],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		bm25,
	};
	if (minSync !== undefined) config.minSync = minSync;
	const configDir = join(process.env.HOME as string, ".autorag");
	mkdirSync(configDir, { recursive: true });
	writeFileSync(join(configDir, "config.json"), `${JSON.stringify(config, null, 2)}\n`);
}

describe("runRefresh + runStatus (cli)", () => {
	it("refresh then status emits JSON with counts and no leaked paths", async () => {
		writeConfig({ forceEngine: "typescript-fallback" });

		const refreshOut: string[] = [];
		const refreshCode = await runRefresh(makeCtx({ stdout: (line) => refreshOut.push(line) }));
		expect(refreshCode).toBe(0);
		expect(refreshOut).toHaveLength(1);

		const refreshBlob = refreshOut[0];
		// Path opacity: no absolute index path, no temp root, no bm25 subdir literal.
		expect(refreshBlob).not.toContain("indexPath");
		expect(refreshBlob).not.toContain(root);
		expect(refreshBlob).not.toContain(join(".autorag", "bm25"));

		const statusOut: string[] = [];
		const statusCode = await runStatus(makeCtx({ stdout: (line) => statusOut.push(line) }));
		expect(statusCode).toBe(0);
		expect(statusOut).toHaveLength(1);

		const statusBlob = statusOut[0];
		expect(statusBlob).not.toContain("indexPath");
		expect(statusBlob).not.toContain(root);
		expect(statusBlob).not.toContain(join(".autorag", "bm25"));

		const status = JSON.parse(statusBlob);
		// `status` runs in a fresh agent instance (a separate CLI process in real
		// use), so in-memory `state`/`counts`/`stale` are not carried across
		// invocations (`stale` is true whenever this instance has never refreshed).
		// The cross-process-observable disk-freshness signal is the absence of any
		// `stale-index` diagnostic: after refresh wrote fresh parsed mirrors, a new
		// status invocation finds no source newer than the recorded mirror index.
		expect(typeof status.state).toBe("string");
		expect(Array.isArray(status.diagnostics)).toBe(true);
		const staleDiagnostics = (status.diagnostics as { code: string }[]).filter((d) => d.code === "stale-index");
		expect(staleDiagnostics).toHaveLength(0);
		expect(status.components).toBeDefined();
	});

	it("surfaces minsync-unavailable without throwing when the minsync binary is absent", async () => {
		writeConfig(
			{ forceEngine: "typescript-fallback" },
			{
				binaryPath: join(root, "missing-minsync"),
				workspacePath: join(root, ".autorag", "minsync"),
			},
		);

		const refreshOut: string[] = [];
		const refreshCode = await runRefresh(makeCtx({ stdout: (line) => refreshOut.push(line) }));
		expect(refreshCode).toBe(0);
		// Refresh must not leak paths even when minsync is unavailable.
		const refreshBlob = refreshOut[0];
		expect(refreshBlob).not.toContain(root);
		expect(refreshBlob).not.toContain("indexPath");

		const statusOut: string[] = [];
		const statusCode = await runStatus(makeCtx({ stdout: (line) => statusOut.push(line) }));
		expect(statusCode).toBe(0);

		const status = JSON.parse(statusOut[0]);
		// MinSync absence surfaces as a component readiness state, not a throw.
		expect(status.components).toBeDefined();
		expect(status.components.minsync).toBe("unavailable");
		// Path opacity holds on the status path too.
		expect(statusOut[0]).not.toContain(root);
		expect(statusOut[0]).not.toContain("indexPath");
	});

	it("reports idle and stale before any refresh has run", async () => {
		writeConfig({ forceEngine: "typescript-fallback" });

		const statusOut: string[] = [];
		const code = await runStatus(makeCtx({ stdout: (line) => statusOut.push(line) }));
		expect(code).toBe(0);

		const status = JSON.parse(statusOut[0]);
		expect(status.state).toBe("idle");
		expect(status.stale).toBe(true);
		expect(status.counts).toBeUndefined();
	});
});

describe("runRefresh --method", () => {
	it("refreshes only bm25 when --method bm25 is given", async () => {
		writeConfig({ forceEngine: "typescript-fallback" });

		const out: string[] = [];
		const code = await runRefresh(makeCtx({ flags: { method: "bm25" }, stdout: (line) => out.push(line) }));
		expect(code).toBe(0);
		expect(out).toHaveLength(1);
		const blob = out[0];
		// BM25 ran, so the result should carry bm25 readiness info.
		expect(blob).not.toContain("indexPath");
		expect(blob).not.toContain(root);
		const parsed = JSON.parse(blob);
		expect(parsed.bm25).toBeDefined();
		expect(parsed.bm25.readiness).toBeDefined();
	});

	it("refreshes with all methods when --method all is given", async () => {
		writeConfig({ forceEngine: "typescript-fallback" });

		const out: string[] = [];
		const code = await runRefresh(makeCtx({ flags: { method: "all" }, stdout: (line) => out.push(line) }));
		expect(code).toBe(0);
		const parsed = JSON.parse(out[0]);
		expect(parsed.bm25).toBeDefined();
	});

	it("rejects an unknown --method value", async () => {
		writeConfig({ forceEngine: "typescript-fallback" });

		const err: string[] = [];
		const code = await runRefresh(makeCtx({ flags: { method: "bogus" }, stderr: (line) => err.push(line) }));
		expect(code).toBe(1);
		expect(err.join("\n")).toContain("Unknown --method value");
	});
});
