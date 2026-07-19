import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runIndex } from "../../src/cli/commands/index.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let tmpDir: string;
let previousHome: string | undefined;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-cli-index-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(tmpDir, "home");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(tmpDir, { recursive: true, force: true });
});

interface Captured {
	readonly stdout: string[];
	readonly stderr: string[];
	readonly ctx: CommandContext;
}

function makeCtx(opts: {
	positionals?: string[];
	flags?: Record<string, string | boolean | undefined>;
	json?: boolean;
	cwd?: string;
	promptYesNo?: (question: string) => Promise<boolean>;
}): Captured {
	const stdout: string[] = [];
	const stderr: string[] = [];
	const ctx: CommandContext = {
		positionals: opts.positionals ?? [],
		flags: opts.flags ?? {},
		json: opts.json ?? false,
		debug: false,
		cwd: opts.cwd ?? tmpDir,
		stdout: (line: string) => {
			stdout.push(line);
		},
		stderr: (line: string) => {
			stderr.push(line);
		},
		promptYesNo: opts.promptYesNo,
	};
	return { stdout, stderr, ctx };
}

interface WorkspaceFixture {
	readonly workspace: string;
	readonly parsed: string;
	readonly bm25: string;
	readonly minsync: string;
	readonly bin: string;
	readonly datasources: string;
	readonly memoryFile: string;
	readonly outsideFile: string;
	readonly siblingDir: string;
	readonly siblingFile: string;
}

function seedWorkspace(): WorkspaceFixture {
	const workspace = join(tmpDir, "workspace");
	const autorag = join(workspace, ".autorag");
	mkdirSync(autorag, { recursive: true });

	const parsed = join(autorag, "parsed");
	const bm25 = join(autorag, "bm25");
	const minsync = join(autorag, "minsync");
	const bin = join(autorag, "bin");
	const datasources = join(autorag, "datasources");

	for (const dir of [parsed, bm25, minsync, bin, datasources]) {
		mkdirSync(dir, { recursive: true });
	}
	// Each reset target contains a file so the directory is non-empty.
	writeFileSync(join(parsed, "index.json"), "{}");
	writeFileSync(join(bm25, "fallback-index.json"), "{}");
	writeFileSync(join(minsync, "data.bin"), "bytes");
	writeFileSync(join(bin, "runner"), "binary");
	writeFileSync(join(datasources, "source.json"), "{}");

	const memoryFile = join(autorag, "memory.json");
	writeFileSync(
		memoryFile,
		'{"version":4,"curatedResults":[],"evidenceChunks":[],"feedbackSignals":[],"signalDefaults":{"explicitWeight":1,"followupWeight":0.5,"retryWeight":-0.5,"implicitCap":0.5},"warnings":[],"insights":[],"pendingInsightSignals":[]}',
	);

	// Files outside .autorag that must never be touched by reset.
	const outsideFile = join(workspace, "important.txt");
	writeFileSync(outsideFile, "keep me");
	const siblingDir = join(tmpDir, "sibling-outside");
	mkdirSync(siblingDir, { recursive: true });
	const siblingFile = join(siblingDir, "keep.txt");
	writeFileSync(siblingFile, "sibling keep");

	return { workspace, parsed, bm25, minsync, bin, datasources, memoryFile, outsideFile, siblingDir, siblingFile };
}

describe("runIndex", () => {
	it("returns exit 2 for an unknown subcommand", async () => {
		const { ctx, stderr } = makeCtx({ positionals: ["frobnicate"], flags: { yes: true } });
		const code = await runIndex(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("returns exit 2 when no subcommand is given", async () => {
		const { ctx, stderr } = makeCtx({ positionals: [], flags: { yes: true } });
		const code = await runIndex(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("Usage");
	});

	it("reset --yes removes parsed/bm25/minsync and preserves bin/datasources/memory", async () => {
		const fx = seedWorkspace();
		const { ctx, stdout } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		expect(existsSync(fx.parsed)).toBe(false);
		expect(existsSync(fx.bm25)).toBe(false);
		expect(existsSync(fx.minsync)).toBe(false);

		expect(existsSync(fx.bin)).toBe(true);
		expect(existsSync(fx.datasources)).toBe(true);
		expect(existsSync(fx.memoryFile)).toBe(true);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.ok).toBe(true);
		expect(parsed.action).toBe("reset");
		expect(parsed.removed).toEqual(["parsed", "bm25", "minsync"]);
	});

	it("declined reset returns exit 2 and removes nothing", async () => {
		const fx = seedWorkspace();
		const { ctx, stderr } = makeCtx({
			positionals: ["reset"],
			flags: {},
			cwd: fx.workspace,
			promptYesNo: async () => false,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(2);
		expect(stderr.join("\n")).toContain("declined");

		// Nothing removed.
		expect(existsSync(fx.parsed)).toBe(true);
		expect(existsSync(fx.bm25)).toBe(true);
		expect(existsSync(fx.minsync)).toBe(true);
		expect(existsSync(fx.bin)).toBe(true);
		expect(existsSync(fx.datasources)).toBe(true);
	});

	it("reset without --yes and without promptYesNo returns exit 2 and removes nothing", async () => {
		const fx = seedWorkspace();
		const { ctx } = makeCtx({
			positionals: ["reset"],
			flags: {},
			cwd: fx.workspace,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(2);
		expect(existsSync(fx.parsed)).toBe(true);
		expect(existsSync(fx.bm25)).toBe(true);
		expect(existsSync(fx.minsync)).toBe(true);
	});

	it("path-escape guard: reset never deletes files outside .autorag", async () => {
		const fx = seedWorkspace();
		const { ctx } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true },
			cwd: fx.workspace,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		// Files outside .autorag (workspace root and a tmp sibling) survive.
		expect(existsSync(fx.outsideFile)).toBe(true);
		expect(existsSync(fx.siblingFile)).toBe(true);
		expect(existsSync(fx.siblingDir)).toBe(true);
	});

	it("refuses to reset when .autorag is a symlink pointing outside the workspace", async () => {
		// Attacker/mis-config: .autorag is a symlink to a dir outside the workspace.
		// rmSync would follow it and delete the target's contents; the guard must refuse.
		const workspace = join(tmpDir, "ws");
		mkdirSync(workspace, { recursive: true });
		const escapeTarget = join(tmpDir, "escape-target");
		mkdirSync(join(escapeTarget, "parsed"), { recursive: true });
		const sentinel = join(escapeTarget, "parsed", "SENTINEL.txt");
		writeFileSync(sentinel, "must survive");
		symlinkSync(escapeTarget, join(workspace, ".autorag"));

		const { ctx, stderr } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true },
			cwd: workspace,
		});
		const code = await runIndex(ctx);

		expect(code).toBe(1);
		expect(stderr.join("\n")).toContain("symlink");
		// The symlink target's contents are untouched.
		expect(existsSync(sentinel)).toBe(true);
	});

	it("rebuild --yes removes then re-creates the parsed index via refresh", async () => {
		const fx = seedWorkspace();
		// Provide a parseable source file so refresh rebuilds the parsed mirror.
		const docs = join(fx.workspace, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "Alpha content for rebuild\n");

		const { ctx, stdout } = makeCtx({
			positionals: ["rebuild"],
			flags: { yes: true, "search-paths": docs },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		// Reset removed the stale targets; refresh re-created at least the parsed mirror.
		expect(existsSync(fx.parsed)).toBe(true);
		// bin/datasources/memory still intact.
		expect(existsSync(fx.bin)).toBe(true);
		expect(existsSync(fx.datasources)).toBe(true);
		expect(existsSync(fx.memoryFile)).toBe(true);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.ok).toBe(true);
		expect(parsed.action).toBe("rebuild");
		expect(parsed.removed).toEqual(["parsed", "bm25", "minsync"]);
		expect(parsed.rebuilt).toBeDefined();
	});
});

describe("runIndex --method scoped reset", () => {
	it("reset --method bm25 removes only the bm25 index", async () => {
		const fx = seedWorkspace();
		const { ctx, stdout } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true, method: "bm25" },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		expect(existsSync(fx.bm25)).toBe(false);
		expect(existsSync(fx.parsed)).toBe(true);
		expect(existsSync(fx.minsync)).toBe(true);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.action).toBe("reset");
		expect(parsed.removed).toEqual(["bm25"]);
	});

	it("reset --method minsync removes only the minsync index", async () => {
		const fx = seedWorkspace();
		const { ctx, stdout } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true, method: "minsync" },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		expect(existsSync(fx.minsync)).toBe(false);
		expect(existsSync(fx.bm25)).toBe(true);
		expect(existsSync(fx.parsed)).toBe(true);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.removed).toEqual(["minsync"]);
	});

	it("reset --method bm25,minsync removes both but not parsed", async () => {
		const fx = seedWorkspace();
		const { ctx, stdout } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true, method: "bm25,minsync" },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		expect(existsSync(fx.bm25)).toBe(false);
		expect(existsSync(fx.minsync)).toBe(false);
		expect(existsSync(fx.parsed)).toBe(true);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.removed).toEqual(["bm25", "minsync"]);
	});

	it("reset --method all removes all three targets", async () => {
		const fx = seedWorkspace();
		const { ctx, stdout } = makeCtx({
			positionals: ["reset"],
			flags: { yes: true, method: "all" },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		expect(existsSync(fx.parsed)).toBe(false);
		expect(existsSync(fx.bm25)).toBe(false);
		expect(existsSync(fx.minsync)).toBe(false);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.removed).toEqual(["parsed", "bm25", "minsync"]);
	});

	it("rebuild --method bm25 removes and rebuilds only bm25", async () => {
		const fx = seedWorkspace();
		const docs = join(fx.workspace, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "Scoped rebuild content\n");

		const { ctx, stdout } = makeCtx({
			positionals: ["rebuild"],
			flags: { yes: true, method: "bm25", "search-paths": docs },
			cwd: fx.workspace,
			json: true,
		});

		const code = await runIndex(ctx);
		expect(code).toBe(0);

		// Only bm25 was reset; refresh re-creates the bm25 index dir.
		// Parsed and minsync were not reset and survive from seed.
		expect(existsSync(fx.bm25)).toBe(true);
		expect(existsSync(fx.parsed)).toBe(true);
		expect(existsSync(fx.minsync)).toBe(true);

		const parsed = JSON.parse(stdout[0]);
		expect(parsed.action).toBe("rebuild");
		expect(parsed.removed).toEqual(["bm25"]);
		expect(parsed.rebuilt).toBeDefined();
	});
});
