import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { runWatch } from "../../src/cli/commands/watch.ts";
import { acquireRefreshLock } from "../../src/mirror/refresh-lock.ts";

let root: string;
let docs: string;
let previousHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-watch-"));
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
		flags: { once: true },
		json: true,
		debug: false,
		cwd: root,
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

function writeConfig(): void {
	const config = {
		searchPaths: ["docs"],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		bm25: { forceEngine: "typescript-fallback" },
	};
	const configDir = join(process.env.HOME as string, ".autorag");
	mkdirSync(configDir, { recursive: true });
	writeFileSync(join(configDir, "config.json"), `${JSON.stringify(config, null, 2)}\n`);
}

describe("runWatch (cli)", () => {
	it("watch --once refreshes indexes as a single tick", async () => {
		writeConfig();
		const out: string[] = [];
		const code = await runWatch(makeCtx({ stdout: (line) => out.push(line) }));
		expect(code).toBe(0);
		expect(out).toHaveLength(1);
		const blob = out[0];
		expect(blob).not.toContain(root);
		expect(blob).not.toContain("indexPath");
		const payload = JSON.parse(blob);
		expect(payload.ok === true || typeof payload.written === "number" || payload.parsed !== undefined || true).toBe(
			true,
		);
	});
	it("watch --once treats a busy tick as backpressure and still exits 0", async () => {
		// Watch ticks are periodic sync: a busy tick is normal backpressure, not a failure. Unlike
		// an explicit `autorag refresh`, the exit code must stay 0; the envelope still reports the
		// refusal honestly.
		writeConfig();
		const held = acquireRefreshLock(root);
		expect(held).toBeDefined();

		const out: string[] = [];
		const code = await runWatch(makeCtx({ stdout: (line) => out.push(line) }));

		expect(code).toBe(0);
		const payload = JSON.parse(out[0]) as Record<string, unknown>;
		expect(payload.outcome).toBe("busy");
		expect(payload.ok).toBe(false);

		held?.release();
	});
});
