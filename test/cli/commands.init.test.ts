import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runInit } from "../../src/cli/commands/init.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-init-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: {},
		json: false,
		debug: false,
		cwd: root,
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

describe("runInit", () => {
	it("writes autorag.config.json with the provided keys folded in", async () => {
		const stdout: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: {
					"search-paths": "docs, notes",
					workspace: root,
					"memory-path": join(root, "memory.json"),
					"model-provider": "openai",
					"model-id": "gpt-4o",
				},
				stdout: (line) => stdout.push(line),
			}),
		);

		expect(code).toBe(0);
		const file = join(root, "autorag.config.json");
		expect(existsSync(file)).toBe(true);

		const config = JSON.parse(readFileSync(file, "utf-8"));
		expect(config.searchPaths).toEqual(["docs", "notes"]);
		expect(config.workspacePath).toBe(root);
		expect(config.memoryPath).toBe(join(root, "memory.json"));
		expect(config.model).toEqual({ provider: "openai", id: "gpt-4o" });
	});

	it("returns 2 and writes an error when the config already exists without --force", async () => {
		writeFileSync(join(root, "autorag.config.json"), `${JSON.stringify({ existing: true })}\n`);

		const stderr: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
				stderr: (line) => stderr.push(line),
			}),
		);

		expect(code).toBe(2);
		expect(stderr.length).toBeGreaterThan(0);
		// The pre-existing file must be untouched.
		const config = JSON.parse(readFileSync(join(root, "autorag.config.json"), "utf-8"));
		expect(config.existing).toBe(true);
	});

	it("overwrites the existing config when --force is set", async () => {
		writeFileSync(join(root, "autorag.config.json"), `${JSON.stringify({ old: true })}\n`);

		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "new", force: true },
			}),
		);

		expect(code).toBe(0);
		const config = JSON.parse(readFileSync(join(root, "autorag.config.json"), "utf-8"));
		expect(config.searchPaths).toEqual(["new"]);
		expect(config.old).toBeUndefined();
	});

	it("emits a JSON envelope with the written filename in --json mode", async () => {
		const stdout: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
				json: true,
				stdout: (line) => stdout.push(line),
			}),
		);

		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);
		const envelope = JSON.parse(stdout[0]);
		expect(envelope.ok).toBe(true);
		expect(envelope.wrote).toEqual(["autorag.config.json"]);
	});

	it("emits a human line in non-json mode", async () => {
		const stdout: string[] = [];
		const code = await runInit(
			makeCtx({
				flags: { "search-paths": "docs" },
				stdout: (line) => stdout.push(line),
			}),
		);

		expect(code).toBe(0);
		expect(stdout).toHaveLength(1);
		expect(stdout[0]).toContain("autorag.config.json");
		expect(stdout[0]).not.toContain("{");
	});
});
