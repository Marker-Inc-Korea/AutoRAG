import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runDuplicates } from "../../src/cli/commands/duplicates.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let root: string;
let previousHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-duplicates-"));
	previousHome = process.env.HOME;
	process.env.HOME = join(root, "home");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	rmSync(root, { recursive: true, force: true });
});

function context(json = false): { ctx: CommandContext; stdout: string[]; stderr: string[] } {
	const stdout: string[] = [];
	const stderr: string[] = [];
	return {
		stdout,
		stderr,
		ctx: {
			positionals: [],
			flags: { "search-paths": join(root, "docs") },
			json,
			debug: false,
			cwd: root,
			stdout: (line) => stdout.push(line),
			stderr: (line) => stderr.push(line),
		},
	};
}

describe("autorag duplicates", () => {
	it("prints a safe cleanup plan without changing files", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs);
		const first = join(docs, "first.txt");
		const second = join(docs, "second.txt");
		writeFileSync(first, "same");
		writeFileSync(second, "same");
		const { ctx, stdout } = context();

		const code = await runDuplicates(ctx, async () => ({
			dir: docs,
			files: [
				{ path: first, content_hash: "hash" },
				{ path: second, content_hash: "hash" },
			],
			families: [],
			errors: [],
		}));

		expect(code).toBe(0);
		expect(stdout.join("\n")).toContain("review/archive");
		expect(stdout.join("\n")).toContain(first);
		expect(stdout.join("\n")).toContain(second);
		expect(readFileSync(first, "utf8")).toBe("same");
		expect(readFileSync(second, "utf8")).toBe("same");
	});

	it("emits machine-readable exact duplicate groups", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs);
		const { ctx, stdout } = context(true);
		const code = await runDuplicates(ctx, async () => ({
			dir: docs,
			files: [
				{ path: "a.txt", content_hash: "hash" },
				{ path: "b.txt", content_hash: "hash" },
			],
			families: [],
			errors: [],
		}));
		expect(code).toBe(0);
		const output = JSON.parse(stdout[0]);
		expect(output.ok).toBe(true);
		expect(output.exactGroups).toHaveLength(1);
	});
});
