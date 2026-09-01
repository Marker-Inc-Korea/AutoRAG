import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DiscrawlClient } from "../../../../src/datasource/skills/discrawl/client.ts";

let root: string;
let binDir: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-discrawl-direct-"));
	binDir = join(root, "bin");
	binaryPath = join(binDir, "discrawl");
	logPath = join(root, "discrawl-calls.jsonl");
	mkdirSync(binDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeDiscrawl(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args: process.argv.slice(2) }) + "\\n");
process.stdout.write(process.env.DISCRAWL_FAKE_OUTPUT ?? "{}");
`,
	);
	chmodSync(binaryPath, 0o755);
}

function loggedArgs(): readonly (readonly string[])[] {
	if (!existsSync(logPath)) return [];
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => (JSON.parse(line) as { args: string[] }).args);
}

describe("DiscrawlClient direct CLI execution", () => {
	it("invokes discrawl without injecting a managed --config into the empty AutoRAG workspace", async () => {
		writeFakeDiscrawl();
		const client = new DiscrawlClient({
			binaryPath,
			root,
			env: {
				PATH: `${binDir}:${process.env.PATH ?? ""}`,
				DISCRAWL_FAKE_OUTPUT: JSON.stringify({ messages: 0 }),
			},
		});

		const result = await client.sync();

		const args = loggedArgs()[0] ?? [];
		expect(args).not.toContain("--config");
		expect(args.join(" ")).not.toContain(".autorag/datasources/discrawl");
		expect(result.ok).toBe(true);
	});
});
