import type * as Fs from "node:fs";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/index.ts";
import { parsedMirrorRoot } from "../../src/mirror/index.ts";
import { ParseError } from "../../src/parser/errors.ts";
import { PlainTextParser } from "../../src/parser/index.ts";

const fsMock = vi.hoisted(() => ({
	writeFileSyncHook: undefined as ((path: unknown) => void) | undefined,
}));

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof Fs>("node:fs");
	return {
		...actual,
		writeFileSync: (...args: Parameters<typeof actual.writeFileSync>) => {
			fsMock.writeFileSyncHook?.(args[0]);
			return actual.writeFileSync(...args);
		},
	};
});

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-isolation-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "a.txt"), "Alpha content\n");
	writeFileSync(join(docs, "b.txt"), "Bravo content\n");
	writeFileSync(join(docs, "c.txt"), "Charlie content\n");
});

afterEach(() => {
	fsMock.writeFileSyncHook = undefined;
	vi.restoreAllMocks();
	rmSync(root, { recursive: true, force: true });
});

function makeAgent(): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
	});
}

function tmpFiles(): string[] {
	const found: string[] = [];
	const walk = (directory: string): void => {
		for (const entry of readdirSync(directory, { withFileTypes: true })) {
			const path = join(directory, entry.name);
			if (entry.isDirectory()) walk(path);
			else if (entry.name.endsWith(".tmp")) found.push(path);
		}
	};
	walk(parsedMirrorRoot(root));
	return found;
}

describe("refresh isolates per-file failures", () => {
	it("completes ok with exactly one parser-crashed diagnostic when a parser throws a plain Error", async () => {
		vi.spyOn(PlainTextParser.prototype, "parse").mockImplementation(async function (this: PlainTextParser, input) {
			if (input.virtualPath.endsWith("b.txt")) throw new Error("third-party parser bug");
			return { markdown: new TextDecoder().decode(input.bytes), metadata: { parser: this.name } };
		});
		const agent = makeAgent();

		const result = await agent.refresh(true, { methods: ["parsed"] });
		const status = await agent.getRefreshStatus();

		expect(result.diagnostics.filter((d) => d.code === "parser-crashed")).toHaveLength(1);
		expect(result.written).toBe(2);
		expect(result.skipped).toBe(1);
		expect(status.state).toBe("success");
		expect(JSON.stringify(result.diagnostics)).not.toContain(root);
	});

	it("still reports a typed ParseError as parser-failed, not parser-crashed", async () => {
		vi.spyOn(PlainTextParser.prototype, "parse").mockImplementation(async function (this: PlainTextParser, input) {
			if (input.virtualPath.endsWith("b.txt")) throw new ParseError(this.name, input.virtualPath, "bad bytes");
			return { markdown: new TextDecoder().decode(input.bytes), metadata: { parser: this.name } };
		});

		const result = await makeAgent().refresh(true, { methods: ["parsed"] });

		expect(result.diagnostics.filter((d) => d.code === "parser-failed")).toHaveLength(1);
		expect(result.diagnostics.some((d) => d.code === "parser-crashed")).toBe(false);
	});
});

describe("refresh aborts on workspace-level failure", () => {
	it("fails with a workspace error on ENOSPC, records it in status, and leaves no tmp file", async () => {
		fsMock.writeFileSyncHook = (path) => {
			if (String(path).includes(parsedMirrorRoot(root)) && String(path).endsWith(".tmp")) {
				const error: NodeJS.ErrnoException = new Error("ENOSPC: no space left on device");
				error.code = "ENOSPC";
				throw error;
			}
		};
		const agent = makeAgent();

		await expect(agent.refresh(true, { methods: ["parsed"] })).rejects.toMatchObject({
			name: "MirrorWorkspaceError",
			code: "ENOSPC",
		});
		fsMock.writeFileSyncHook = undefined;
		const status = await agent.getRefreshStatus();

		expect(status.state).toBe("failed");
		expect(tmpFiles()).toEqual([]);
	});
});
