import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { syncMinSyncWorkspace } from "../../src/minsync/workspace.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";
import { seedMinSyncStage } from "../helpers/seed-minsync-stage.ts";

let root: string;
let source: string;
let minsyncWorkspace: string;
let minsyncBinary: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-staging-longpath-"));
	source = join(root, "docs");
	minsyncWorkspace = join(root, ".autorag", "minsync");
	minsyncBinary = join(root, "fake-minsync.mjs");
	mkdirSync(source, { recursive: true });
	mkdirSync(minsyncWorkspace, { recursive: true });
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
if (args[0] === "stage") {
  createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({
    stateDir: args[args.indexOf("--state-dir") + 1],
  });
} else if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  const embedderAt = args.indexOf("--embedder");
  const embedderId = embedderAt === -1 ? "openai" : args[embedderAt + 1];
  writeFileSync(config, "[embedder]\\nid = \\"" + embedderId + "\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
} else if (args[0] === "sync") {
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ files_processed: 1, files_processed_paths: ["files/docs/ok.txt.md"] }));
} else if (args[0] === "--version") {
  console.log("minsync 0.7.6");
} else {
  console.log(JSON.stringify({ results: [] }));
}
`,
		{ mode: 0o755 },
	);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("MinSync workspace staging ENAMETOOLONG isolation (#2223 / AutoRAG-app#489)", () => {
	it("stages a 215-unit filename cleanly because the temp file uses a fixed short UUID name", async () => {
		if (process.platform === "win32") return;
		const longName = `${"a".repeat(211)}.txt`;
		const longSource = join(source, longName);
		const okSource = join(source, "ok.txt");
		writeFileSync(longSource, "long filename content\n");
		writeFileSync(okSource, "normal document\n");

		const parsedDir = join(root, ".autorag", "parsed");
		mkdirSync(parsedDir, { recursive: true });
		const longParsed = join(parsedDir, "long.md");
		const okParsed = join(parsedDir, "ok.md");
		writeFileSync(longParsed, "long filename content\n");
		writeFileSync(okParsed, "normal document\n");

		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				[`/docs/${longName}`]: {
					virtualPath: `/docs/${longName}`,
					sourcePath: longSource,
					outputPath: longParsed,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 22,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
				"/docs/ok.txt": {
					virtualPath: "/docs/ok.txt",
					sourcePath: okSource,
					outputPath: okParsed,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 16,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});

		const result = syncMinSyncWorkspace(root, { workspacePath: minsyncWorkspace });
		expect(result.stagingFailed).toEqual([]);
		expect(existsSync(join(minsyncWorkspace, "files", "docs", `${longName}.md`))).toBe(true);
		expect(readFileSync(join(minsyncWorkspace, "files", "docs", "ok.txt.md"), "utf8")).toBe("normal document\n");
	});

	it("isolates a 253-unit filename whose final .md exceeds NAME_MAX and still indexes other documents", async () => {
		if (process.platform === "win32") return;
		// 253-char source is legal on APFS/ext4 (<=255), but "<name>.md" is 256 (>255).
		const maxName = `${"b".repeat(249)}.txt`;
		writeFileSync(join(source, maxName), "too long once .md is appended\n");
		writeFileSync(join(source, "ok.txt"), "normal document content\n");

		const agent = new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { binaryPath: minsyncBinary, workspacePath: minsyncWorkspace, autoInstall: false },
		});

		const result = await agent.refresh(false, { methods: ["parsed", "minsync"] });
		expect(result.minsync?.ok).toBe(true);
		expect(existsSync(join(minsyncWorkspace, "files", "docs", "ok.txt.md"))).toBe(true);
		const failed = result.diagnostics.filter((d) => d.code === "minsync-staging-failed");
		expect(failed).toHaveLength(1);
		expect(failed[0]?.source).toBe(`/docs/${maxName}`);
		expect(failed[0]?.severity).toBe("warning");
	});

	it("isolates a deep macOS NFD Korean path exceeding PATH_MAX without aborting refresh (#2223)", async () => {
		if (process.platform !== "darwin") return;
		const longWs = join(root, `ws_${"w".repeat(120)}`);
		const longMinsyncWs = join(longWs, ".autorag", "minsync");
		mkdirSync(longWs, { recursive: true });
		const seg = "소상공인지원사업컨설팅결과보고서폴더_가나다라마바사아자차카타파하".normalize("NFD");
		const deepDir = join(source, `00.${seg}`, `01.${seg}`, `02.${seg}`);
		mkdirSync(deepDir, { recursive: true });
		const deepFile = join(deepDir, "최종제출자료_벤처기업확인_유효기간_안내문.md".normalize("NFD"));
		writeFileSync(deepFile, "# notice\nventure certificate is valid for 3 years\n");
		writeFileSync(join(source, "short.md"), "# short\nshort path document\n");

		const agent = new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(longWs, "memory.json"),
			workspacePath: longWs,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { binaryPath: minsyncBinary, workspacePath: longMinsyncWs, autoInstall: false },
		});

		const result = await agent.refresh(false, { methods: ["parsed", "minsync"] });
		expect(result.minsync?.ok).toBe(true);
		expect(existsSync(join(longMinsyncWs, "files", "docs", "short.md.md"))).toBe(true);
		const failed = result.diagnostics.filter((d) => d.code === "minsync-staging-failed");
		expect(failed).toHaveLength(1);
		expect(failed[0]?.severity).toBe("warning");
	});
});
