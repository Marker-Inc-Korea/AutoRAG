import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	renameSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MinSyncStageSession } from "../../src/minsync/stage.ts";
import { syncMinSyncWorkspace } from "../../src/minsync/workspace.ts";
import { detectMirrorStaleness, loadMirrorIndex, syncParsedMirrors } from "../../src/mirror/index.ts";
import { mirrorStageKey } from "../../src/mirror/index-store.ts";
import { parsedOutputPath } from "../../src/mirror/paths.ts";
import { ParseError } from "../../src/parser/errors.ts";
import { type ParseInput, type ParseOutput, ParserRegistry, PlainTextParser } from "../../src/parser/index.ts";
import { writeFakeMinSync } from "../helpers/fake-minsync.ts";

let root: string;
let source: string;
class VersionedParser extends PlainTextParser {
	version = "1";
	readonly attempts: string[] = [];
	async parse(input: ParseInput): Promise<ParseOutput> {
		this.attempts.push(input.virtualPath);
		return { markdown: `${new TextDecoder().decode(input.bytes).trim()} parsed-${this.version}\n` };
	}
}

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-stage-regression-"));
	source = join(root, "docs");
	mkdirSync(source);
	writeFileSync(join(source, "a.txt"), "Alpha\n");
	writeFileSync(join(source, "b.txt"), "Bravo\n");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));
function output(key: string): string {
	const entry = loadMirrorIndex(root).entries[`/docs/${key}.txt`];
	if (!entry) throw new Error(`missing ${key}`);
	return entry.outputPath;
}

describe("MinSync-owned incremental parsing", () => {
	it("leaves the staged corpus untouched on repeated warm refreshes", async () => {
		const parser = new VersionedParser();
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await syncParsedMirrors(options);
		const staged = syncMinSyncWorkspace(root).entries.map((entry) => entry.minSyncPath);
		const before = staged.map((path) => statSync(path, { bigint: true }).mtimeNs);
		parser.attempts.length = 0;
		await syncParsedMirrors(options);
		syncMinSyncWorkspace(root);
		expect(parser.attempts).toEqual([]);
		expect(staged.map((path) => statSync(path, { bigint: true }).mtimeNs)).toEqual(before);
	});

	it("keeps the last good mirror when a source changes during parsing, including interrupted recovery", async () => {
		const options = { root, searchPaths: [source] };
		await syncParsedMirrors({ ...options, registry: new ParserRegistry([new VersionedParser()]) });
		const priorOutput = output("a");
		writeFileSync(join(source, "a.txt"), "New source\n");
		class ChangingParser extends VersionedParser {
			async parse(input: ParseInput): Promise<ParseOutput> {
				const result = await super.parse(input);
				if (input.virtualPath === "/docs/a.txt") writeFileSync(join(source, "a.txt"), "Still changing\n");
				return result;
			}
		}
		const controller = new AbortController();
		await expect(
			syncParsedMirrors({
				...options,
				registry: new ParserRegistry([new ChangingParser()]),
				signal: controller.signal,
				onProgress: (progress) => {
					if (progress.processed === 1) controller.abort();
				},
			}),
		).rejects.toBeDefined();
		expect(existsSync(join(root, ".autorag", "parsed", "index.json"))).toBe(false);
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-1\n");
		expect(output("a")).toBe(priorOutput);
		expect(await detectMirrorStaleness(options)).toContainEqual(expect.objectContaining({ source: "/docs/a.txt" }));
		await syncParsedMirrors({ ...options, registry: new ParserRegistry([new VersionedParser()]) });
		expect(readFileSync(output("a"), "utf8")).toBe("Still changing parsed-1\n");
	});

	it.skipIf(process.platform === "win32")("does not reparse content after a permissions-only change", async () => {
		const parser = new VersionedParser();
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await syncParsedMirrors(options);
		chmodSync(join(source, "a.txt"), 0o600);
		parser.attempts.length = 0;
		expect(await detectMirrorStaleness(options)).toEqual([]);
		await syncParsedMirrors(options);
		expect(parser.attempts).toEqual([]);
	});

	it.each([`/docs/${"한".repeat(500)}.txt`, "/docs/unpaired-\ud800.txt"])(
		"recovers stage outputs for unusual virtual path %s",
		async (virtualPath) => {
			const binary = join(root, "minsync.mjs");
			writeFakeMinSync(binary);
			const session = new MinSyncStageSession(binary, join(root, ".autorag", "parsed", "stage"), root);
			const key = mirrorStageKey(virtualPath);
			const version = JSON.stringify({
				virtualPath,
				sourcePath: join(source, "a.txt"),
				parserName: "plain-text",
				sourceMtimeNs: 1,
				sourceSizeBytes: 6,
			});
			try {
				await session.open({
					stage: "parse",
					source_id: "autorag-parsed-mirror",
					fingerprint: "test",
					output_root: join(root, ".autorag", "parsed", "files"),
					items: [{ key, version }],
				});
				const outputPath = parsedOutputPath(root, virtualPath);
				writeFileSync(outputPath, "Long path content\n");
				await session.commit(key, version, { path: basename(outputPath) });
				await session.finish();
			} finally {
				await session.close();
			}
			expect(readFileSync(loadMirrorIndex(root).entries[virtualPath]!.outputPath, "utf8")).toBe(
				"Long path content\n",
			);
		},
	);

	it("changes execution limits without reparsing healthy outputs but retries prior parser failures", async () => {
		class FailingParser extends VersionedParser {
			fail = true;
			async parse(input: ParseInput): Promise<ParseOutput> {
				const result = await super.parse(input);
				if (this.fail && input.virtualPath === "/docs/a.txt")
					throw new ParseError(this.name, input.virtualPath, "timeout");
				return result;
			}
		}
		const parser = new FailingParser();
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await syncParsedMirrors({
			...options,
			parserOptions: {
				fileTimeoutMs: 1000,
				maxHeapMb: 128,
				ocr: { enabled: false, cachePath: "/old", timeoutMs: 1000 },
			},
		});
		parser.fail = false;
		parser.attempts.length = 0;
		await syncParsedMirrors({
			...options,
			parserOptions: {
				fileTimeoutMs: 2000,
				maxHeapMb: 256,
				ocr: { enabled: false, cachePath: "/new", timeoutMs: 2000 },
			},
		});
		expect(parser.attempts).toEqual(["/docs/a.txt"]);
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-1\n");
		expect(readFileSync(output("b"), "utf8")).toBe("Bravo parsed-1\n");
	});

	it("rebuilds unchanged sources when the parser implementation version changes", async () => {
		const parser = new VersionedParser();
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await syncParsedMirrors(options);
		parser.version = "2";
		expect(await detectMirrorStaleness(options)).toContainEqual(
			expect.objectContaining({ source: "/docs/a.txt", reason: "parser-changed" }),
		);
		await syncParsedMirrors(options);
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-2\n");
		expect(readFileSync(output("b"), "utf8")).toBe("Bravo parsed-2\n");
	});

	it("repairs a damaged mirror without parsing the healthy sibling again", async () => {
		const parser = new VersionedParser();
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await syncParsedMirrors(options);
		writeFileSync(output("a"), "corrupt\n");
		parser.attempts.length = 0;
		await syncParsedMirrors(options);
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-1\n");
		expect(parser.attempts).toEqual(["/docs/a.txt"]);
	});

	it("invalidates parsed outputs when output-affecting parser options change", async () => {
		const parser = new VersionedParser();
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await syncParsedMirrors({ ...options, parserOptions: { languages: ["en"] } });
		parser.attempts.length = 0;
		await syncParsedMirrors({ ...options, parserOptions: { languages: ["ko"] } });
		expect(parser.attempts).toEqual(["/docs/a.txt", "/docs/b.txt"]);
	});

	it("resumes committed parsing without the metadata cache, preserving it while its source root is offline", async () => {
		const parser = new VersionedParser();
		const controller = new AbortController();
		const reason = new Error("cancel after first commit");
		const options = { root, searchPaths: [source], registry: new ParserRegistry([parser]) };
		await expect(
			syncParsedMirrors({
				...options,
				signal: controller.signal,
				onProgress: (p) => {
					if (p.processed === 1) controller.abort(reason);
				},
			}),
		).rejects.toBe(reason);
		expect(existsSync(join(root, ".autorag", "parsed", "index.json"))).toBe(false);
		renameSync(source, `${source}-offline`);
		const offline = await syncParsedMirrors(options);
		expect(offline.deleted).toBe(0);
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-1\n");
		renameSync(`${source}-offline`, source);
		parser.attempts.length = 0;
		await syncParsedMirrors(options);
		expect(parser.attempts).toEqual(["/docs/b.txt"]);
		expect(readFileSync(output("b"), "utf8")).toBe("Bravo parsed-1\n");
	});

	it("preserves invalid-output and ordinary-failure identities while their root is unavailable", async () => {
		class CrashingParser extends VersionedParser {
			async parse(input: ParseInput): Promise<ParseOutput> {
				if (input.virtualPath === "/docs/b.txt") throw new Error("crashed");
				return super.parse(input);
			}
		}
		const options = { root, searchPaths: [source], registry: new ParserRegistry([new CrashingParser()]) };
		await syncParsedMirrors(options);
		const damagedPath = output("a");
		writeFileSync(damagedPath, "corrupt\n");
		renameSync(source, `${source}-offline`);
		const offline = await syncParsedMirrors(options);
		expect(offline.deleted).toBe(0);
		expect(offline.diagnostics.some((d) => d.code === "deleted-mirror")).toBe(false);
		expect([...(loadMirrorIndex(root).known ?? [])].sort()).toEqual(["/docs/a.txt", "/docs/b.txt"]);
		expect(readFileSync(damagedPath, "utf8")).toBe("corrupt\n");
		renameSync(`${source}-offline`, source);
		const parser = new VersionedParser();
		await syncParsedMirrors({ ...options, registry: new ParserRegistry([parser]) });
		expect(parser.attempts).toEqual(["/docs/a.txt", "/docs/b.txt"]);
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-1\n");
		expect(readFileSync(output("b"), "utf8")).toBe("Bravo parsed-1\n");
	});

	it("does not commit a source that changes while its parser is running", async () => {
		class ChangingParser extends VersionedParser {
			async parse(input: ParseInput): Promise<ParseOutput> {
				const parsed = await super.parse(input);
				if (input.virtualPath === "/docs/a.txt") writeFileSync(join(source, "a.txt"), "Changed during parsing\n");
				return parsed;
			}
		}
		const result = await syncParsedMirrors({
			root,
			searchPaths: [source],
			registry: new ParserRegistry([new ChangingParser()]),
		});
		expect(result.diagnostics).toContainEqual(
			expect.objectContaining({ code: "source-changed", source: "/docs/a.txt" }),
		);
		expect(loadMirrorIndex(root).entries["/docs/a.txt"]).toBeUndefined();
		await syncParsedMirrors({ root, searchPaths: [source], registry: new ParserRegistry([new VersionedParser()]) });
		expect(readFileSync(output("a"), "utf8")).toBe("Changed during parsing parsed-1\n");
	});

	it("rejects a competing refresh without overwriting the active run's metadata", async () => {
		let release!: () => void;
		let entered!: () => void;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const started = new Promise<void>((resolve) => {
			entered = resolve;
		});
		class WaitingParser extends VersionedParser {
			async parse(input: ParseInput): Promise<ParseOutput> {
				entered();
				await gate;
				return super.parse(input);
			}
		}
		const options = { root, searchPaths: [source], registry: new ParserRegistry([new WaitingParser()]) };
		const active = syncParsedMirrors(options);
		await started;
		try {
			await expect(syncParsedMirrors(options)).rejects.toMatchObject({ code: "lock_failed" });
		} finally {
			release();
			await active;
		}
		expect(readFileSync(output("a"), "utf8")).toBe("Alpha parsed-1\n");
		expect(readFileSync(output("b"), "utf8")).toBe("Bravo parsed-1\n");
	});
});
