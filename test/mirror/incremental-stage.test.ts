import { mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { detectMirrorStaleness, loadMirrorIndex, syncParsedMirrors } from "../../src/mirror/index.ts";
import { type ParseInput, type ParseOutput, ParserRegistry, PlainTextParser } from "../../src/parser/index.ts";

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
		rmSync(join(root, ".autorag", "parsed", "index.json"));
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
