import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readJsonLines, readQrels, readTopicsTsv, writeJsonAtomic } from "../../benchmark/miracl/jsonl.ts";

describe("MIRACL input parsing", () => {
	const roots: string[] = [];
	const makeRoot = () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-miracl-"));
		roots.push(root);
		return root;
	};

	afterEach(() => {
		roots.splice(0).forEach((root) => rmSync(root, { recursive: true, force: true }));
		vi.doUnmock("node:fs/promises");
		vi.resetModules();
	});

	it("parses strict topics and qrels", async () => {
		const root = makeRoot();
		mkdirSync(root, { recursive: true });
		writeFileSync(join(root, "topics.tsv"), "q1\t한국어 질문\n");
		writeFileSync(join(root, "qrels.txt"), "q1 Q0 d1 2\nq1 Q0 d2 0\n");
		expect(await readTopicsTsv(join(root, "topics.tsv"))).toEqual([{ queryId: "q1", text: "한국어 질문" }]);
		expect(await readQrels(join(root, "qrels.txt"))).toEqual([
			{ queryId: "q1", documentId: "d1", relevance: 2 },
			{ queryId: "q1", documentId: "d2", relevance: 0 },
		]);
	});

	it("rejects duplicate topic ids and malformed relevance", async () => {
		const root = makeRoot();
		writeFileSync(join(root, "topics.tsv"), "q1\ta\nq1\tb\n");
		writeFileSync(join(root, "qrels.txt"), "q1 Q0 d1 NaN\n");
		await expect(readTopicsTsv(join(root, "topics.tsv"))).rejects.toThrow("duplicate query id");
		await expect(readQrels(join(root, "qrels.txt"))).rejects.toThrow("finite integer");
	});

	it("rejects blank topic identifiers and incomplete qrel columns", async () => {
		const root = makeRoot();
		writeFileSync(join(root, "topics.tsv"), " \tquestion\n");
		writeFileSync(join(root, "qrels.txt"), "q1 Q0 \t1\n");
		await expect(readTopicsTsv(join(root, "topics.tsv"))).rejects.toThrow("query id");
		await expect(readQrels(join(root, "qrels.txt"))).rejects.toThrow("exactly four columns");
	});

	it("rejects duplicate qrel pairs", async () => {
		const root = makeRoot();
		const path = join(root, "qrels.txt");
		writeFileSync(path, "q1 Q0 d1 1\nq1 Q0 d1 2\n");
		await expect(readQrels(path)).rejects.toThrow("duplicate qrel");
	});

	it("rejects empty, oversized, and malformed input lines", async () => {
		const root = makeRoot();
		const emptyPath = join(root, "empty.tsv");
		const oversizedPath = join(root, "oversized.tsv");
		const malformedPath = join(root, "malformed.txt");
		writeFileSync(emptyPath, "");
		writeFileSync(oversizedPath, `q1\t${"a".repeat(16 * 1024 * 1024 + 1)}\n`);
		writeFileSync(malformedPath, "q1 Q0 d1\n");
		await expect(readTopicsTsv(emptyPath)).rejects.toThrow("is empty");
		await expect(readTopicsTsv(oversizedPath)).rejects.toThrow("exceeds 16 MiB");
		await expect(readQrels(malformedPath)).rejects.toThrow("exactly four columns");
	});

	it("reads valid JSONL and rejects blank or malformed JSONL lines", async () => {
		const root = makeRoot();
		const validPath = join(root, "valid.jsonl");
		const blankPath = join(root, "blank.jsonl");
		const malformedPath = join(root, "malformed.jsonl");
		writeFileSync(validPath, '{"id":"one"}\r\n{"id":"two"}\n');
		writeFileSync(blankPath, '\n');
		writeFileSync(malformedPath, '{"id":}\n');
		expect(await readJsonLines<{ id: string }>(validPath)).toEqual([{ id: "one" }, { id: "two" }]);
		await expect(readJsonLines(blankPath)).rejects.toThrow("must not be blank");
		await expect(readJsonLines(malformedPath)).rejects.toThrow("invalid JSON");
	});

	it("writes a private JSON file without replacing an existing destination", async () => {
		const root = makeRoot();
		const path = join(root, "result.json");
		await writeJsonAtomic(path, { status: "new" });
		expect(readFileSync(path, "utf8")).toBe('{"status":"new"}\n');
		if (process.platform !== "win32") {
			expect(statSync(path).mode & 0o777).toBe(0o600);
		}
		await expect(writeJsonAtomic(path, { status: "replacement" })).rejects.toThrow("destination already exists");
		expect(readFileSync(path, "utf8")).toBe('{"status":"new"}\n');
	});

	it("cleans up the temporary file when publication fails", async () => {
		const root = makeRoot();
		const path = join(root, "result.json");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rename: async () => {
				throw new Error("publication failed");
			},
			link: async () => {
				throw new Error("publication failed");
			},
		}));
		const { writeJsonAtomic: writeWithFailedPublication } = await import("../../benchmark/miracl/jsonl.ts");
		await expect(writeWithFailedPublication(path, { status: "new" })).rejects.toThrow("publication failed");
		expect(existsSync(`${path}.tmp-${process.pid}`)).toBe(false);
	});

	it("does not overwrite a destination created during publication", async () => {
		const root = makeRoot();
		const path = join(root, "result.json");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		const createCompetingDestination = async (destinationPath: string) => {
			if (destinationPath === path) {
				await actual.writeFile(path, '"competing"\n', "utf8");
			}
		};
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rename: async (temporaryPath: string, destinationPath: string) => {
				await createCompetingDestination(destinationPath);
				return actual.rename(temporaryPath, destinationPath);
			},
			link: async (temporaryPath: string, destinationPath: string) => {
				await createCompetingDestination(destinationPath);
				return actual.link(temporaryPath, destinationPath);
			},
		}));
		const { writeJsonAtomic: writeWithConcurrentDestination } = await import("../../benchmark/miracl/jsonl.ts");
		await expect(writeWithConcurrentDestination(path, { status: "new" })).rejects.toThrow("destination already exists");
		expect(readFileSync(path, "utf8")).toBe('"competing"\n');
		expect(existsSync(`${path}.tmp-${process.pid}`)).toBe(false);
	});
});
