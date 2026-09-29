import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createRecentsStore, RECENTS_FILENAME } from "../src/main/recents-store";

let root: string;

beforeEach(async () => {
	root = await mkdtemp(join(tmpdir(), "autorag-recents-test-"));
});

afterEach(async () => {
	await rm(root, { recursive: true, force: true });
});

describe("recents store", () => {
	it("lists recorded paths most-recent first", async () => {
		// Given a store and two recorded files
		const store = createRecentsStore({ directory: root });
		await store.record(join(root, "first.pdf"));
		await store.record(join(root, "second.xlsx"));

		// When listing
		const listed = await store.list();

		// Then the newest record leads
		expect(listed).toEqual([join(root, "second.xlsx"), join(root, "first.pdf")]);
	});

	it("moves a re-opened path back to the front without duplicating it", async () => {
		// Given three opened files
		const store = createRecentsStore({ directory: root });
		const first = join(root, "first.pdf");
		const second = join(root, "second.xlsx");
		const third = join(root, "third.docx");
		await store.record(first);
		await store.record(second);
		await store.record(third);

		// When the oldest file is opened again
		await store.record(first);

		// Then it leads and appears exactly once
		expect(await store.list()).toEqual([first, third, second]);
	});

	it("survives a new store instance over the same directory", async () => {
		// Given a recorded file
		const filePath = join(root, "persisted.pdf");
		await createRecentsStore({ directory: root }).record(filePath);

		// When a fresh store reads the same directory
		const listed = await createRecentsStore({ directory: root }).list();

		// Then the record is still there
		expect(listed).toEqual([filePath]);
	});

	it("keeps only the newest paths when the limit is exceeded", async () => {
		// Given a store capped at three
		const store = createRecentsStore({ directory: root, limit: 3 });

		// When four files are opened in order
		for (const name of ["a.txt", "b.txt", "c.txt", "d.txt"]) {
			await store.record(join(root, name));
		}

		// Then the oldest drops off
		expect(await store.list()).toEqual([join(root, "d.txt"), join(root, "c.txt"), join(root, "b.txt")]);
	});

	it("keeps OS-native paths verbatim, including Windows paths", async () => {
		// Given a store and Windows-style absolute paths
		const store = createRecentsStore({ directory: root });
		const windowsPath = "C:\\Users\\me\\Documents\\Q3 budget.xlsx";
		const uncPath = "\\\\server\\share\\report.pdf";
		await store.record(windowsPath);
		await store.record(uncPath);

		// When listing through a fresh store
		const listed = await createRecentsStore({ directory: root }).list();

		// Then both paths round-trip unchanged
		expect(listed).toEqual([uncPath, windowsPath]);
	});

	it("starts empty when the store file does not exist", async () => {
		// Given a directory with no store file
		// When listing
		const listed = await createRecentsStore({ directory: root }).list();

		// Then the list is empty
		expect(listed).toEqual([]);
	});

	it("starts empty when the store file is corrupt", async () => {
		// Given a store file with invalid JSON
		await writeFile(join(root, RECENTS_FILENAME), "{ not json", "utf8");

		// When listing
		const listed = await createRecentsStore({ directory: root }).list();

		// Then the corrupt file degrades to an empty list
		expect(listed).toEqual([]);
	});

	it("drops malformed entries from a stored file", async () => {
		// Given a stored array with one valid path and several malformed entries
		await writeFile(
			join(root, RECENTS_FILENAME),
			JSON.stringify([join(root, "kept.pdf"), "", 42, null, { path: "nope" }]),
			"utf8",
		);

		// When listing
		const listed = await createRecentsStore({ directory: root }).list();

		// Then only the valid path survives
		expect(listed).toEqual([join(root, "kept.pdf")]);
	});

	it("serializes concurrent records without losing any", async () => {
		// Given a store and five records issued at once
		const store = createRecentsStore({ directory: root });
		const paths = ["a", "b", "c", "d", "e"].map((name) => join(root, `${name}.txt`));

		// When all records are awaited together
		await Promise.all(paths.map((path) => store.record(path)));

		// Then every path is present and the persisted file parses
		const listed = await store.list();
		expect([...listed].sort()).toEqual([...paths].sort());
		expect(JSON.parse(await readFile(join(root, RECENTS_FILENAME), "utf8"))).toEqual(listed);
	});

	it("creates the store directory when it does not exist yet", async () => {
		// Given a store directory two levels below an existing root
		const nested = join(root, "userData", "recents");
		await mkdir(join(root, "userData"), { recursive: true });

		// When recording a file
		await createRecentsStore({ directory: nested }).record(join(root, "deep.pdf"));

		// Then the file landed in the created directory
		expect(await createRecentsStore({ directory: nested }).list()).toEqual([join(root, "deep.pdf")]);
	});

	it("caps the default history at 100 entries, dropping the oldest", async () => {
		// Given a store left at its default limit and 101 opened files
		const store = createRecentsStore({ directory: root });
		for (let index = 0; index < 101; index += 1) {
			await store.record(join(root, `file-${String(index).padStart(3, "0")}.txt`));
		}

		// When listing
		const listed = await store.list();

		// Then the newest 100 survive, newest first, and the first one is gone
		expect(listed).toHaveLength(100);
		expect(listed[0]).toBe(join(root, "file-100.txt"));
		expect(listed.at(-1)).toBe(join(root, "file-001.txt"));
		expect(listed).not.toContain(join(root, "file-000.txt"));
	});
});
