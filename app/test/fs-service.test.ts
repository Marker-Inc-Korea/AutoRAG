import { access, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createFsService, FsRenameError, type FsServiceDeps } from "../src/main/fs-service";

interface StubbedDeps {
	readonly deps: FsServiceDeps;
	readonly trashed: string[];
	readonly revealed: string[];
	readonly quickLooked: string[];
	readonly opened: string[];
	readonly clipboardText: string[];
}

function makeDeps(homeDir: string): StubbedDeps {
	const trashed: string[] = [];
	const revealed: string[] = [];
	const quickLooked: string[] = [];
	const opened: string[] = [];
	const clipboardText: string[] = [];
	return {
		deps: {
			shell: {
				trashItem: async (path: string) => {
					trashed.push(path);
				},
				showItemInFolder: (path: string) => {
					revealed.push(path);
				},
				openPath: async (path: string) => {
					opened.push(path);
					return "";
				},
			},
			clipboard: {
				writeText: (text: string) => {
					clipboardText.push(text);
				},
			},
			homeDir,
			spawnQuickLook: (path: string) => {
				quickLooked.push(path);
			},
			osKind: async () => null,
			icons: { icons: async () => new Map() },
		},
		trashed,
		revealed,
		quickLooked,
		opened,
		clipboardText,
	};
}

function makeFailingTrashDeps(homeDir: string, message: string): FsServiceDeps {
	return {
		shell: {
			trashItem: async () => {
				throw new Error(message);
			},
			showItemInFolder: () => { },
			openPath: async () => "",
		},
		clipboard: { writeText: () => { } },
		homeDir,
	};
}

let root: string;

beforeEach(async () => {
	root = await mkdtemp(join(tmpdir(), "autorag-fs-test-"));
});

afterEach(async () => {
	await rm(root, { recursive: true, force: true });
});

describe("listDir", () => {
	it("returns files and folders with the contract shape, folders getting size null", async () => {
		// Given a directory with one file and one folder
		const dir = join(root, "listing");
		await mkdir(join(dir, "sub"), { recursive: true });
		await writeFile(join(dir, "Notes.TXT"), "hello");

		// When listing the directory
		const listing = await createFsService(makeDeps(root).deps).listDir(dir);

		// Then the path echoes the input and both entries carry the contract shape
		expect(listing.path).toBe(dir);
		const file = listing.entries.find((entry) => entry.name === "Notes.TXT");
		const folder = listing.entries.find((entry) => entry.name === "sub");
		expect(file).toBeDefined();
		expect(file?.kind).toBe("file");
		expect(file?.ext).toBe("txt");
		expect(file?.size).toBe(5);
		expect(file?.isSymlink).toBe(false);
		expect(Number.isNaN(Date.parse(file?.modifiedAt ?? ""))).toBe(false);
		expect(folder).toBeDefined();
		expect(folder?.kind).toBe("folder");
		expect(folder?.ext).toBe("");
		expect(folder?.size).toBeNull();
	});

	it("flags symlinks", async () => {
		// Given a file and a symlink pointing at it
		const dir = join(root, "links");
		await mkdir(dir, { recursive: true });
		await writeFile(join(dir, "target.txt"), "x");
		await symlink(join(dir, "target.txt"), join(dir, "link.txt"));

		// When listing the directory
		const listing = await createFsService(makeDeps(root).deps).listDir(dir);

		// Then the symlink entry is flagged and the real file is not
		expect(listing.entries.find((entry) => entry.name === "link.txt")?.isSymlink).toBe(true);
		expect(listing.entries.find((entry) => entry.name === "target.txt")?.isSymlink).toBe(false);
	});
});

describe("stat", () => {
	it("returns an FsEntry for a single file", async () => {
		// Given a file
		const filePath = join(root, "single.md");
		await writeFile(filePath, "abc");

		// When statting it
		const entry = await createFsService(makeDeps(root).deps).stat(filePath);

		// Then the entry describes the file
		expect(entry).toMatchObject({
			name: "single.md",
			path: filePath,
			kind: "file",
			ext: "md",
			size: 3,
			isSymlink: false,
		});
	});
});

describe("copy", () => {
	it("copies into the destination with Finder-style collision naming", async () => {
		// Given a source file and a destination already holding "a.txt" and "a copy.txt"
		const srcDir = join(root, "src");
		const destDir = join(root, "dest");
		await mkdir(srcDir, { recursive: true });
		await mkdir(destDir, { recursive: true });
		const source = join(srcDir, "a.txt");
		await writeFile(source, "payload");
		await writeFile(join(destDir, "a.txt"), "old");
		await writeFile(join(destDir, "a copy.txt"), "old copy");

		// When copying the source into the destination
		const result = await createFsService(makeDeps(root).deps).copy([source], destDir);

		// Then the copy lands as "a copy 2.txt" with the original content
		expect(result.ok).toEqual([source]);
		expect(result.failed).toEqual([]);
		expect(await readFile(join(destDir, "a copy 2.txt"), "utf8")).toBe("payload");
	});

	it("uses ' copy' for the first collision", async () => {
		// Given a destination holding only the original name
		const srcDir = join(root, "src1");
		const destDir = join(root, "dest1");
		await mkdir(srcDir, { recursive: true });
		await mkdir(destDir, { recursive: true });
		const source = join(srcDir, "b.txt");
		await writeFile(source, "payload");
		await writeFile(join(destDir, "b.txt"), "old");

		// When copying the source into the destination
		await createFsService(makeDeps(root).deps).copy([source], destDir);

		// Then the copy lands as "b copy.txt"
		expect(await readFile(join(destDir, "b copy.txt"), "utf8")).toBe("payload");
	});

	it("reports per-path failures verbatim instead of throwing", async () => {
		// Given a path that does not exist
		const destDir = join(root, "dest2");
		await mkdir(destDir, { recursive: true });
		const missing = join(root, "missing.txt");

		// When copying it
		const result = await createFsService(makeDeps(root).deps).copy([missing], destDir);

		// Then the batch result names the failed path with a non-empty reason
		expect(result.ok).toEqual([]);
		expect(result.failed).toHaveLength(1);
		expect(result.failed[0]?.path).toBe(missing);
		expect(result.failed[0]?.message.length).toBeGreaterThan(0);
	});
});

describe("duplicate", () => {
	it("duplicates in place as '<name> copy' without touching the original", async () => {
		// Given a file
		const filePath = join(root, "doc.txt");
		await writeFile(filePath, "v1");

		// When duplicating it twice
		const service = createFsService(makeDeps(root).deps);
		await service.duplicate([filePath]);
		await service.duplicate([filePath]);

		// Then "doc copy.txt" and "doc copy 2.txt" exist next to the intact original
		expect(await readFile(join(root, "doc copy.txt"), "utf8")).toBe("v1");
		expect(await readFile(join(root, "doc copy 2.txt"), "utf8")).toBe("v1");
		expect(await readFile(filePath, "utf8")).toBe("v1");
	});
});

describe("rename", () => {
	it("rejects an empty name with a typed invalid-name error", async () => {
		// Given a file
		const filePath = join(root, "a.txt");
		await writeFile(filePath, "x");
		const service = createFsService(makeDeps(root).deps);

		// When renaming to an empty or whitespace-only name, Then a typed error is raised
		for (const badName of ["", "   "]) {
			const failure = await service.rename(filePath, badName).catch((error: unknown) => error);
			expect(failure).toBeInstanceOf(FsRenameError);
			expect((failure as FsRenameError).code).toBe("invalid-name");
		}
	});

	it("rejects names containing path separators", async () => {
		// Given a file
		const filePath = join(root, "a.txt");
		await writeFile(filePath, "x");

		// When renaming to a name with a path separator, Then a typed invalid-name error is raised
		const failure = await createFsService(makeDeps(root).deps)
			.rename(filePath, "nested/a.txt")
			.catch((error: unknown) => error);
		expect(failure).toBeInstanceOf(FsRenameError);
		expect((failure as FsRenameError).code).toBe("invalid-name");
	});

	it("rejects a collision with an existing name", async () => {
		// Given two files
		const aPath = join(root, "a.txt");
		const bPath = join(root, "b.txt");
		await writeFile(aPath, "a");
		await writeFile(bPath, "b");

		// When renaming a.txt over b.txt, Then a typed name-collision error is raised and b.txt survives
		const failure = await createFsService(makeDeps(root).deps)
			.rename(aPath, "b.txt")
			.catch((error: unknown) => error);
		expect(failure).toBeInstanceOf(FsRenameError);
		expect((failure as FsRenameError).code).toBe("name-collision");
		expect(await readFile(bPath, "utf8")).toBe("b");
	});

	it("renames to a fresh name and returns the new entry", async () => {
		// Given a file
		const oldPath = join(root, "old.txt");
		await writeFile(oldPath, "content");

		// When renaming it
		const entry = await createFsService(makeDeps(root).deps).rename(oldPath, "new.txt");

		// Then the new entry points at the new path and the old path is gone
		expect(entry.name).toBe("new.txt");
		expect(entry.path).toBe(join(root, "new.txt"));
		expect(entry.kind).toBe("file");
		await expect(access(oldPath)).rejects.toThrow();
		expect(await readFile(entry.path, "utf8")).toBe("content");
	});
});

describe("in-app clipboard and move", () => {
	it("starts empty and round-trips a cut entry", async () => {
		// Given a fresh service
		const service = createFsService(makeDeps(root).deps);

		// Then the clipboard starts empty
		expect(await service.clipboardGet()).toBeNull();

		// When setting a cut clipboard
		await service.clipboardSet({ op: "cut", paths: [join(root, "clip.txt")] });

		// Then it round-trips
		expect(await service.clipboardGet()).toEqual({ op: "cut", paths: [join(root, "clip.txt")] });
	});

	it("moves the clipboard cut paths into the destination (cut + paste)", async () => {
		// Given a file captured in the clipboard as cut
		const srcDir = join(root, "cut-src");
		const destDir = join(root, "cut-dest");
		await mkdir(srcDir, { recursive: true });
		await mkdir(destDir, { recursive: true });
		const source = join(srcDir, "move-me.txt");
		await writeFile(source, "moving");
		const service = createFsService(makeDeps(root).deps);
		await service.clipboardSet({ op: "cut", paths: [source] });

		// When pasting: reading the clipboard and moving its paths into the destination
		const clipboard = await service.clipboardGet();
		expect(clipboard?.op).toBe("cut");
		const result = await service.move(clipboard?.paths ?? [], destDir);

		// Then the file lives in the destination and the source is gone
		expect(result.ok).toEqual([source]);
		expect(await readFile(join(destDir, "move-me.txt"), "utf8")).toBe("moving");
		await expect(access(source)).rejects.toThrow();
	});

	it("applies collision naming when the destination already has the name", async () => {
		// Given a destination already holding the same file name
		const srcDir = join(root, "move-src");
		const destDir = join(root, "move-dest");
		await mkdir(srcDir, { recursive: true });
		await mkdir(destDir, { recursive: true });
		const source = join(srcDir, "same.txt");
		await writeFile(source, "incoming");
		await writeFile(join(destDir, "same.txt"), "existing");

		// When moving the source into the destination
		await createFsService(makeDeps(root).deps).move([source], destDir);

		// Then it lands as "same copy.txt"
		expect(await readFile(join(destDir, "same copy.txt"), "utf8")).toBe("incoming");
		expect(await readFile(join(destDir, "same.txt"), "utf8")).toBe("existing");
	});
});

describe("trash", () => {
	it("routes paths through the injected shell (stubbed, no real Trash)", async () => {
		// Given a real file and a stubbed shell
		const filePath = join(root, "trash-me.txt");
		await writeFile(filePath, "x");
		const stubs = makeDeps(root);

		// When trashing it
		const result = await createFsService(stubs.deps).trash([filePath]);

		// Then the shell was asked to trash the path; the stub leaves the file on disk
		expect(result.ok).toEqual([filePath]);
		expect(stubs.trashed).toEqual([filePath]);
		expect(await readFile(filePath, "utf8")).toBe("x");
	});

	it("reports the shell error verbatim in the batch result", async () => {
		// Given a shell whose trashItem always fails
		const filePath = join(root, "trash-fail.txt");
		await writeFile(filePath, "x");

		// When trashing
		const result = await createFsService(makeFailingTrashDeps(root, "trash boom")).trash([filePath]);

		// Then the failure carries the path and the verbatim shell message
		expect(result.ok).toEqual([]);
		expect(result.failed).toEqual([{ path: filePath, message: "trash boom" }]);
	});
});

describe("search", () => {
	it("finds files and folders by case-insensitive name substring across locations", async () => {
		// Given a home whose Documents tree holds a matching file and a matching folder
		const home = join(root, "home");
		await mkdir(join(home, "Documents", "inner"), { recursive: true });
		await mkdir(join(home, "Downloads"), { recursive: true });
		await writeFile(join(home, "Documents", "inner", "report-final.txt"), "r");
		await mkdir(join(home, "Documents", "Quarterly Reports"));
		await writeFile(join(home, "Downloads", "unrelated.txt"), "u");

		// When searching for "report"
		const results = await createFsService(makeDeps(home).deps).search("REPORT");

		// Then both the file and the folder match, attributed to Documents, and unrelated files do not
		const names = results.map((hit) => hit.entry.name).sort();
		expect(names).toEqual(["Quarterly Reports", "report-final.txt"]);
		expect(results.every((hit) => hit.location === "Documents")).toBe(true);
	});

	it("returns nothing when no name matches", async () => {
		// Given a home with files that do not match the query
		const home = join(root, "home-empty");
		await mkdir(join(home, "Documents"), { recursive: true });
		await writeFile(join(home, "Documents", "alpha.txt"), "a");

		// When searching for a missing substring, Then the result is empty
		expect(await createFsService(makeDeps(home).deps).search("zzz-nothing")).toEqual([]);
	});
});

describe("locations", () => {
	it("lists existing home dirs as available and missing ones as unavailable", async () => {
		// Given a home with Desktop and Documents but no Downloads, iCloud, or third-party drives
		const home = join(root, "home-locations");
		await mkdir(join(home, "Desktop"), { recursive: true });
		await mkdir(join(home, "Documents"), { recursive: true });

		// When listing locations
		const locations = await createFsService(makeDeps(home).deps).locations();

		// Then Desktop/Documents are available, Downloads and iCloud are listed but unavailable
		const byName = new Map(locations.map((location) => [location.name, location]));
		expect(byName.get("Desktop")).toMatchObject({ path: join(home, "Desktop"), section: "favorites", available: true });
		expect(byName.get("Documents")).toMatchObject({ available: true });
		expect(byName.get("Downloads")).toMatchObject({ available: false });
		expect(byName.get("iCloud Drive")).toMatchObject({ section: "cloud", available: false });

		// And Google Drive / Dropbox are omitted entirely when their well-known paths do not exist
		expect(byName.has("Google Drive")).toBe(false);
		expect(byName.has("Dropbox")).toBe(false);
	});
});

describe("shell and clipboard side effects", () => {
	it("reveal forwards the path to shell.showItemInFolder", async () => {
		// Given a stubbed shell
		const stubs = makeDeps(root);
		const filePath = join(root, "reveal.txt");
		await writeFile(filePath, "x");

		// When revealing the file
		await createFsService(stubs.deps).reveal(filePath);

		// Then the shell was asked to reveal it
		expect(stubs.revealed).toEqual([filePath]);
	});

	it("quickLook forwards the path to the injected spawner", async () => {
		// Given a stubbed qlmanage spawner
		const stubs = makeDeps(root);
		const filePath = join(root, "peek.txt");
		await writeFile(filePath, "x");

		// When opening Quick Look
		await createFsService(stubs.deps).quickLook(filePath);

		// Then the spawner received the path
		expect(stubs.quickLooked).toEqual([filePath]);
	});

	it("open forwards the path to shell.openPath, the OS default application", async () => {
		// Given a stubbed shell
		const stubs = makeDeps(root);
		const filePath = join(root, "deck.key");
		await writeFile(filePath, "x");

		// When opening the file like a double-click
		await createFsService(stubs.deps).open(filePath);

		// Then the shell was asked to open it with the OS default app
		expect(stubs.opened).toEqual([filePath]);
	});

	it("open rejects with the verbatim shell error message", async () => {
		// Given a shell that failed to open the path
		const deps: FsServiceDeps = {
			shell: {
				trashItem: async () => { },
				showItemInFolder: () => { },
				openPath: async () => "The file “deck.key” does not exist.",
			},
			clipboard: { writeText: () => { } },
			homeDir: root,
		};

		// When opening the file, the real message surfaces verbatim
		await expect(createFsService(deps).open(join(root, "deck.key"))).rejects.toThrow(
			"The file “deck.key” does not exist.",
		);
	});

	it("copyPathsToClipboard writes newline-joined paths to the OS clipboard", async () => {
		// Given a stubbed OS clipboard
		const stubs = makeDeps(root);

		// When copying two paths
		await createFsService(stubs.deps).copyPathsToClipboard(["/a/one.txt", "/b/two.txt"]);

		// Then the clipboard received them joined by newlines
		expect(stubs.clipboardText).toEqual(["/a/one.txt\n/b/two.txt"]);
	});
});

describe("osKind", () => {
	it("annotates each file with the OS-detected kind", async () => {
		// Given a folder with a video and a markdown file
		const dir = await mkdtemp(join(tmpdir(), "fs-oskind-"));
		await writeFile(join(dir, "clip.mp4"), "x");
		await writeFile(join(dir, "note.md"), "x");
		const deps: FsServiceDeps = {
			...makeDeps(dir).deps,
			osKind: async (_path, ext) => (ext === "mp4" ? "MPEG-4 movie" : ext === "md" ? "Markdown Document" : null),
		};

		// When listing the folder
		const listing = await createFsService(deps).listDir(dir);

		// Then each file carries its OS kind
		expect(listing.entries.find((entry) => entry.name === "clip.mp4")?.osKind).toBe("MPEG-4 movie");
		expect(listing.entries.find((entry) => entry.name === "note.md")?.osKind).toBe("Markdown Document");
		await rm(dir, { recursive: true, force: true });
	});

	it("leaves folders without an OS kind", async () => {
		// Given a folder holding a subfolder
		const dir = await mkdtemp(join(tmpdir(), "fs-oskind-folder-"));
		await mkdir(join(dir, "sub"));

		// When listing it
		const listing = await createFsService(makeDeps(dir).deps).listDir(dir);

		// Then the subfolder has no OS kind
		expect(listing.entries.find((entry) => entry.name === "sub")?.osKind).toBeNull();
		await rm(dir, { recursive: true, force: true });
	});

	it("degrades to null when the OS lookup fails, keeping the entry", async () => {
		// Given a folder whose OS lookup throws
		const dir = await mkdtemp(join(tmpdir(), "fs-oskind-fail-"));
		await writeFile(join(dir, "clip.mp4"), "x");
		const deps: FsServiceDeps = {
			...makeDeps(dir).deps,
			osKind: async () => {
				throw new Error("mdls: boom");
			},
		};

		// When listing it
		const listing = await createFsService(deps).listDir(dir);

		// Then the entry survives with no OS kind
		const entry = listing.entries.find((candidate) => candidate.name === "clip.mp4");
		expect(entry?.osKind).toBeNull();
		expect(entry?.ext).toBe("mp4");
		await rm(dir, { recursive: true, force: true });
	});
});

describe("icons", () => {
	it("annotates each file with the OS tile icon", async () => {
		// Given a folder whose provider produced an icon for the video only
		const dir = await mkdtemp(join(tmpdir(), "fs-icons-"));
		await writeFile(join(dir, "clip.mp4"), "x");
		await writeFile(join(dir, "note.txt"), "x");
		const deps: FsServiceDeps = {
			...makeDeps(dir).deps,
			icons: {
				icons: async (targets) =>
					new Map(
						targets.filter((target) => target.path.endsWith(".mp4")).map((target) => [target.path, "data:image/png;base64,AAAA"]),
					),
			},
		};

		// When listing the folder
		const listing = await createFsService(deps).listDir(dir);

		// Then the video carries the icon and the other file keeps the letter tile
		expect(listing.entries.find((entry) => entry.name === "clip.mp4")?.iconDataUrl).toBe("data:image/png;base64,AAAA");
		expect(listing.entries.find((entry) => entry.name === "note.txt")?.iconDataUrl).toBeNull();
		await rm(dir, { recursive: true, force: true });
	});
});
