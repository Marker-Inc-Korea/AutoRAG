import { writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { describe, expect, it } from "vitest";
import { ICON_LIMIT, type IconTarget, THUMBNAIL_CHUNK, createIconProvider } from "../src/main/file-icon";

const BYTES = new Uint8Array([1, 2, 3, 4]);
const target = (path: string, modifiedAt = "2026-09-29T00:00:00.000Z"): IconTarget => ({ path, modifiedAt });
const writeThumbnail = (path: string, outDir: string) =>
	writeFile(join(outDir, `${basename(path)}.png`), BYTES);

describe("createIconProvider", () => {
	it("generates one batch per listing and returns a data URL per produced thumbnail", async () => {
		// Given a macOS generator that records its batches
		const batches: string[][] = [];
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				batches.push([...paths]);
				for (const path of paths) await writeThumbnail(path, outDir);
			},
		});

		// When two files are listed together
		const icons = await provider.icons([target("/docs/a.png"), target("/docs/b.mp4")]);

		// Then one process covered both
		expect(batches).toEqual([["/docs/a.png", "/docs/b.mp4"]]);
		expect(icons.get("/docs/a.png")).toBe(`data:image/png;base64,${Buffer.from(BYTES).toString("base64")}`);
		expect(icons.get("/docs/b.mp4")).toBe(`data:image/png;base64,${Buffer.from(BYTES).toString("base64")}`);
	});

	it("reuses the cached icon for an unchanged file and regenerates a changed one", async () => {
		// Given a provider whose generator counts runs
		let runs = 0;
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				runs += 1;
				for (const path of paths) await writeThumbnail(path, outDir);
			},
		});

		// When the same file is listed twice unchanged, then changes
		await provider.icons([target("/docs/a.png", "t1")]);
		await provider.icons([target("/docs/a.png", "t1")]);
		expect(runs).toBe(1);
		await provider.icons([target("/docs/a.png", "t2")]);

		// Then only the changed file triggered a new run
		expect(runs).toBe(2);
	});

	it("caps one listing at the icon limit", async () => {
		// Given more files than the limit
		const planned: string[] = [];
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				planned.push(...paths);
				for (const path of paths) await writeThumbnail(path, outDir);
			},
		});
		const many = Array.from({ length: ICON_LIMIT + 5 }, (_value, index) => target(`/docs/f${index}.png`));

		// When they are listed
		const icons = await provider.icons(many);

		// Then exactly the limit was planned and nothing beyond it got an icon
		expect(planned).toHaveLength(ICON_LIMIT);
		expect(icons.size).toBe(ICON_LIMIT);
	});

	it("splits a large listing into small sequential chunks", async () => {
		// Given more files than one chunk holds
		const batches: string[][] = [];
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				batches.push([...paths]);
				for (const path of paths) await writeThumbnail(path, outDir);
			},
		});
		const many = Array.from({ length: THUMBNAIL_CHUNK * 2 + 1 }, (_value, index) => target(`/docs/f${index}.png`));

		// When they are listed
		const icons = await provider.icons(many);

		// Then the generator never saw more than one chunk at a time
		expect(batches.map((batch) => batch.length)).toEqual([THUMBNAIL_CHUNK, THUMBNAIL_CHUNK, 1]);
		expect(icons.size).toBe(THUMBNAIL_CHUNK * 2 + 1);
	});

	it("skips a crashed chunk and keeps the rest", async () => {
		// Given a generator whose first chunk crashes like qlmanage -t does
		let call = 0;
		const warnings: string[] = [];
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				call += 1;
				if (call === 1) throw new Error("Segmentation fault: 11");
				for (const path of paths) await writeThumbnail(path, outDir);
			},
			warn: (message) => {
				warnings.push(message);
			},
		});
		const many = Array.from({ length: THUMBNAIL_CHUNK + 1 }, (_value, index) => target(`/docs/f${index}.png`));

		// When they are listed
		const icons = await provider.icons(many);

		// Then only the crashed chunk went without icons and the crash surfaced verbatim
		expect(icons.size).toBe(1);
		expect(icons.get(`/docs/f${THUMBNAIL_CHUNK}.png`)).toBeDefined();
		expect(warnings.join("\n")).toContain("Segmentation fault: 11");
	});

	it("does not regenerate icons for files that produced none", async () => {
		// Given a file with no OS thumbnail (qlmanage writes nothing)
		let generateCalls = 0;
		const provider = createIconProvider({
			platform: "darwin",
			generate: async () => {
				generateCalls += 1;
			},
		});

		// When the same file is listed twice
		expect((await provider.icons([target("/docs/app.zip")])).size).toBe(0);
		expect((await provider.icons([target("/docs/app.zip")])).size).toBe(0);

		// Then the OS was only asked once — per-file nulls are cached too
		expect(generateCalls).toBe(1);
	});

	it("skips a file the generator produced nothing for", async () => {
		// Given a generator that only writes one of two files
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				await writeThumbnail(paths[0] ?? "", outDir);
			},
		});

		// When both are listed
		const icons = await provider.icons([target("/docs/a.png"), target("/docs/unsupported.bin")]);

		// Then only the produced one has an icon
		expect([...icons.keys()]).toEqual(["/docs/a.png"]);
	});

	it("logs a generation failure verbatim and returns no icons", async () => {
		// Given a generator that fails every chunk
		const warnings: string[] = [];
		const provider = createIconProvider({
			platform: "darwin",
			generate: async () => {
				throw new Error("qlmanage: failed to generate thumbnails");
			},
			warn: (message) => {
				warnings.push(message);
			},
		});

		// When a file is listed
		const icons = await provider.icons([target("/docs/a.png")]);

		// Then the listing survives with no icon and the real message is visible
		expect(icons.size).toBe(0);
		expect(warnings.join("\n")).toContain("qlmanage: failed to generate thumbnails");
	});

	it("uses the OS type icon off macOS", async () => {
		// Given a Windows provider with a system icon lookup
		const asked: string[] = [];
		const provider = createIconProvider({
			platform: "win32",
			systemIcon: async (path) => {
				asked.push(path);
				return path.endsWith(".mp4") ? "data:image/png;base64,AAAA" : null;
			},
		});

		// When two files are listed
		const icons = await provider.icons([target("C:\\docs\\a.mp4"), target("C:\\docs\\b.bin")]);

		// Then each was asked and only the answered one has an icon
		expect(asked).toEqual(["C:\\docs\\a.mp4", "C:\\docs\\b.bin"]);
		expect([...icons.keys()]).toEqual(["C:\\docs\\a.mp4"]);
	});
});
