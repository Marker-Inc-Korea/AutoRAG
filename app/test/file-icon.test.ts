import { writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { describe, expect, it } from "vitest";
import { ICON_LIMIT, type IconTarget, createIconProvider } from "../src/main/file-icon";

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
		let planned: readonly string[] = [];
		const provider = createIconProvider({
			platform: "darwin",
			generate: async (paths, outDir) => {
				planned = [...paths];
				for (const path of paths) await writeThumbnail(path, outDir);
			},
		});
		const many = Array.from({ length: ICON_LIMIT + 5 }, (_value, index) => target(`/docs/f${index}.png`));

		// When they are listed
		const icons = await provider.icons(many);

		// Then exactly the limit was generated and nothing beyond it got an icon
		expect(planned).toHaveLength(ICON_LIMIT);
		expect(icons.size).toBe(ICON_LIMIT);
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
		// Given a generator that fails
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
