import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createFsService, mapDupeyFamily, type FsServiceDeps } from "../src/main/fs-service";
import { createDupeyProbe, DUPEY_INSTALL_COMMAND } from "../src/main/dupey";
import type { DupeyScanResult } from "@autorag/librarian";
import type { FinderEntry } from "../src/renderer/src/data/entries";
import {
	applyVersionStacks,
	buildVersionFamilies,
	defaultIndexIncluded,
	EMPTY_VERSION_FAMILIES,
	relationLabel,
	type VersionFamilyData,
} from "../src/renderer/src/state/version-family";

function entry(path: string, name = path.split("/").pop() ?? ""): FinderEntry {
	return {
		name,
		path,
		kind: "file",
		fileKind: "xlsx",
		dateLabel: "Sep 13, 16:48",
		sizeLabel: "84 KB",
		kindLabel: "Excel Spreadsheet",
		location: path.slice(0, path.lastIndexOf("/")) || path,
		modifiedValue: 0,
		sizeValue: 0,
	};
}

const HEAD = "Documents/Finance/2026 Q3/Q3_마케팅예산_v3.xlsx";
const SAME_FOLDER_MEMBER = "Documents/Finance/2026 Q3/Q3_마케팅예산_v2.xlsx";
const CROSS_FOLDER_MEMBER = "Downloads/Q3_마케팅예산_v3 (1).xlsx";

function families(): VersionFamilyData[] {
	return [
		{
			head: HEAD,
			members: [
				{ path: SAME_FOLDER_MEMBER, relation: "near" },
				{ path: CROSS_FOLDER_MEMBER, relation: "exact" },
			],
			entriesByPath: {
				[HEAD]: entry(HEAD),
				[SAME_FOLDER_MEMBER]: entry(SAME_FOLDER_MEMBER),
				[CROSS_FOLDER_MEMBER]: entry(CROSS_FOLDER_MEMBER),
			},
		},
	];
}

describe("applyVersionStacks", () => {
	it("passes rows through when there are no families", () => {
		const rows = [entry(HEAD)];
		const out = applyVersionStacks(rows, {
			families: EMPTY_VERSION_FAMILIES,
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out).toEqual([{ entry: rows[0], stackCount: 0, stackOpen: false, child: false, relation: null }]);
	});

	it("hides same-folder members from the flat list", () => {
		const out = applyVersionStacks([entry(HEAD), entry(SAME_FOLDER_MEMBER)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out.map((row) => row.entry.path)).toEqual([HEAD]);
	});

	it("keeps cross-folder members as their own flat rows", () => {
		const out = applyVersionStacks([entry(CROSS_FOLDER_MEMBER)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out.map((row) => row.entry.path)).toEqual([CROSS_FOLDER_MEMBER]);
		expect(out[0]?.child).toBe(false);
	});

	it("counts members on the head and expands on selection", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [HEAD],
		});
		expect(out[0]?.stackCount).toBe(2);
		expect(out[0]?.stackOpen).toBe(true);
		expect(out.map((row) => row.entry.path)).toEqual([HEAD, SAME_FOLDER_MEMBER, CROSS_FOLDER_MEMBER]);
		expect(out[1]?.child).toBe(true);
		expect(out[1]?.relation).toBe("near");
		expect(out[2]?.relation).toBe("exact");
	});

	it("auto-expands when a member is selected", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [CROSS_FOLDER_MEMBER],
		});
		expect(out[0]?.stackOpen).toBe(true);
	});

	it("expands through the manual pin", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set([HEAD]),
			selectedKeys: [],
		});
		expect(out[0]?.stackOpen).toBe(true);
	});

	it("stays closed when nothing selects or pins it", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out[0]?.stackOpen).toBe(false);
		expect(out.map((row) => row.entry.path)).toEqual([HEAD]);
	});
});

describe("relationLabel", () => {
	it("labels the relation alone in the same folder", () => {
		expect(relationLabel("exact", "Documents/Finance/2026 Q3", "Documents/Finance/2026 Q3")).toBe("동일본");
	});

	it("appends the folder when the member lives elsewhere", () => {
		expect(relationLabel("near", "Downloads", "Documents/Finance/2026 Q3")).toBe("유사본 · Downloads");
	});
});

describe("defaultIndexIncluded", () => {
	it("excludes stack members by default", () => {
		const built = buildVersionFamilies(families());
		expect(defaultIndexIncluded(SAME_FOLDER_MEMBER, built)).toBe(false);
		expect(defaultIndexIncluded(CROSS_FOLDER_MEMBER, built)).toBe(false);
		expect(defaultIndexIncluded(HEAD, built)).toBe(true);
		expect(defaultIndexIncluded("Documents/other.txt", built)).toBe(true);
	});
});

function scan(family: Record<string, unknown>[]): DupeyScanResult {
	return { dir: "/x", files: [], families: family as unknown as DupeyScanResult["families"], errors: [] };
}

describe("mapDupeyFamily", () => {
	it("uses the pick keeper as head", () => {
		const family = {
			id: 0,
			relation: "mixed",
			files: ["/x/a.txt", "/x/b.txt"],
			members: [
				{ path: "/x/a.txt", relation: "exact", exact_hash: "h1", joined_with: "/x/b.txt" },
				{ path: "/x/b.txt", relation: "exact", exact_hash: "h1" },
			],
			pick: { ranked: [{ path: "/x/b.txt", rank: 1 }] },
		};
		const mapped = mapDupeyFamily(family as never);
		expect(mapped?.head).toBe("/x/b.txt");
		expect(mapped?.members).toEqual([{ path: "/x/a.txt", relation: "exact" }]);
	});

	it("marks hash-equal members exact regardless of the joined edge", () => {
		const family = {
			id: 0,
			relation: "mixed",
			files: ["/x/head.txt", "/x/c.txt", "/x/d.txt"],
			members: [
				{ path: "/x/head.txt", exact_hash: "same" },
				{ path: "/x/c.txt", exact_hash: "same", relation: "near", joined_with: "/x/d.txt" },
				{ path: "/x/d.txt", exact_hash: "other", relation: "contains" },
			],
			pick: { ranked: [{ path: "/x/head.txt" }] },
		};
		const mapped = mapDupeyFamily(family as never);
		expect(mapped?.members).toEqual([
			{ path: "/x/c.txt", relation: "exact" },
			{ path: "/x/d.txt", relation: "contains" },
		]);
	});

	it("clamps unknown relations to near", () => {
		const family = {
			id: 0,
			relation: "mixed",
			files: ["/x/head.txt", "/x/e.txt"],
			members: [{ path: "/x/head.txt" }, { path: "/x/e.txt", relation: "weird" }],
			pick: { ranked: [{ path: "/x/head.txt" }] },
		};
		const mapped = mapDupeyFamily(family as never);
		expect(mapped?.members).toEqual([{ path: "/x/e.txt", relation: "near" }]);
	});

	it("returns null without a head", () => {
		expect(mapDupeyFamily({ id: 0, relation: "mixed", files: [], members: [] } as never)).toBeNull();
	});
});

const dupeyAvailable = { status: async () => ({ available: true, version: "dupey 0.1.2", error: null }) };

async function serviceWith(scanResult: DupeyScanResult, home: string) {
	const deps: FsServiceDeps = {
		shell: {
			trashItem: async () => {},
			showItemInFolder: () => {},
		},
		clipboard: { writeText: () => {} },
		homeDir: home,
		dupey: dupeyAvailable,
		scanDuplicates: async () => scanResult,
	};
	return createFsService(deps);
}

describe("dupey probe", () => {
	it("reports the version when the CLI answers", async () => {
		const probe = createDupeyProbe({ run: async () => "dupey 0.1.2" });
		expect(await probe.status()).toEqual({ available: true, version: "dupey 0.1.2", error: null });
	});

	it("reports the spawn failure when the CLI is missing", async () => {
		const probe = createDupeyProbe({
			run: async () => {
				throw new Error("spawn dupey ENOENT");
			},
		});
		const status = await probe.status();
		expect(status.available).toBe(false);
		expect(status.error).toContain("ENOENT");
	});

	it("caches the result within the TTL and re-probes after it", async () => {
		let calls = 0;
		let clock = 0;
		const probe = createDupeyProbe({
			run: async () => {
				calls += 1;
				return "dupey 0.1.2";
			},
			ttlMs: 1000,
			now: () => clock,
		});
		await probe.status();
		clock = 500;
		await probe.status();
		expect(calls).toBe(1);
		clock = 1500;
		await probe.status();
		expect(calls).toBe(2);
	});
});

describe("fs service versionFamilies", () => {

	it("maps dupey families into entries and caches per root", async () => {
		const home = await mkdtemp(join(tmpdir(), "vf-home-"));
		await mkdir(join(home, "Desktop"), { recursive: true });
		await writeFile(join(home, "Desktop", "a.txt"), "same");
		await writeFile(join(home, "Desktop", "b.txt"), "same");
		const scanFixture = scan([
			{
				id: 0,
				relation: "mixed",
				files: [join(home, "Desktop", "a.txt"), join(home, "Desktop", "b.txt")],
				members: [
					{ path: join(home, "Desktop", "a.txt"), exact_hash: "h" },
					{ path: join(home, "Desktop", "b.txt"), exact_hash: "h" },
				],
				pick: { ranked: [{ path: join(home, "Desktop", "a.txt") }] },
			},
		]);
		let scans = 0;
		const deps: FsServiceDeps = {
			shell: { trashItem: async () => {}, showItemInFolder: () => {} },
			clipboard: { writeText: () => {} },
			homeDir: home,
			dupey: dupeyAvailable,
			scanDuplicates: async () => {
				scans++;
				return scanFixture;
			},
		};
		const service = createFsService(deps);
		const result = await service.versionFamilies();
		expect(result.error).toBeNull();
		expect(result.families).toHaveLength(1);
		expect(result.families[0]?.head).toBe(join(home, "Desktop", "a.txt"));
		expect(result.families[0]?.members).toEqual([{ path: join(home, "Desktop", "b.txt"), relation: "exact" }]);
		expect(result.families[0]?.entries.map((e) => e.path)).toEqual([
			join(home, "Desktop", "a.txt"),
			join(home, "Desktop", "b.txt"),
		]);

		await service.versionFamilies();
		expect(scans).toBe(1);
	});

	it("reports dupey-missing with the install command instead of degrading silently", async () => {
		const home = await mkdtemp(join(tmpdir(), "vf-home-"));
		let scans = 0;
		const service = createFsService({
			shell: { trashItem: async () => {}, showItemInFolder: () => {} },
			clipboard: { writeText: () => {} },
			homeDir: home,
			dupey: { status: async () => ({ available: false, version: null, error: "spawn dupey ENOENT" }) },
			scanDuplicates: async () => {
				scans += 1;
				return scan([]);
			},
		});
		const result = await service.versionFamilies();
		expect(result.families).toEqual([]);
		expect(result.error?.code).toBe("dupey-missing");
		expect(result.error?.installCommand).toBe(DUPEY_INSTALL_COMMAND);
		expect(result.error?.message).toContain("ENOENT");
		expect(scans).toBe(0);
	});

	it("reports scan-failed when dupey runs but a root scan breaks", async () => {
		const home = await mkdtemp(join(tmpdir(), "vf-home-"));
		await mkdir(join(home, "Desktop"), { recursive: true });
		const service = createFsService({
			shell: { trashItem: async () => {}, showItemInFolder: () => {} },
			clipboard: { writeText: () => {} },
			homeDir: home,
			dupey: dupeyAvailable,
			scanDuplicates: async () => {
				throw new Error("dupey scan exploded");
			},
		});
		const result = await service.versionFamilies();
		expect(result.families).toEqual([]);
		expect(result.error?.code).toBe("scan-failed");
		expect(result.error?.message).toContain("dupey scan exploded");
		expect(result.error?.installCommand).toBeNull();
	});

	it("skips families whose files disappeared", async () => {
		const home = await mkdtemp(join(tmpdir(), "vf-home-"));
		const result = scan([
			{
				id: 0,
				relation: "mixed",
				files: [join(home, "gone.txt"), join(home, "gone2.txt")],
				members: [
					{ path: join(home, "gone.txt"), exact_hash: "h" },
					{ path: join(home, "gone2.txt"), exact_hash: "h" },
				],
				pick: { ranked: [{ path: join(home, "gone.txt") }] },
			},
		]);
		const service = await serviceWith(result, home);
		const scanResult = await service.versionFamilies();
		expect(scanResult.families).toEqual([]);
		expect(scanResult.error).toBeNull();
	});
});
