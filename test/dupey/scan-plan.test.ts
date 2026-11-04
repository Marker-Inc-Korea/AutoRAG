import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { listCloudRoots, planDupeyScans } from "../../src/dupey/scan-plan.ts";
import type { CloudFileProbe } from "../../src/filesystem/cloud-placeholder.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-scan-plan-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("planDupeyScans", () => {
	it("skips a root that is a cloud folder or lies inside one", () => {
		const cloud = join(root, "OneDrive-School");
		const inside = join(cloud, "Documents");
		mkdirSync(inside, { recursive: true });

		const plan = planDupeyScans([cloud, inside], [{ path: cloud, kind: "file-provider" }]);

		expect(plan.targets).toEqual([]);
		expect(plan.skipped).toEqual([cloud, inside]);
	});

	it("scans a local root and excludes the cloud and network folders nested in it by name", () => {
		const local = join(root, "local");
		const drive = join(local, "GoogleDrive-me");
		const share = join(local, "mnt", "share");
		mkdirSync(drive, { recursive: true });
		mkdirSync(share, { recursive: true });

		const plan = planDupeyScans(
			[local],
			[
				{ path: drive, kind: "file-provider" },
				{ path: share, kind: "network" },
			],
		);

		expect(plan.skipped).toEqual([]);
		expect(plan.targets).toEqual([{ path: local, excludeDirs: ["GoogleDrive-me", "share"] }]);
	});

	it("keeps scanning every root when no cloud root is known", () => {
		const local = join(root, "local");
		mkdirSync(local);

		expect(planDupeyScans([local], [])).toEqual({ targets: [{ path: local, excludeDirs: [] }], skipped: [] });
	});
});

describe("listCloudRoots", () => {
	const failing: CloudFileProbe = {
		cloudRoots: async () => {
			throw new Error("mount hung");
		},
		datalessAmong: async () => new Set(),
		providerOf: async () => undefined,
		canRelease: () => false,
		evict: async () => ({ ok: true }),
	};

	it("reads a probe that fails as no roots instead of failing the scan", async () => {
		expect(await listCloudRoots(failing, [root])).toEqual([]);
	});

	it("reads a missing probe as no roots", async () => {
		expect(await listCloudRoots(undefined, [root])).toEqual([]);
	});
});
