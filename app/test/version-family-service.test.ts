import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { DupeyScanResult } from "@autorag/librarian";
import { createDupeyProbe, DUPEY_INSTALL_COMMAND } from "../src/main/dupey";
import {
	createVersionFamilyService,
	DEFAULT_SCAN_INTERVAL_MINUTES,
	mapDupeyFamily,
	type VersionFamilyServiceDeps,
} from "../src/main/version-family-service";
import { createFileVersionFamilyStore } from "../src/main/version-family-store";
import type { FsEntry, FsLocation, FsVersionFamiliesResult } from "../src/shared/fs-contract";

const dupeyAvailable = { status: async () => ({ available: true, version: "dupey 0.1.2", error: null }) };
const dupeyMissing = { status: async () => ({ available: false, version: null, error: "spawn dupey ENOENT" }) };

function fsEntry(path: string): FsEntry {
	return {
		name: path.split("/").pop() ?? path,
		path,
		kind: "file",
		ext: "txt",
		size: 4,
		modifiedAt: "2026-09-29T00:00:00.000Z",
		isSymlink: false,
	};
}

function scanOf(families: readonly Record<string, unknown>[]): DupeyScanResult {
	return { dir: "/x", files: [], families: families as unknown as DupeyScanResult["families"], errors: [] };
}

function family(head: string, member: string, relation = "exact"): Record<string, unknown> {
	return {
		id: 0,
		relation: "mixed",
		files: [head, member],
		members: [
			{ path: head, exact_hash: "h" },
			{ path: member, exact_hash: "h", relation },
		],
		pick: { ranked: [{ path: head }] },
	};
}

interface Harness {
	readonly deps: VersionFamilyServiceDeps;
	readonly updates: FsVersionFamiliesResult[];
	readonly timers: { ms: number; fire: () => void }[];
	readonly scans: string[];
	readonly writes: number;
}

function harness(overrides: Partial<VersionFamilyServiceDeps> = {}): Harness {
	const updates: FsVersionFamiliesResult[] = [];
	const timers: { ms: number; fire: () => void }[] = [];
	const scans: string[] = [];
	const state = { writes: 0 };
	let snapshot: unknown = null;
	const locations: FsLocation[] = [
		{ name: "Desktop", path: "/home/Desktop", section: "favorites", available: true },
	];
	const deps: VersionFamilyServiceDeps = {
		locations: async () => locations,
		store: {
			read: async () => (snapshot as never) ?? null,
			write: async (value) => {
				snapshot = value;
				state.writes += 1;
			},
		},
		buildEntry: async (path) => fsEntry(path),
		dupey: dupeyAvailable,
		scanDuplicates: async (dir) => {
			scans.push(dir);
			return scanOf([family("/home/Desktop/a.txt", "/home/Desktop/b.txt")]);
		},
		intervalMinutes: () => DEFAULT_SCAN_INTERVAL_MINUTES,
		onUpdate: (result) => updates.push(result),
		now: () => new Date("2026-09-29T10:00:00.000Z"),
		setTimer: (handler, ms) => {
			timers.push({ ms, fire: handler });
			return timers.length;
		},
		clearTimer: () => {},
		...overrides,
	};
	return { deps, updates, timers, scans, get writes() { return state.writes; } };
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

describe("mapDupeyFamily", () => {
	it("uses the pick keeper as head", () => {
		const mapped = mapDupeyFamily(family("/x/b.txt", "/x/a.txt") as never);
		expect(mapped?.head).toBe("/x/b.txt");
		expect(mapped?.members).toEqual([{ path: "/x/a.txt", relation: "exact" }]);
	});

	it("clamps unknown relations to near", () => {
		const mapped = mapDupeyFamily({
			id: 0,
			relation: "mixed",
			files: ["/x/head.txt", "/x/e.txt"],
			members: [{ path: "/x/head.txt" }, { path: "/x/e.txt", relation: "weird" }],
			pick: { ranked: [{ path: "/x/head.txt" }] },
		} as never);
		expect(mapped?.members).toEqual([{ path: "/x/e.txt", relation: "near" }]);
	});

	it("returns null without a head", () => {
		expect(mapDupeyFamily({ id: 0, relation: "mixed", files: [], members: [] } as never)).toBeNull();
	});
});

describe("version-family service — SSOT", () => {
	it("serves the persisted snapshot without scanning", async () => {
		const h = harness({
			store: {
				read: async () => ({
					version: 1 as const,
					scannedAt: "2026-09-28T00:00:00.000Z",
					locations: ["/home/Desktop"],
					families: [{ head: "/home/Desktop/a.txt", members: [{ path: "/home/Desktop/b.txt", relation: "exact" as const }], entries: [] }],
				}),
				write: async () => {},
			},
		});
		const service = createVersionFamilyService(h.deps);
		const result = await service.result();
		expect(h.scans).toEqual([]);
		expect(result.scannedAt).toBe("2026-09-28T00:00:00.000Z");
		expect(result.families).toHaveLength(1);
		expect(result.error).toBeNull();
	});

	it("scans, persists, and publishes on refresh", async () => {
		const h = harness();
		const service = createVersionFamilyService(h.deps);
		const result = await service.refresh();
		expect(h.scans).toEqual(["/home/Desktop"]);
		expect(result.families).toHaveLength(1);
		expect(result.scannedAt).toBe("2026-09-29T10:00:00.000Z");
		expect(h.writes).toBe(1);
		expect(h.updates).toHaveLength(1);
	});

	it("reports dupey-missing and keeps the last snapshot", async () => {
		const h = harness({ dupey: dupeyMissing });
		const service = createVersionFamilyService(h.deps);
		const result = await service.refresh();
		expect(result.families).toEqual([]);
		expect(result.error?.code).toBe("dupey-missing");
		expect(result.error?.installCommand).toBe(DUPEY_INSTALL_COMMAND);
		expect(h.scans).toEqual([]);
		expect(h.writes).toBe(0);
	});

	it("reports scan-failed and keeps the stored snapshot when every root fails", async () => {
		const h = harness({
			store: {
				read: async () => ({
					version: 1 as const,
					scannedAt: "2026-09-28T00:00:00.000Z",
					locations: ["/home/Desktop"],
					families: [{ head: "/home/Desktop/old.txt", members: [{ path: "/home/Desktop/old2.txt", relation: "near" as const }], entries: [] }],
				}),
				write: async () => {
					throw new Error("write should not happen");
				},
			},
			scanDuplicates: async () => {
				throw new Error("dupey scan exploded");
			},
		});
		const service = createVersionFamilyService(h.deps);
		const result = await service.refresh();
		expect(result.error?.code).toBe("scan-failed");
		expect(result.error?.message).toContain("dupey scan exploded");
		expect(result.families.map((f) => f.head)).toEqual(["/home/Desktop/old.txt"]);
		expect(result.scannedAt).toBe("2026-09-28T00:00:00.000Z");
	});
});

describe("version-family service — schedule", () => {
	it("arms the timer at the configured interval on start", async () => {
		const h = harness();
		const service = createVersionFamilyService(h.deps);
		await service.start();
		expect(h.timers).toHaveLength(1);
		expect(h.timers[0]?.ms).toBe(DEFAULT_SCAN_INTERVAL_MINUTES * 60_000);
		expect(h.scans).toEqual(["/home/Desktop"]);
	});

	it("re-arms with the new interval on reschedule", async () => {
		let minutes = 60;
		const cleared: unknown[] = [];
		const h = harness({
			intervalMinutes: () => minutes,
			clearTimer: (handle) => cleared.push(handle),
		});
		const service = createVersionFamilyService(h.deps);
		await service.start();
		minutes = 15;
		service.reschedule();
		expect(h.timers.map((t) => t.ms)).toEqual([3_600_000, 900_000]);
		expect(cleared).toEqual([1]);
	});

	it("scans again when the interval fires", async () => {
		const h = harness();
		const service = createVersionFamilyService(h.deps);
		await service.start();
		h.timers[0]?.fire();
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(h.scans).toEqual(["/home/Desktop", "/home/Desktop"]);
	});

	it("stops the timer", async () => {
		const cleared: unknown[] = [];
		const h = harness({ clearTimer: (handle) => cleared.push(handle) });
		const service = createVersionFamilyService(h.deps);
		await service.start();
		service.stop();
		expect(cleared).toEqual([1]);
	});
});

describe("file version-family store", () => {
	it("round-trips a snapshot through the userData directory", async () => {
		const dir = await mkdtemp(join(tmpdir(), "vf-store-"));
		const store = createFileVersionFamilyStore(dir);
		expect(await store.read()).toBeNull();
		const snapshot = {
			version: 1 as const,
			scannedAt: "2026-09-29T10:00:00.000Z",
			locations: ["/home/Desktop"],
			families: [
				{
					head: "/home/Desktop/a.txt",
					members: [{ path: "/home/Desktop/b.txt", relation: "exact" as const }],
					entries: [fsEntry("/home/Desktop/a.txt")],
				},
			],
		};
		await store.write(snapshot);
		expect(await store.read()).toEqual(snapshot);
		expect(JSON.parse(await readFile(join(dir, "version-families.json"), "utf8"))).toEqual(snapshot);
	});

	it("returns null for a corrupt snapshot", async () => {
		const dir = await mkdtemp(join(tmpdir(), "vf-store-"));
		await writeFile(join(dir, "version-families.json"), "{ not json", "utf8");
		expect(await createFileVersionFamilyStore(dir).read()).toBeNull();
	});
});
