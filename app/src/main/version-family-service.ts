/**
 * dupey version-family scanning, scheduling, and the SSOT hand-off.
 *
 * Flow: dupey CLI -> persisted snapshot (userData) -> UI. The UI is always
 * served from the snapshot; scans run once at startup and then on a repeating
 * interval that the app Settings control. Nothing here reacts to file
 * mutations — the interval is the only refresh trigger.
 */

import { scanWithDupey, type DupeyScanResult } from "@autorag/librarian";
import type {
	FsEntry,
	FsLocation,
	FsVersionFamiliesResult,
	FsVersionFamily,
	FsVersionFamilyError,
	FsVersionMember,
	FsVersionRelation,
} from "../shared/fs-contract";
import { createDupeyProbe, DUPEY_INSTALL_COMMAND, type DupeyProbe } from "./dupey";
import type { StoredVersionFamilies, VersionFamilyStore } from "./version-family-store";

export const DEFAULT_SCAN_INTERVAL_MINUTES = 60;

interface DupeyMemberInfo {
	readonly path?: string;
	readonly relation?: string;
	readonly exact_hash?: string;
	readonly joined_with?: string;
}

function clampRelation(value: string | undefined): FsVersionRelation {
	return value === "exact" || value === "contains" ? value : "near";
}

/** dupey family (pick-keeper head + per-member relations) to the app contract. */
export function mapDupeyFamily(
	family: DupeyScanResult["families"][number],
): { readonly head: string; readonly members: readonly FsVersionMember[] } | null {
	const head =
		(family.pick as { readonly ranked?: readonly { readonly path?: string }[] } | undefined)?.ranked?.[0]
			?.path ?? family.files[0];
	if (typeof head !== "string" || head.length === 0) return null;
	const memberInfos = new Map<string, DupeyMemberInfo>();
	for (const member of family.members as readonly DupeyMemberInfo[]) {
		if (typeof member?.path === "string") memberInfos.set(member.path, member);
	}
	const headHash = memberInfos.get(head)?.exact_hash;
	const members: FsVersionMember[] = [];
	for (const path of family.files) {
		if (path === head) continue;
		const info = memberInfos.get(path);
		const relation =
			info?.exact_hash !== undefined && headHash !== undefined && info.exact_hash === headHash
				? "exact"
				: clampRelation(info?.relation);
		members.push({ path, relation });
	}
	return { head, members };
}

export interface VersionFamilyServiceDeps {
	readonly locations: () => Promise<readonly FsLocation[]>;
	readonly store: VersionFamilyStore;
	readonly buildEntry: (path: string) => Promise<FsEntry>;
	/** Defaults to the dupey CLI; tests inject fixture scans. */
	readonly scanDuplicates?: (dir: string) => Promise<DupeyScanResult>;
	readonly dupey?: DupeyProbe;
	/** Current interval in minutes, read from app Settings on every arm. */
	readonly intervalMinutes: () => number;
	/** Fired after every refresh so the renderer can adopt the new snapshot. */
	readonly onUpdate?: (result: FsVersionFamiliesResult) => void;
	readonly now?: () => Date;
	readonly setTimer?: (handler: () => void, ms: number) => unknown;
	readonly clearTimer?: (handle: unknown) => void;
}

export interface VersionFamilyService {
	/** Snapshot read (memory, then disk). Never scans. */
	result(): Promise<FsVersionFamiliesResult>;
	/** Runs a scan, persists the snapshot, and reports the outcome. */
	refresh(): Promise<FsVersionFamiliesResult>;
	/** Startup scan plus the repeating interval. */
	start(): Promise<FsVersionFamiliesResult>;
	stop(): void;
	/** Re-reads the interval from Settings and re-arms the timer. */
	reschedule(): void;
}

export function createVersionFamilyService(deps: VersionFamilyServiceDeps): VersionFamilyService {
	const scanDuplicates = deps.scanDuplicates ?? ((dir: string) => scanWithDupey(dir));
	const dupey = deps.dupey ?? createDupeyProbe();
	const now = deps.now ?? (() => new Date());
	const setTimer = deps.setTimer ?? ((handler: () => void, ms: number) => setInterval(handler, ms));
	const clearTimer = deps.clearTimer ?? ((handle: unknown) => clearInterval(handle as NodeJS.Timeout));

	let snapshot: StoredVersionFamilies | null | undefined;
	let lastError: FsVersionFamilyError | null = null;
	let timer: unknown = null;

	async function load(): Promise<StoredVersionFamilies | null> {
		if (snapshot === undefined) snapshot = await deps.store.read();
		return snapshot;
	}

	function envelope(stored: StoredVersionFamilies | null): FsVersionFamiliesResult {
		return {
			families: stored?.families ?? [],
			scannedAt: stored?.scannedAt ?? null,
			error: lastError,
		};
	}

	async function result(): Promise<FsVersionFamiliesResult> {
		return envelope(await load());
	}

	async function scanLocations(): Promise<{
		readonly families: readonly FsVersionFamily[];
		readonly locations: readonly string[];
		readonly errors: readonly string[];
	}> {
		const families: FsVersionFamily[] = [];
		const locations: string[] = [];
		const errors: string[] = [];
		for (const location of await deps.locations()) {
			if (!location.available) continue;
			locations.push(location.path);
			try {
				const scan = await scanDuplicates(location.path);
				for (const raw of scan.families) {
					const mapped = mapDupeyFamily(raw);
					if (mapped === null || mapped.members.length === 0) continue;
					const entries: FsEntry[] = [];
					try {
						entries.push(await deps.buildEntry(mapped.head));
					} catch {
						continue;
					}
					for (const member of mapped.members) {
						try {
							entries.push(await deps.buildEntry(member.path));
						} catch {
							continue;
						}
					}
					const members = mapped.members.filter((member) =>
						entries.some((entry) => entry.path === member.path),
					);
					if (members.length === 0) continue;
					families.push({ head: mapped.head, members, entries });
				}
			} catch (error) {
				const message = `dupey scan failed for ${location.path}: ${
					error instanceof Error ? error.message : String(error)
				}`;
				console.error(message);
				errors.push(message);
			}
		}
		return { families, locations, errors };
	}

	async function refresh(): Promise<FsVersionFamiliesResult> {
		const status = await dupey.status();
		if (!status.available) {
			lastError = {
				code: "dupey-missing",
				message: `dupey CLI is required for version stacks: ${status.error ?? "not found on PATH"}`,
				installCommand: DUPEY_INSTALL_COMMAND,
			};
			const current = envelope(await load());
			deps.onUpdate?.(current);
			return current;
		}
		const { families, locations, errors } = await scanLocations();
		if (families.length > 0 || errors.length === 0) {
			const stored: StoredVersionFamilies = {
				version: 1,
				scannedAt: now().toISOString(),
				locations,
				families,
			};
			await deps.store.write(stored);
			snapshot = stored;
		}
		lastError =
			errors.length === 0 ? null : { code: "scan-failed", message: errors.join("\n"), installCommand: null };
		const current = envelope(await load());
		deps.onUpdate?.(current);
		return current;
	}

	function arm(): void {
		if (timer !== null) {
			clearTimer(timer);
			timer = null;
		}
		const minutes = Math.max(1, Math.floor(deps.intervalMinutes()));
		timer = setTimer(() => {
			void refresh();
		}, minutes * 60_000);
	}

	return {
		result,
		refresh,
		async start(): Promise<FsVersionFamiliesResult> {
			arm();
			return refresh();
		},
		stop(): void {
			if (timer !== null) {
				clearTimer(timer);
				timer = null;
			}
		},
		reschedule: arm,
	};
}
