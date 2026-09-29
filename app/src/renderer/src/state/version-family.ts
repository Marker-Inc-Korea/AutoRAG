/**
 * Version families (handoff README §2 "File rows", prototype `FAMS`).
 *
 * A family groups duplicate or near-duplicate documents behind one head row.
 * Members whose head lives in the same folder are hidden from the flat list
 * and appear only when the stack expands; members in other folders keep their
 * own flat row and also appear as children when the stack is open.
 */

import type { FinderEntry } from "../data/entries";
import { basename, dirname } from "./paths";

export type VersionRelation = "exact" | "near" | "contains";

export interface VersionMember {
	readonly path: string;
	readonly relation: VersionRelation;
}

export interface VersionFamilyData {
	readonly head: string;
	readonly members: readonly VersionMember[];
	readonly entriesByPath: Readonly<Record<string, FinderEntry>>;
}

export const RELATION_LABELS: Readonly<Record<VersionRelation, string>> = {
	exact: "동일본",
	near: "유사본",
	contains: "포함본",
};

export interface ResolvedMembership {
	readonly family: VersionFamilyData;
	readonly relation: VersionRelation;
}

export interface VersionFamilies {
	readonly byHead: ReadonlyMap<string, VersionFamilyData>;
	readonly memberOf: ReadonlyMap<string, ResolvedMembership>;
}

export interface VersionFamilyError {
	readonly code: "dupey-missing" | "scan-failed";
	readonly message: string;
	readonly installCommand: string | null;
}

export interface VersionFamiliesResult {
	readonly families: readonly VersionFamilyData[];
	readonly scannedAt: string | null;
	readonly error: VersionFamilyError | null;
}

export const EMPTY_VERSION_FAMILIES: VersionFamilies = {
	byHead: new Map(),
	memberOf: new Map(),
};

export const EMPTY_VERSION_FAMILIES_RESULT: VersionFamiliesResult = {
	families: [],
	scannedAt: null,
	error: null,
};

export function buildVersionFamilies(families: readonly VersionFamilyData[]): VersionFamilies {
	const byHead = new Map<string, VersionFamilyData>();
	const memberOf = new Map<string, ResolvedMembership>();
	for (const family of families) {
		if (byHead.has(family.head)) continue;
		byHead.set(family.head, family);
		for (const member of family.members) {
			if (member.path === family.head || memberOf.has(member.path)) continue;
			memberOf.set(member.path, { family, relation: member.relation });
		}
	}
	return { byHead, memberOf };
}

/** One rendered row after the version-stack pass. */
export interface StackRow {
	readonly entry: FinderEntry;
	readonly stackCount: number;
	readonly stackOpen: boolean;
	readonly child: boolean;
	readonly relation: VersionRelation | null;
}

export interface StackOptions {
	readonly families: VersionFamilies;
	readonly manualOpen: ReadonlySet<string>;
	readonly selectedKeys: readonly string[];
}

/** Kind-cell label for a child row: relation plus the folder when it differs. */
export function relationLabel(relation: VersionRelation, memberLocation: string, listPath: string): string {
	const folder = basename(memberLocation);
	return memberLocation === listPath ? RELATION_LABELS[relation] : `${RELATION_LABELS[relation]} · ${folder}`;
}

/**
 * Applies the prototype's `curRows` stack rules to a sorted flat listing:
 *
 * - A member whose head is in the same folder is hidden from the flat list.
 * - A head row carries the live member count and opens when selected
 *   (itself or any member) or pinned open through the badge.
 * - Open stacks append their member rows, cross-folder ones included, as
 *   indented children right after the head.
 */
export function applyVersionStacks(
	rows: readonly FinderEntry[],
	options: StackOptions,
): readonly StackRow[] {
	const { families, manualOpen, selectedKeys } = options;
	if (families.byHead.size === 0) return rows.map(stackRow);

	const selected = new Set(selectedKeys);
	const out: StackRow[] = [];
	for (const entry of rows) {
		const membership = families.memberOf.get(entry.path);
		if (membership !== undefined) {
			const headDir = dirname(membership.family.head);
			if (headDir === entry.location) continue;
		}
		const family = families.byHead.get(entry.path);
		if (family === undefined) {
			out.push(stackRow(entry));
			continue;
		}
		const members = family.members.filter(
			(member) => family.entriesByPath[member.path] !== undefined,
		);
		const open =
			members.length > 0 &&
			(manualOpen.has(entry.path) ||
				selected.has(entry.path) ||
				members.some((member) => selected.has(member.path)));
		out.push({ entry, stackCount: members.length, stackOpen: open, child: false, relation: null });
		if (open) {
			for (const member of members) {
				const memberEntry = family.entriesByPath[member.path];
				if (memberEntry === undefined) continue;
				out.push({
					entry: memberEntry,
					stackCount: 0,
					stackOpen: false,
					child: true,
					relation: member.relation,
				});
			}
		}
	}
	return out;
}

function stackRow(entry: FinderEntry): StackRow {
	return { entry, stackCount: 0, stackOpen: false, child: false, relation: null };
}

/** Stack members default to excluded from indexing (handoff README §2). */
export function defaultIndexIncluded(path: string, families: VersionFamilies): boolean {
	return !families.memberOf.has(path);
}
