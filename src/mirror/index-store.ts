import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { lookupMinSyncInPath } from "../minsync/binary.ts";
import { minSyncBinaryMissingError } from "../minsync/errors.ts";
import { executableName } from "../minsync/installer.ts";
import { type MinSyncStageInspectReport, readMinSyncStage } from "../minsync/stage.ts";
import { parsedMirrorRoot } from "./paths.ts";

const mirrorSourceSchema = z.object({
	virtualPath: z.string(),
	sourcePath: z.string(),
	parserName: z.string(),
	sourceMtimeNs: z.number(),
	sourceSizeBytes: z.number(),
	parserVersion: z.string().optional(),
	sourceVersion: z.string().optional(),
});

/** A read-only application view of MinSync-owned committed source metadata. */
export interface ParsedMirrorEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly outputPath: string;
	readonly parserName: string;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly updatedAt: string;
	readonly parserVersion?: string;
	readonly sourceVersion?: string;
	readonly parserFingerprint?: string;
}

export type ParsedMirrorSkipReason =
	| "duplicate-excluded"
	| "user-excluded"
	| "parser-skipped"
	| "parser-failed"
	| "source-unreadable";

/** Projected from engine outcomes; never stored or used to schedule work by the host. */
export interface ParsedMirrorSkipEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly reason: ParsedMirrorSkipReason;
	readonly sourceMtimeNs: number;
	readonly sourceSizeBytes: number;
	readonly parserName?: string;
	readonly parserVersion?: string;
	readonly parserFingerprint?: string;
}

export interface ParsedMirrorIndex {
	readonly version: 1;
	readonly entries: Readonly<Record<string, ParsedMirrorEntry>>;
	readonly skipped?: Readonly<Record<string, ParsedMirrorSkipEntry>>;
	/** All engine-held identities, including missing outputs and ordinary failures; never reusable outputs. */
	readonly known?: readonly string[];
}

export function emptyMirrorIndex(): ParsedMirrorIndex {
	return { version: 1, entries: {} };
}

export function mirrorStageKey(virtualPath: string): string {
	return createHash("sha256").update(virtualPath).digest("hex");
}

export function loadMirrorIndex(
	root: string,
	options: { readonly minSyncBinaryPath?: string } = {},
): ParsedMirrorIndex {
	const stateDir = join(parsedMirrorRoot(root), "stage");
	if (!existsSync(stateDir)) return emptyMirrorIndex();
	const binaryPath =
		options.minSyncBinaryPath ??
		lookupMinSyncInPath(process.env) ??
		join(root, ".autorag", "bin", executableName(process.platform));
	if (!existsSync(binaryPath)) throw minSyncBinaryMissingError();
	return mirrorIndexFromStage(
		root,
		readMinSyncStage(binaryPath, stateDir, root, {
			stage: "parse",
			source_id: "autorag-parsed-mirror",
			output_root: join(parsedMirrorRoot(root), "files"),
		}),
	);
}

/** Decode only the host's own opaque source payload, never the engine's disk format. */
export function mirrorSourceMetadata(
	version: string | null,
): Omit<ParsedMirrorEntry, "outputPath" | "updatedAt"> | undefined {
	if (version === null) return undefined;
	try {
		const parsed = mirrorSourceSchema.safeParse(JSON.parse(version));
		return parsed.success ? parsed.data : undefined;
	} catch {
		return undefined;
	}
}

export function mirrorIndexFromStage(root: string, report: MinSyncStageInspectReport): ParsedMirrorIndex {
	const entries: Record<string, ParsedMirrorEntry> = {};
	const skipped: Record<string, ParsedMirrorSkipEntry> = {};
	for (const item of report.items) {
		const source = mirrorSourceMetadata(item.version);
		if (source === undefined || mirrorStageKey(source.virtualPath) !== item.key) continue;
		entries[source.virtualPath] = {
			...source,
			outputPath: join(parsedMirrorRoot(root), "files", item.output.path),
			updatedAt: item.updated_at,
			parserFingerprint: item.fingerprint,
		};
	}
	for (const item of report.skipped) {
		const source = mirrorSourceMetadata(item.version);
		if (source === undefined || mirrorStageKey(source.virtualPath) !== item.key || !isSkipReason(item.reason))
			continue;
		skipped[source.virtualPath] = { ...source, reason: item.reason, parserFingerprint: item.fingerprint };
	}
	for (const item of report.failed) {
		if (!item.deferred || !isSkipReason(item.error)) continue;
		const source = mirrorSourceMetadata(item.version);
		if (source === undefined || mirrorStageKey(source.virtualPath) !== item.key) continue;
		skipped[source.virtualPath] = { ...source, reason: item.error };
	}
	const known = report.known.flatMap((item) => {
		const source = mirrorSourceMetadata(item.version);
		return source !== undefined && mirrorStageKey(source.virtualPath) === item.key ? [source.virtualPath] : [];
	});
	return { version: 1, entries, skipped, known };
}

function isSkipReason(value: string): value is ParsedMirrorSkipReason {
	return (
		value === "duplicate-excluded" ||
		value === "user-excluded" ||
		value === "parser-skipped" ||
		value === "parser-failed" ||
		value === "source-unreadable"
	);
}
