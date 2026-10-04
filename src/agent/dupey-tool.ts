import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { DupeyScanFamily, DupeyScanResult } from "../dupey/index.ts";

export const SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME = "scan_duplicate_documents";

/** Caps keep a pathological scan from flooding the model's context. */
const MAX_RENDERED_FAMILIES = 200;
const MAX_RENDERED_MEMBERS = 50;

export interface ScanDuplicateDocumentsDetails {
	readonly scans: readonly DupeyScanResult[];
	readonly familyCount: number;
	readonly exactDuplicateCount: number;
}

export interface ScanDuplicateDocumentsProvider {
	scanDuplicateDocuments(): Promise<ScanDuplicateDocumentsDetails>;
}

const scanDuplicateDocumentsSchema = Type.Object({});

export function createScanDuplicateDocumentsTool(
	provider: ScanDuplicateDocumentsProvider,
): AgentTool<typeof scanDuplicateDocumentsSchema, ScanDuplicateDocumentsDetails> {
	return {
		name: SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME,
		label: "Scan Duplicate Documents",
		description:
			"Scan configured local document roots with dupey and report exact, near, and containment families with their files, relation, similarity scores, and the selected latest candidate. Read-only: never moves or deletes files.",
		parameters: scanDuplicateDocumentsSchema,
		async execute(): Promise<AgentToolResult<ScanDuplicateDocumentsDetails>> {
			const details = await provider.scanDuplicateDocuments();
			return {
				content: [{ type: "text", text: renderScanReport(details) }],
				details,
			};
		},
	};
}

/**
 * Model-facing report. `details` is UI/log-only in pi-agent-core, so the family
 * members, relation, scores, and dupey's selected candidate must be rendered into
 * the text the model actually reads.
 */
function renderScanReport(details: ScanDuplicateDocumentsDetails): string {
	const errors = details.scans.reduce((count, scan) => count + scan.errors.length, 0);
	const lines = [
		`dupey scanned ${details.scans.length} configured root(s).`,
		`families=${details.familyCount} exactDuplicates=${details.exactDuplicateCount} extractionErrors=${errors}`,
		"Review exact families before cleanup; near/contains families are not safe deletion evidence.",
	];
	let rendered = 0;
	let omitted = 0;
	for (const scan of details.scans) {
		if (scan.families.length === 0) continue;
		lines.push("", `Root: ${scan.dir}`);
		for (const family of scan.families) {
			if (rendered >= MAX_RENDERED_FAMILIES) {
				omitted += 1;
				continue;
			}
			rendered += 1;
			lines.push(...renderFamily(family));
		}
	}
	if (omitted > 0) {
		lines.push(`… ${omitted} more families omitted; run \`autorag duplicates --json\` for the full list.`);
	}
	return lines.join("\n");
}

function renderFamily(family: DupeyScanFamily): string[] {
	const members: readonly Record<string, unknown>[] =
		family.members.length > 0 ? family.members : family.files.map((path) => ({ path }));
	const lines = [`  [${family.relation}] family ${family.id}: ${members.length} file(s)`];
	for (const member of members.slice(0, MAX_RENDERED_MEMBERS)) {
		const path = readString(member, "path");
		if (path === undefined) continue;
		lines.push(`    - ${path}${formatMemberScores(member)}`);
	}
	if (members.length > MAX_RENDERED_MEMBERS) {
		lines.push(`    … ${members.length - MAX_RENDERED_MEMBERS} more file(s) omitted`);
	}
	lines.push(`    selected: ${selectedCandidate(family) ?? "(none)"}`);
	return lines;
}

/** dupey's rank-1 candidate for the family, with its score and selection reasons. */
function selectedCandidate(family: DupeyScanFamily): string | undefined {
	const ranked: unknown = family.pick?.ranked;
	if (!Array.isArray(ranked)) return undefined;
	const top: unknown = ranked[0];
	if (!isKeyedRecord(top)) return undefined;
	const path = readString(top, "path");
	if (path === undefined) return undefined;
	const score = formatScore(readNumber(top, "score"));
	const reasons = readReasons(top);
	const scorePart = score === undefined ? "" : `score=${score}`;
	const detail = [scorePart, reasons].filter((part) => part.length > 0).join("; ");
	return detail.length === 0 ? path : `${path} (${detail})`;
}

function formatMemberScores(member: Record<string, unknown>): string {
	const parts: string[] = [];
	const near = formatScore(readNumber(member, "near_score"));
	const jaccard = formatScore(readNumber(member, "jaccard"));
	const containment = formatScore(readNumber(member, "containment"));
	if (near !== undefined) parts.push(`near=${near}`);
	if (jaccard !== undefined) parts.push(`jaccard=${jaccard}`);
	if (containment !== undefined) parts.push(`containment=${containment}`);
	return parts.length === 0 ? "" : ` (${parts.join(", ")})`;
}

function readReasons(source: Record<string, unknown>): string {
	const reasons: unknown = source.reasons;
	if (!Array.isArray(reasons)) return "";
	const rendered = reasons.flatMap((reason) => {
		if (!isKeyedRecord(reason)) return [];
		const name = readString(reason, "name");
		if (name === undefined) return [];
		const detail = readString(reason, "detail");
		return [detail === undefined ? name : `${name} — ${detail}`];
	});
	return rendered.length === 0 ? "" : `reason: ${rendered.join(", ")}`;
}

function isKeyedRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(source: Record<string, unknown>, key: string): string | undefined {
	const value = source[key];
	return typeof value === "string" ? value : undefined;
}

function readNumber(source: Record<string, unknown>, key: string): number | undefined {
	const value = source[key];
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function formatScore(value: number | undefined): string | undefined {
	return value === undefined ? undefined : value.toFixed(2);
}
