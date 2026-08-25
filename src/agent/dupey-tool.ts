import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { DupeyScanResult } from "../dupey/index.ts";

export const SCAN_DUPLICATE_DOCUMENTS_TOOL_NAME = "scan_duplicate_documents";

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
			"Scan configured local document roots with dupey and report exact, near, and containment families. Read-only: never moves or deletes files.",
		parameters: scanDuplicateDocumentsSchema,
		async execute(): Promise<AgentToolResult<ScanDuplicateDocumentsDetails>> {
			const details = await provider.scanDuplicateDocuments();
			const errors = details.scans.reduce((count, scan) => count + scan.errors.length, 0);
			return {
				content: [
					{
						type: "text",
						text: [
							`dupey scanned ${details.scans.length} configured root(s).`,
							`families=${details.familyCount} exactDuplicates=${details.exactDuplicateCount} extractionErrors=${errors}`,
							"Review exact families before cleanup; near/contains families are not safe deletion evidence.",
						].join("\n"),
					},
				],
				details,
			};
		},
	};
}
