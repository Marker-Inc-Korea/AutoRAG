import { createHash } from "node:crypto";

export interface EvidenceIdInput {
	readonly method: string;
	readonly source: string;
	readonly retrievalResultId?: string;
	readonly chunkIndex?: number;
	readonly lineNumber?: number;
	readonly excerpt?: string;
	readonly content?: string;
}

export interface NormalizedEvidenceRef extends EvidenceIdInput {
	readonly stableEvidenceId: string;
}

export function normalizeEvidenceText(text: string): string {
	return text.replace(/\r\n?/gu, "\n").trim().normalize("NFC");
}

export function isPathOpaqueIdentifier(value: string): boolean {
	if (value.length === 0) return false;
	if (value.includes("/") || value.includes("\\")) return false;
	if (/^[A-Za-z]:/u.test(value)) return false;
	return true;
}

export function stableEvidenceId(input: EvidenceIdInput): string {
	const method = input.method.trim() || "unknown";
	if (input.retrievalResultId && isPathOpaqueIdentifier(input.retrievalResultId)) {
		return input.retrievalResultId.startsWith(`${method}:`)
			? input.retrievalResultId
			: `${method}:${input.retrievalResultId}`;
	}

	const evidenceText = input.excerpt ?? input.content;
	if (evidenceText === undefined) {
		throw new Error("Evidence ID requires excerpt or content when retrievalResultId is absent or path-like");
	}
	const payload = [
		method,
		input.source,
		input.chunkIndex ?? "",
		input.lineNumber ?? "",
		normalizeEvidenceText(evidenceText),
	].join("\0");
	const digest = createHash("sha256").update(payload).digest("hex").slice(0, 24);
	return `${method}:${digest}`;
}

export function normalizeEvidenceRef(input: EvidenceIdInput): NormalizedEvidenceRef {
	return {
		...input,
		stableEvidenceId: stableEvidenceId(input),
	};
}
