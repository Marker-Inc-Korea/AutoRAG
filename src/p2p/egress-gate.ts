import type { SearchDocumentResult, SearchDocumentsResponse } from "../agent/search-documents.ts";
import { scanOutboundPayload } from "./injection-classifier.ts";
import { redactPII } from "./pii-gate.ts";
import type { PolicyResolution } from "./policy.ts";
import type { PeerQueryResponse } from "./wire.ts";
import { wireSourceId } from "./wire.ts";

export type EgressPolicyResolver = (source: string, peerFingerprint?: string) => PolicyResolution;

export interface BuildPeerResponseOptions {
	readonly response: SearchDocumentsResponse;
	/** Sources returned by the server's policy-filtered retrieval pipeline. */
	readonly observedSources: ReadonlySet<string>;
	readonly resolvePolicy: EgressPolicyResolver;
	readonly peerFingerprint: string;
	readonly workspaceRoots: readonly string[];
	readonly pseudonymize: boolean;
}

interface EgressDiagnostic {
	readonly code: string;
	readonly message: string;
}

const WIRE_SOURCE_PATTERN = /^\/[a-z0-9-]+(?:\/|$)/u;

function diagnostic(code: string, message: string): EgressDiagnostic {
	return { code, message };
}

function isAllowedFileResolution(resolution: PolicyResolution | undefined): boolean {
	return (
		resolution !== undefined &&
		(resolution.tier === "always" || resolution.tier === "peers") &&
		resolution.allowed === true
	);
}

function isSearchDocumentResult(value: unknown): value is SearchDocumentResult {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const candidate = value as Record<string, unknown>;
	return (
		typeof candidate.number === "number" &&
		Number.isSafeInteger(candidate.number) &&
		candidate.number >= 1 &&
		typeof candidate.title === "string" &&
		candidate.title.length > 0 &&
		typeof candidate.summary === "string" &&
		Array.isArray(candidate.evidence)
	);
}

function excerptFor(result: SearchDocumentResult): string {
	const firstEvidence = result.evidence[0];
	return typeof firstEvidence?.excerpt === "string" ? firstEvidence.excerpt : "";
}

function rejectedResponse(diagnostics: readonly EgressDiagnostic[]): PeerQueryResponse {
	return {
		v: 1,
		status: "rejected",
		answer: "",
		results: [],
		files: [],
		diagnostics: [...diagnostics],
	};
}

/**
 * Build the wire response for one remote search.
 *
 * A result source is admissible only when it is an exact member of the
 * server-owned observed-source registry. The model response is therefore
 * never used to expand the allowlist; it can only name one of the sources the
 * server actually observed during this run.
 */
export function buildPeerResponse(options: BuildPeerResponseOptions): PeerQueryResponse {
	const pseudonymMap = new Map<string, string>();
	const diagnostics: EgressDiagnostic[] = [];
	const rawAnswer = typeof options.response?.answer === "string" ? options.response.answer : "";
	const answer = redactPII(rawAnswer, {
		pseudonymize: options.pseudonymize,
		map: pseudonymMap,
	}).text;
	const survivingResults: PeerQueryResponse["results"] = [];
	const outboundTexts = [answer];

	const rawResults = Array.isArray(options.response?.results) ? options.response.results : [];
	for (const rawResult of rawResults) {
		if (!isSearchDocumentResult(rawResult) || typeof rawResult.source !== "string") {
			diagnostics.push(
				diagnostic("source-unmappable", "A result without a valid server-resolvable source was dropped."),
			);
			continue;
		}

		// Do not normalize or derive the candidate before this exact membership
		// check. In particular, model-provided path variants must not become
		// observed sources through canonicalization.
		if (!options.observedSources.has(rawResult.source)) {
			diagnostics.push(
				diagnostic("source-unmappable", "A result source was not returned by the server retrieval registry."),
			);
			continue;
		}

		let resolution: PolicyResolution | undefined;
		try {
			resolution = options.resolvePolicy(rawResult.source, options.peerFingerprint);
		} catch {
			resolution = undefined;
		}
		if (!isAllowedFileResolution(resolution)) {
			diagnostics.push(diagnostic("policy-denied", "A result source is not allowed for this peer."));
			continue;
		}

		let source: string;
		try {
			source = wireSourceId(rawResult.source);
		} catch {
			diagnostics.push(diagnostic("source-unmappable", "A result source could not be mapped to a wire id."));
			continue;
		}
		if (!WIRE_SOURCE_PATTERN.test(source)) {
			diagnostics.push(diagnostic("source-unmappable", "A result source produced an invalid wire id."));
			continue;
		}

		const summary = redactPII(rawResult.summary, {
			pseudonymize: options.pseudonymize,
			map: pseudonymMap,
		}).text;
		const excerpt = redactPII(excerptFor(rawResult), {
			pseudonymize: options.pseudonymize,
			map: pseudonymMap,
		}).text;
		outboundTexts.push(summary, excerpt);
		survivingResults.push({
			number: rawResult.number,
			title: rawResult.title,
			summary,
			source,
			excerpt,
		});
	}

	const scan = scanOutboundPayload(outboundTexts, [...options.workspaceRoots]);
	if (!scan.ok) {
		return rejectedResponse([
			...diagnostics,
			diagnostic(scan.code, "The peer response was rejected by the outbound payload gate."),
		]);
	}

	return {
		v: 1,
		status: "ok",
		answer,
		results: survivingResults,
		files: [],
		diagnostics,
	};
}
