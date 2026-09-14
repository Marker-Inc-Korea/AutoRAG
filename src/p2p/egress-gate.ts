import type { SearchDocumentResult, SearchDocumentsResponse } from "../agent/search-documents.ts";
import { scanOutboundPayload } from "./injection-classifier.ts";
import { redactPII } from "./pii-gate.ts";
import type { PolicyResolution } from "./policy.ts";
import type { PeerQueryResponse } from "./wire.ts";

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

function maskDeniedEvidence(
	answer: string,
	dropped: readonly { readonly title: string; readonly summary: string; readonly excerpt: string }[],
): string {
	let masked = answer;
	for (const item of dropped) {
		for (const piece of [item.summary, item.excerpt, item.title]) {
			if (piece.length >= 8 && masked.includes(piece)) {
				masked = masked.split(piece).join("[redacted]");
			}
		}
	}
	return masked;
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
	if (options.observedSources.size === 0) {
		return rejectedResponse([diagnostic("policy-denied", "No policy-allowed retrieval sources were observed.")]);
	}
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

		if (!rawResult.source.startsWith("/")) {
			diagnostics.push(diagnostic("source-unmappable", "A result source is not a canonical virtual source."));
			continue;
		}
		const source = rawResult.source;

		const title = redactPII(rawResult.title, {
			pseudonymize: options.pseudonymize,
			map: pseudonymMap,
		}).text;
		const summary = redactPII(rawResult.summary, {
			pseudonymize: options.pseudonymize,
			map: pseudonymMap,
		}).text;
		const excerpt = redactPII(excerptFor(rawResult), {
			pseudonymize: options.pseudonymize,
			map: pseudonymMap,
		}).text;
		outboundTexts.push(title, summary, excerpt);
		survivingResults.push({
			number: rawResult.number,
			title,
			summary,
			source,
			excerpt,
		});
	}

	const droppedEvidence = rawResults.flatMap((rawResult) => {
		if (!isSearchDocumentResult(rawResult) || typeof rawResult.source !== "string") return [];
		if (survivingResults.some((result) => result.number === rawResult.number)) return [];
		return [{ title: rawResult.title, summary: rawResult.summary, excerpt: excerptFor(rawResult) }];
	});
	const maskedAnswer = maskDeniedEvidence(answer, droppedEvidence);
	outboundTexts[0] = maskedAnswer;

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
		answer: maskedAnswer,
		results: survivingResults,
		files: [],
		diagnostics,
	};
}
