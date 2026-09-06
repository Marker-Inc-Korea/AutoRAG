import type { RetrievalResult } from "../retrieval/types.ts";
import type { PolicyResolution } from "./policy.ts";

/** The per-method result map returned by ParallelRetriever. */
export type ResultsByMethod = Map<string, RetrievalResult[]>;

export type PolicyResolver = (source: string, peerFingerprint?: string) => PolicyResolution;

function isAllowedResolution(resolution: PolicyResolution): boolean {
	return (resolution.tier === "always" || resolution.tier === "peers") && resolution.allowed;
}

/**
 * Narrows retrieval results to sources allowed for a peer by the trusted P2P
 * policy. Sources are passed to the resolver exactly as retrieval produced
 * them, so file virtual paths and datasource identifiers share one matcher.
 *
 * The input map and all input result objects remain untouched. Returned result
 * objects are deep copies because retrieval results may be shared with local
 * consumers while this filtered map is sent through a remote-session path.
 */
export function filterRetrievalResultsByPolicy(
	results: ResultsByMethod,
	resolvePolicy: PolicyResolver,
	peerFingerprint: string,
): ResultsByMethod {
	const filtered = new Map<string, RetrievalResult[]>();

	for (const [method, candidates] of results) {
		const allowed = candidates
			.filter((candidate) => isAllowedResolution(resolvePolicy(candidate.source, peerFingerprint)))
			.map((candidate) => structuredClone(candidate));
		filtered.set(method, allowed);
	}

	return filtered;
}
