/**
 * Skip reporting for the retrieval pipeline.
 *
 * A method that throws is skipped, not fatal: healthy methods still answer. The
 * helpers here turn those skips into a first-class, path-opaque report of the
 * retrieval *surfaces* (local MinSync files, a datasource) that were not
 * searched, with a stable reason a machine consumer can branch on.
 */

import type {
	RetrievalMethodDescriptor,
	RetrievalSkipAction,
	RetrievalSkipReason,
	RetrievalUnsearchedSurface,
} from "./types.ts";

/** Surface label shared by every MinSync-backed local method (vector, BM25, hybrid). */
export const MINSYNC_SURFACE = "minsync";

/** Registered method names that read the same local MinSync workspace. */
const MINSYNC_METHOD_NAMES: ReadonlySet<string> = new Set(["minsync", "hybrid", "bm25"]);

/** One method that did not run, already mapped onto its surface and cause. */
export interface RetrievalSkip {
	readonly method: string;
	readonly surface: string;
	readonly reason: RetrievalSkipReason;
}

const SKIP_ACTIONS: Readonly<Record<RetrievalSkipReason, RetrievalSkipAction>> = {
	"sync-in-progress": "retry",
	"binary-missing": "install-binary",
	"embedder-unavailable": "prepare-embedder",
	"identity-mismatch": "reindex",
	"method-error": "retry",
};

const SKIP_CAUSES: Readonly<Record<RetrievalSkipReason, string>> = {
	"sync-in-progress": "an index sync holds the workspace lock",
	"binary-missing": "the retrieval binary is not installed",
	"embedder-unavailable": "the local embedder is unavailable",
	"identity-mismatch": "the index was built with a different embedding identity or dimension",
	"method-error": "the retrieval method failed",
};

/**
 * Map a method descriptor onto the surface a caller reasons about: datasource
 * methods keep their datasource id, MinSync-backed local methods collapse into
 * one shared surface, and anything else reports under its own method name.
 */
export function retrievalSurfaceFor(descriptor: RetrievalMethodDescriptor): string {
	const datasourceId = descriptor.datasourceId;
	if (typeof datasourceId === "string" && datasourceId !== "") return datasourceId;
	return MINSYNC_METHOD_NAMES.has(descriptor.name) ? MINSYNC_SURFACE : descriptor.name;
}

/**
 * Classify a thrown retrieval error into a stable skip reason. The error text is
 * inspected but never carried into the result: failure messages can hold real
 * paths, and the retrieval contract keeps skip reports path-opaque.
 */
export function classifyRetrievalSkip(error: unknown): RetrievalSkipReason {
	const text = (
		error instanceof Error ? `${error.name}: ${error.message}` : typeof error === "string" ? error : ""
	).toLowerCase();
	if (text.includes("sync is in progress") || text.includes("another sync") || text.includes("lock")) {
		return "sync-in-progress";
	}
	if (text.includes("dimension") || text.includes("identity") || text.includes("reindex")) {
		return "identity-mismatch";
	}
	if (text.includes("embedder") || text.includes("embedding")) return "embedder-unavailable";
	if (
		text.includes("enoent") ||
		text.includes("no such file") ||
		text.includes("not installed") ||
		text.includes("missing binary") ||
		text.includes("binary missing") ||
		text.includes("spawn")
	) {
		return "binary-missing";
	}
	return "method-error";
}

/** The stable recovery hint for a skip reason. */
export function retrievalSkipAction(reason: RetrievalSkipReason): RetrievalSkipAction {
	return SKIP_ACTIONS[reason];
}

/** Path-opaque sentence describing why a surface was not searched. */
export function retrievalSkipMessage(surface: string, reason: RetrievalSkipReason): string {
	const subject = surface === MINSYNC_SURFACE ? "Local MinSync sources were" : `Retrieval surface "${surface}" was`;
	return `${subject} not searched because ${SKIP_CAUSES[reason]}.`;
}

/**
 * Collapse per-method skips into one entry per (surface, reason) pair, so the
 * report reads "these sources were not searched, for this reason" instead of a
 * bare list of method names. Deterministically ordered for stable output.
 */
export function groupUnsearchedSurfaces(skips: readonly RetrievalSkip[]): RetrievalUnsearchedSurface[] {
	const groups = new Map<string, { surface: string; reason: RetrievalSkipReason; methods: Set<string> }>();
	for (const skip of skips) {
		const key = `${skip.surface}\u0000${skip.reason}`;
		const existing = groups.get(key);
		if (existing) existing.methods.add(skip.method);
		else groups.set(key, { surface: skip.surface, reason: skip.reason, methods: new Set([skip.method]) });
	}
	return Array.from(groups.values())
		.map(({ surface, reason, methods }) => ({
			surface,
			methods: Array.from(methods).sort(),
			reason,
			action: retrievalSkipAction(reason),
			message: retrievalSkipMessage(surface, reason),
		}))
		.sort((a, b) => a.surface.localeCompare(b.surface) || a.reason.localeCompare(b.reason));
}
