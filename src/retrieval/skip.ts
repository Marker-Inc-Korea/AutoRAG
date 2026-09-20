/**
 * Skip reporting for the retrieval pipeline.
 *
 * A method that throws is skipped, not fatal: healthy methods still answer. The
 * helpers here turn those skips into a first-class report of the retrieval
 * *surfaces* (local MinSync files, a datasource) that were not searched.
 *
 * The reason is the underlying error, verbatim. AutoRAG does not classify,
 * rewrite, or suppress it: the operator searching their own machine is the one
 * who has to debug the failure, so they get the CLI's own words — exit codes,
 * stderr, paths and all.
 */

import type { RetrievalMethodDescriptor, RetrievalUnsearchedSurface } from "./types.ts";

/** Surface label shared by every MinSync-backed local method (vector, BM25, hybrid). */
export const MINSYNC_SURFACE = "minsync";

/** Registered method names that read the same local MinSync workspace. */
const MINSYNC_METHOD_NAMES: ReadonlySet<string> = new Set(["minsync", "hybrid", "bm25"]);

/** One method that did not run, with the error that stopped it. */
export interface RetrievalSkip {
	readonly method: string;
	readonly surface: string;
	/** The underlying failure, verbatim. */
	readonly reason: string;
}

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
 * Render a thrown value as the text a human needs to debug it. Nothing is
 * stripped: whatever the method (or the CLI behind it) said comes through,
 * including paths and exit status. The JS stack is left out because it points at
 * AutoRAG's own frames, not at the failure the operator has to fix.
 */
export function describeRetrievalError(error: unknown): string {
	if (error instanceof Error) {
		return error.message.length > 0 ? `${error.name}: ${error.message}` : error.name;
	}
	if (typeof error === "string") return error;
	try {
		return JSON.stringify(error) ?? String(error);
	} catch {
		return String(error);
	}
}

/**
 * Collapse per-method skips into one entry per surface. Distinct failures on
 * the same surface are all preserved verbatim in the reason; methods are
 * merged. Deterministically ordered.
 */
export function groupUnsearchedSurfaces(skips: readonly RetrievalSkip[]): RetrievalUnsearchedSurface[] {
	const groups = new Map<string, { readonly reasons: Set<string>; readonly methods: Set<string> }>();
	for (const skip of skips) {
		const existing = groups.get(skip.surface);
		if (existing) {
			existing.methods.add(skip.method);
			existing.reasons.add(skip.reason);
		} else {
			groups.set(skip.surface, { reasons: new Set([skip.reason]), methods: new Set([skip.method]) });
		}
	}
	return Array.from(groups.entries())
		.map(([surface, { reasons, methods }]) => ({
			surface,
			methods: Array.from(methods).sort(),
			reason: Array.from(reasons).sort().join("; "),
		}))
		.sort((a, b) => a.surface.localeCompare(b.surface) || a.reason.localeCompare(b.reason));
}
