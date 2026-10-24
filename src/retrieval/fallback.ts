import { MinSyncRequiredError } from "../minsync/errors.ts";
import { describeRetrievalError } from "./skip.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "./types.ts";

/**
 * One logical retrieval method that prefers its primary and degrades to a
 * fallback only when the primary cannot run.
 *
 * Datasources used to register a hybrid method next to its lexical and semantic
 * components and let the merger reconcile them. That ran the same backend
 * several times and let the merger count a hybrid and its own component as
 * independent corroboration. Registering one chain instead means the pipeline
 * sees a single method per datasource: the hybrid when it works, a component
 * only when it does not.
 *
 * The chain reports the primary's descriptor, so selection, diagnostics and
 * surface grouping behave as if the primary were registered alone. A primary
 * that answers with no hits is an answer, not a failure: the fallback is not
 * consulted, because a lexical-only retry cannot find what a hybrid missed.
 */
export class FallbackRetrievalMethod implements RetrievalMethod {
	private readonly primary: RetrievalMethod;
	private readonly fallbacks: readonly RetrievalMethod[];

	constructor(primary: RetrievalMethod, fallbacks: readonly RetrievalMethod[]) {
		this.primary = primary;
		this.fallbacks = fallbacks;
	}

	describe(): RetrievalMethodDescriptor {
		return this.primary.describe();
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		// The hook belongs to this chain's caller; the wrapped methods must not report on its behalf.
		const { onFallback, ...inner } = options;
		try {
			return await this.primary.retrieve(query, inner);
		} catch (primaryError) {
			// An abort is the caller giving up, and a missing required MinSync must end the
			// operation; neither is "primary unavailable".
			if (options.signal?.aborted || primaryError instanceof MinSyncRequiredError) throw primaryError;
			const fallbackFrom = this.primary.describe().name;
			const fallbackReason = describeRetrievalError(primaryError);
			for (const fallback of this.fallbacks) {
				try {
					const results = await fallback.retrieve(query, inner);
					onFallback?.({ method: fallbackFrom, fallback: fallback.describe().name, reason: fallbackReason });
					return results.map((result) => ({
						...result,
						metadata: { ...result.metadata, fallbackFrom, fallbackReason },
					}));
				} catch (fallbackError) {
					if (options.signal?.aborted || fallbackError instanceof MinSyncRequiredError) throw fallbackError;
				}
			}
			// Nothing answered: the primary's own error is the actionable one.
			throw primaryError;
		}
	}
}
