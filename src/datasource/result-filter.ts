/**
 * Result filter that applies datasource access gating to the shared retrieval
 * result stream.
 *
 * The filter sits *after* {@link ParallelRetriever} and *before* merging: it
 * takes the per-method result map and the {@link RetrievalMethod}s that
 * produced it, then for every datasource-backed method either drops all of
 * its results (when the trusted access context denies the descriptor) or
 * keeps only the sources the context allows for the current user scope.
 *
 * Non-datasource methods (those whose descriptor has no `datasourceId`) pass
 * through completely untouched — they are gated by their own retrieval
 * pipeline, not by the datasource access context.
 *
 * Security invariants:
 *  - Deny is always an explicit `false` (an empty result array), never
 *    `undefined`-as-deny. {@link DatasourceAccessContext} itself returns
 *    explicit booleans from its predicate; this filter never passes
 *    `undefined` to `matchesVirtualPathScope` as a deny signal.
 *  - The filter never grants access. It only narrows results using the
 *    trusted, server-supplied {@link DatasourceAccessContext}; model/tool
 *    arguments (such as `userScope`) can only *further restrict* the trusted
 *    allow-scopes, never widen them.
 */

import type { RetrievalMethod, RetrievalResult } from "../retrieval/types.ts";
import type { DatasourceAccessContext } from "./access-context.ts";

/** Per-method retrieval results keyed by method name. */
export type ResultsByMethod = Map<string, RetrievalResult[]>;

/**
 * Applies datasource access gating to a per-method result map.
 *
 * The instance form lets callers wire a {@link DatasourceSkillRegistry} for
 * future descriptor enrichment; the v1 filter only needs the
 * {@link RetrievalMethodDescriptor} carried by each method, which already
 * satisfies {@link DatasourceAccessible} structurally (`datasourceId?` +
 * `tags?`).
 */
export class DatasourceResultFilter {
	/**
	 * Filter `byMethod` according to datasource access rules.
	 *
	 * For each method present in both `byMethod` and `methods`:
	 *  - non-datasource methods (`descriptor.datasourceId` unset) are passed
	 *    through with their result array unchanged;
	 *  - datasource methods whose descriptor is denied by `ctx` are reduced to
	 *    an empty result array (`[]` — explicit deny, never `undefined`);
	 *  - datasource methods whose descriptor is allowed are filtered so only
	 *    sources matching `ctx.allowedSourcesPredicate(userScope)` survive.
	 *
	 * Result entries for methods not present in `methods` are passed through
	 * unchanged (the filter cannot classify them and must not fabricate a
	 * descriptor). A new {@link ResultsByMethod} is returned; the input map is
	 * not mutated.
	 *
	 * @param byMethod   Per-method results from the retriever.
	 * @param methods    The {@link RetrievalMethod}s that produced them.
	 * @param ctx        Trusted, server-supplied access context.
	 * @param userScope  Optional user/model-requested scope; when provided it
	 *   is intersected with the trusted allow-scopes and can only narrow
	 *   results. `undefined` means "no extra user restriction".
	 */
	filter(
		byMethod: ResultsByMethod,
		methods: readonly RetrievalMethod[],
		ctx: DatasourceAccessContext,
		userScope?: string,
	): ResultsByMethod {
		const descriptors = new Map<string, RetrievalMethod>();
		for (const method of methods) {
			descriptors.set(method.describe().name, method);
		}

		// One predicate per filter call; userScope is forwarded as-is. The
		// access context treats `undefined` userScope as "no extra restriction"
		// and never passes it to `matchesVirtualPathScope` as a deny value.
		const predicate = ctx.allowedSourcesPredicate(userScope);

		const out: ResultsByMethod = new Map();
		for (const [name, results] of byMethod) {
			const method = descriptors.get(name);
			if (method === undefined) {
				// No descriptor available: cannot classify as datasource.
				out.set(name, results);
				continue;
			}
			const descriptor = method.describe();
			if (descriptor.datasourceId === undefined) {
				// Non-datasource method: pass through untouched.
				out.set(name, results);
				continue;
			}
			if (!ctx.isAccessible(descriptor)) {
				// Explicit deny: empty array, never undefined.
				out.set(name, []);
				continue;
			}
			// Allowed descriptor: keep only sources the context permits.
			out.set(
				name,
				results.filter((r) => predicate(r.source)),
			);
		}
		return out;
	}
}
