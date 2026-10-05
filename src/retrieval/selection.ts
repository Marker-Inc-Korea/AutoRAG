/**
 * Selection gating for the retrieval pipeline.
 *
 * `searchSelected` lets a caller narrow a search to specific datasources,
 * specific methods, or the local (non-datasource) surfaces *before* any backend
 * is invoked. This module owns that narrowing so the engine can reuse its
 * existing parallel-retrieve / datasource-filter / merge chain on the reduced
 * method list instead of fanning out to every backend and filtering afterwards.
 *
 * Security invariants:
 *  - Unauthorized datasource methods are never selected, whether or not the
 *    caller passed a selection — an unselected search still excludes them, so
 *    no denied backend is executed.
 *  - A caller-supplied selection can only *narrow* the trusted method set; it
 *    can never widen it. Unknown or unauthorized method/datasource names are
 *    rejected with a typed error, never silently resolved.
 *  - The trusted {@link DatasourceAccessContext} remains the sole authority:
 *    the `authorizedDatasourceIds` catalog is itself built from that context.
 */

import type { DatasourceAccessContext } from "../datasource/access-context.ts";
import type { RetrievalMethod } from "./types.ts";

/**
 * Model-free, credential-free descriptor of one authorized configured
 * datasource, for catalog/listing surfaces. Carries no credentials, config
 * paths, or raw instance metadata — only the identity, capability tags, and
 * the opaque source scope strings the access context authorizes.
 */
export interface DatasourceCatalogEntry {
	/** Stable datasource id, e.g. `"kakao"`. */
	readonly datasourceId: string;
	/** Skill name, e.g. `"kakao"`. */
	readonly name: string;
	/** Skill kind, e.g. `"chat"` or `"chat-export"`. */
	readonly type: string;
	/** Human-readable description. */
	readonly description: string;
	/** Authorization tags that made this datasource accessible. */
	readonly tags: readonly string[];
	/** Capability names such as `"keyword"`, `"scoped"`, `"polling"`. */
	readonly capabilities: readonly string[];
	readonly status: "active" | "stub";
	/**
	 * Authorized slash-hierarchical source scope strings discovered for this
	 * datasource. Scope-capable datasources expose only sources allowed by the
	 * trusted scope predicate; unscoped datasources expose their opaque sources.
	 */
	readonly sourceScopes: readonly string[];
}

/**
 * Caller request narrowing which retrieval methods run.
 *
 * Semantics (unambiguous, deny-widening):
 *  - `datasourceIds` omitted ⇒ every authorized datasource method is eligible.
 *  - `datasourceIds` present ⇒ ONLY methods of those datasources are eligible,
 *    plus the local (non-datasource) methods only when `local === true`.
 *  - `local === false` ⇒ no local method is ever eligible.
 *  - `methods` present ⇒ intersected with the eligible set; every named method
 *    must exist and (for datasource methods) be authorized.
 */
export interface RetrievalSelection {
	/** Datasource ids to run. Omit to run all authorized datasources. */
	readonly datasourceIds?: readonly string[];
	/** Method names to run, intersected with the datasource/local selection. */
	readonly methods?: readonly string[];
	/** Whether local (non-datasource) methods run. See the semantics above. */
	readonly local?: boolean;
}

export type RetrievalSelectionErrorCode =
	| "unknown-method"
	| "unauthorized-method"
	| "unknown-datasource"
	| "unauthorized-datasource";

/** Thrown when a caller selection names an unknown or unauthorized target. */
export class RetrievalSelectionError extends Error {
	readonly code: RetrievalSelectionErrorCode;

	constructor(code: RetrievalSelectionErrorCode, message: string) {
		super(message);
		this.name = "RetrievalSelectionError";
		this.code = code;
	}
}

/**
 * Reduce the registered methods to the ones a selection authorizes to run.
 *
 * The returned list is built entirely from `methods`; a caller cannot conjure a
 * method that is not registered. Datasource methods denied by `ctx` are dropped
 * from the eligible set unconditionally, so they are never invoked. Explicit
 * selections that name an unknown method or an unauthorized/unknown datasource
 * throw {@link RetrievalSelectionError} rather than degrading silently.
 *
 * @param methods    Every registered retrieval method.
 * @param ctx        Trusted, server-supplied access context.
 * @param selection  Caller narrowing request.
 * @param authorizedDatasourceIds Authorized datasource ids from the catalog;
 *   used to distinguish an unknown id from an unauthorized one. Method-less
 *   authorized datasources belong here and select cleanly (yielding nothing).
 */
export function resolveSelectedMethods(
	methods: readonly RetrievalMethod[],
	ctx: DatasourceAccessContext,
	selection: RetrievalSelection,
	authorizedDatasourceIds: readonly string[],
): RetrievalMethod[] {
	const byName = new Map<string, RetrievalMethod>();
	for (const method of methods) {
		const name = method.describe().name;
		if (!byName.has(name)) byName.set(name, method);
	}
	const authorizedIds = new Set(authorizedDatasourceIds);

	if (selection.methods !== undefined) {
		for (const name of selection.methods) {
			const method = byName.get(name);
			if (method === undefined) {
				throw new RetrievalSelectionError("unknown-method", `Unknown retrieval method "${name}" was selected.`);
			}
			const descriptor = method.describe();
			if (descriptor.datasourceId !== undefined && !ctx.isAccessible(descriptor)) {
				throw new RetrievalSelectionError(
					"unauthorized-method",
					`Retrieval method "${name}" belongs to an unauthorized datasource and cannot be selected.`,
				);
			}
		}
	}

	const explicitDatasources = selection.datasourceIds !== undefined;
	if (selection.datasourceIds !== undefined) {
		for (const id of selection.datasourceIds) {
			if (authorizedIds.has(id)) continue;
			const knownMethod = methods.some((method) => method.describe().datasourceId === id);
			throw new RetrievalSelectionError(
				knownMethod ? "unauthorized-datasource" : "unknown-datasource",
				`Datasource "${id}" is not authorized for this runtime and cannot be selected.`,
			);
		}
	}

	const selectedIds = selection.datasourceIds === undefined ? undefined : new Set(selection.datasourceIds);
	const includeLocal = explicitDatasources ? selection.local === true : selection.local !== false;
	const methodFilter = selection.methods === undefined ? undefined : new Set(selection.methods);

	const out: RetrievalMethod[] = [];
	for (const method of methods) {
		const descriptor = method.describe();
		if (descriptor.datasourceId !== undefined) {
			// Denied datasources are dropped even when nothing was selected.
			if (!ctx.isAccessible(descriptor)) continue;
			if (selectedIds !== undefined && !selectedIds.has(descriptor.datasourceId)) continue;
		} else if (!includeLocal) {
			continue;
		}
		if (methodFilter !== undefined && !methodFilter.has(descriptor.name)) continue;
		out.push(method);
	}
	return out;
}

/**
 * Authorized datasource ids derived directly from a method set and access
 * context. Used as the fallback catalog for a standalone engine that has no
 * datasource-skill catalog wired in; method-less datasources are unknowable
 * here and therefore treated as unknown.
 */
export function derivedAuthorizedDatasourceIds(
	methods: readonly RetrievalMethod[],
	ctx: DatasourceAccessContext,
): string[] {
	const ids = new Set<string>();
	for (const method of methods) {
		const descriptor = method.describe();
		if (descriptor.datasourceId === undefined) continue;
		if (!ctx.isAccessible(descriptor)) continue;
		ids.add(descriptor.datasourceId);
	}
	return [...ids];
}
