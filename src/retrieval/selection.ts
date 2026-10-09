/**
 * Selection gating for the retrieval pipeline.
 *
 * `searchSelected` lets a caller narrow a search to specific datasources,
 * specific methods, or the local (non-datasource) surfaces *before* any backend
 * is invoked. This module owns that narrowing so the engine can reuse its
 * existing parallel-retrieve / scope-filter / merge chain on the reduced
 * method list instead of fanning out to every backend and filtering afterwards.
 *
 * Every configured datasource is searchable without permission setup. A
 * caller-supplied selection can only *narrow* the registered method set: it can
 * never add a method that is not registered. Unknown method or datasource names
 * are rejected with a typed error, never silently resolved.
 */

import type { RetrievalMethod } from "./types.ts";

/**
 * Model-free, credential-free descriptor of one configured datasource, for
 * catalog/listing surfaces. Carries no credentials, config paths, or raw
 * instance metadata — only the identity, descriptive tags, and the opaque
 * source scope strings the datasource exposes.
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
	/** Descriptive tags (metadata only). */
	readonly tags: readonly string[];
	/** Capability names such as `"keyword"`, `"scoped"`, `"polling"`. */
	readonly capabilities: readonly string[];
	readonly status: "active" | "stub";
	/**
	 * Slash-hierarchical source scope strings discovered for this datasource.
	 * Scope-capable datasources expose their instance source scopes; unscoped
	 * datasources expose their opaque sources.
	 */
	readonly sourceScopes: readonly string[];
}

/**
 * Caller request narrowing which retrieval methods run.
 *
 * Semantics (unambiguous, narrowing-only):
 *  - `datasourceIds` omitted ⇒ every datasource method is eligible.
 *  - `datasourceIds` present ⇒ ONLY methods of those datasources are eligible,
 *    plus the local (non-datasource) methods only when `local === true`.
 *  - `local === false` ⇒ no local method is ever eligible.
 *  - `methods` present ⇒ intersected with the eligible set; every named method
 *    must exist.
 */
export interface RetrievalSelection {
	/** Datasource ids to run. Omit to run all configured datasources. */
	readonly datasourceIds?: readonly string[];
	/** Method names to run, intersected with the datasource/local selection. */
	readonly methods?: readonly string[];
	/** Whether local (non-datasource) methods run. See the semantics above. */
	readonly local?: boolean;
}

export type RetrievalSelectionErrorCode = "unknown-method" | "unknown-datasource";

/** Thrown when a caller selection names an unknown method or datasource. */
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
 * method that is not registered. Explicit selections that name an unknown
 * method or an unknown datasource throw {@link RetrievalSelectionError} rather
 * than degrading silently.
 *
 * @param methods       Every registered retrieval method.
 * @param selection     Caller narrowing request.
 * @param datasourceIds Configured datasource ids (from the catalog), used to
 *   distinguish a known datasource from an unknown one. Method-less
 *   datasources belong here and select cleanly (yielding nothing).
 */
export function resolveSelectedMethods(
	methods: readonly RetrievalMethod[],
	selection: RetrievalSelection,
	datasourceIds: readonly string[],
): RetrievalMethod[] {
	const byName = new Map<string, RetrievalMethod>();
	for (const method of methods) {
		const name = method.describe().name;
		if (!byName.has(name)) byName.set(name, method);
	}
	const knownIds = new Set(datasourceIds);

	if (selection.methods !== undefined) {
		for (const name of selection.methods) {
			if (!byName.has(name)) {
				throw new RetrievalSelectionError("unknown-method", `Unknown retrieval method "${name}" was selected.`);
			}
		}
	}

	const explicitDatasources = selection.datasourceIds !== undefined;
	if (selection.datasourceIds !== undefined) {
		for (const id of selection.datasourceIds) {
			if (knownIds.has(id)) continue;
			throw new RetrievalSelectionError(
				"unknown-datasource",
				`Datasource "${id}" is not configured and cannot be selected.`,
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
 * Configured datasource ids derived directly from a method set. Used as the
 * fallback catalog for a standalone engine that has no datasource-skill catalog
 * wired in; method-less datasources are unknowable here and therefore treated
 * as unknown.
 */
export function derivedDatasourceIds(methods: readonly RetrievalMethod[]): string[] {
	const ids = new Set<string>();
	for (const method of methods) {
		const datasourceId = method.describe().datasourceId;
		if (datasourceId === undefined) continue;
		ids.add(datasourceId);
	}
	return [...ids];
}
