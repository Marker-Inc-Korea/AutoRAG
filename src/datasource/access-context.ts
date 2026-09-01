/**
 * Trusted, server-bound access context for the datasource layer.
 *
 * Constructed from server-supplied allow-tags and allow-scopes ONLY — never
 * from model or tool arguments. Tags authorize datasource skills; scopes are
 * applied only to skills that advertise the `scoped` capability.
 */

import { matchesVirtualPathScope, normalizeVirtualPath } from "../retrieval/scope.ts";
import type { DatasourceAccessible } from "./types.ts";

export interface DatasourceAccessContextOptions {
	/** Trusted allow-tags. Undefined or empty ⇒ default-deny (isDenyAll). */
	readonly allowedTags?: readonly string[];
	/** Trusted allow-scopes as slash-hierarchical paths or globs. */
	readonly allowedScopes?: readonly string[];
}

/**
 * Access context for gating datasource descriptors and sources.
 *
 * Non-datasource descriptors (those without a `datasourceId`) are NOT gated
 * by this context: {@link isAccessible} returns `true` for them so plain
 * retrieval methods (e.g. `posix`) pass through unchanged.
 */
export class DatasourceAccessContext {
	readonly allowedTags: readonly string[];
	readonly allowedScopes: readonly string[];
	private readonly denyAll: boolean;

	constructor(options: DatasourceAccessContextOptions = {}) {
		const tags = options.allowedTags ?? [];
		const scopes = options.allowedScopes ?? [];
		this.allowedTags = tags;
		this.allowedScopes = scopes;
		// Default-deny when no trusted allow-tags are configured.
		this.denyAll = tags.length === 0;
	}

	/** True when the context denies everything (no trusted allow-tags). */
	get isDenyAll(): boolean {
		return this.denyAll;
	}

	/**
	 * Whether a descriptor is accessible.
	 *
	 * Non-datasource descriptors (no `datasourceId`) pass through (`true`) —
	 * they are not gated by this context. Datasource descriptors require at
	 * least one tag intersecting the trusted allow-tags.
	 */
	isAccessible(descriptor: DatasourceAccessible): boolean {
		// Non-datasource descriptors are not gated: pass through explicitly.
		if (descriptor.datasourceId === undefined) return true;
		if (this.denyAll) return false;
		const tags = descriptor.tags ?? [];
		if (tags.length === 0) return false;
		return tags.some((tag) => this.allowedTags.includes(tag));
	}

	/**
	 * Scope-capable datasource results are filtered by trusted and user scopes.
	 * Datasources without the `scoped` capability bypass this predicate and
	 * are authorized only at the skill/tag level.
	 */
	allowedSourcesPredicate(_userScope?: string): (source: string) => boolean {
		if (this.denyAll) return () => false;
		const trustedScopes = this.allowedScopes;
		const normalizedUserScope = _userScope === undefined ? undefined : normalizeVirtualPath(_userScope);
		return (source: string): boolean => {
			if (source.includes("#")) return false;
			const inTrusted =
				trustedScopes.length === 0 ? true : trustedScopes.some((scope) => matchesVirtualPathScope(source, scope));
			if (!inTrusted) return false;
			return normalizedUserScope === undefined ? true : matchesVirtualPathScope(source, normalizedUserScope);
		};
	}
}
