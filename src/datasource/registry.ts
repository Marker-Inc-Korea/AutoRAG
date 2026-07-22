/**
 * Server-bound registry of datasource skills.
 *
 * Skills are constructed from trusted, server-supplied configuration and
 * registered here so the retrieval pipeline can enumerate them, look them up
 * by tag, and resolve the concrete instances a given access context permits.
 *
 * The registry never grants access: {@link resolveInstances} only *narrows*
 * the registered skills to those a {@link DatasourceAccessContext} already
 * allows. Model/tool arguments are not consulted.
 */

import type { RetrievalMethod } from "../retrieval/types.ts";
import type { DatasourceAccessContext } from "./access-context.ts";
import { buildDatasourceInstanceSource } from "./scope.ts";
import type { DatasourceInstance, DatasourceSkill, DatasourceSkillDescriptor, PollingMetadata } from "./types.ts";

/** A registered skill paired with its cached descriptor. */
export interface RegisteredDatasourceSkill {
	readonly skill: DatasourceSkill;
	readonly descriptor: DatasourceSkillDescriptor;
}

/**
 * Registry of {@link DatasourceSkill}s keyed by stable skill id
 * ({@link DatasourceSkillDescriptor.name}).
 */
export class DatasourceSkillRegistry {
	private readonly skills = new Map<string, RegisteredDatasourceSkill>();

	/**
	 * Register a datasource skill. Throws if a skill with the same
	 * {@link DatasourceSkillDescriptor.name} is already registered.
	 */
	register(skill: DatasourceSkill): void {
		const descriptor = skill.describe();
		const id = descriptor.name;
		if (id.length === 0) {
			throw new Error("Cannot register a datasource skill with an empty name");
		}
		if (this.skills.has(id)) {
			throw new Error(`Datasource skill "${id}" is already registered`);
		}
		this.skills.set(id, { skill, descriptor });
	}

	/** All registered skills in insertion order. */
	list(): readonly RegisteredDatasourceSkill[] {
		return Array.from(this.skills.values());
	}

	/** Skills whose descriptor carries the given tag. */
	byTag(tag: string): readonly RegisteredDatasourceSkill[] {
		return this.list().filter((entry) => entry.descriptor.tags.includes(tag));
	}

	/** Look up a registered skill by id, or `undefined`. */
	get(id: string): RegisteredDatasourceSkill | undefined {
		return this.skills.get(id);
	}

	/**
	 * Resolve the concrete datasource instances reachable under `ctx`.
	 *
	 * Only skills whose descriptor {@link DatasourceAccessContext.isAccessible}
	 * allows contribute instances, and only their declared instance ids are
	 * materialized. Each instance carries its opaque slash-hierarchical
	 * {@link DatasourceInstance.sourcePath} (e.g. `/kakao/acct-1`), built from
	 * the trusted skill name and instance id — never from model input.
	 */
	resolveInstances(ctx: DatasourceAccessContext): readonly DatasourceInstance[] {
		const out: DatasourceInstance[] = [];
		const predicate = ctx.allowedSourcesPredicate();
		for (const { skill, descriptor } of this.skills.values()) {
			if (!ctx.isAccessible(descriptor)) continue;
			const instanceIds = descriptor.instances ?? [];
			const polling = safePolling(skill);
			for (const id of instanceIds) {
				const sourcePath = buildDatasourceInstanceSource(descriptor.name, id);
				if (!predicate(sourcePath)) continue;
				out.push({
					id,
					skill,
					descriptor,
					sourcePath,
					polling,
				});
			}
		}
		return out;
	}

	/**
	 * Convenience: all retrieval methods exposed by accessible skills, for
	 * feeding the shared retriever. Methods from denied skills are excluded.
	 */
	accessibleMethods(ctx: DatasourceAccessContext): readonly RetrievalMethod[] {
		const out: RetrievalMethod[] = [];
		for (const { skill, descriptor } of this.skills.values()) {
			if (!ctx.isAccessible(descriptor)) continue;
			out.push(...skill.retrievalMethods());
		}
		return out;
	}
}

/** Defensive polling accessor: never throws on a misbehaving skill. */
function safePolling(skill: DatasourceSkill): PollingMetadata | undefined {
	try {
		return skill.polling();
	} catch {
		return undefined;
	}
}
