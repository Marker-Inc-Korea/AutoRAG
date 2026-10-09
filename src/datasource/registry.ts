/**
 * Server-bound registry of datasource skills.
 *
 * Skills are constructed from trusted, server-supplied configuration and
 * registered here so the retrieval pipeline can enumerate them, look them up
 * by tag (descriptive metadata), and resolve the concrete instances they expose.
 *
 * Every configured/connected datasource is searchable without permission setup;
 * there is no access context. The ordinary query `scope` narrows results later
 * in the pipeline, not here. Model/tool arguments are not consulted.
 */

import type { RetrievalMethod } from "../retrieval/types.ts";
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

	/** Skills whose descriptor carries the given tag (descriptive metadata). */
	byTag(tag: string): readonly RegisteredDatasourceSkill[] {
		return this.list().filter((entry) => entry.descriptor.tags.includes(tag));
	}

	/** Look up a registered skill by id, or `undefined`. */
	get(id: string): RegisteredDatasourceSkill | undefined {
		return this.skills.get(id);
	}

	/**
	 * Resolve every configured datasource instance.
	 *
	 * All registered skills contribute, and only their declared instance ids are
	 * materialized. Each instance carries its opaque slash-hierarchical
	 * {@link DatasourceInstance.sourcePath} (e.g. `/kakao/acct-1`), built from
	 * the trusted skill name and instance id — never from model input.
	 */
	resolveInstances(): readonly DatasourceInstance[] {
		const out: DatasourceInstance[] = [];
		for (const { skill, descriptor } of this.skills.values()) {
			const instanceIds = descriptor.instances ?? [];
			const polling = safePolling(skill);
			for (const id of instanceIds) {
				out.push({
					id,
					skill,
					descriptor,
					sourcePath: buildDatasourceInstanceSource(descriptor.name, id),
					polling,
				});
			}
		}
		return out;
	}

	/**
	 * All retrieval methods exposed by registered skills, for feeding the shared
	 * retriever.
	 */
	retrievalMethods(): readonly RetrievalMethod[] {
		const out: RetrievalMethod[] = [];
		for (const { skill } of this.skills.values()) {
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
