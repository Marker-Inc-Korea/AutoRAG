import type { RetrievalMethod } from "../retrieval/types.ts";
import type {
	DatasourceGrepHit,
	DatasourceGrepOptions,
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SearchFilterSpec,
	SourceDescription,
} from "./types.ts";

/**
 * Adds operator-authored context to a configured datasource without changing
 * its backend, access identity, or retrieval methods.
 */
export class DescribedDatasourceSkill implements DatasourceSkill {
	private readonly skill: DatasourceSkill;
	private readonly description: string;

	constructor(skill: DatasourceSkill, description: string) {
		this.skill = skill;
		this.description = description;
	}

	describe(): DatasourceSkillDescriptor {
		return { ...this.skill.describe(), description: this.description };
	}

	polling(): PollingMetadata {
		return this.skill.polling();
	}

	index(options?: DatasourceIndexOptions): Promise<DatasourceIndexResult> {
		return this.skill.index(options);
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return this.skill.retrievalMethods();
	}

	searchFilters(): Promise<readonly SearchFilterSpec[]> {
		return this.skill.searchFilters?.() ?? Promise.resolve([]);
	}

	grep(pattern: string, options?: DatasourceGrepOptions): Promise<readonly DatasourceGrepHit[]> {
		return this.skill.grep?.(pattern, options) ?? Promise.resolve([]);
	}

	describeSources(): readonly SourceDescription[] {
		return this.skill.describeSources();
	}

	skillManifest(): DatasourceSkillManifest {
		const manifest = this.skill.skillManifest();
		return {
			...manifest,
			description: this.description,
			content: `${manifest.content}\n\n## Operator description\n\n${this.description}`,
		};
	}
}
