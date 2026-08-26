import type { RetrievalMethod } from "../retrieval/types.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
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

	index(): Promise<DatasourceIndexResult> {
		return this.skill.index();
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return this.skill.retrievalMethods();
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
