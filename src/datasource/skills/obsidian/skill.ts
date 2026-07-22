/**
 * Obsidian vault datasource skill (issue #1314).
 *
 * Thin composition of ObsidianConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/obsidian/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { ObsidianConnector, type ObsidianConnectorOptions } from "./connector.ts";

export const OBSIDIAN_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "obsidian",
	skillType: "obsidian-vault",
	description: "Obsidian vault datasource",
	capabilities: ["notes", "filesystem"],
	defaultTags: ["obsidian", "notes"],
	contentType: "note",
	manifestDescription:
		"Search indexed Obsidian vault notes, including tags, wiki links, and embeds. Use for questions about personal notes and knowledge-base content.",
	manifestNotes: ["Only the server-configured vault is indexed; note metadata includes tags, links, and embeds."],
};

export interface ObsidianSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: ObsidianConnector;
	readonly connectorOptions?: ObsidianConnectorOptions;
}

export class ObsidianSkill extends ConnectorDatasourceSkill {
	constructor(options: ObsidianSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(OBSIDIAN_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new ObsidianConnector(connectorOptions ?? {}),
		});
	}
}
