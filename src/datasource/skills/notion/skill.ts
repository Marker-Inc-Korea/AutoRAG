/**
 * Notion workspace datasource skill (issue #1302).
 *
 * Thin composition of NotionConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/notion/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { NotionConnector, type NotionConnectorOptions } from "./connector.ts";

export const NOTION_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "notion",
	skillType: "notion-workspace",
	description: "Notion workspace datasource",
	capabilities: ["documents", "api", "polling"],
	defaultTags: ["notion", "documents"],
	contentType: "document",
	manifestDescription:
		"Search indexed Notion pages and databases shared with the integration. Use for questions about Notion docs, wikis, and database entries.",
	manifestNotes: ["Only pages shared with the server-configured integration token are indexed."],
};

export interface NotionSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: NotionConnector;
	readonly connectorOptions?: NotionConnectorOptions;
}

export class NotionSkill extends ConnectorDatasourceSkill {
	constructor(options: NotionSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(NOTION_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new NotionConnector(connectorOptions ?? {}),
		});
	}
}
