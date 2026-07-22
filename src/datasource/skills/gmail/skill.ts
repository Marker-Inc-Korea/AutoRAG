/**
 * Gmail datasource skill (issue #1304).
 *
 * Thin composition of GmailConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/gmail/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import type { DatasourceConnector } from "../../connector.ts";
import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { GmailConnector, type GmailConnectorOptions } from "./connector.ts";

export const GMAIL_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "gmail",
	skillType: "gmail-account",
	description: "Gmail datasource",
	capabilities: ["email", "api", "polling"],
	defaultTags: ["gmail", "email", "pii"],
	contentType: "email",
	manifestDescription:
		"Search indexed Gmail messages from the authorized account and labels. Use for questions about email conversations, decisions, or attachments discussed over email.",
	manifestNotes: ["Label and folder visibility is bounded by the server-configured OAuth token."],
};

export interface GmailSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/**
	 * Trusted pre-built connector; wins over {@link connectorOptions}. Accepts
	 * any mail-backed connector (Gmail REST or the himalaya CLI bridge).
	 */
	readonly connector?: DatasourceConnector;
	readonly connectorOptions?: GmailConnectorOptions;
}

export class GmailSkill extends ConnectorDatasourceSkill {
	constructor(options: GmailSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(GMAIL_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new GmailConnector(connectorOptions ?? {}),
		});
	}
}
