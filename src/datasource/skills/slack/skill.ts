/**
 * Slack workspace datasource skill (issue #1300).
 *
 * Thin composition of SlackConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/slack/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { SlackConnector, type SlackConnectorOptions } from "./connector.ts";

export const SLACK_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "slack",
	skillType: "slack-workspace",
	description: "Slack workspace datasource",
	capabilities: ["chat", "api", "polling"],
	defaultTags: ["slack", "chat", "pii"],
	contentType: "chat",
	manifestDescription:
		"Search indexed Slack messages across authorized workspace channels. Use for questions about Slack conversations, decisions made in channels, or who said what.",
	manifestNotes: [
		"Message history windows and channel visibility are bounded by the server-configured bot token scopes.",
	],
};

export interface SlackSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: SlackConnector;
	readonly connectorOptions?: SlackConnectorOptions;
}

export class SlackSkill extends ConnectorDatasourceSkill {
	constructor(options: SlackSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(SLACK_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new SlackConnector(connectorOptions ?? {}),
		});
	}
}
