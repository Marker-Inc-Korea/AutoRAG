/**
 * Discord guild datasource skill (issue #1305).
 *
 * Thin composition of DiscordConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/discord/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { DiscordConnector, type DiscordConnectorOptions } from "./connector.ts";

export const DISCORD_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "discord",
	skillType: "discord-guild",
	description: "Discord guild datasource",
	capabilities: ["chat", "api", "polling"],
	defaultTags: ["discord", "chat", "pii"],
	contentType: "chat",
	manifestDescription:
		"Search indexed Discord messages across authorized guild channels and threads. Use for questions about Discord conversations or community discussions.",
	manifestNotes: ["Channel visibility and history windows are bounded by the server-configured bot permissions."],
};

export interface DiscordSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: DiscordConnector;
	readonly connectorOptions?: DiscordConnectorOptions;
}

export class DiscordSkill extends ConnectorDatasourceSkill {
	constructor(options: DiscordSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(DISCORD_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new DiscordConnector(connectorOptions ?? {}),
		});
	}
}
