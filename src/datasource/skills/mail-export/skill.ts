/**
 * Local mail export datasource skill (issue #1311).
 *
 * Thin composition of MailExportConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/mail-export/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { MailExportConnector, type MailExportConnectorOptions } from "./connector.ts";

export const MAIL_EXPORT_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "mail-export",
	skillType: "mail-export",
	description: "Local mail export datasource",
	capabilities: ["email", "filesystem"],
	defaultTags: ["mail-export", "email", "pii"],
	contentType: "email",
	manifestDescription:
		"Search indexed local mail exports (mbox/eml archives). Use for questions about archived email conversations.",
	manifestNotes: [
		"Only the server-configured export paths are indexed; archives are read locally and never uploaded.",
	],
};

export interface MailExportSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: MailExportConnector;
	readonly connectorOptions?: MailExportConnectorOptions;
}

export class MailExportSkill extends ConnectorDatasourceSkill {
	constructor(options: MailExportSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(MAIL_EXPORT_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new MailExportConnector(connectorOptions ?? {}),
		});
	}
}
