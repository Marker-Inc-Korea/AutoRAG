/**
 * Google Drive datasource skill (issue #1301).
 *
 * Thin composition of GDriveConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/gdrive/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import type { DatasourceConnector } from "../../connector.ts";
import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { GDriveConnector, type GDriveConnectorOptions } from "./connector.ts";

export const GDRIVE_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "gdrive",
	skillType: "google-drive",
	description: "Google Drive datasource",
	capabilities: ["documents", "api", "polling"],
	defaultTags: ["gdrive", "documents", "pii"],
	contentType: "document",
	manifestDescription:
		"Search indexed Google Drive documents (Docs, Sheets, plain text). Use for questions about files stored in the authorized Drive account or folder.",
	manifestNotes: [
		"Shared-drive permissions are enforced by the server-configured OAuth token; unexported binary formats are not indexed.",
	],
};

export interface GDriveSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/**
	 * Trusted pre-built connector; wins over {@link connectorOptions}. Accepts
	 * any drive-backed connector (Drive REST or the rclone CLI bridge).
	 */
	readonly connector?: DatasourceConnector;
	readonly connectorOptions?: GDriveConnectorOptions;
}

export class GDriveSkill extends ConnectorDatasourceSkill {
	constructor(options: GDriveSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(GDRIVE_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new GDriveConnector(connectorOptions ?? {}),
		});
	}
}
