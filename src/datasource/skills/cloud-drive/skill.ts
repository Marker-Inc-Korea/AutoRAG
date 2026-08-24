import type { DatasourceConnector } from "../../connector.ts";
import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { RcloneConnector, type RcloneConnectorOptions } from "../gdrive/rclone-connector.ts";

export const CLOUD_DRIVE_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "cloud-drive",
	skillType: "rclone-drive",
	description: "rclone cloud-drive datasource",
	capabilities: ["documents", "external-cli", "incremental", "polling"],
	requiresExternalCli: true,
	defaultTags: ["cloud-drive", "documents", "pii"],
	contentType: "document",
	manifestDescription:
		"Search indexed documents from an authorized rclone remote such as Google Drive, OneDrive, iCloud Drive, Dropbox, SMB, SFTP, or WebDAV.",
	manifestNotes: [
		"Configure and authenticate the remote with `rclone config`; AutoRAG never receives or stores provider credentials.",
		"Google Drive is Tier-1 supported. OneDrive and other rclone remotes use the same manifest contract. iCloud Drive is experimental and requires periodic Apple ID reauthentication.",
		"Indexing is incremental: `rclone lsjson` inventories metadata, then only added or changed indexable files are mirrored. Search uses the last completed snapshot while a failed sync is retried.",
		"Use `load_datasource_skill` before `search_datasource_documents`; pass only a natural-language query and optionally a narrowing scope such as `/cloud-drive/<instance>/**`.",
	],
};

export interface CloudDriveSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	readonly provider?: "google-drive" | "onedrive" | "icloud" | string;
	readonly connector?: DatasourceConnector;
	readonly connectorOptions?: RcloneConnectorOptions;
}

export class CloudDriveSkill extends ConnectorDatasourceSkill {
	constructor(options: CloudDriveSkillOptions = {}) {
		const { connector, connectorOptions, provider: _provider, ...rest } = options;
		super(CLOUD_DRIVE_SKILL_DEFINITION, {
			...rest,
			connector:
				connector ??
				new RcloneConnector({
					...connectorOptions,
					skillName: "cloud-drive",
					instanceId: rest.instanceId,
					workspaceRoot: rest.workspaceRoot,
				}),
		});
	}
}
