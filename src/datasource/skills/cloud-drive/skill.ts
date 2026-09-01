import type { DatasourceConnector } from "../../connector.ts";
import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { RcloneConnector, type RcloneConnectorOptions } from "./rclone-connector.ts";

export const CLOUD_DRIVE_SKILL_DEFINITION = createCloudDriveSkillDefinition("cloud-drive");

export interface CloudDriveSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Unique model-visible datasource name for this configured connection. */
	readonly skillName?: string;
	readonly provider?: "google-drive" | "onedrive" | "icloud" | string;
	readonly connector?: DatasourceConnector;
	readonly connectorOptions?: RcloneConnectorOptions;
}

export class CloudDriveSkill extends ConnectorDatasourceSkill {
	constructor(options: CloudDriveSkillOptions = {}) {
		const { connector, connectorOptions, provider, skillName = "cloud-drive", ...rest } = options;
		super(createCloudDriveSkillDefinition(skillName, provider), {
			...rest,
			connector:
				connector ??
				new RcloneConnector({
					...connectorOptions,
					skillName,
					instanceId: rest.instanceId,
					workspaceRoot: rest.workspaceRoot,
				}),
		});
	}
}

function createCloudDriveSkillDefinition(skillName: string, provider?: string): ConnectorSkillDefinition {
	const providerLabel = providerDisplayName(provider);
	return {
		skillName,
		skillType: "rclone-drive",
		description: `${providerLabel} datasource (${skillName})`,
		capabilities: ["documents", "external-cli", "incremental", "polling"],
		requiresExternalCli: true,
		defaultTags: ["cloud-drive", provider ?? "rclone", "documents", "pii"],
		contentType: "document",
		manifestDescription: `Search indexed documents from the authorized ${providerLabel} connection named "${skillName}".`,
		manifestNotes: [
			"Configure and authenticate the remote with `rclone config`; AutoRAG never receives or stores provider credentials.",
			"Google Drive is Tier-1 supported. OneDrive and other rclone remotes use the same manifest contract. iCloud Drive is experimental and requires periodic Apple ID reauthentication.",
			"Indexing is incremental: `rclone lsjson` inventories metadata, then only added or changed indexable files are mirrored. Search uses the last completed snapshot while a failed sync is retried.",
			`This connection is independently addressable. Load \`datasource-${skillName}\`, then call \`search_datasource_documents\` with a natural-language query and \`topK\`.`,
			"You can also invoke `rclone` directly through `bash` for the full CLI surface: `rclone lsjson <remote>:<path>`, `rclone copy <remote>:<path> <dest>`, `rclone --help`.",
		],
	};
}

function providerDisplayName(provider: string | undefined): string {
	switch (provider) {
		case "google-drive":
			return "Google Drive";
		case "onedrive":
			return "OneDrive";
		case "icloud":
			return "iCloud Drive (experimental)";
		default:
			return provider ?? "rclone cloud-drive";
	}
}
