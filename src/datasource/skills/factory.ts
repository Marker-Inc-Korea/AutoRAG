/**
 * Config-driven factory for the built-in connector-backed datasource skills.
 *
 * The CLI/server layer passes the trusted `datasources` section of
 * `config.json`; this factory materializes one skill per configured entry.
 * Model/tool arguments never reach this factory — it is part of the trusted
 * configuration path. Unknown skill names and disabled entries are skipped
 * and reported so setup surfaces actionable (but path-opaque) feedback.
 */

import { AliasedDatasourceSkill } from "../aliased-skill.ts";
import { DescribedDatasourceSkill } from "../described-skill.ts";
import type { DatasourceSkill } from "../types.ts";
import { ClawGalleryClient, type ClawGalleryOptions, ClawGallerySkill } from "./clawgallery/index.ts";
import { CloudDriveSkill } from "./cloud-drive/index.ts";
import { DiscrawlClient, type DiscrawlOptions, DiscrawlSkill } from "./discrawl/index.ts";
import { type GDriveConnectorOptions, GDriveSkill } from "./gdrive/index.ts";
import { RcloneConnector, type RcloneConnectorOptions } from "./gdrive/rclone-connector.ts";
import { type GitHubConnectorOptions, GitHubSkill } from "./github/index.ts";
import { HimalayaConnector, type HimalayaConnectorOptions } from "./gmail/himalaya-connector.ts";
import { type GmailConnectorOptions, GmailSkill } from "./gmail/index.ts";
import { KatokClient, type KatokOptions, KatokSkill } from "./katok/index.ts";
import { type MailExportConnectorOptions, MailExportSkill } from "./mail-export/index.ts";
import { type NotcrawlOptions, NotionSkill } from "./notion/index.ts";
import { ObsidianSkill } from "./obsidian/index.ts";
import { type RssConnectorOptions, RssSkill } from "./rss/index.ts";
import { SlackSkill, type SlacrawlOptions } from "./slack/index.ts";
import { type SpotlightConnectorOptions, SpotlightSkill } from "./spotlight/index.ts";
import { type TelecrawlOptions, TelecrawlSkill } from "./telecrawl/index.ts";
import { type WacrawlOptions, WacrawlSkill } from "./wacrawl/index.ts";

/** One configured datasource entry (the trusted `datasources.<name>` value). */
export interface DatasourceSkillConfig {
	readonly enabled?: boolean;
	/** Operator-authored context shown to the agent for this connection. */
	readonly description?: string;
	/** Built-in datasource template used when the config key is a connection alias. */
	readonly type?: string;
	readonly instanceId?: string;
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	/** Connector-specific options (token env names, repos, feeds, paths, …). */
	readonly connector?: Record<string, unknown>;
	/** Optional chat/channel allowlist; absent means all channels. */
	readonly channels?: {
		readonly ids?: readonly string[];
		readonly names?: readonly string[];
	};
}

/** The trusted `datasources` config section: skill name → config. */
export type DatasourcesConfig = Readonly<Record<string, DatasourceSkillConfig | boolean>>;

export interface BuildDatasourceSkillsResult {
	readonly skills: readonly DatasourceSkill[];
	/** Names that were configured but not recognized. */
	readonly unknown: readonly string[];
}

type SkillBuilder = (
	config: DatasourceSkillConfig,
	workspaceRoot: string | undefined,
	registrationName: string,
) => DatasourceSkill;

const BUILDERS: Readonly<Record<string, SkillBuilder>> = {
	telegram: (config, _workspaceRoot, registrationName) =>
		new TelecrawlSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			channelIds: config.channels?.ids,
			channelNames: config.channels?.names,
			connectorOptions: config.connector as TelecrawlOptions,
		}),
	whatsapp: (config, _workspaceRoot, registrationName) =>
		new WacrawlSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			channelIds: config.channels?.ids,
			channelNames: config.channels?.names,
			connectorOptions: config.connector as WacrawlOptions,
		}),
	slack: (config, _workspaceRoot, registrationName) =>
		new SlackSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			channelIds: config.channels?.ids,
			channelNames: config.channels?.names,
			connectorOptions: config.connector as SlacrawlOptions,
		}),
	discord: (config, workspaceRoot, registrationName) => {
		const connector = (config.connector ?? {}) as DiscrawlOptions & { readonly embedLimit?: number };
		const clientOptions: DiscrawlOptions = {
			...connector,
			...(connector.root === undefined && workspaceRoot !== undefined ? { root: workspaceRoot } : {}),
		};
		return new DiscrawlSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			channelIds: config.channels?.ids,
			channelNames: config.channels?.names,
			client: new DiscrawlClient(clientOptions),
			...(connector.embeddingModel !== undefined ? { embeddingModel: connector.embeddingModel } : {}),
			...(connector.defaultMode !== undefined ? { defaultMode: connector.defaultMode } : {}),
			...(connector.embedLimit !== undefined ? { embedLimit: connector.embedLimit } : {}),
		});
	},
	clawgallery: (config, workspaceRoot, registrationName) => {
		const connector = (config.connector ?? {}) as ClawGalleryOptions;
		const clientOptions: ClawGalleryOptions = {
			...connector,
			...(connector.configDir === undefined && workspaceRoot !== undefined
				? { configDir: `${workspaceRoot}/.autorag/datasources/clawgallery/${registrationName}` }
				: {}),
		};
		return new ClawGallerySkill({
			client: new ClawGalleryClient(clientOptions),
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			...(connector.defaultMode !== undefined ? { defaultMode: connector.defaultMode } : {}),
			...(connector.syncVisual !== undefined ? { syncVisual: connector.syncVisual } : {}),
			...(connector.vdrBackend !== undefined ? { vdrBackend: connector.vdrBackend } : {}),
		});
	},
	notion: (config, _workspaceRoot, registrationName) =>
		new NotionSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			connectorOptions: config.connector as NotcrawlOptions,
		}),
	kakao: (config) =>
		new KatokSkill({
			client: new KatokClient(config.connector as KatokOptions),
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
		}),
	github: (config, workspaceRoot, registrationName) =>
		new GitHubSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: config.connector as GitHubConnectorOptions,
		}),
	gdrive: (config, workspaceRoot, registrationName) => {
		const connector = config.connector as
			| (GDriveConnectorOptions & RcloneConnectorOptions & { backend?: string })
			| undefined;
		// `backend: "rclone"` routes through the external rclone CLI (Google
		// Drive or any rclone remote) instead of the Drive REST API.
		if (connector?.backend === "rclone") {
			const { backend: _backend, ...rcloneOptions } = connector;
			return new GDriveSkill({
				...common(config, workspaceRoot),
				skillName: registrationName,
				connector: new RcloneConnector({
					...rcloneOptions,
					skillName: registrationName,
					instanceId: config.instanceId,
					workspaceRoot,
				}),
			});
		}
		return new GDriveSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: connector as GDriveConnectorOptions,
		});
	},
	"cloud-drive": (config, workspaceRoot, registrationName) =>
		new CloudDriveSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			provider: typeof config.connector?.provider === "string" ? config.connector.provider : undefined,
			connectorOptions: config.connector as RcloneConnectorOptions,
		}),
	gmail: (config, workspaceRoot, registrationName) => {
		const connector = config.connector as
			| (GmailConnectorOptions & HimalayaConnectorOptions & { backend?: string })
			| undefined;
		// `backend: "himalaya"` routes through the external himalaya CLI (any
		// IMAP/Maildir account it has configured) instead of the Gmail REST API.
		if (connector?.backend === "himalaya") {
			const { backend: _backend, ...himalayaOptions } = connector;
			return new GmailSkill({
				...common(config, workspaceRoot),
				skillName: registrationName,
				connector: new HimalayaConnector(himalayaOptions),
			});
		}
		return new GmailSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: connector as GmailConnectorOptions,
		});
	},
	"mail-export": (config, workspaceRoot, registrationName) =>
		new MailExportSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: config.connector as MailExportConnectorOptions,
		}),
	obsidian: (config, workspaceRoot, registrationName) => {
		const connector = config.connector as { vaultPath?: string; binaryPath?: string } | undefined;
		return new ObsidianSkill({
			...common(config, workspaceRoot),
			datasourceId: registrationName,
			...(connector?.vaultPath !== undefined ? { vaultPath: connector.vaultPath } : {}),
			...(connector?.binaryPath !== undefined ? { binaryPath: connector.binaryPath } : {}),
		});
	},
	rss: (config, workspaceRoot, registrationName) =>
		new RssSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: config.connector as RssConnectorOptions,
		}),
	spotlight: (config, workspaceRoot, registrationName) =>
		new SpotlightSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: config.connector as SpotlightConnectorOptions,
		}),
};

/** Skill names this factory can build. */
export const BUILTIN_DATASOURCE_SKILL_NAMES: readonly string[] = Object.keys(BUILDERS);

function common(config: DatasourceSkillConfig, workspaceRoot: string | undefined) {
	return {
		...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
		...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
		...(config.tags !== undefined ? { tags: config.tags } : {}),
		...(workspaceRoot !== undefined ? { workspaceRoot } : {}),
	};
}

/**
 * Build datasource skills from the trusted config section. `false` or
 * `{ enabled: false }` entries are skipped silently; unrecognized names are
 * collected in `unknown` (never thrown) so callers can surface setup
 * feedback without failing agent construction.
 */
export function buildDatasourceSkills(
	config: DatasourcesConfig | undefined,
	workspaceRoot?: string,
): BuildDatasourceSkillsResult {
	const skills: DatasourceSkill[] = [];
	const unknown: string[] = [];
	for (const [name, raw] of Object.entries(config ?? {})) {
		if (raw === false) continue;
		const entry: DatasourceSkillConfig = raw === true ? {} : raw;
		if (entry.enabled === false) continue;
		const templateName = entry.type ?? name;
		const builder = BUILDERS[templateName];
		if (builder === undefined) {
			unknown.push(name);
			continue;
		}
		const normalizedEntry =
			entry.instanceId === undefined && entry.type !== undefined ? { ...entry, instanceId: name } : entry;
		const skill = builder(normalizedEntry, workspaceRoot, name);
		const hasChannelFilter = (entry.channels?.ids?.length ?? 0) > 0 || (entry.channels?.names?.length ?? 0) > 0;
		const aliased =
			name !== templateName || hasChannelFilter
				? new AliasedDatasourceSkill(skill, {
						alias: name,
						...(entry.channels?.ids !== undefined ? { channelIds: entry.channels.ids } : {}),
						...(entry.channels?.names !== undefined ? { channelNames: entry.channels.names } : {}),
					})
				: skill;
		skills.push(
			typeof entry.description === "string" && entry.description.trim().length > 0
				? new DescribedDatasourceSkill(aliased, entry.description.trim())
				: aliased,
		);
	}
	return { skills, unknown };
}
