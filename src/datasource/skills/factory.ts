/**
 * Config-driven factory for the built-in connector-backed datasource skills.
 *
 * The CLI/server layer passes the trusted `datasources` section of
 * `config.json`; this factory materializes one skill per configured entry.
 * Model/tool arguments never reach this factory — it is part of the trusted
 * configuration path. Unknown skill names and disabled entries are skipped
 * and reported so setup surfaces actionable (but path-opaque) feedback.
 */

import type { DatasourceSkill } from "../types.ts";
import { CloudDriveSkill } from "./cloud-drive/index.ts";
import { DiscrawlClient, type DiscrawlOptions, DiscrawlSkill } from "./discrawl/index.ts";
import { type GDriveConnectorOptions, GDriveSkill } from "./gdrive/index.ts";
import { RcloneConnector, type RcloneConnectorOptions } from "./gdrive/rclone-connector.ts";
import { type GitHubConnectorOptions, GitHubSkill } from "./github/index.ts";
import { HimalayaConnector, type HimalayaConnectorOptions } from "./gmail/himalaya-connector.ts";
import { type GmailConnectorOptions, GmailSkill } from "./gmail/index.ts";
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
	readonly instanceId?: string;
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	/** Connector-specific options (token env names, repos, feeds, paths, …). */
	readonly connector?: Record<string, unknown>;
}

/** The trusted `datasources` config section: skill name → config. */
export type DatasourcesConfig = Readonly<Record<string, DatasourceSkillConfig | boolean>>;

export interface BuildDatasourceSkillsResult {
	readonly skills: readonly DatasourceSkill[];
	/** Names that were configured but not recognized. */
	readonly unknown: readonly string[];
}

type SkillBuilder = (config: DatasourceSkillConfig, workspaceRoot: string | undefined) => DatasourceSkill;

const BUILDERS: Readonly<Record<string, SkillBuilder>> = {
	telegram: (config) =>
		new TelecrawlSkill({
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			connectorOptions: config.connector as TelecrawlOptions,
		}),
	whatsapp: (config) =>
		new WacrawlSkill({
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			connectorOptions: config.connector as WacrawlOptions,
		}),
	slack: (config) =>
		new SlackSkill({
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			connectorOptions: config.connector as SlacrawlOptions,
		}),
	discord: (config, workspaceRoot) => {
		const connector = (config.connector ?? {}) as DiscrawlOptions & { readonly embedLimit?: number };
		const clientOptions: DiscrawlOptions = {
			...connector,
			...(connector.root === undefined && workspaceRoot !== undefined ? { root: workspaceRoot } : {}),
		};
		return new DiscrawlSkill({
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			client: new DiscrawlClient(clientOptions),
			...(connector.embeddingModel !== undefined ? { embeddingModel: connector.embeddingModel } : {}),
			...(connector.defaultMode !== undefined ? { defaultMode: connector.defaultMode } : {}),
			...(connector.embedLimit !== undefined ? { embedLimit: connector.embedLimit } : {}),
		});
	},
	notion: (config) =>
		new NotionSkill({
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			connectorOptions: config.connector as NotcrawlOptions,
		}),
	github: (config, workspaceRoot) =>
		new GitHubSkill({
			...common(config, workspaceRoot),
			connectorOptions: config.connector as GitHubConnectorOptions,
		}),
	gdrive: (config, workspaceRoot) => {
		const connector = config.connector as
			| (GDriveConnectorOptions & RcloneConnectorOptions & { backend?: string })
			| undefined;
		// `backend: "rclone"` routes through the external rclone CLI (Google
		// Drive or any rclone remote) instead of the Drive REST API.
		if (connector?.backend === "rclone") {
			const { backend: _backend, ...rcloneOptions } = connector;
			return new GDriveSkill({
				...common(config, workspaceRoot),
				connector: new RcloneConnector({
					...rcloneOptions,
					skillName: "gdrive",
					instanceId: config.instanceId,
					workspaceRoot,
				}),
			});
		}
		return new GDriveSkill({
			...common(config, workspaceRoot),
			connectorOptions: connector as GDriveConnectorOptions,
		});
	},
	"cloud-drive": (config, workspaceRoot) =>
		new CloudDriveSkill({
			...common(config, workspaceRoot),
			provider: typeof config.connector?.provider === "string" ? config.connector.provider : undefined,
			connectorOptions: config.connector as RcloneConnectorOptions,
		}),
	gmail: (config, workspaceRoot) => {
		const connector = config.connector as
			| (GmailConnectorOptions & HimalayaConnectorOptions & { backend?: string })
			| undefined;
		// `backend: "himalaya"` routes through the external himalaya CLI (any
		// IMAP/Maildir account it has configured) instead of the Gmail REST API.
		if (connector?.backend === "himalaya") {
			const { backend: _backend, ...himalayaOptions } = connector;
			return new GmailSkill({
				...common(config, workspaceRoot),
				connector: new HimalayaConnector(himalayaOptions),
			});
		}
		return new GmailSkill({ ...common(config, workspaceRoot), connectorOptions: connector as GmailConnectorOptions });
	},
	"mail-export": (config, workspaceRoot) =>
		new MailExportSkill({
			...common(config, workspaceRoot),
			connectorOptions: config.connector as MailExportConnectorOptions,
		}),
	obsidian: (config, workspaceRoot) => {
		const connector = config.connector as { vaultPath?: string; binaryPath?: string } | undefined;
		return new ObsidianSkill({
			...common(config, workspaceRoot),
			...(connector?.vaultPath !== undefined ? { vaultPath: connector.vaultPath } : {}),
			...(connector?.binaryPath !== undefined ? { binaryPath: connector.binaryPath } : {}),
		});
	},
	rss: (config, workspaceRoot) =>
		new RssSkill({ ...common(config, workspaceRoot), connectorOptions: config.connector as RssConnectorOptions }),
	spotlight: (config, workspaceRoot) =>
		new SpotlightSkill({
			...common(config, workspaceRoot),
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
		const builder = BUILDERS[name];
		if (builder === undefined) {
			unknown.push(name);
			continue;
		}
		skills.push(builder(entry, workspaceRoot));
	}
	return { skills, unknown };
}
