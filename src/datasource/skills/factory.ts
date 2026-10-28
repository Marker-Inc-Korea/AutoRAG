/**
 * Config-driven factory for the built-in connector-backed datasource skills.
 *
 * The CLI/server layer passes the trusted `datasources` section of
 * `config.json`; this factory materializes one skill per configured entry.
 * Model/tool arguments never reach this factory — it is part of the trusted
 * configuration path. Unknown skill names and disabled entries are skipped
 * and reported so setup surfaces actionable (but path-opaque) feedback.
 */

import { homedir } from "node:os";
import { join } from "node:path";
import type { Embedder } from "../../embedding/embedder.ts";
import { AliasedDatasourceSkill } from "../aliased-skill.ts";
import { DescribedDatasourceSkill } from "../described-skill.ts";
import type { EmbeddingEndpointSource } from "../embedding-env.ts";
import { resolveSidecarOptions } from "../sidecar/config.ts";
import { CRAWLER_DESCRIPTORS, type CrawlerDescriptor } from "../sidecar/descriptors.ts";
import type { DatasourceSkill } from "../types.ts";
import { ClawGalleryClient, type ClawGalleryOptions, ClawGallerySkill } from "./clawgallery/index.ts";
import { CloudDriveSkill, type RcloneConnectorOptions } from "./cloud-drive/index.ts";
import {
	DiscordAttachmentStore,
	DiscrawlClient,
	type DiscrawlEmbeddingRuntime,
	type DiscrawlOptions,
	DiscrawlSkill,
	discordAttachmentsRoot,
} from "./discrawl/index.ts";
import { resolveDiscrawlSidecarOptions } from "./discrawl/sidecar.ts";
import { type GitHubConnectorOptions, GitHubSkill } from "./github/index.ts";
import { type GitHubGistConnectorOptions, GitHubGistSkill } from "./github-gist/index.ts";
import { HisterSkill } from "./hister/index.ts";
import { LarkSkill } from "./lark/index.ts";
import { LazykatokClient, type LazykatokOptions, LazykatokSkill } from "./lazykatok/index.ts";
import { type MailExportConnectorOptions, MailExportSkill } from "./mail-export/index.ts";
import { type MailcrawlOptions, MailcrawlSkill } from "./mailcrawl/index.ts";
import { type NotcrawlOptions, NotionSkill } from "./notion/index.ts";
import { type ObsidianMinSyncOptions, ObsidianSkill } from "./obsidian/index.ts";
import { type RssConnectorOptions, RssSkill } from "./rss/index.ts";
import { SlackSkill } from "./slack/index.ts";
import { type SpotlightConnectorOptions, SpotlightSkill } from "./spotlight/index.ts";
import { TelecrawlSkill } from "./telecrawl/index.ts";
import { WacrawlSkill } from "./wacrawl/index.ts";

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

/**
 * What the shared embedding backend gives a datasource. AutoRAG-owned semantic stores (chat sidecars, the
 * gist vector index) call the `embedder`; an external CLI that embeds for itself (discrawl) is pointed at the
 * same backend through `discrawl`.
 */
export interface DatasourceEmbeddingOptions {
	readonly embedder?: Embedder;
	readonly discrawl?: DiscrawlEmbeddingRuntime | (() => Promise<DiscrawlEmbeddingRuntime | undefined>);
	/** For CLIs that embed for themselves through env (mailcrawl, lazykatok): where the shared backend is. */
	readonly cliEndpoint?: EmbeddingEndpointSource;
	/** MinSync binary, embedder and chunking shared with the local-file index; vault indexes run on it too. */
	readonly minSync?: ObsidianMinSyncOptions;
}

type SkillBuilder = (
	config: DatasourceSkillConfig,
	workspaceRoot: string | undefined,
	registrationName: string,
	embedding: DatasourceEmbeddingOptions,
) => DatasourceSkill | undefined;

/**
 * Chat crawlers (telegram, whatsapp, slack) share one construction shape: the CLI
 * client options, the channel allowlist, and the optional read-only sidecar.
 */
function chatCrawlerArgs(
	descriptor: CrawlerDescriptor,
	defaultCrawlerDb: string,
	config: DatasourceSkillConfig,
	workspaceRoot: string | undefined,
	registrationName: string,
	embedding: DatasourceEmbeddingOptions,
) {
	const sidecar = resolveSidecarOptions({
		descriptor,
		connector: config.connector,
		workspaceRoot,
		registrationName,
		instanceId: config.instanceId ?? "default",
		defaultCrawlerDb,
		...(embedding.embedder === undefined ? {} : { embedder: embedding.embedder }),
	});
	return {
		datasourceId: registrationName,
		...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
		...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
		...(config.tags !== undefined ? { tags: config.tags } : {}),
		channelIds: config.channels?.ids,
		channelNames: config.channels?.names,
		...(sidecar === undefined ? {} : { sidecar }),
		connectorOptions: {
			...config.connector,
			...(workspaceRoot === undefined ? {} : { workspacePath: workspaceRoot }),
		},
	};
}

function histerConnectorOptions(connector: Record<string, unknown> | undefined): {
	readonly baseUrl?: string;
	readonly tokenEnv?: string;
	readonly timeoutMs?: number;
	readonly semantic?: boolean;
} {
	if (connector === undefined) return {};
	const { baseUrl, tokenEnv, timeoutMs, semantic } = connector;
	return {
		...(typeof baseUrl === "string" ? { baseUrl } : {}),
		...(typeof tokenEnv === "string" ? { tokenEnv } : {}),
		...(typeof timeoutMs === "number" ? { timeoutMs } : {}),
		...(typeof semantic === "boolean" ? { semantic } : {}),
	};
}

type DiscordConnector = DiscrawlOptions & { readonly embedLimit?: number; readonly downloadAttachments?: boolean };

function discordClient(
	connector: DiscordConnector,
	workspaceRoot: string | undefined,
	embedding: DatasourceEmbeddingOptions,
): DiscrawlClient {
	const { downloadAttachments: _download, embedLimit: _limit, ...clientConnector } = connector;
	return new DiscrawlClient({
		...clientConnector,
		...(connector.root === undefined && workspaceRoot !== undefined ? { root: workspaceRoot } : {}),
		...(embedding.discrawl === undefined ? {} : { embeddingRuntime: embedding.discrawl }),
	});
}

function discordAttachmentStore(
	registrationName: string,
	config: DatasourceSkillConfig,
	client: DiscrawlClient,
	guildId: string | undefined,
): DiscordAttachmentStore {
	return new DiscordAttachmentStore({
		client,
		root: discordAttachmentsRoot(registrationName),
		...(guildId === undefined ? {} : { guildId }),
		...(config.channels?.ids === undefined ? {} : { channelIds: config.channels.ids }),
		...(config.channels?.names === undefined ? {} : { channelNames: config.channels.names }),
	});
}

export interface DiscordAttachmentConnection {
	readonly name: string;
	/** Whether refreshes download automatically (`connector.downloadAttachments !== false`, default on). */
	readonly autoDownload: boolean;
	readonly store: DiscordAttachmentStore;
}

/**
 * Every enabled Discord connection's attachments folder, built exactly as a refresh builds it, so
 * `autorag attachments` and the refresh always agree on where files live.
 */
export function discordAttachmentConnections(
	config: DatasourcesConfig | undefined,
	workspaceRoot?: string,
): readonly DiscordAttachmentConnection[] {
	const out: DiscordAttachmentConnection[] = [];
	for (const [name, raw] of Object.entries(config ?? {})) {
		if (raw === false) continue;
		const entry: DatasourceSkillConfig = raw === true ? {} : raw;
		if (entry.enabled === false || (entry.type ?? name) !== "discord") continue;
		const connector = (entry.connector ?? {}) as DiscordConnector;
		out.push({
			name,
			autoDownload: connector.downloadAttachments !== false,
			store: discordAttachmentStore(name, entry, discordClient(connector, workspaceRoot, {}), connector.guildId),
		});
	}
	return out;
}

const BUILDERS: Readonly<Record<string, SkillBuilder>> = {
	telegram: (config, workspaceRoot, registrationName, embedding) =>
		new TelecrawlSkill(
			chatCrawlerArgs(
				CRAWLER_DESCRIPTORS.telegram,
				join(homedir(), ".telecrawl", "telecrawl.db"),
				config,
				workspaceRoot,
				registrationName,
				embedding,
			),
		),
	whatsapp: (config, workspaceRoot, registrationName, embedding) =>
		new WacrawlSkill(
			chatCrawlerArgs(
				CRAWLER_DESCRIPTORS.whatsapp,
				join(homedir(), ".wacrawl", "wacrawl.db"),
				config,
				workspaceRoot,
				registrationName,
				embedding,
			),
		),
	slack: (config, workspaceRoot, registrationName, embedding) =>
		new SlackSkill(
			chatCrawlerArgs(
				CRAWLER_DESCRIPTORS.slack,
				join(homedir(), ".slacrawl", "slacrawl.db"),
				config,
				workspaceRoot,
				registrationName,
				embedding,
			),
		),
	discord: (config, workspaceRoot, registrationName, embedding) => {
		const connector = (config.connector ?? {}) as DiscordConnector;
		const sidecar = resolveDiscrawlSidecarOptions({
			connector: config.connector,
			workspaceRoot,
			registrationName,
			instanceId: config.instanceId ?? "default",
		});
		const client = discordClient(connector, workspaceRoot, embedding);
		return new DiscrawlSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			channelIds: config.channels?.ids,
			channelNames: config.channels?.names,
			...(sidecar === undefined ? {} : { sidecar }),
			client,
			...(connector.embeddingModel !== undefined ? { embeddingModel: connector.embeddingModel } : {}),
			...(connector.defaultMode !== undefined ? { defaultMode: connector.defaultMode } : {}),
			...(connector.embedLimit !== undefined ? { embedLimit: connector.embedLimit } : {}),
			...(connector.downloadAttachments === false
				? {}
				: { attachments: discordAttachmentStore(registrationName, config, client, connector.guildId) }),
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
	notion: (config, workspaceRoot, registrationName) =>
		new NotionSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			connectorOptions: {
				...(config.connector as NotcrawlOptions),
				...(workspaceRoot === undefined ? {} : { workspacePath: workspaceRoot }),
			},
		}),
	lark: (config, _workspaceRoot, registrationName) =>
		new LarkSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			...(config.channels?.ids !== undefined ? { chatIds: config.channels.ids } : {}),
			...larkConnectorOptions(config.connector),
		}),
	hister: (config, _workspaceRoot, registrationName) =>
		new HisterSkill({
			datasourceId: registrationName,
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			...histerConnectorOptions(config.connector),
		}),
	kakao: (config, workspaceRoot, registrationName, embedding) => {
		const connector = (config.connector ?? {}) as LazykatokOptions;
		const dataDir =
			typeof connector.workspacePath === "string" && connector.workspacePath.length > 0
				? connector.workspacePath
				: join(homedir(), "Library", "Application Support", "katok");
		const sidecar = resolveSidecarOptions({
			descriptor: CRAWLER_DESCRIPTORS.kakao,
			connector: config.connector,
			workspaceRoot,
			registrationName,
			instanceId: config.instanceId ?? "default",
			defaultCrawlerDb: join(dataDir, "archive.sqlite3"),
			...(embedding.embedder === undefined ? {} : { embedder: embedding.embedder }),
		});
		return new LazykatokSkill({
			client: new LazykatokClient({
				...connector,
				...(embedding.cliEndpoint === undefined ? {} : { embeddingEndpoint: embedding.cliEndpoint }),
			}),
			...(config.instanceId !== undefined ? { instanceId: config.instanceId } : {}),
			...(config.pollingIntervalMs !== undefined ? { pollingIntervalMs: config.pollingIntervalMs } : {}),
			...(config.tags !== undefined ? { tags: config.tags } : {}),
			channelIds: config.channels?.ids,
			channelNames: config.channels?.names,
			...(sidecar === undefined ? {} : { sidecar }),
		});
	},
	github: (config, workspaceRoot, registrationName) =>
		new GitHubSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: config.connector as GitHubConnectorOptions,
		}),
	"github-gist": (config, workspaceRoot, registrationName, embedding) => {
		// `connector.semantic: false` (or `{ enabled: false }`) keeps the gist lexical-only: no
		// vector index is built and no semantic method is registered. It is a skill option, not a
		// connector option, so it is taken out before the connector sees the rest.
		const { semantic, ...connectorOptions } = (config.connector ?? {}) as GitHubGistConnectorOptions & {
			readonly semantic?: unknown;
		};
		const semanticOff =
			semantic === false ||
			(typeof semantic === "object" && semantic !== null && (semantic as { enabled?: unknown }).enabled === false);
		return new GitHubGistSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions,
			...(semanticOff
				? { semantic: { enabled: false } }
				: embedding.embedder === undefined
					? {}
					: { semantic: { embedder: embedding.embedder } }),
		});
	},
	"cloud-drive": (config, workspaceRoot, registrationName) =>
		new CloudDriveSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			provider: typeof config.connector?.provider === "string" ? config.connector.provider : undefined,
			connectorOptions: {
				...(config.connector as RcloneConnectorOptions),
			},
		}),
	"mail-export": (config, workspaceRoot, registrationName) =>
		new MailExportSkill({
			...common(config, workspaceRoot),
			skillName: registrationName,
			connectorOptions: config.connector as MailExportConnectorOptions,
		}),
	mailcrawl: (config, workspaceRoot, registrationName, embedding) =>
		new MailcrawlSkill({
			...common(config, workspaceRoot),
			datasourceId: registrationName,
			...(config.connector as MailcrawlOptions),
			...(embedding.cliEndpoint === undefined ? {} : { embeddingEndpoint: embedding.cliEndpoint }),
		}),
	obsidian: (config, workspaceRoot, registrationName, embedding) => {
		const connector = config.connector as { vaultPath?: string } | undefined;
		return new ObsidianSkill({
			...common(config, workspaceRoot),
			datasourceId: registrationName,
			...(connector?.vaultPath === undefined ? {} : { vaultPath: connector.vaultPath }),
			...(embedding.minSync === undefined ? {} : { minSync: embedding.minSync }),
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

/** Skills whose backend runs on one platform only; elsewhere they are neither offered nor built. */
const PLATFORM_ONLY_SKILLS: Readonly<Record<string, NodeJS.Platform>> = { kakao: "darwin" };

/** Whether the skill's backend exists on `platform`. Names with no platform restriction are always supported. */
export function datasourceSkillSupported(name: string, platform: NodeJS.Platform = process.platform): boolean {
	return (PLATFORM_ONLY_SKILLS[name] ?? platform) === platform;
}

/** Skill names this factory can build on `platform`. */
export function builtinDatasourceSkillNames(platform: NodeJS.Platform = process.platform): readonly string[] {
	return Object.keys(BUILDERS).filter((name) => datasourceSkillSupported(name, platform));
}

/** Skill names this factory can build on the running platform. */
export const BUILTIN_DATASOURCE_SKILL_NAMES: readonly string[] = builtinDatasourceSkillNames();

function larkConnectorOptions(connector: Record<string, unknown> | undefined): {
	readonly binaryPath?: string;
	readonly embedderBaseUrl?: string;
} {
	if (connector === undefined) return {};
	const binaryPath = connector.binaryPath;
	const embedderBaseUrl = connector.embedderBaseUrl;
	return {
		...(typeof binaryPath === "string" ? { binaryPath } : {}),
		...(typeof embedderBaseUrl === "string" ? { embedderBaseUrl } : {}),
	};
}

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
	embedding: DatasourceEmbeddingOptions = {},
): BuildDatasourceSkillsResult {
	const skills: DatasourceSkill[] = [];
	const unknown: string[] = [];
	for (const [name, raw] of Object.entries(config ?? {})) {
		if (raw === false) continue;
		if (raw !== true && (typeof raw !== "object" || raw === null || Array.isArray(raw))) {
			unknown.push(name);
			continue;
		}
		const entry: DatasourceSkillConfig = raw === true ? {} : raw;
		if (entry.enabled === false) continue;
		const templateName = entry.type ?? name;
		const builder =
			Object.hasOwn(BUILDERS, templateName) && datasourceSkillSupported(templateName)
				? BUILDERS[templateName]
				: undefined;
		if (builder === undefined) {
			unknown.push(name);
			continue;
		}
		const normalizedEntry =
			entry.instanceId === undefined && entry.type !== undefined ? { ...entry, instanceId: name } : entry;
		const skill = builder(normalizedEntry, workspaceRoot, name, embedding);
		if (skill === undefined) {
			unknown.push(name);
			continue;
		}
		// Lark applies chat ids as a message-search flag. The generic alias
		// filter would drop document hits, which have no chat id.
		const larkOwnsChannels = templateName === "lark";
		const hasChannelFilter =
			!larkOwnsChannels && ((entry.channels?.ids?.length ?? 0) > 0 || (entry.channels?.names?.length ?? 0) > 0);
		let aliased: DatasourceSkill = skill;
		if (name !== templateName || hasChannelFilter) {
			aliased = new AliasedDatasourceSkill(skill, {
				alias: name,
				...(!larkOwnsChannels && entry.channels?.ids !== undefined ? { channelIds: entry.channels.ids } : {}),
				...(!larkOwnsChannels && entry.channels?.names !== undefined ? { channelNames: entry.channels.names } : {}),
			});
		}
		skills.push(
			typeof entry.description === "string" && entry.description.trim().length > 0
				? new DescribedDatasourceSkill(aliased, entry.description.trim())
				: aliased,
		);
	}
	return { skills, unknown };
}
