export { DatasourceAccessContext, type DatasourceAccessContextOptions } from "./access-context.ts";
export {
	DatasourceChunkStore,
	type DatasourceChunkStoreOptions,
	datasourceChunkStorePath,
	type ScoredChunk,
	type StoredChunk,
} from "./chunk-store.ts";
export {
	type ConnectorDocument,
	type ConnectorFailureReason,
	type ConnectorFetchFail,
	type ConnectorFetchOk,
	type ConnectorFetchResult,
	connectorFailureToDiagnosticCode,
	type DatasourceConnector,
	sanitizeIdSegment,
	sanitizeOpaqueText,
} from "./connector.ts";
export {
	ConnectorDatasourceSkill,
	ConnectorLexicalMethod,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "./connector-skill.ts";
export { mapDatasourceDiagnostic, mapDatasourceDiagnostics } from "./diagnostics.ts";
export { type CronParseResult, isDue, parseCronExpr } from "./polling.ts";
export { DatasourceSkillRegistry, type RegisteredDatasourceSkill } from "./registry.ts";
export { DatasourceResultFilter, type ResultsByMethod } from "./result-filter.ts";
export {
	buildDatasourceChunkSource,
	buildDatasourceInstanceSource,
	DATASOURCE_CHUNKS_SEGMENT,
	datasourceSourceHasFragment,
	datasourceSourcePath,
	isDatasourceSource,
	matchesDatasourceScope,
	matchesVirtualPathScope,
	normalizeVirtualPath,
} from "./scope.ts";
export {
	DiscordConnector,
	type DiscordConnectorOptions,
	DiscordSkill,
	type DiscordSkillOptions,
} from "./skills/discord/index.ts";
export {
	BUILTIN_DATASOURCE_SKILL_NAMES,
	type BuildDatasourceSkillsResult,
	buildDatasourceSkills,
	type DatasourceSkillConfig,
	type DatasourcesConfig,
} from "./skills/factory.ts";
export {
	GDriveConnector,
	type GDriveConnectorOptions,
	GDriveSkill,
	type GDriveSkillOptions,
} from "./skills/gdrive/index.ts";
export {
	GitHubConnector,
	type GitHubConnectorOptions,
	GitHubSkill,
	type GitHubSkillOptions,
} from "./skills/github/index.ts";
export {
	GmailConnector,
	type GmailConnectorOptions,
	GmailSkill,
	type GmailSkillOptions,
} from "./skills/gmail/index.ts";
export type {
	KatokFailureReason,
	KatokHit,
	KatokOptions,
	KatokSearchMode,
	KatokSearchOptions,
	KatokSearchResult,
	KatokSkillClient,
	KatokSkillOptions,
} from "./skills/katok/index.ts";
export { KatokBm25Method, KatokClient, KatokSemanticMethod, KatokSkill } from "./skills/katok/index.ts";
export {
	MailExportConnector,
	type MailExportConnectorOptions,
	MailExportSkill,
	type MailExportSkillOptions,
} from "./skills/mail-export/index.ts";
export {
	NotionConnector,
	type NotionConnectorOptions,
	NotionSkill,
	type NotionSkillOptions,
} from "./skills/notion/index.ts";
export {
	ObsidianConnector,
	type ObsidianConnectorOptions,
	ObsidianSkill,
	type ObsidianSkillOptions,
} from "./skills/obsidian/index.ts";
export {
	RssConnector,
	type RssConnectorOptions,
	type RssFeedConfig,
	RssSkill,
	type RssSkillOptions,
} from "./skills/rss/index.ts";
export {
	SlackConnector,
	type SlackConnectorOptions,
	SlackSkill,
	type SlackSkillOptions,
} from "./skills/slack/index.ts";
export type {
	DatasourceAccessible,
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceIndexFail,
	DatasourceIndexOk,
	DatasourceIndexResult,
	DatasourceInstance,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "./types.ts";
