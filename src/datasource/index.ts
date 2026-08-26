export { DatasourceAccessContext, type DatasourceAccessContextOptions } from "./access-context.ts";
export { type AliasDatasourceSkillOptions, AliasedDatasourceSkill } from "./aliased-skill.ts";
export {
	DatasourceChunkStore,
	type DatasourceChunkStoreOptions,
	datasourceChunkStorePath,
	type ScoredChunk,
	type StoredChunk,
} from "./chunk-store.ts";
export {
	boundDiagnosticText,
	type ConnectorDocument,
	type ConnectorFailureReason,
	type ConnectorFetchFail,
	type ConnectorFetchOk,
	type ConnectorFetchResult,
	connectorFailureToDiagnosticCode,
	type DatasourceConnector,
	sanitizeIdSegment,
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
export { createCrawlerManagedCliProvider } from "./crawler-managed-config.ts";
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
	CLOUD_DRIVE_SKILL_DEFINITION,
	CloudDriveSkill,
	type CloudDriveSkillOptions,
} from "./skills/cloud-drive/index.ts";
export {
	DEFAULT_DISCRAWL_EMBEDDING_MODEL,
	DEFAULT_DISCRAWL_MODE,
	DEFAULT_DISCRAWL_SOURCE,
	DiscrawlClient,
	type DiscrawlFailureReason,
	DiscrawlFtsMethod,
	DiscrawlHybridMethod,
	type DiscrawlOptions,
	type DiscrawlSearchHit,
	type DiscrawlSearchMode,
	type DiscrawlSearchOptions,
	type DiscrawlSearchResult,
	DiscrawlSemanticMethod,
	DiscrawlSkill,
	type DiscrawlSkillClient,
	type DiscrawlSkillOptions,
	type DiscrawlSourceKind,
	ENGLISH_ONLY_EMBEDDING_MODELS,
} from "./skills/discrawl/index.ts";
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
export {
	KatokBm25Method,
	KatokClient,
	KatokSemanticMethod,
	KatokSkill,
	createKatokManagedCliProvider,
} from "./skills/katok/index.ts";
export {
	MailExportConnector,
	type MailExportConnectorOptions,
	MailExportSkill,
	type MailExportSkillOptions,
} from "./skills/mail-export/index.ts";
export {
	NotcrawlClient,
	type NotcrawlOptions,
	NotionSkill,
	type NotionSkillOptions,
} from "./skills/notion/index.ts";
export {
	ObsidianBm25Method,
	ObsidianSemanticMethod,
	ObsidianSkill,
	type ObsidianSkillClient,
	type ObsidianSkillOptions,
	QmdClient,
	type QmdOptions,
	type QmdSearchHit,
	type QmdSearchMode,
	type QmdSearchResult,
} from "./skills/obsidian/index.ts";
export {
	RssConnector,
	type RssConnectorOptions,
	type RssFeedConfig,
	RssSkill,
	type RssSkillOptions,
} from "./skills/rss/index.ts";
export {
	SlackSkill,
	type SlackSkillOptions,
	SlacrawlClient,
	type SlacrawlOptions,
} from "./skills/slack/index.ts";
export {
	SpotlightConnector,
	type SpotlightConnectorOptions,
	SpotlightSkill,
	type SpotlightSkillOptions,
} from "./skills/spotlight/index.ts";
export {
	TelecrawlClient,
	type TelecrawlOptions,
	TelecrawlSkill,
	type TelecrawlSkillOptions,
} from "./skills/telecrawl/index.ts";
export { WacrawlClient, type WacrawlOptions, WacrawlSkill, type WacrawlSkillOptions } from "./skills/wacrawl/index.ts";
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
