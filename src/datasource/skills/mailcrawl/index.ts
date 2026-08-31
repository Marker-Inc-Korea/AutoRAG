export { MailcrawlClient, type MailcrawlIndexClient, type MailcrawlSearchClient } from "./client.ts";
export { createMailcrawlManagedCliProvider } from "./config.ts";
export { MAILCRAWL_MODES, MailcrawlMethod } from "./methods.ts";
export { MailcrawlSkill, type MailcrawlSkillClient, type MailcrawlSkillOptions } from "./skill.ts";
export type {
	MailcrawlFailure,
	MailcrawlFailureReason,
	MailcrawlIndexInfo,
	MailcrawlOptions,
	MailcrawlSearchHit,
	MailcrawlSearchMode,
	MailcrawlSearchResult,
	MailcrawlSyncInfo,
	MailcrawlSyncResult,
} from "./types.ts";
