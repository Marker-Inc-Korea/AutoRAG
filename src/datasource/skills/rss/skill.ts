/**
 * RSS/news datasource skill (issue #1316).
 *
 * Thin composition of RssConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/rss/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { RssConnector, type RssConnectorOptions } from "./connector.ts";

export const RSS_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "rss",
	skillType: "rss-feeds",
	description: "RSS/news datasource",
	capabilities: ["news", "api", "polling"],
	defaultTags: ["rss", "news", "public"],
	contentType: "article",
	manifestDescription:
		"Search indexed RSS/news feed articles from the configured feeds. Use for questions about recent news items or blog posts the server subscribes to.",
	manifestNotes: ["Feeds are polled with a dedupe window so re-delivered items are indexed once per window."],
};

export interface RssSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: RssConnector;
	readonly connectorOptions?: RssConnectorOptions;
	/** Dedupe window for re-delivered feed items. Default 24 hours. */
	readonly dedupeWindowMs?: number;
}

const DEFAULT_DEDUPE_WINDOW_MS = 24 * 60 * 60 * 1000;

export class RssSkill extends ConnectorDatasourceSkill {
	constructor(options: RssSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(RSS_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new RssConnector(connectorOptions ?? {}),
			dedupeWindowMs: options.dedupeWindowMs ?? DEFAULT_DEDUPE_WINDOW_MS,
		});
	}
}
