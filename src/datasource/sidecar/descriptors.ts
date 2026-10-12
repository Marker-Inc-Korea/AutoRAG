/**
 * Declarative description of a crawler's read-only SQLite schema. The sidecar
 * engine (index.ts) is crawler-agnostic: everything that differs between
 * slacrawl, telecrawl, wacrawl and notcrawl is expressed here as SQL fragments.
 *
 * Every SQL fragment is a trusted constant authored in this file. User input
 * (the search query, workspace id) is only ever bound as a parameter.
 *
 * The crawler database is opened read-only and ATTACHed; the sidecar owns a
 * separate database file, so a crawler that rebuilds, migrates, purges or
 * vacuums its own file can never lose or corrupt sidecar state.
 */

export interface CrawlerDescriptor {
	readonly id: string;
	/**
	 * True when the crawler's own TOML config (`connector.configPath`) can relocate its archive via
	 * `db_path`. The sidecar then follows that setting instead of assuming the default archive.
	 */
	readonly archivePathInConfig?: boolean;
	/** `message_key` stored in the sidecar for each row; unique per message within one crawler DB. */
	readonly keyExpression: string;
	/**
	 * Public chunk id for a stored key. MUST equal the id the crawler's CLI client derives for the
	 * same message (see each skill's `parseHits`), so the retrieval merger deduplicates a message
	 * found by both the CLI-backed and the sidecar methods.
	 */
	readonly chunkId: (key: string, metadata: Readonly<Record<string, string>>) => string;
	/** FROM + JOINs exposing alias `m` (message-like row) and any lookup aliases used below. */
	readonly fromClause: string;
	/** Row visibility predicate: excludes soft-deleted rows. May reference a `:scope` parameter when `scopeColumn` is set. */
	readonly liveWhere: string;
	/** Column that the optional workspace/account scope filters on; omitted when the crawler has one account per DB. */
	readonly scopeExpression?: string;
	/** SQL returning the embeddable/displayed body of the row. */
	readonly textExpression: string;
	/** Optional extra text prepended to the embedding input (title, channel). */
	readonly titleExpression?: string;
	/** Seconds-since-epoch expression used for `publishedAt`; omitted when unknown. */
	readonly epochSecondsExpression?: string;
	/**
	 * FTS5 relation joined to the message row. `joinOn` ties the FTS row to alias `m`;
	 * `table` is the FTS table (also the argument to bm25()).
	 */
	readonly fts: { readonly table: string; readonly joinOn: string };
	/** Extra result columns mapped into hit metadata: a SQL expression and the metadata key it fills. */
	readonly metadata: readonly { readonly sql: string; readonly key: string }[];
	/** Metadata keys the channel allowlist (`channels.ids/names`) matches against. */
	readonly channelIdKey: "channelId" | "chatId";
	readonly channelNameKey: "channelName" | "chatName";
	/** Path-opaque label for the hierarchy: ["workspaces", ws, "channels", name] etc. */
	readonly hierarchy: (metadata: Readonly<Record<string, unknown>>) => readonly string[];
}

const str = (value: unknown): string | undefined => (typeof value === "string" && value.length > 0 ? value : undefined);

const slack: CrawlerDescriptor = {
	id: "slack",
	archivePathInConfig: true,
	keyExpression: "m.channel_id || '|' || m.ts",
	// slacrawl search emits no message id; the CLI client derives `<channel>-<ts with dots as dashes>`.
	chunkId: (_key, metadata) => `${metadata.channelId ?? ""}-${(metadata.timestamp ?? "").replaceAll(".", "-")}`,
	fromClause:
		"messages m left join channels c on c.id = m.channel_id and c.workspace_id = m.workspace_id " +
		"left join users u on u.id = m.user_id",
	liveWhere: "(m.deleted_ts is null or m.deleted_ts = '')",
	scopeExpression: "m.workspace_id",
	textExpression: "coalesce(nullif(m.normalized_text, ''), m.text)",
	titleExpression: "c.name",
	epochSecondsExpression: "cast(m.ts as real)",
	fts: { table: "message_fts", joinOn: "message_fts.message_key = m.channel_id || '|' || m.ts" },
	metadata: [
		{ sql: "m.channel_id", key: "channelId" },
		{ sql: "c.name", key: "channelName" },
		{ sql: "m.workspace_id", key: "workspaceId" },
		{ sql: "m.user_id", key: "userId" },
		{ sql: "coalesce(nullif(u.real_name, ''), u.name)", key: "senderName" },
		{ sql: "m.ts", key: "timestamp" },
	],
	channelIdKey: "channelId",
	channelNameKey: "channelName",
	hierarchy: (metadata) => [
		"workspaces",
		str(metadata.workspaceId) ?? "unknown",
		"channels",
		str(metadata.channelName) ?? str(metadata.channelId) ?? "unknown",
	],
};

/** telecrawl and wacrawl share one message schema (`messages` + external-content `messages_fts` keyed by rowid). */
function chatCrawler(id: string): CrawlerDescriptor {
	return {
		id,
		keyExpression: "m.event_id",
		// telecrawl/wacrawl search rows carry event_id, which the CLI client uses verbatim.
		chunkId: (key) => key,
		fromClause: "messages m",
		liveWhere: "(m.deleted_at is null or m.deleted_at = 0)",
		textExpression: "coalesce(m.text, '')",
		titleExpression: "m.chat_name",
		epochSecondsExpression: "m.ts",
		fts: { table: "messages_fts", joinOn: "messages_fts.rowid = m.rowid" },
		metadata: [
			{ sql: "m.chat_jid", key: "chatId" },
			{ sql: "m.chat_name", key: "chatName" },
			{ sql: "m.sender_name", key: "senderName" },
			{ sql: "m.ts", key: "timestamp" },
		],
		channelIdKey: "chatId",
		channelNameKey: "chatName",
		hierarchy: (metadata) => ["chats", str(metadata.chatName) ?? str(metadata.chatId) ?? "unknown"],
	};
}

export const CRAWLER_DESCRIPTORS = {
	slack,
	telegram: chatCrawler("telegram"),
	whatsapp: chatCrawler("whatsapp"),
} as const satisfies Readonly<Record<string, CrawlerDescriptor>>;

export type SidecarCrawlerId = keyof typeof CRAWLER_DESCRIPTORS;
