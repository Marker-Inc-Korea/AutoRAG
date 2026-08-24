import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import type { CrawlerHit, CrawlerSearchOptions, CrawlerSearchResult, CrawlerSyncResult } from "./crawler-types.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "./scope.ts";
import type {
	DatasourceDiagnosticCode,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "./types.ts";

export interface CrawlerSkillClient {
	sync(signal?: AbortSignal): Promise<CrawlerSyncResult>;
	search(query: string, options?: CrawlerSearchOptions): Promise<CrawlerSearchResult>;
}

export interface CrawlerDatasourceDefinition {
	readonly datasourceId: string;
	readonly skillType: string;
	readonly description: string;
	readonly defaultTags: readonly string[];
	readonly contentType: string;
	readonly manifestDescription: string;
	readonly backendName: string;
}

export interface CrawlerDatasourceSkillOptions {
	readonly client: CrawlerSkillClient;
	readonly datasourceId?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly lastIndexedAt?: number;
	/** Omitted/empty means search every chat/channel. */
	readonly channelIds?: readonly string[];
	readonly channelNames?: readonly string[];
}

const DEFAULT_INSTANCE_ID = "default";
const DEFAULT_POLLING_INTERVAL_MS = 15 * 60 * 1000;

export class CrawlerDatasourceSkill implements DatasourceSkill {
	private readonly definition: CrawlerDatasourceDefinition;
	private readonly client: CrawlerSkillClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly pollingIntervalMs: number;
	private readonly tags: readonly string[];
	private readonly channelIds: ReadonlySet<string>;
	private readonly channelNames: ReadonlySet<string>;
	private lastIndexedAt: number | undefined;

	constructor(definition: CrawlerDatasourceDefinition, options: CrawlerDatasourceSkillOptions) {
		this.definition =
			options.datasourceId === undefined
				? definition
				: {
						...definition,
						datasourceId: options.datasourceId,
						description: `${definition.description} (${options.datasourceId})`,
					};
		this.client = options.client;
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.pollingIntervalMs = options.pollingIntervalMs ?? DEFAULT_POLLING_INTERVAL_MS;
		this.tags = options.tags ?? definition.defaultTags;
		this.lastIndexedAt = options.lastIndexedAt;
		this.channelIds = new Set(options.channelIds ?? []);
		this.channelNames = new Set(options.channelNames ?? []);
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: this.definition.datasourceId,
			id: this.definition.datasourceId,
			type: this.definition.skillType,
			description: this.definition.description,
			capabilities: ["external-cli", "polling", "incremental", "fts5", "lexical"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: this.definition.datasourceId,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: this.pollingIntervalMs > 0 ? "poll" : "none",
			...(this.pollingIntervalMs > 0 ? { intervalMs: this.pollingIntervalMs } : {}),
			lastIndexedAt: this.lastIndexedAt,
		};
	}

	async index(): Promise<DatasourceIndexResult> {
		try {
			const result = await this.client.sync();
			if (!result.ok) return this.fail(failureCode(result.reason), result.reason);
			this.lastIndexedAt = Date.now();
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: this.definition.datasourceId,
				chunkCount: result.count,
				indexedAt: this.lastIndexedAt,
				diagnostics: [],
			};
		} catch {
			return this.fail("datasource-unavailable", "spawn-error");
		}
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [
			new CrawlerLexicalMethod({
				client: this.client,
				datasourceId: this.definition.datasourceId,
				skillType: this.definition.skillType,
				instanceId: this.instanceId,
				tags: this.tags,
				backendName: this.definition.backendName,
				channelIds: this.channelIds,
				channelNames: this.channelNames,
			}),
		];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath(this.definition.datasourceId, instanceId),
			datasourceId: this.definition.datasourceId,
			skill: this.definition.datasourceId,
			instanceId,
			contentType: this.definition.contentType,
			metadata: { datasourceId: this.definition.datasourceId, instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const scopes = this.instances
			.map((instanceId) => `- \`${datasourceSourcePath(this.definition.datasourceId, instanceId)}\``)
			.join("\n");
		return {
			name: `datasource-${this.definition.datasourceId}`,
			description: this.definition.manifestDescription,
			content: [
				`# ${this.definition.description} (${this.definition.skillType})`,
				"",
				`This datasource is indexed and searched through the external \`${this.definition.backendName}\` CLI. AutoRAG never opens the source application's private database directly.`,
				"",
				"## When to use",
				this.definition.manifestDescription,
				"",
				"## How to search",
				"Call `search_datasource_documents` with a query and optional narrowing scope:",
				scopes,
				"",
				"`scope` can only narrow within already-authorized scopes; it can never widen access.",
				"",
				this.channelIds.size === 0 && this.channelNames.size === 0
					? "Channel selection: all channels/chats are searchable."
					: `Channel selection: only configured ids/names are searchable (${[
							...this.channelIds,
							...this.channelNames,
						].join(", ")}).`,
			].join("\n"),
		};
	}

	private fail(code: DatasourceDiagnosticCode, reason: string): DatasourceIndexResult {
		const message = `${this.definition.backendName} ${reason}`;
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: this.definition.datasourceId,
			indexedAt: Date.now(),
			diagnostics: [
				{
					code,
					severity: code === "datasource-unavailable" ? "warning" : "error",
					message,
					instanceId: this.instanceId,
					source: this.definition.datasourceId,
				},
			],
			error: reason,
			code,
			message,
		};
	}
}

type CrawlerLexicalMethodOptions = {
	readonly client: CrawlerSkillClient;
	readonly datasourceId: string;
	readonly skillType: string;
	readonly instanceId: string;
	readonly tags: readonly string[];
	readonly backendName: string;
	readonly channelIds: ReadonlySet<string>;
	readonly channelNames: ReadonlySet<string>;
};

class CrawlerLexicalMethod implements RetrievalMethod {
	private readonly options: CrawlerLexicalMethodOptions;

	constructor(options: CrawlerLexicalMethodOptions) {
		this.options = options;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: `${this.options.datasourceId}-fts`,
			type: "bm25",
			description: `FTS5 retrieval over ${this.options.skillType} via ${this.options.backendName}`,
			status: "active",
			capabilities: ["lexical", "fts5", "scoped", "external-cli", "path-opaque-sources"],
			datasourceId: this.options.datasourceId,
			tags: this.options.tags,
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		let result: CrawlerSearchResult;
		try {
			result = await this.options.client.search(trimmed, options);
		} catch {
			return [];
		}
		if (!result.ok) return [];
		const mapped: RetrievalResult[] = [];
		for (const hit of result.hits) {
			const source = datasourceSourcePath(this.options.datasourceId, this.options.instanceId, hit.id);
			if (!matchesScope(source, options.scope, options.allowedScopes) || !matchesChannel(hit, this.options))
				continue;
			mapped.push({
				id: `${this.options.datasourceId}:${this.options.instanceId}:${hit.id}`,
				content: hit.content,
				source,
				score: hit.score,
				metadata: {
					...(hit.metadata ?? {}),
					method: `${this.options.datasourceId}-fts`,
					datasourceId: this.options.datasourceId,
					instanceId: this.options.instanceId,
					backend: this.options.backendName,
					...(hit.title !== undefined ? { title: hit.title } : {}),
					...(hit.hierarchy !== undefined ? { hierarchy: hit.hierarchy.join("/") } : {}),
					...(hit.publishedAt !== undefined ? { publishedAt: hit.publishedAt } : {}),
				},
			});
		}
		return mapped;
	}
}

function matchesChannel(
	hit: CrawlerHit,
	options: Pick<CrawlerLexicalMethodOptions, "channelIds" | "channelNames">,
): boolean {
	if (options.channelIds.size === 0 && options.channelNames.size === 0) return true;
	const metadata = hit.metadata ?? {};
	const id = [metadata.channelId, metadata.chatId].find((value): value is string => typeof value === "string");
	const name = [metadata.channelName, metadata.chatName].find((value): value is string => typeof value === "string");
	return (id !== undefined && options.channelIds.has(id)) || (name !== undefined && options.channelNames.has(name));
}

function matchesScope(
	source: string,
	scope: string | undefined,
	allowedScopes: readonly string[] | undefined,
): boolean {
	if (!matchesDatasourceScope(source, scope)) return false;
	if (allowedScopes === undefined || allowedScopes.length === 0) return true;
	return allowedScopes.some((entry) => matchesDatasourceScope(source, entry));
}

function failureCode(reason: string): DatasourceDiagnosticCode {
	return reason === "binary-missing" || reason === "spawn-error" || reason === "timeout" || reason === "aborted"
		? "datasource-unavailable"
		: "datasource-index-failed";
}
