import { isAbsolute, join, resolve } from "node:path";
import { FallbackRetrievalMethod } from "../../../retrieval/fallback.ts";
import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { grepMailcrawlArchiveSqlite } from "../../grep.ts";
import { datasourceSourcePath } from "../../scope.ts";
import { datasourceSearchToolName } from "../../tool-naming.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceGrepHit,
	DatasourceGrepOptions,
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "../../types.ts";
import { MailcrawlClient, type MailcrawlIndexClient } from "./client.ts";
import { validateMailcrawlInstanceId } from "./config.ts";
import { MailcrawlMethod } from "./methods.ts";
import type { MailcrawlOptions, MailcrawlSyncResult } from "./types.ts";

const TAGS = ["mailcrawl", "email", "pii"] as const;
export interface MailcrawlSkillClient extends MailcrawlIndexClient {
	sync(signal?: AbortSignal): Promise<MailcrawlSyncResult>;
}
export interface MailcrawlSkillOptions extends MailcrawlOptions {
	readonly client?: MailcrawlSkillClient;
	readonly datasourceId?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly connectorOptions?: MailcrawlOptions;
}
export class MailcrawlSkill implements DatasourceSkill {
	private readonly client: MailcrawlSkillClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly interval: number;
	private readonly tags: readonly string[];
	private readonly account: string | undefined;
	private readonly mailbox: string | undefined;
	private readonly dataDir: string | undefined;
	private readonly envDataDir: string | undefined;
	private lastIndexedAt: number | undefined;
	constructor(options: MailcrawlSkillOptions = {}) {
		const connectorOptions = options.connectorOptions ?? options;
		this.instanceId = options.instanceId ?? "default";
		this.instances = options.instances?.length ? options.instances : [this.instanceId];
		validateMailcrawlInstanceId(this.instanceId);
		for (const instance of this.instances) validateMailcrawlInstanceId(instance);
		this.interval = options.pollingIntervalMs ?? 15 * 60 * 1000;
		this.tags = options.tags ?? TAGS;
		this.account = connectorOptions.account;
		this.mailbox = connectorOptions.mailbox;
		this.dataDir = connectorOptions.dataDir;
		this.envDataDir = connectorOptions.env?.MAILCRAWL_DATA_DIR;
		this.client = options.client ?? new MailcrawlClient({ ...connectorOptions, instanceId: this.instanceId });
	}
	describe(): DatasourceSkillDescriptor {
		return {
			name: "mailcrawl",
			id: "mailcrawl",
			type: "mailcrawl-archive",
			description: "Local email archive datasource via the external mailcrawl CLI",
			capabilities: ["email", "external-cli", "polling", "incremental", "bm25", "semantic", "hybrid"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: "mailcrawl",
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}
	polling(): PollingMetadata {
		return {
			mode: this.interval > 0 ? "poll" : "none",
			...(this.interval > 0 ? { intervalMs: this.interval } : {}),
			lastIndexedAt: this.lastIndexedAt,
		};
	}
	async index(options?: DatasourceIndexOptions): Promise<DatasourceIndexResult> {
		try {
			const result = await this.client.sync(options?.signal);
			if (!result.ok) {
				return this.fail(
					result.reason === "binary-missing" || result.reason === "spawn-error" || result.reason === "timeout"
						? "datasource-unavailable"
						: result.reason === "remote-embedding-rejected"
							? "datasource-embedding-egress-rejected"
							: "datasource-index-failed",
					result.reason,
				);
			}
			const indexed = await this.client.index(options?.signal);
			const diagnostics: DatasourceDiagnostic[] = [];
			if (!indexed.ok) {
				diagnostics.push({
					code:
						indexed.reason === "remote-embedding-rejected"
							? "datasource-embedding-egress-rejected"
							: "datasource-index-failed",
					severity: "warning",
					message: `mailcrawl semantic index unavailable (${indexed.reason}); BM25 remains available`,
					instanceId: this.instanceId,
					source: "mailcrawl",
				});
			}
			this.lastIndexedAt = Date.now();
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: "mailcrawl",
				chunkCount: result.data.chunksAdded ?? result.data.messages,
				indexedAt: this.lastIndexedAt,
				diagnostics,
			};
		} catch {
			return this.fail("datasource-unavailable", "mailcrawl sync failed unexpectedly");
		}
	}
	/** Hybrid search first; BM25 only when mailcrawl cannot run hybrid (for example, no usable semantic index). */
	retrievalMethods(): readonly RetrievalMethod[] {
		const hybrid = new MailcrawlMethod(this.client, "hybrid", this.instanceId, this.tags, this.account, this.mailbox);
		const bm25 = new MailcrawlMethod(this.client, "bm25", this.instanceId, this.tags, this.account, this.mailbox);
		return [new FallbackRetrievalMethod(hybrid, [bm25])];
	}
	/** Exact substring/regex grep over mailcrawl's parsed email chunks via read-only SQLite sidecar. */
	async grep(pattern: string, options?: DatasourceGrepOptions): Promise<readonly DatasourceGrepHit[]> {
		if (pattern.length === 0) return [];
		options?.signal?.throwIfAborted();
		const dir = this.dataDir ?? this.envDataDir ?? process.env.MAILCRAWL_DATA_DIR ?? ".mailcrawl";
		const resolvedDir = isAbsolute(dir) ? dir : resolve(dir);
		return grepMailcrawlArchiveSqlite({
			dbPath: join(resolvedDir, "archive.sqlite"),
			datasourceId: "mailcrawl",
			instanceId: this.instanceId,
			account: this.account,
			mailbox: this.mailbox,
			pattern,
			grepOptions: options,
		});
	}
	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath("mailcrawl", instanceId),
			datasourceId: "mailcrawl",
			skill: "mailcrawl",
			instanceId,
			contentType: "email",
			metadata: { datasourceId: "mailcrawl", instanceId, tags: this.tags },
		}));
	}
	skillManifest(): DatasourceSkillManifest {
		const scopes = this.instances.map((id) => `- \`${datasourceSourcePath("mailcrawl", id)}\``).join("\n");
		return {
			name: "datasource-mailcrawl",
			description: "Search local email synchronized and indexed by mailcrawl.",
			content: `# mailcrawl email datasource\n\nUse this skill for authorized local email retrieval. AutoRAG invokes the external \`mailcrawl\` CLI for search/sync and may read its archive read-only for exact-match \`grep\`.\n\n## Indexing\nIndexing is server-managed and incremental; do not trigger sync yourself.\n\n## Search\nCall the dedicated \`${datasourceSearchToolName("mailcrawl")}\` tool with a natural-language query, or call \`grep\` with \`path="/mailcrawl"\` for exact substring/regex matches (invoice numbers, order IDs, error codes). Available scopes:\n${scopes}\n\nThe selectable retrieval modes are BM25, semantic, and hybrid. Scope arguments can only narrow trusted access. Source identifiers remain opaque and hierarchical.\n\n## Native CLI\nThe external \`mailcrawl\` CLI owns the archive and index. When the dedicated tool cannot express what you need, call mailcrawl directly through \`bash\`:\n- \`mailcrawl search --mode hybrid --limit 20 --json "<query>"\` (\`bm25\` and \`semantic\` modes also available; \`--account\` / \`--mailbox\` narrow the archive)\n- \`mailcrawl grep --literal --ignore-case --json "<pattern>"\` (exact substring/regex match over parsed emails)\nNever pass datasource virtual paths (\`/mailcrawl/...\`) to bash; they are not OS paths.`,
		};
	}
	private fail(code: DatasourceDiagnosticCode, message: string): DatasourceIndexResult {
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: "mailcrawl",
			indexedAt: Date.now(),
			diagnostics: [
				{
					code,
					severity: code === "datasource-unavailable" ? "warning" : "error",
					message: `mailcrawl ${message}`,
					instanceId: this.instanceId,
					source: "mailcrawl",
				},
			],
			error: message,
			code,
			message,
		};
	}
}
