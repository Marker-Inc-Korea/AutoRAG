import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { datasourceSourcePath } from "../../scope.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
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
	async index(): Promise<DatasourceIndexResult> {
		try {
			const result = await this.client.sync();
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
			const indexed = await this.client.index();
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
	retrievalMethods(): readonly RetrievalMethod[] {
		return [
			new MailcrawlMethod(this.client, "bm25", this.instanceId, this.tags, this.account, this.mailbox),
			new MailcrawlMethod(this.client, "semantic", this.instanceId, this.tags, this.account, this.mailbox),
			new MailcrawlMethod(this.client, "hybrid", this.instanceId, this.tags, this.account, this.mailbox),
		];
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
		return {
			name: "datasource-mailcrawl",
			description: "Search local email synchronized and indexed by mailcrawl.",
			content: [
				"# mailcrawl email datasource",
				"",
				"AutoRAG invokes the external `mailcrawl` CLI and never opens its SQLite archive directly.",
				"",
				"## Indexing",
				"Indexing is incremental via `mailcrawl sync`; do not trigger sync yourself.",
				"",
				"## How to search",
				"Call `search_datasource_documents` with a natural-language query and `topK`.",
				"",
				"## Native CLI",
				"You can also invoke `mailcrawl` directly through `bash` when you need its full surface:",
				"- `mailcrawl search <query> --json` — BM25 search",
				"- `mailcrawl search <query> --semantic --json` — semantic search",
				"- `mailcrawl doctor --json` — health check",
				"- `mailcrawl --help` — full command reference",
				"",
				"## Configuration",
				"mailcrawl uses its own native archive by default. Server-side connector overrides: `dataDir` (archive location), `account`/`mailbox` (sync scope), `backend`/`source` (ingest backend), `himalayaConfig` (explicit himalaya config), `binaryPath`.",
				"",
				"Source identifiers are opaque and hierarchical.",
			].join("\n"),
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
