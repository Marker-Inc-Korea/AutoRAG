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
import { QmdClient } from "./client.ts";
import {
	ObsidianBm25Method,
	type ObsidianSearchClient,
	ObsidianSemanticMethod,
} from "./methods.ts";
import { toQmdCollectionName } from "./paths.ts";
import type {
	QmdEmbedResult,
	QmdEnsureResult,
	QmdOptions,
	QmdUpdateResult,
} from "./types.ts";

export interface ObsidianSkillClient extends ObsidianSearchClient {
	ensureCollection(): Promise<QmdEnsureResult>;
	update(): Promise<QmdUpdateResult>;
	embed(): Promise<QmdEmbedResult>;
}

export interface ObsidianSkillOptions {
	readonly client?: ObsidianSkillClient;
	readonly vaultPath?: string;
	readonly binaryPath?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly lastIndexedAt?: number;
	readonly workspaceRoot?: string;
	readonly collectionName?: string;
	readonly timeoutMs?: number;
	readonly connectorOptions?: { readonly vaultPath?: string };
}

const OBSIDIAN_DATASOURCE_ID = "obsidian";
const OBSIDIAN_SKILL_TYPE = "obsidian-vault";
const DEFAULT_INSTANCE_ID = "default";
const DEFAULT_POLLING_INTERVAL_MS = 15 * 60 * 1000;
const DEFAULT_OBSIDIAN_TAGS = ["obsidian", "notes"] as const;

export class ObsidianSkill implements DatasourceSkill {
	private readonly client: ObsidianSkillClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly pollingIntervalMs: number;
	private readonly tags: readonly string[];
	private readonly vaultPath: string | undefined;
	private lastIndexedAt: number | undefined;
	private lastError: string | undefined;

	constructor(options: ObsidianSkillOptions = {}) {
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.pollingIntervalMs = options.pollingIntervalMs ?? DEFAULT_POLLING_INTERVAL_MS;
		this.tags = options.tags ?? DEFAULT_OBSIDIAN_TAGS;
		this.lastIndexedAt = options.lastIndexedAt;
		this.vaultPath = options.vaultPath ?? options.connectorOptions?.vaultPath;
		const clientOptions: QmdOptions = {
			...(options.binaryPath !== undefined ? { binaryPath: options.binaryPath } : {}),
			...(this.vaultPath !== undefined ? { vaultPath: this.vaultPath } : {}),
			...(options.workspaceRoot !== undefined ? { workspaceRoot: options.workspaceRoot } : {}),
			instanceId: this.instanceId,
			collectionName: options.collectionName ?? toQmdCollectionName(this.instanceId),
			...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
		};
		this.client = options.client ?? new QmdClient(clientOptions);
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: OBSIDIAN_DATASOURCE_ID,
			id: OBSIDIAN_DATASOURCE_ID,
			type: OBSIDIAN_SKILL_TYPE,
			description: "Obsidian vault datasource via the external qmd CLI",
			capabilities: ["notes", "external-cli", "polling", "bm25", "semantic", "incremental"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: OBSIDIAN_DATASOURCE_ID,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: this.pollingIntervalMs > 0 ? "poll" : "none",
			...(this.pollingIntervalMs > 0 ? { intervalMs: this.pollingIntervalMs } : {}),
			lastIndexedAt: this.lastIndexedAt,
			...(this.lastError !== undefined ? { lastError: this.lastError } : {}),
		};
	}

	async index(): Promise<DatasourceIndexResult> {
		try {
			const ensure = await this.client.ensureCollection();
			if (!ensure.ok) {
				return this.fail(qmdFailureCode(ensure.reason), ensure);
			}
			const update = await this.client.update();
			if (!update.ok) {
				return this.fail(qmdFailureCode(update.reason), update);
			}
			const diagnostics: DatasourceDiagnostic[] = [];
			const embed = await this.client.embed();
			if (!embed.ok) {
				diagnostics.push({
					code: "datasource-index-failed",
					severity: "warning",
					message: `qmd embed failed (${embed.reason}); lexical search remains available`,
					instanceId: this.instanceId,
					source: OBSIDIAN_DATASOURCE_ID,
				});
			}
			this.lastIndexedAt = Date.now();
			this.lastError = undefined;
			const chunkCount =
				update.data.indexed + update.data.updated + update.data.unchanged > 0
					? update.data.indexed + update.data.updated + update.data.unchanged
					: update.data.indexed + update.data.updated;
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: OBSIDIAN_DATASOURCE_ID,
				chunkCount,
				indexedAt: this.lastIndexedAt,
				diagnostics,
			};
		} catch {
			return this.fail("datasource-unavailable", {
				ok: false,
				reason: "spawn-error",
				stdout: "",
				stderr: "qmd command failed",
				code: null,
			});
		}
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [
			new ObsidianBm25Method({ client: this.client, instanceId: this.instanceId, tags: this.tags }),
			new ObsidianSemanticMethod({ client: this.client, instanceId: this.instanceId, tags: this.tags }),
		];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath(OBSIDIAN_DATASOURCE_ID, instanceId),
			datasourceId: OBSIDIAN_DATASOURCE_ID,
			skill: OBSIDIAN_DATASOURCE_ID,
			instanceId,
			contentType: "note",
			metadata: {
				datasourceId: OBSIDIAN_DATASOURCE_ID,
				instanceId,
				tags: this.tags,
				...(this.vaultPath !== undefined ? { vaultPath: this.vaultPath } : {}),
			},
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const instanceScopes = this.instances
			.map((instanceId) => `- \`${datasourceSourcePath(OBSIDIAN_DATASOURCE_ID, instanceId)}\``)
			.join("\n");
		const cadence =
			this.pollingIntervalMs > 0
				? `roughly every ${Math.round(this.pollingIntervalMs / 60000)} minute(s) when auto-refresh runs`
				: "on manual refresh only";
		return {
			name: `datasource-${OBSIDIAN_DATASOURCE_ID}`,
			description:
				"Search an indexed Obsidian vault (markdown notes) via qmd incremental BM25 + semantic indexes. Use for questions about personal notes and knowledge-base content.",
			content: [
				`# Obsidian vault datasource (${OBSIDIAN_SKILL_TYPE})`,
				"",
				"This skill indexes and searches an Obsidian vault through the external `qmd` CLI (incremental update, BM25 search, vector search). AutoRAG does not reimplement vault walking for query-time retrieval.",
				"",
				"## When to use",
				"Use this skill when the question is about Obsidian notes, vault folders, tags, or knowledge-base content.",
				"",
				"## Indexing",
				`Indexing is server-managed and refreshed ${cadence} via \`qmd update\` (+ \`qmd embed\` for semantic). You do not trigger indexing; just search.`,
				"",
				"## How to search",
				"Call `search_datasource_documents` with a natural-language `query`. Optionally pass `topK` and a narrowing `scope`. Available authorized scopes:",
				instanceScopes.length > 0 ? instanceScopes : "- (no authorized instances)",
				"",
				"`scope` can only narrow within already-authorized scopes; it can never widen access.",
				"",
				"## Output rules",
				"Datasource source identifiers such as `/obsidian/<instance>/chunks/<id>` are stable. Result metadata may carry real vault file paths; cite them when helpful. Privacy is the operator's responsibility: run AutoRAG with a local LLM if results must not leave this machine.",
			].join("\n"),
		};
	}

	private fail(
		code: DatasourceDiagnosticCode,
		result: { ok: false; reason: string; stdout?: string; stderr: string; code: number | null },
	): DatasourceIndexResult {
		const message =
			result.stderr.length > 0 ? `${result.reason}: ${boundDiagnostic(result.stderr)}` : result.reason;
		this.lastError = message;
		const diagnostic: DatasourceDiagnostic = {
			code,
			severity: code === "datasource-unavailable" || code === "datasource-empty" ? "warning" : "error",
			message,
			instanceId: this.instanceId,
			source: OBSIDIAN_DATASOURCE_ID,
		};
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: OBSIDIAN_DATASOURCE_ID,
			indexedAt: Date.now(),
			diagnostics: [diagnostic],
			error: result.reason,
			code,
			message,
		};
	}
}

function qmdFailureCode(reason: string): DatasourceDiagnosticCode {
	if (reason === "not-configured") return "datasource-unavailable";
	if (reason === "binary-missing") return "datasource-unavailable";
	return "datasource-index-failed";
}

function boundDiagnostic(value: string): string {
	const trimmed = value.trim();
	return trimmed.length > 500 ? `${trimmed.slice(0, 500)}...` : trimmed;
}
