/**
 * Shared {@link DatasourceSkill} implementation for connector-backed
 * datasources (Slack, Discord, Notion, GitHub, Google Drive, Gmail/IMAP,
 * local mail exports, Obsidian vaults, RSS/news).
 *
 * Composition: a trusted {@link DatasourceConnector} fetches documents, a
 * {@link DatasourceChunkStore} persists and lexically indexes them, and this
 * class exposes the standard skill surface — descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, retrieval methods routed
 * through the shared pipeline, opaque source descriptions, and a
 * progressive-disclosure agent-skill manifest.
 *
 * Security invariants:
 *  - Constructed only from trusted, server-supplied configuration.
 *  - Sources are opaque slash-hierarchical paths `/<skill>/<instance>/…`.
 *  - `index()` never throws; failures surface as sanitized diagnostics.
 *  - Retrieval honors `scope` and `allowedScopes` narrowing; tool arguments
 *    can never widen access (enforced upstream by DatasourceAccessContext).
 */

import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import { DatasourceChunkStore } from "./chunk-store.ts";
import type { ConnectorDocument, DatasourceConnector } from "./connector.ts";
import { connectorFailureToDiagnosticCode, sanitizeOpaqueText } from "./connector.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "./scope.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "./types.ts";

/** Static identity/behavior shared by all instances of one skill kind. */
export interface ConnectorSkillDefinition {
	/** Stable skill name and datasource id, e.g. `"slack"`. */
	readonly skillName: string;
	/** Skill kind, e.g. `"slack-workspace"`. */
	readonly skillType: string;
	/** One-line descriptor description. */
	readonly description: string;
	/** Capability tags such as `"chat"`, `"api"`, `"polling"`. */
	readonly capabilities: readonly string[];
	/** Default authorization tags when the server config supplies none. */
	readonly defaultTags: readonly string[];
	/** Content type reported on source descriptions, e.g. `"chat"`. */
	readonly contentType: string;
	/** Short "when to use" text for the system-prompt skills block. */
	readonly manifestDescription: string;
	/** Extra path-opaque manifest body lines (after the standard sections). */
	readonly manifestNotes?: readonly string[];
}

export interface ConnectorSkillOptions {
	readonly connector: DatasourceConnector;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	/** Poll interval in ms. Default 15 minutes; `0` disables polling. */
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly lastIndexedAt?: number;
	/** Workspace root for chunk persistence; omitted ⇒ in-memory only. */
	readonly workspaceRoot?: string;
	readonly maxChunkChars?: number;
	/**
	 * Dedupe window in ms: when > 0, documents whose `docId` was already
	 * indexed within the window keep their previous timestamp identity and
	 * duplicates inside one fetch are dropped (RSS/news re-delivery).
	 */
	readonly dedupeWindowMs?: number;
}

const DEFAULT_POLLING_INTERVAL_MS = 15 * 60 * 1000;
const DEFAULT_INSTANCE_ID = "default";

export class ConnectorDatasourceSkill implements DatasourceSkill {
	protected readonly definition: ConnectorSkillDefinition;
	protected readonly connector: DatasourceConnector;
	protected readonly instanceId: string;
	protected readonly instances: readonly string[];
	protected readonly pollingIntervalMs: number;
	protected readonly tags: readonly string[];
	protected readonly store: DatasourceChunkStore;
	protected readonly dedupeWindowMs: number;
	private lastIndexedAt: number | undefined;
	private lastPolledAt: number | undefined;
	private lastError: string | undefined;
	/** docId → last-seen epoch ms, for the dedupe window. */
	private readonly seenDocs = new Map<string, number>();

	constructor(definition: ConnectorSkillDefinition, options: ConnectorSkillOptions) {
		this.definition = definition;
		this.connector = options.connector;
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.pollingIntervalMs = options.pollingIntervalMs ?? DEFAULT_POLLING_INTERVAL_MS;
		this.tags = options.tags ?? definition.defaultTags;
		this.lastIndexedAt = options.lastIndexedAt;
		this.dedupeWindowMs = options.dedupeWindowMs ?? 0;
		this.store = new DatasourceChunkStore({
			skillName: definition.skillName,
			instanceId: this.instanceId,
			workspaceRoot: options.workspaceRoot,
			maxChunkChars: options.maxChunkChars,
		});
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: this.definition.skillName,
			id: this.definition.skillName,
			type: this.definition.skillType,
			description: this.definition.description,
			capabilities: this.definition.capabilities,
			tags: this.tags,
			status: "active",
			datasourceId: this.definition.skillName,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: this.pollingIntervalMs > 0 ? "poll" : "none",
			...(this.pollingIntervalMs > 0 ? { intervalMs: this.pollingIntervalMs } : {}),
			lastIndexedAt: this.lastIndexedAt,
			...(this.lastPolledAt !== undefined ? { lastPolledAt: this.lastPolledAt } : {}),
			...(this.lastError !== undefined ? { lastError: this.lastError } : {}),
		};
	}

	async index(): Promise<DatasourceIndexResult> {
		this.lastPolledAt = Date.now();
		let fetched: Awaited<ReturnType<DatasourceConnector["fetch"]>>;
		try {
			fetched = await this.connector.fetch();
		} catch {
			return this.fail("datasource-unavailable", "connector failed unexpectedly");
		}
		if (!fetched.ok) {
			const code = connectorFailureToDiagnosticCode(fetched.reason);
			const message = sanitizeOpaqueText(fetched.message ?? fetched.reason);
			return this.fail(code, `${fetched.reason}: ${message}`);
		}
		const documents = this.applyDedupeWindow(fetched.documents);
		const chunkCount = this.store.replaceDocuments(documents);
		this.lastIndexedAt = Date.now();
		this.lastError = undefined;
		const diagnostics: DatasourceDiagnostic[] = (fetched.warnings ?? []).map((warning) => ({
			code: "datasource-index-failed",
			severity: "warning",
			message: sanitizeOpaqueText(warning),
			instanceId: this.instanceId,
			source: this.definition.skillName,
		}));
		if (chunkCount === 0) {
			diagnostics.push({
				code: "datasource-empty",
				severity: "info",
				message: "datasource fetch returned no indexable documents",
				instanceId: this.instanceId,
				source: this.definition.skillName,
			});
		}
		return {
			ok: true,
			instanceId: this.instanceId,
			skill: this.definition.skillName,
			chunkCount,
			indexedAt: this.lastIndexedAt,
			diagnostics,
		};
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [
			new ConnectorLexicalMethod({
				skillName: this.definition.skillName,
				skillType: this.definition.skillType,
				instanceId: this.instanceId,
				tags: this.tags,
				store: this.store,
			}),
		];
	}

	describeSources(): readonly SourceDescription[] {
		const out: SourceDescription[] = this.instances.map((instanceId) => ({
			source: datasourceSourcePath(this.definition.skillName, instanceId),
			datasourceId: this.definition.skillName,
			skill: this.definition.skillName,
			instanceId,
			contentType: this.definition.contentType,
			metadata: {
				datasourceId: this.definition.skillName,
				instanceId,
				tags: this.tags,
			},
		}));
		for (const hierarchy of this.store.hierarchies()) {
			if (hierarchy.length === 0) continue;
			out.push({
				source: `${datasourceSourcePath(this.definition.skillName, this.instanceId)}/${hierarchy.join("/")}`,
				datasourceId: this.definition.skillName,
				skill: this.definition.skillName,
				instanceId: this.instanceId,
				contentType: this.definition.contentType,
				metadata: {
					datasourceId: this.definition.skillName,
					instanceId: this.instanceId,
					hierarchy,
				},
			});
		}
		return out;
	}

	skillManifest(): DatasourceSkillManifest {
		const skillName = this.definition.skillName;
		const instanceScopes = this.instances
			.map((instanceId) => `- \`${datasourceSourcePath(skillName, instanceId)}\``)
			.join("\n");
		const cadence =
			this.pollingIntervalMs > 0
				? `roughly every ${Math.round(this.pollingIntervalMs / 60000)} minute(s) when auto-refresh runs`
				: "on manual refresh only";
		return {
			name: `datasource-${skillName}`,
			description: this.definition.manifestDescription,
			content: [
				`# ${this.definition.description} (${this.definition.skillType})`,
				"",
				"## When to use",
				this.definition.manifestDescription,
				"",
				"## Indexing",
				`Indexing is server-managed and refreshed ${cadence}. You do not trigger indexing; just search.`,
				"",
				"## How to search",
				"Call `search_datasource_documents` with a natural-language `query`. Optionally pass `topK` and a narrowing `scope`. Available authorized scopes:",
				instanceScopes.length > 0 ? instanceScopes : "- (no authorized instances)",
				"",
				"`scope` can only narrow within already-authorized scopes; it can never widen access.",
				"",
				"## Output rules",
				`Datasource source identifiers such as \`/${skillName}/<instance>/chunks/<id>\` are internal and opaque. Never put them, real file paths, tokens, account IDs, or e-mail addresses in the visible answer.`,
				...(this.definition.manifestNotes !== undefined && this.definition.manifestNotes.length > 0
					? ["", ...this.definition.manifestNotes]
					: []),
			].join("\n"),
		};
	}

	private applyDedupeWindow(documents: readonly ConnectorDocument[]): readonly ConnectorDocument[] {
		if (this.dedupeWindowMs <= 0) return documents;
		const now = Date.now();
		for (const [docId, seenAt] of this.seenDocs) {
			if (now - seenAt > this.dedupeWindowMs) this.seenDocs.delete(docId);
		}
		const out: ConnectorDocument[] = [];
		const inBatch = new Set<string>();
		for (const document of documents) {
			if (inBatch.has(document.docId)) continue;
			inBatch.add(document.docId);
			this.seenDocs.set(document.docId, now);
			out.push(document);
		}
		return out;
	}

	private fail(code: DatasourceDiagnosticCode, message: string): DatasourceIndexResult {
		const sanitized = sanitizeOpaqueText(message);
		this.lastError = sanitized;
		const diagnostic: DatasourceDiagnostic = {
			code,
			severity: code === "datasource-unavailable" || code === "datasource-empty" ? "warning" : "error",
			message: sanitized,
			instanceId: this.instanceId,
			source: this.definition.skillName,
		};
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: this.definition.skillName,
			indexedAt: Date.now(),
			diagnostics: [diagnostic],
			error: sanitized,
			code,
			message: sanitized,
		};
	}
}

interface ConnectorLexicalMethodOptions {
	readonly skillName: string;
	readonly skillType: string;
	readonly instanceId: string;
	readonly tags: readonly string[];
	readonly store: DatasourceChunkStore;
}

const DEFAULT_TOP_K = 20;

/**
 * Lexical retrieval over a connector skill's chunk store. Failures collapse
 * to an empty result set; retrieval never throws. Sources are opaque
 * `/<skill>/<instance>/chunks/<chunk-id>` paths and are matched against both
 * the user scope and the trusted allowed scopes.
 */
export class ConnectorLexicalMethod implements RetrievalMethod {
	private readonly options: ConnectorLexicalMethodOptions;

	constructor(options: ConnectorLexicalMethodOptions) {
		this.options = options;
	}

	describe(): RetrievalMethodDescriptor {
		const { skillName, skillType, tags } = this.options;
		return {
			name: `${skillName}-lexical`,
			type: "bm25",
			description: `Lexical retrieval over indexed ${skillType} chunks`,
			status: "active",
			capabilities: ["lexical", "scoped", "path-opaque-sources"],
			datasourceId: skillName,
			tags: [...tags],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		const topK = options.topK ?? DEFAULT_TOP_K;
		let hits: ReturnType<DatasourceChunkStore["search"]>;
		try {
			hits = this.options.store.search(trimmed, Math.max(topK * 3, topK));
		} catch {
			return [];
		}
		const { skillName, instanceId } = this.options;
		const mapped: RetrievalResult[] = [];
		for (const { chunk, score } of hits) {
			const source = datasourceSourcePath(skillName, instanceId, chunk.chunkId);
			if (!this.matchesScope(source, options.scope, options.allowedScopes)) continue;
			mapped.push({
				id: `${skillName}:${instanceId}:${chunk.chunkId}`,
				content: chunk.content,
				source,
				score,
				metadata: {
					...chunk.metadata,
					method: `${skillName}-lexical`,
					datasourceId: skillName,
					instanceId,
					chunkId: chunk.chunkId,
					...(chunk.title !== undefined ? { title: chunk.title } : {}),
					...(chunk.hierarchy.length > 0 ? { hierarchy: chunk.hierarchy.join("/") } : {}),
					...(chunk.publishedAt !== undefined ? { publishedAt: chunk.publishedAt } : {}),
				},
			});
			if (mapped.length >= topK) break;
		}
		return mapped;
	}

	private matchesScope(
		source: string,
		scope: string | undefined,
		allowedScopes: readonly string[] | undefined,
	): boolean {
		if (!matchesDatasourceScope(source, scope)) return false;
		if (allowedScopes === undefined || allowedScopes.length === 0) return true;
		return allowedScopes.some((entry) => matchesDatasourceScope(source, entry));
	}
}
