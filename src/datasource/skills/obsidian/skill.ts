import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { datasourceSearchToolName } from "../../tool-naming.ts";
import type {
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "../../types.ts";
import { ObsidianBm25Method, ObsidianSemanticMethod } from "./methods.ts";
import { OBSIDIAN_DATASOURCE_ID, OBSIDIAN_SKILL_TYPE, obsidianInstanceSource } from "./paths.ts";
import { type ObsidianMinSyncOptions, ObsidianVaultIndex } from "./vault-index.ts";

export interface ObsidianSkillOptions {
	readonly datasourceId?: string;
	readonly vaultPath?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly lastIndexedAt?: number;
	/** Where `.autorag/datasources/obsidian/<instance>` is kept. Required to index or search. */
	readonly workspaceRoot?: string;
	/** The install's MinSync binary, embedder and chunking, shared with the local-file index. */
	readonly minSync?: ObsidianMinSyncOptions;
	readonly connectorOptions?: { readonly vaultPath?: string };
}

const DEFAULT_INSTANCE_ID = "default";
const DEFAULT_POLLING_INTERVAL_MS = 15 * 60 * 1000;
const DEFAULT_OBSIDIAN_TAGS = ["obsidian", "notes"] as const;

/**
 * An Obsidian vault as a datasource: its markdown notes are indexed by MinSync (BM25 and
 * semantic) in a workspace of their own, and a note's source is its vault-relative path, so a
 * folder is an ordinary `scope`.
 */
export class ObsidianSkill implements DatasourceSkill {
	private readonly vaultIndex: ObsidianVaultIndex;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly pollingIntervalMs: number;
	private readonly tags: readonly string[];
	private lastIndexedAt: number | undefined;
	private lastError: string | undefined;

	constructor(options: ObsidianSkillOptions = {}) {
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.pollingIntervalMs = options.pollingIntervalMs ?? DEFAULT_POLLING_INTERVAL_MS;
		this.tags = options.tags ?? DEFAULT_OBSIDIAN_TAGS;
		this.lastIndexedAt = options.lastIndexedAt;
		this.vaultIndex = new ObsidianVaultIndex({
			instanceId: this.instanceId,
			vaultPath: options.vaultPath ?? options.connectorOptions?.vaultPath,
			workspaceRoot: options.workspaceRoot,
			...(options.minSync === undefined ? {} : { minSync: options.minSync }),
		});
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: OBSIDIAN_DATASOURCE_ID,
			id: OBSIDIAN_DATASOURCE_ID,
			type: OBSIDIAN_SKILL_TYPE,
			description: "Obsidian vault datasource, indexed by MinSync",
			capabilities: ["notes", "polling", "lexical", "semantic", "incremental", "scoped"],
			tags: this.tags,
			status: "active",
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
			...(this.lastError === undefined ? {} : { lastError: this.lastError }),
		};
	}

	async index(options?: DatasourceIndexOptions): Promise<DatasourceIndexResult> {
		const outcome = await this.vaultIndex.sync(options?.signal);
		const indexedAt = Date.now();
		if (!outcome.ok) {
			this.lastError = outcome.message;
			return {
				ok: false,
				instanceId: this.instanceId,
				skill: OBSIDIAN_DATASOURCE_ID,
				indexedAt,
				diagnostics: [
					{
						code: outcome.code,
						severity: outcome.code === "datasource-index-failed" ? "error" : "warning",
						message: outcome.message,
						instanceId: this.instanceId,
						source: OBSIDIAN_DATASOURCE_ID,
					},
				],
				error: outcome.code,
				code: outcome.code,
				message: outcome.message,
			};
		}
		this.lastIndexedAt = indexedAt;
		this.lastError = undefined;
		return {
			ok: true,
			instanceId: this.instanceId,
			skill: OBSIDIAN_DATASOURCE_ID,
			chunkCount: outcome.notes,
			indexedAt,
			diagnostics: outcome.diagnostics,
		};
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		const methodOptions = { index: this.vaultIndex, tags: this.tags };
		return [new ObsidianBm25Method(methodOptions), new ObsidianSemanticMethod(methodOptions)];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: obsidianInstanceSource(instanceId),
			datasourceId: OBSIDIAN_DATASOURCE_ID,
			skill: OBSIDIAN_DATASOURCE_ID,
			instanceId,
			contentType: "note",
			metadata: { datasourceId: OBSIDIAN_DATASOURCE_ID, instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const instanceScopes = this.instances
			.map((instanceId) => `- \`${obsidianInstanceSource(instanceId)}\``)
			.join("\n");
		const folderExample = `${obsidianInstanceSource(this.instances[0] ?? this.instanceId)}/<folder>`;
		const cadence =
			this.pollingIntervalMs > 0
				? `roughly every ${Math.round(this.pollingIntervalMs / 60000)} minute(s) when auto-refresh runs`
				: "on manual refresh only";
		return {
			name: `datasource-${OBSIDIAN_DATASOURCE_ID}`,
			description:
				"Search an Obsidian vault (markdown notes) with BM25 and semantic retrieval, optionally limited to a folder. Use for questions about personal notes and knowledge-base content.",
			content: [
				`# Obsidian vault datasource (${OBSIDIAN_SKILL_TYPE})`,
				"",
				"The vault's markdown notes are indexed by MinSync in a workspace of their own, separate from the local-file index. Hidden folders (`.obsidian`, `.trash`), non-markdown attachments, and symlinks are not indexed.",
				"",
				"## When to use",
				"Use this skill when the question is about Obsidian notes, vault folders, or knowledge-base content.",
				"",
				"## Indexing",
				`Indexing is server-managed and refreshed ${cadence}; only new, changed, and deleted notes are re-indexed. You do not trigger indexing; just search.`,
				"",
				"## How to search",
				`Call the dedicated \`${datasourceSearchToolName(OBSIDIAN_DATASOURCE_ID)}\` tool with a natural-language \`query\`. Optionally pass \`topK\` and a \`scope\`. Each note's source is its vault-relative path, so a folder is a scope. Available vault scopes:`,
				instanceScopes.length > 0 ? instanceScopes : "- (no vaults configured)",
				"",
				`Narrow to a folder with \`${folderExample}\`: it matches that folder and everything below it. A glob needs an explicit trailing \`/**\`, for example \`${obsidianInstanceSource(this.instances[0] ?? this.instanceId)}/**/Daily/**\`. \`#\`, \`%\`, \`*\`, and \`\\\` in a note or folder name appear as \`%23\`, \`%25\`, \`%2A\`, and \`%5C\`.`,
				"",
				"## Output rules",
				"Result metadata carries the note's `file` (vault-relative path), `title`, and real `path`; cite them when they help the user find the note. Privacy is the operator's responsibility: run AutoRAG with a local LLM if results must not leave this machine.",
			].join("\n"),
		};
	}
}
