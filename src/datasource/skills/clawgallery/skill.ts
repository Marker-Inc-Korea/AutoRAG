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
import { ClawGalleryMethod, type ClawGallerySearchClient } from "./methods.ts";
import { CLAWGALLERY_DATASOURCE_ID, CLAWGALLERY_SOURCE_KIND } from "./paths.ts";
import type {
	ClawGalleryIndexResult,
	ClawGalleryOptions,
	ClawGallerySearchMode,
	ClawGalleryVdrResult,
} from "./types.ts";

export interface ClawGallerySkillClient extends ClawGallerySearchClient {
	bootstrap(): Promise<ClawGalleryIndexResult>;
	syncVisual(): Promise<ClawGalleryVdrResult>;
}
export interface ClawGallerySkillOptions {
	readonly client: ClawGallerySkillClient;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly defaultMode?: ClawGallerySearchMode;
	readonly syncVisual?: boolean;
	readonly vdrBackend?: ClawGalleryOptions["vdrBackend"];
}

const DEFAULT_INSTANCE = "default";
const DEFAULT_INTERVAL = 15 * 60 * 1000;
const DEFAULT_TAGS = ["clawgallery", "screenshots", "images"] as const;

export class ClawGallerySkill implements DatasourceSkill {
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly tags: readonly string[];
	private readonly mode: ClawGallerySearchMode;
	private readonly interval: number;
	private readonly sync: boolean;
	private lastIndexedAt: number | undefined;
	private readonly options: ClawGallerySkillOptions;

	constructor(options: ClawGallerySkillOptions) {
		this.options = options;
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE;
		this.instances = options.instances?.length ? options.instances : [this.instanceId];
		this.tags = options.tags ?? DEFAULT_TAGS;
		this.mode = options.defaultMode ?? "hybrid";
		this.interval = options.pollingIntervalMs ?? DEFAULT_INTERVAL;
		this.sync = options.syncVisual ?? true;
	}
	describe(): DatasourceSkillDescriptor {
		return {
			name: CLAWGALLERY_DATASOURCE_ID,
			id: CLAWGALLERY_DATASOURCE_ID,
			type: "clawgallery",
			description: "Screenshot and photo gallery via the external ClawGallery CLI",
			capabilities: ["images", "external-cli", "polling", "incremental", "keyword", "lexical", "semantic", "hybrid"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: CLAWGALLERY_DATASOURCE_ID,
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
		const bootstrap = await this.options.client.bootstrap();
		if (!bootstrap.ok) {
			const code = codeFor(bootstrap, "datasource-index-failed");
			const message = `clawgallery bootstrap failed (${bootstrap.reason})`;
			return {
				ok: false,
				instanceId: this.instanceId,
				skill: CLAWGALLERY_DATASOURCE_ID,
				indexedAt: Date.now(),
				diagnostics: [
					{
						code,
						severity: code === "datasource-unavailable" ? "warning" : "error",
						message,
						instanceId: this.instanceId,
						source: CLAWGALLERY_DATASOURCE_ID,
					},
				],
				error: message,
				code,
				message,
			};
		}
		const diagnostics: DatasourceDiagnostic[] = [];
		if (this.sync) {
			const visual = await this.options.client.syncVisual();
			if (!visual.ok)
				diagnostics.push({
					code: codeFor(visual, "datasource-index-failed"),
					severity: "warning",
					message: `clawgallery visual sync failed (${visual.reason}); existing search index remains available`,
					instanceId: this.instanceId,
					source: CLAWGALLERY_DATASOURCE_ID,
				});
			else if (visual.data.failed > 0)
				diagnostics.push({
					code: "datasource-index-failed",
					severity: "warning",
					message: `clawgallery visual sync left ${visual.data.failed} failed image(s)`,
					instanceId: this.instanceId,
					source: CLAWGALLERY_DATASOURCE_ID,
				});
		}
		this.lastIndexedAt = Date.now();
		if (bootstrap.data.indexed === 0)
			diagnostics.push({
				code: "datasource-empty",
				severity: "info",
				message: "clawgallery bootstrap found no changed images",
				instanceId: this.instanceId,
				source: CLAWGALLERY_DATASOURCE_ID,
			});
		return {
			ok: true,
			instanceId: this.instanceId,
			skill: CLAWGALLERY_DATASOURCE_ID,
			chunkCount: bootstrap.data.indexed,
			indexedAt: this.lastIndexedAt,
			diagnostics,
		};
	}
	retrievalMethods(): readonly RetrievalMethod[] {
		const order: ClawGallerySearchMode[] =
			this.mode === "keyword"
				? ["keyword", "hybrid", "lexical", "embedding"]
				: this.mode === "lexical"
					? ["lexical", "hybrid", "keyword", "embedding"]
					: this.mode === "embedding"
						? ["embedding", "hybrid", "keyword", "lexical"]
						: ["hybrid", "keyword", "lexical", "embedding"];
		return order.map((mode) => new ClawGalleryMethod(mode, this.options.client, this.instanceId, this.tags));
	}
	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath(CLAWGALLERY_SOURCE_KIND, instanceId),
			datasourceId: CLAWGALLERY_DATASOURCE_ID,
			skill: CLAWGALLERY_DATASOURCE_ID,
			instanceId,
			contentType: "image",
			metadata: { datasourceId: CLAWGALLERY_DATASOURCE_ID, instanceId, tags: this.tags },
		}));
	}
	skillManifest(): DatasourceSkillManifest {
		return {
			name: `datasource-${CLAWGALLERY_DATASOURCE_ID}`,
			description:
				"Search screenshots and photos by filename, caption, text-like visual concepts, or semantic visual meaning.",
			content: [
				"# ClawGallery datasource (clawgallery)",
				"",
				"The external `clawgallery` CLI owns image discovery, captions, V-SPLADE, and VDR state. AutoRAG never reads images.jsonl or vdr.sqlite3.",
				"",
				"Indexing is incremental: refresh runs `bootstrap`, then trusted configuration may run `vdr sync`. Do not trigger captioning or renaming from AutoRAG.",
				"",
				"## Search mode rules",
				"",
				"- `keyword` searches image paths and captions and remains available without a vector index.",
				"- `lexical` requires a sparse V-SPLADE index created with `clawgallery vdr sync --backend vsplade`.",
				"- `embedding` requires a dense VDR index created with `clawgallery vdr sync --backend mlx` (or another dense backend).",
				"- Never use `embedding` just because a V-SPLADE index exists: V-SPLADE is sparse lexical retrieval, not dense embedding retrieval.",
				"- Never use `lexical` unless a sparse V-SPLADE index exists; it must not fall back to dense vectors.",
				"- `hybrid` is the default and asks ClawGallery to RRF-fuse keyword results with every available dense and sparse vector channel.",
				"",
				"Use `search_datasource_documents` with a natural-language query. Prefer hybrid/default search unless you know which index capability is available. If an explicit vector mode is needed, inspect `clawgallery vdr status --json` first when capability details are available.",
				"",
				"With only V-SPLADE synced, use lexical or hybrid. With only dense VDR synced, use embedding or hybrid. When both are synced, use hybrid so dense VDR and sparse V-SPLADE results can coexist and be RRF-combined.",
				"",
				`Optionally narrow \`scope\` to an authorized \`/${CLAWGALLERY_SOURCE_KIND}/<instance>/**\`.`,
			].join("\n"),
		};
	}
}

function codeFor(result: { readonly reason: string }, fallback: DatasourceDiagnosticCode): DatasourceDiagnosticCode {
	return ["binary-missing", "spawn-error", "timeout", "aborted"].includes(result.reason)
		? "datasource-unavailable"
		: fallback;
}
