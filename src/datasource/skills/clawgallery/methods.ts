import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { matchesDatasourceScope } from "../../scope.ts";
import { CLAWGALLERY_DATASOURCE_ID, clawGallerySourcePath } from "./paths.ts";
import type {
	ClawGalleryHit,
	ClawGallerySearchMode,
	ClawGallerySearchOptions,
	ClawGallerySearchResult,
} from "./types.ts";

export interface ClawGallerySearchClient {
	search(
		mode: ClawGallerySearchMode,
		query: string,
		options?: ClawGallerySearchOptions,
	): Promise<ClawGallerySearchResult>;
}
export interface ClawGalleryMethodOptions {
	readonly client: ClawGallerySearchClient;
	readonly instanceId: string;
	readonly tags?: readonly string[];
}

export class ClawGalleryMethod implements RetrievalMethod {
	private readonly mode: ClawGallerySearchMode;
	private readonly client: ClawGallerySearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(
		mode: ClawGallerySearchMode,
		client: ClawGallerySearchClient,
		instanceId: string,
		tags: readonly string[],
	) {
		this.mode = mode;
		this.client = client;
		this.instanceId = instanceId;
		this.tags = tags;
	}
	describe(): RetrievalMethodDescriptor {
		return {
			name: `clawgallery-${this.mode}`,
			type: this.mode === "embedding" ? "vector" : this.mode === "hybrid" ? "hybrid" : "bm25",
			description: `ClawGallery ${this.mode} retrieval through the external CLI`,
			status: "active",
			capabilities: [this.mode, "scoped", "external-cli", "incremental"],
			datasourceId: CLAWGALLERY_DATASOURCE_ID,
			tags: [...this.tags],
		};
	}
	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (!trimmed) return [];
		let result: ClawGallerySearchResult;
		try {
			result = await this.client.search(this.mode, trimmed, { ...options, topK: options.topK ?? 20 });
		} catch {
			return [];
		}
		if (!result.ok) return [];
		const out: RetrievalResult[] = [];
		for (const hit of result.hits) {
			const source = clawGallerySourcePath(this.instanceId, hit.imageId);
			if (!matchesDatasourceScope(source, options.scope)) continue;
			if (
				options.allowedScopes?.length &&
				!options.allowedScopes.some((scope) => matchesDatasourceScope(source, scope))
			)
				continue;
			out.push(toResult(hit, source, this.mode, this.instanceId));
			if (out.length >= (options.topK ?? 20)) break;
		}
		return out;
	}
}

function toResult(
	hit: ClawGalleryHit,
	source: string,
	mode: ClawGallerySearchMode,
	instanceId: string,
): RetrievalResult {
	return {
		id: `clawgallery:${instanceId}:${hit.imageId}`,
		content: hit.content,
		source,
		score: hit.score,
		metadata: {
			...(hit.metadata ?? {}),
			datasourceId: CLAWGALLERY_DATASOURCE_ID,
			instanceId,
			mode,
			imageId: hit.imageId,
			...(hit.path ? { path: hit.path } : {}),
			...(hit.title ? { title: hit.title } : {}),
			...(hit.caption ? { caption: hit.caption } : {}),
		},
	};
}
