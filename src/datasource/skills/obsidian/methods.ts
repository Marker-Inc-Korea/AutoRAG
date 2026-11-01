import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { OBSIDIAN_DATASOURCE_ID } from "./paths.ts";
import type { ObsidianSearchMode, ObsidianVaultIndex } from "./vault-index.ts";

export interface ObsidianMethodOptions {
	readonly index: ObsidianVaultIndex;
	readonly tags: readonly string[];
}

const COMMON_CAPABILITIES = ["scoped", "incremental", "path-opaque-sources"] as const;

/** One MinSync query mode over one vault. A failure reaches the pipeline verbatim and is reported as unsearched. */
class ObsidianMinSyncMethod implements RetrievalMethod {
	private readonly index: ObsidianVaultIndex;
	private readonly tags: readonly string[];
	private readonly mode: ObsidianSearchMode;

	constructor(mode: ObsidianSearchMode, options: ObsidianMethodOptions) {
		this.mode = mode;
		this.index = options.index;
		this.tags = options.tags;
	}

	describe(): RetrievalMethodDescriptor {
		return this.mode === "sparse"
			? {
					name: "obsidian-sparse",
					type: "sparse",
					description: "SPLADE sparse retrieval over an Obsidian vault, indexed by MinSync",
					status: "active",
					capabilities: ["lexical", ...COMMON_CAPABILITIES],
					datasourceId: OBSIDIAN_DATASOURCE_ID,
					tags: this.tags,
				}
			: {
					name: "obsidian-semantic",
					type: "vector",
					description: "Semantic vector retrieval over an Obsidian vault, indexed by MinSync",
					status: "active",
					capabilities: ["semantic", ...COMMON_CAPABILITIES],
					datasourceId: OBSIDIAN_DATASOURCE_ID,
					tags: this.tags,
				};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return this.index.query(this.mode, query, {
			...(options.topK === undefined ? {} : { topK: options.topK }),
			...(options.scope === undefined ? {} : { scope: options.scope }),
			...(options.signal === undefined ? {} : { signal: options.signal }),
		});
	}
}

export class ObsidianSparseMethod extends ObsidianMinSyncMethod {
	constructor(options: ObsidianMethodOptions) {
		super("sparse", options);
	}
}

export class ObsidianSemanticMethod extends ObsidianMinSyncMethod {
	constructor(options: ObsidianMethodOptions) {
		super("semantic", options);
	}
}
