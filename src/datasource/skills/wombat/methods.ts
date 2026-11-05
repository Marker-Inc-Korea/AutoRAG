import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceCliError } from "../../errors.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";
import type { WombatSearchClient } from "./client.ts";
import type { WombatSearchMode } from "./types.ts";

export const WOMBAT_MODES: readonly WombatSearchMode[] = ["keyword", "semantic", "hybrid"];

const METHOD_TYPE: Readonly<Record<WombatSearchMode, "bm25" | "vector" | "hybrid">> = {
	keyword: "bm25",
	semantic: "vector",
	hybrid: "hybrid",
};

/** Retrieval over indexed coding-agent sessions (Codex, Claude Code, OpenCode, pi, omp) through the external wombat CLI. */
export class WombatMethod implements RetrievalMethod {
	private readonly client: WombatSearchClient;
	private readonly mode: WombatSearchMode;
	private readonly instanceId: string;
	private readonly tags: readonly string[];
	private readonly agents: readonly string[] | undefined;

	constructor(
		client: WombatSearchClient,
		mode: WombatSearchMode,
		instanceId: string,
		tags: readonly string[],
		agents?: readonly string[],
	) {
		this.client = client;
		this.mode = mode;
		this.instanceId = instanceId;
		this.tags = tags;
		this.agents = agents;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: `wombat-${this.mode}`,
			type: METHOD_TYPE[this.mode],
			description: `${this.mode} retrieval over coding-agent session conversations via the external wombat CLI`,
			status: "active",
			capabilities: ["coding-agent", this.mode, "scoped", "external-cli", "path-opaque-sources"],
			datasourceId: "wombat",
			tags: this.tags,
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		if (!query.trim()) return [];
		const topK = options.topK ?? 20;
		// An agent allow-list is applied after the CLI answers, so over-fetch to keep topK results after it.
		const result = await this.client.search(this.mode, query, {
			topK: this.agents === undefined ? topK : topK * 4,
			signal: options.signal,
		});
		// A wombat failure reaches the caller verbatim: the pipeline reports this datasource as unsearched.
		if (!result.ok) throw datasourceCliError("wombat", `search --mode ${this.mode}`, result);
		return result.hits
			.filter((hit) => {
				if (this.agents !== undefined && !this.agents.includes(hit.agent)) return false;
				return matchesDatasourceScope(datasourceSourcePath("wombat", this.instanceId, hit.chunkId), options.scope);
			})
			.map((hit) => ({
				id: `wombat:${this.instanceId}:${hit.chunkId}`,
				content: hit.snippet,
				source: datasourceSourcePath("wombat", this.instanceId, hit.chunkId),
				score: hit.score,
				metadata: {
					method: `wombat-${this.mode}`,
					datasourceId: "wombat",
					instanceId: this.instanceId,
					mode: this.mode,
					chunkId: hit.chunkId,
					agent: hit.agent,
					sessionId: hit.sessionId,
					sessionKey: hit.catalogKey,
					title: hit.title,
					cwd: hit.cwd,
					sessionPath: hit.sourcePath,
					startTurn: hit.startTurn,
					endTurn: hit.endTurn,
					startedAt: hit.startedAt,
					endedAt: hit.endedAt,
				},
			}))
			.slice(0, topK);
	}
}
