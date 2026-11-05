import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceCliError } from "../../errors.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";
import type { WombatSearchClient } from "./client.ts";
import type { WombatSearchHit, WombatSearchMode } from "./types.ts";

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
				content: `${sessionLabel(hit)}\n${hit.snippet}`,
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

/** A value placed in the one-line label: no line breaks, so a title or path cannot add lines to a result. */
function oneLine(value: string): string {
	return value.replace(/\s+/gu, " ").trim();
}

/**
 * The first line of every hit: which session the snippet came from. It carries what `wombat show` and
 * `wombat trace` take (`<agent>:<session id>`), the turn range to pass to `--turn`, and enough context
 * (date, title, project) to tell two sessions apart. Parts the CLI did not report are left out.
 */
function sessionLabel(hit: WombatSearchHit): string {
	const turns =
		hit.startTurn === undefined
			? undefined
			: hit.endTurn === undefined || hit.endTurn === hit.startTurn
				? `turn ${hit.startTurn}`
				: `turns ${hit.startTurn}-${hit.endTurn}`;
	const parts = [
		oneLine(hit.catalogKey),
		turns,
		hit.startedAt === undefined ? undefined : hit.startedAt.slice(0, 10),
		hit.title.length === 0 ? undefined : oneLine(hit.title),
		hit.cwd === undefined ? undefined : oneLine(hit.cwd),
	].filter((part): part is string => part !== undefined && part.length > 0);
	return `[${parts.join(" | ")}]`;
}
