import type { Workspace } from "@nomadamas/agentdir";
import { agentdirGrep } from "../../agentdir/grep-core.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

/** Workspace handle or a (possibly async) provider resolved at retrieve time. */
export type WorkspaceProvider = Workspace | (() => Workspace | Promise<Workspace>);

/**
 * POSIX-style content retrieval over the agentdir virtual tree.
 *
 * Wraps the shared `agentdirGrep` core (the same code that backs the `grep`
 * agent tool) as a pluggable `RetrievalMethod`, so the previously-dead
 * RetrievalMethodRegistry / ParallelRetriever / ResultMerger pipeline becomes
 * a live retrieval path. Results carry virtual paths only — never source paths.
 */
export class AgentdirPosixMethod implements RetrievalMethod {
	private readonly provider: WorkspaceProvider;

	constructor(provider: WorkspaceProvider) {
		this.provider = provider;
	}

	private async workspace(): Promise<Workspace> {
		return typeof this.provider === "function" ? this.provider() : this.provider;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "posix",
			type: "posix",
			description: "agentdir virtual-tree content search (rglob + readBytes + regex)",
			status: "active",
			capabilities: ["regex", "literal", "glob-scope", "virtual-paths"],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const ws = await this.workspace();
		const hits = await agentdirGrep(ws, query, {
			pathGlob: options.scope,
			maxResults: options.topK,
		});
		return hits.map((hit) => ({
			id: `${hit.virtualPath}:${hit.lineNumber}`,
			content: hit.line,
			source: hit.virtualPath,
			score: hit.score,
			metadata: { lineNumber: hit.lineNumber, matchCount: hit.matchCount, method: "posix" },
		}));
	}
}
