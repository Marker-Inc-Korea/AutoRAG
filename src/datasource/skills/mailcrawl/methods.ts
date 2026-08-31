import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../../../retrieval/types.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";
import type { MailcrawlSearchClient } from "./client.ts";
import type { MailcrawlSearchMode, MailcrawlSearchResult } from "./types.ts";

const modes: readonly MailcrawlSearchMode[] = ["bm25", "semantic", "hybrid"];
export class MailcrawlMethod implements RetrievalMethod {
	private readonly client: MailcrawlSearchClient;
	private readonly mode: MailcrawlSearchMode;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(client: MailcrawlSearchClient, mode: MailcrawlSearchMode, instanceId: string, tags: readonly string[]) {
		this.client = client;
		this.mode = mode;
		this.instanceId = instanceId;
		this.tags = tags;
	}
	describe(): RetrievalMethodDescriptor {
		return { name: `mailcrawl-${this.mode}`, type: this.mode === "bm25" ? "bm25" : this.mode === "semantic" ? "vector" : "hybrid", description: `${this.mode} retrieval over archived email via the external mailcrawl CLI`, status: "active", capabilities: ["email", this.mode, "scoped", "external-cli", "path-opaque-sources"], datasourceId: "mailcrawl", tags: this.tags };
	}
	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		if (!query.trim()) return [];
		let result: MailcrawlSearchResult;
		try { result = await this.client.search(this.mode, query, { topK: options.topK, signal: options.signal }); } catch { return []; }
		if (!result.ok) return [];
		return result.hits.filter((hit) => {
			const source = datasourceSourcePath("mailcrawl", this.instanceId, hit.chunkId);
			return matchesDatasourceScope(source, options.scope) && (options.allowedScopes === undefined || options.allowedScopes.length === 0 || options.allowedScopes.some((scope) => matchesDatasourceScope(source, scope)));
		}).map((hit) => {
			const source = datasourceSourcePath("mailcrawl", this.instanceId, hit.chunkId);
			return { id: `mailcrawl:${this.instanceId}:${hit.chunkId}`, content: hit.snippet, source, score: hit.score, metadata: { method: `mailcrawl-${this.mode}`, datasourceId: "mailcrawl", instanceId: this.instanceId, mode: this.mode, chunkId: hit.chunkId, messageId: hit.messageId, threadId: hit.threadId, accountId: hit.accountId, mailbox: hit.mailbox, subject: hit.subject, from: hit.from, to: hit.to, date: hit.date } };
		}).slice(0, options.topK ?? 20);
	}
}
export const MAILCRAWL_MODES = modes;

