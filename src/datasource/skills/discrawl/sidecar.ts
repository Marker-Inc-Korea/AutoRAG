import { datasourceCliError } from "../../errors.ts";
import { type ResolveSidecarInput, resolveSidecarOptions } from "../../sidecar/config.ts";
import { CRAWLER_DESCRIPTORS } from "../../sidecar/descriptors.ts";
import { CrawlerSidecar, type CrawlerSidecarOptions } from "../../sidecar/index.ts";
import type { DiscrawlStatusResult } from "./types.ts";

export interface DiscrawlStatusClient {
	status?(signal?: AbortSignal): Promise<DiscrawlStatusResult>;
}

export interface DiscrawlSidecarOptions extends Omit<CrawlerSidecarOptions, "crawlerDbPath" | "embedder"> {
	/** Explicit archive path; omitted to discover it through the native client's status command. */
	readonly crawlerDbPath?: string;
}

/** Discord reuses its native embeddings; its sidecar provides lexical ranking only. */
export function resolveDiscrawlSidecarOptions(
	input: Omit<ResolveSidecarInput, "descriptor" | "defaultCrawlerDb">,
): DiscrawlSidecarOptions | undefined {
	const setting = input.connector?.sidecar;
	const object =
		typeof setting === "object" && setting !== null && !Array.isArray(setting)
			? (setting as Record<string, unknown>)
			: {};
	const resolved = resolveSidecarOptions({
		...input,
		descriptor: CRAWLER_DESCRIPTORS.discord,
		// An empty path defers discovery to the same native client's `status` command.
		defaultCrawlerDb: "",
		connector: {
			...input.connector,
			workspace: input.connector?.guildId,
			sidecar: setting === false ? false : { ...object, semantic: false },
		},
	});
	if (resolved === undefined) return undefined;
	const { embedder: _embedder, crawlerDbPath, ...lexical } = resolved;
	return { ...lexical, ...(crawlerDbPath.length > 0 ? { crawlerDbPath } : {}) };
}

/** Resolves the native archive on first search, without requiring indexing or a second vector store. */
export class DiscrawlSidecar {
	private readonly options: DiscrawlSidecarOptions;
	private readonly client: DiscrawlStatusClient;
	private sidecar: CrawlerSidecar | undefined;

	constructor(options: DiscrawlSidecarOptions, client: DiscrawlStatusClient) {
		this.options = options;
		this.client = client;
	}

	async get(signal?: AbortSignal): Promise<CrawlerSidecar> {
		signal?.throwIfAborted();
		if (this.sidecar !== undefined) return this.sidecar;
		let crawlerDbPath = this.options.crawlerDbPath;
		if (crawlerDbPath === undefined || crawlerDbPath.length === 0) {
			if (this.client.status === undefined) {
				throw new Error("Discord BM25 requires an archive path or a discrawl client with status support");
			}
			const result = await this.client.status(signal);
			signal?.throwIfAborted();
			if (!result.ok) throw datasourceCliError("discord", "status", result);
			if (result.data.databasePath === undefined || result.data.databasePath.length === 0) {
				throw new Error("discrawl status did not return database_path; cannot locate the Discord archive");
			}
			crawlerDbPath = result.data.databasePath;
		}
		// Cache only successful discovery; failures and cancelled requests remain retryable.
		this.sidecar ??= new CrawlerSidecar({ ...this.options, crawlerDbPath, embedder: undefined });
		return this.sidecar;
	}

	close(): void {
		this.sidecar?.close();
		this.sidecar = undefined;
	}
}
