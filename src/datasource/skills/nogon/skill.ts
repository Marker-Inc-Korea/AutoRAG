/**
 * Browser web-history datasource backed by a running Nogon server.
 *
 * Nogon (https://github.com/asciimoo/nogon, AGPL-3.0-or-later) is a separate program. It captures the pages a
 * Chromium or Firefox browser visits (through its own extension), keeps the full text, and owns the lexical and
 * semantic indexes, incremental updates, and credentials. AutoRAG only sends search requests to its public HTTP
 * API and maps the answers. It does not link, embed, bundle, copy, or modify Nogon, and never hides that Nogon
 * is the backend: the manifest names it, its license, and its source.
 *
 * Web pages only: every query is confined to `type:web`, and any document Nogon reports with another type
 * (watched local files, remote file snapshots) is dropped. Local directory indexing in Nogon is never used.
 */

import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceCliError } from "../../errors.ts";
import { matchesDatasourceScope, normalizeVirtualPath } from "../../scope.ts";
import { datasourceSearchToolName } from "../../tool-naming.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "../../types.ts";
import { NogonClient, type NogonClientOptions, type NogonFailure, type NogonHit, pageId } from "./client.ts";

export const NOGON_SKILL_TYPE = "nogon-web-history";
const DEFAULT_ID = "nogon";
const DEFAULT_INSTANCE = "default";
const DEFAULT_TAGS = ["nogon", "browser", "web-history", "pii"] as const;
const SAFE_DOMAIN = /^[a-z0-9][a-z0-9.-]*$/u;
const SAFE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/u;

export interface NogonSkillOptions extends NogonClientOptions {
	readonly client?: NogonClient;
	readonly datasourceId?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly tags?: readonly string[];
	/**
	 * Also register Nogon's semantic search. Requires `semantic_search.enable` and an embedding endpoint in the
	 * Nogon server config; Nogon embeds the query text through the endpoint the operator chose there.
	 */
	readonly semantic?: boolean;
}

type Mode = "lexical" | "semantic";

export class NogonSkill implements DatasourceSkill {
	private readonly datasourceId: string;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly tags: readonly string[];
	private readonly semantic: boolean;
	private readonly client: NogonClient | undefined;
	/** A bad `baseUrl` or token setup is reported through `index()`/search, never thrown out of config loading. */
	private readonly configError: string | undefined;
	private lastIndexedAt: number | undefined;
	private lastError: string | undefined;

	constructor(options: NogonSkillOptions = {}) {
		this.datasourceId = options.datasourceId ?? DEFAULT_ID;
		this.instanceId = SAFE_SEGMENT.test(options.instanceId ?? DEFAULT_INSTANCE)
			? (options.instanceId ?? DEFAULT_INSTANCE)
			: DEFAULT_INSTANCE;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.tags = options.tags ?? DEFAULT_TAGS;
		this.semantic = options.semantic === true;
		let client = options.client;
		let configError: string | undefined;
		if (client === undefined) {
			try {
				client = new NogonClient(options);
			} catch (error) {
				configError = error instanceof Error ? error.message : String(error);
			}
		}
		this.client = client;
		this.configError = configError;
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: this.datasourceId,
			id: this.datasourceId,
			type: NOGON_SKILL_TYPE,
			description: "Browser web history (pages you visited) via a Nogon server",
			capabilities: [
				"external-cli",
				"remote-search",
				"lexical",
				"scoped",
				"web-history",
				...(this.semantic ? ["semantic"] : []),
			],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: this.datasourceId,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: "none",
			...(this.lastIndexedAt !== undefined ? { lastIndexedAt: this.lastIndexedAt } : {}),
			...(this.lastError !== undefined ? { lastError: this.lastError } : {}),
		};
	}

	/**
	 * Nogon indexes pages itself the moment the browser extension sends them, so there is nothing to copy.
	 * `index()` is a read-only probe: reachability, authentication, how many web pages are searchable, and whether
	 * Nogon also holds non-web documents that this datasource will never return.
	 */
	async index(options?: DatasourceIndexOptions): Promise<DatasourceIndexResult> {
		if (this.client === undefined)
			return this.fail("datasource-unavailable", this.configError ?? "nogon is not configured");
		const config = await this.client.config(options?.signal);
		if (!config.ok) return this.failFrom(config);
		if (config.value.authenticated === false) {
			return this.fail(
				"datasource-auth-error",
				`nogon requires an access token (authMode ${String(config.value.authMode)}): set connector.tokenEnv to the name of the environment variable that holds Nogon's app.access_token`,
			);
		}
		const web = await this.client.count("type:web", options?.signal);
		if (!web.ok) return this.failFrom(web);
		const other = await this.client.count("-type:web", options?.signal);
		if (!other.ok) return this.failFrom(other);
		this.lastError = undefined;
		this.lastIndexedAt = Date.now();
		const diagnostics: DatasourceDiagnostic[] = [];
		if (web.value === 0) {
			diagnostics.push(
				this.note(
					"nogon holds no web pages yet: install its Chrome/Firefox extension and browse, or run `nogon import browser history`",
					"datasource-empty",
					"info",
				),
			);
		}
		if (other.value > 0) {
			diagnostics.push(
				this.note(
					`nogon also holds ${other.value} non-web document(s) (watched files or file snapshots); they are excluded from this datasource and never searched`,
					"datasource-index-partial",
					"warning",
				),
			);
		}
		if (this.semantic && config.value.semanticEnabled === false) {
			diagnostics.push(
				this.note(
					"connector.semantic is true but the Nogon server has semantic search disabled; set semantic_search.enable in the Nogon config and restart `nogon listen`",
					"datasource-unavailable",
					"warning",
				),
			);
		}
		return {
			ok: true,
			instanceId: this.instanceId,
			skill: this.datasourceId,
			chunkCount: web.value,
			indexedAt: this.lastIndexedAt,
			diagnostics,
		};
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return this.semantic
			? [new NogonMethod(this, "lexical"), new NogonMethod(this, "semantic")]
			: [new NogonMethod(this, "lexical")];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: normalizeVirtualPath(`/${this.datasourceId}/${instanceId}`),
			datasourceId: this.datasourceId,
			skill: this.datasourceId,
			instanceId,
			contentType: "web-page",
			metadata: { datasourceId: this.datasourceId, instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const scopes = this.instances
			.map((instanceId) => `- \`${normalizeVirtualPath(`/${this.datasourceId}/${instanceId}`)}\``)
			.join("\n");
		return {
			name: `datasource-${this.datasourceId}`,
			description:
				"Search the web pages visited in the user's browser (Chrome/Chromium/Firefox history with full page text) through their Nogon server. Use for questions like 'what was that page I read about X' or 'when did I look at Y'.",
			content: [
				"# Browser web history (Nogon)",
				"",
				"Search the pages the user visited in their browser, with the page text Nogon captured at visit time. [Nogon](https://github.com/asciimoo/nogon) is a separate program (AGPL-3.0-or-later). It keeps its own index and credentials. AutoRAG only sends it search requests over HTTP: it does not bundle, link, copy, or modify Nogon, and never hides that Nogon is the backend.",
				"",
				"## When to use",
				"Questions about pages, articles, docs, or sites the user read or opened in a browser; recalling a page by topic, domain, or time. Not for the user's own files: Nogon's local-directory indexing is never searched here, only web pages.",
				"",
				"## How to search",
				`Call the dedicated \`${datasourceSearchToolName(this.datasourceId)}\` tool with a query and optional narrowing scope. Available authorized scopes:`,
				scopes,
				"",
				"`scope` can only narrow within already-authorized scopes; it can never widen access. A scope of `<root>/pages/<domain>/**` restricts the search to one site.",
				"",
				'Plain words are alternatives: a hit may match any of them, so a natural question works. Stopwords (the, where, 의, について, que, ...) are dropped from plain-word queries in every language AutoRAG supports, so a short keyword query with the distinctive word ranks better than a full sentence. Nogon syntax passes through unchanged: `"exact phrase"`, `title:`, `text:`, `domain:`, `url:`, `language:`, `-term`, `secur*`, `added:<7d`, `updated:>=2026-10-01`, `visits:3..`, and `sort:date` / `sort:-date` / `sort:visits`. A query using any such operator is sent as written (plain words ANDed), so write it deliberately. Nogon has no `OR` keyword: to combine alternatives write `(a|b)`.',
				"",
				this.semantic
					? "Semantic (meaning-based) retrieval is also registered for this datasource; Nogon embeds the query through the embedding endpoint its operator configured."
					: "Only keyword (lexical) retrieval is registered for this datasource.",
				"",
				"## Reading a hit",
				"A hit carries the page title, URL, and a snippet around the match. The URL is in the hit content and `metadata.url`; the `/" +
					`${this.datasourceId}/...\` source identity is opaque. To read the page as it is today use \`web_fetch\` on its URL. Visit time is \`metadata.visitedAt\`.`,
				"",
				"## Native CLI",
				"The `nogon` program owns the index, the browser extension, and the data. When the dedicated tool cannot express what you need, call it directly through `bash`:",
				"- `nogon search 'domain:example.com sort:date' --format json --fields title,url,updated --limit 20`",
				"- `nogon doctor` (server, authentication, index health)",
				"- `nogon list-urls` (every indexed URL)",
				`Never pass datasource virtual paths (\`/${this.datasourceId}/...\`) to bash; they are not OS paths.`,
				"",
				"## Privacy",
				"Browsing history is sensitive. Results are not redacted, so use a local model when that matters. If Nogon's semantic search is enabled, page text is sent to the embedding endpoint its operator configured.",
			].join("\n"),
		};
	}

	async search(mode: Mode, query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		if (this.client === undefined) {
			throw datasourceCliError(this.datasourceId, "search", {
				reason: "unavailable",
				stderr: this.configError ?? "nogon is not configured",
			});
		}
		const target = this.scopeTarget(options.scope);
		if (!target.applies) return [];
		const search = {
			...(options.topK === undefined ? {} : { topK: options.topK }),
			...(options.signal === undefined ? {} : { signal: options.signal }),
			...(target.domain === undefined ? {} : { domain: target.domain }),
		};
		const result =
			mode === "lexical"
				? await this.client.searchLexical(trimmed, search)
				: await this.client.searchSemantic(trimmed, search);
		if (!result.ok)
			throw datasourceCliError(this.datasourceId, "search", { reason: result.kind, stderr: result.message });
		const hits: RetrievalResult[] = [];
		for (const hit of result.value) {
			const source = this.sourceFor(hit);
			if (!matchesDatasourceScope(source, options.scope)) continue;
			hits.push(this.toResult(mode, source, hit));
		}
		return hits;
	}

	/** Opaque, stable identity: `/<id>/<instance>/pages/<domain>/<hash of the URL>`. */
	private sourceFor(hit: NogonHit): string {
		const domain = hit.domain.toLowerCase().replace(/[^a-z0-9.-]+/gu, "_");
		return normalizeVirtualPath(`/${this.datasourceId}/${this.instanceId}/pages/${domain}/${pageId(hit.url)}`);
	}

	private toResult(mode: Mode, source: string, hit: NogonHit): RetrievalResult {
		const body = [hit.title, hit.url, hit.snippet.length > 0 ? hit.snippet : (hit.description ?? "")]
			.filter((part) => part.length > 0)
			.join("\n");
		return {
			id: `${this.datasourceId}:${this.instanceId}:${pageId(hit.url)}`,
			content: body,
			source,
			score: hit.score,
			metadata: {
				method: `${this.datasourceId}-${mode}`,
				datasourceId: this.datasourceId,
				instanceId: this.instanceId,
				backend: "nogon",
				url: hit.url,
				domain: hit.domain,
				title: hit.title,
				...(hit.updatedAt === undefined
					? {}
					: { visitedAt: new Date(hit.updatedAt).toISOString(), publishedAt: hit.updatedAt }),
				...(hit.addedAt === undefined ? {} : { firstVisitedAt: new Date(hit.addedAt).toISOString() }),
				...(hit.language === undefined ? {} : { language: hit.language }),
			},
		};
	}

	/** Does the scope reach this datasource at all, and can it be pushed down to Nogon as `domain:`? */
	private scopeTarget(scope: string | undefined): { readonly applies: boolean; readonly domain?: string } {
		if (scope === undefined || scope.trim().length === 0) return { applies: true };
		const normalized = normalizeVirtualPath(scope);
		const root = normalizeVirtualPath(`/${this.datasourceId}/${this.instanceId}`);
		const pages = `${root}/pages`;
		const applies =
			normalized === "/" ||
			normalized.startsWith(root) ||
			root.startsWith(`${normalized}/`) ||
			matchesDatasourceScope(root, normalized) ||
			matchesDatasourceScope(`${pages}/x/x`, normalized);
		if (!applies) return { applies: false };
		if (!normalized.startsWith(`${pages}/`)) return { applies: true };
		const segment = normalized.slice(pages.length + 1).split("/")[0] ?? "";
		return SAFE_DOMAIN.test(segment) ? { applies: true, domain: segment } : { applies: true };
	}

	private failFrom(failure: NogonFailure): DatasourceIndexResult {
		const code: DatasourceDiagnosticCode =
			failure.kind === "auth"
				? "datasource-auth-error"
				: failure.kind === "unavailable"
					? "datasource-unavailable"
					: "datasource-index-failed";
		return this.fail(code, failure.message);
	}

	private fail(code: DatasourceDiagnosticCode, message: string): DatasourceIndexResult {
		this.lastError = message;
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: this.datasourceId,
			indexedAt: Date.now(),
			diagnostics: [this.note(message, code, code === "datasource-unavailable" ? "warning" : "error")],
			error: message,
			code,
			message,
		};
	}

	private note(
		message: string,
		code: DatasourceDiagnosticCode,
		severity: DatasourceDiagnostic["severity"],
	): DatasourceDiagnostic {
		return { code, severity, message, instanceId: this.instanceId, source: this.datasourceId };
	}
}

class NogonMethod implements RetrievalMethod {
	private readonly skill: NogonSkill;
	private readonly mode: Mode;

	constructor(skill: NogonSkill, mode: Mode) {
		this.skill = skill;
		this.mode = mode;
	}

	describe(): RetrievalMethodDescriptor {
		const descriptor = this.skill.describe();
		const id = descriptor.datasourceId ?? DEFAULT_ID;
		return this.mode === "lexical"
			? {
					name: `${id}-lexical`,
					type: "remote",
					description: "Full-text search over visited web pages via the Nogon server",
					status: "active",
					capabilities: ["remote-search", "lexical", "scoped", "external-cli", "path-opaque-sources"],
					datasourceId: id,
					tags: descriptor.tags,
				}
			: {
					name: `${id}-semantic`,
					type: "vector",
					description: "Meaning-based search over visited web pages via the Nogon server's embeddings",
					status: "active",
					capabilities: ["remote-search", "semantic", "scoped", "external-cli", "path-opaque-sources"],
					datasourceId: id,
					tags: descriptor.tags,
				};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return this.skill.search(this.mode, query, options);
	}
}
