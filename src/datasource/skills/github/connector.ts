/**
 * GitHub Issues/PRs connector (issue #1303).
 *
 * Fetches issues and pull requests for a trusted list of `owner/repo`
 * targets through the GitHub REST API. A token is optional (public repos
 * work unauthenticated). Per-repo 404s degrade to warnings; 401 fails as
 * auth and rate limiting fails the whole fetch. Never throws.
 */

import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asNumber, asRecord, asString, httpJson, parseEpochMs, resolveToken } from "../../http.ts";

export interface GitHubConnectorOptions {
	/** GitHub REST base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** Optional token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the token. Default `GITHUB_TOKEN`. */
	readonly tokenEnv?: string;
	/** Repositories to index as `owner/repo`. Required trusted configuration. */
	readonly repos?: readonly string[];
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	readonly maxPages?: number;
	readonly maxDocuments?: number;
}

const DEFAULT_BASE_URL = "https://api.github.com";
const DEFAULT_TOKEN_ENV = "GITHUB_TOKEN";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_DOCUMENTS = 500;

export class GitHubConnector implements DatasourceConnector {
	private readonly options: GitHubConnectorOptions;

	constructor(options: GitHubConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const repos = this.options.repos ?? [];
		if (repos.length === 0) return { ok: false, reason: "not-configured", message: "no repositories configured" };
		const token = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const request = {
			headers: {
				Accept: "application/vnd.github+json",
				"X-GitHub-Api-Version": "2022-11-28",
				"User-Agent": "autorag-datasource",
				...(token !== undefined ? { Authorization: `Bearer ${token}` } : {}),
			},
			timeoutMs: this.options.timeoutMs,
			fetchImpl: this.options.fetchImpl,
			signal,
		};
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxDocuments = this.options.maxDocuments ?? DEFAULT_MAX_DOCUMENTS;

		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		for (const repo of repos) {
			if (documents.length >= maxDocuments) break;
			const [owner, name] = repo.split("/", 2);
			if (owner === undefined || name === undefined || owner.length === 0 || name.length === 0) {
				warnings.push(`repository entry ${repos.indexOf(repo) + 1} is not owner/repo`);
				continue;
			}
			for (let page = 1; page <= maxPages && documents.length < maxDocuments; page += 1) {
				const url = `${baseUrl}/repos/${owner}/${name}/issues?state=all&per_page=100&page=${page}`;
				const result = await httpJson(url, request);
				if (!result.ok) {
					if (result.reason === "auth") return { ok: false, reason: "auth", message: "github: unauthorized" };
					if (result.reason === "rate-limited" || result.reason === "permission") {
						// GitHub reports primary rate limiting as 403; treat both as rate-limited.
						return { ok: false, reason: "rate-limited", message: "github: rate limited or forbidden" };
					}
					warnings.push(`repo ${owner}-${name} fetch failed: ${result.message}`);
					break;
				}
				const items = asArray(result.json);
				if (items.length === 0) break;
				for (const raw of items) {
					if (documents.length >= maxDocuments) break;
					const item = asRecord(raw);
					const number = asNumber(item?.number);
					const title = asString(item?.title);
					if (item === undefined || number === undefined || title === undefined) continue;
					const isPull = asRecord(item.pull_request) !== undefined;
					const body = asString(item.body) ?? "";
					const labels = asArray(item.labels)
						.map((label) => asString(asRecord(label)?.name))
						.filter((label): label is string => label !== undefined);
					documents.push({
						docId: `${owner}-${name}-${number}`,
						hierarchy: [owner, name, isPull ? "pulls" : "issues"],
						title: `#${number} ${title}`,
						content: body.length > 0 ? `${title}\n\n${body}` : title,
						publishedAt: parseEpochMs(item.updated_at),
						metadata: {
							state: asString(item.state) ?? "unknown",
							number,
							kind: isPull ? "pull" : "issue",
							labels,
						},
					});
				}
				if (items.length < 100) break;
			}
		}

		return { ok: true, documents, ...(warnings.length > 0 ? { warnings } : {}) };
	}
}
