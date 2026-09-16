/**
 * GitHub Gist connector (issue #1588).
 *
 * Indexes the authenticated account's gists (public + secret, bounded by the
 * token's scopes) through the GitHub REST API. Incremental: the gist list
 * endpoint returns `id` + `updated_at` cheaply, so a persisted cursor map
 * (`state.json`) decides which gists need a full content fetch and which
 * disappeared (reported via `deletedDocIds`). Token resolution order:
 * explicit `token` > `tokenEnv` (default `GITHUB_TOKEN`) > `gh auth token`
 * when the gh CLI fallback is enabled. Never throws.
 */

import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { ConnectorDocument, ConnectorFetchResult, DatasourceConnector } from "../../connector.ts";
import { asArray, asRecord, asString, httpJson, parseEpochMs, resolveToken } from "../../http.ts";

export interface GitHubGistConnectorOptions {
	/** GitHub REST base; override to point at a mock server in tests. */
	readonly baseUrl?: string;
	/** Personal access token; explicit value wins over {@link tokenEnv}. */
	readonly token?: string;
	/** Env var name holding the token. Default `GITHUB_TOKEN`. */
	readonly tokenEnv?: string;
	/** When true (default), fall back to `gh auth token` for auth. */
	readonly ghCliFallback?: boolean;
	/** Injectable gh-token resolver (tests); defaults to spawning `gh auth token`. */
	readonly ghTokenRunner?: () => Promise<string | undefined>;
	readonly timeoutMs?: number;
	readonly fetchImpl?: typeof fetch;
	/** Persisted incremental cursor; omitted ⇒ every fetch is a full sync. */
	readonly statePath?: string;
	readonly maxPages?: number;
	readonly maxGists?: number;
}

const DEFAULT_BASE_URL = "https://api.github.com";
const DEFAULT_TOKEN_ENV = "GITHUB_TOKEN";
const DEFAULT_MAX_PAGES = 10;
const DEFAULT_MAX_GISTS = 500;
const GH_TOKEN_TIMEOUT_MS = 5_000;

interface GistSyncState {
	readonly version: number;
	readonly gists: Record<string, string>;
}

const STATE_VERSION = 1;

async function defaultGhTokenRunner(): Promise<string | undefined> {
	return new Promise((resolve) => {
		execFile("gh", ["auth", "token"], { timeout: GH_TOKEN_TIMEOUT_MS }, (error, stdout) => {
			if (error) {
				resolve(undefined);
				return;
			}
			const token = stdout.trim();
			resolve(token.length > 0 ? token : undefined);
		});
	});
}

export class GitHubGistConnector implements DatasourceConnector {
	private readonly options: GitHubGistConnectorOptions;

	constructor(options: GitHubGistConnectorOptions = {}) {
		this.options = options;
	}

	async fetch(signal?: AbortSignal): Promise<ConnectorFetchResult> {
		const token = await this.resolveAuth();
		if (token === undefined) {
			return {
				ok: false,
				reason: "not-configured",
				message: "no GitHub token: set GITHUB_TOKEN (or connector.tokenEnv) or authenticate the gh CLI",
			};
		}
		const baseUrl = this.options.baseUrl ?? DEFAULT_BASE_URL;
		const request = {
			headers: {
				Accept: "application/vnd.github+json",
				"X-GitHub-Api-Version": "2022-11-28",
				"User-Agent": "autorag-datasource",
				Authorization: `Bearer ${token}`,
			},
			timeoutMs: this.options.timeoutMs,
			fetchImpl: this.options.fetchImpl,
			signal,
		};
		const maxPages = this.options.maxPages ?? DEFAULT_MAX_PAGES;
		const maxGists = this.options.maxGists ?? DEFAULT_MAX_GISTS;

		// 1. List gists (cheap: id + updated_at only used for the diff).
		const listed = new Map<string, { updatedAt: string }>();
		for (let page = 1; page <= maxPages && listed.size < maxGists; page += 1) {
			const url = `${baseUrl}/gists?per_page=100&page=${page}`;
			const result = await httpJson(url, request);
			if (!result.ok) {
				if (result.reason === "auth") return { ok: false, reason: "auth", message: "github: unauthorized" };
				if (result.reason === "rate-limited" || result.reason === "permission") {
					return { ok: false, reason: "rate-limited", message: "github: rate limited or forbidden" };
				}
				return { ok: false, reason: "api-error", message: `github: gist list failed: ${result.message}` };
			}
			const items = asArray(result.json);
			if (items.length === 0) break;
			for (const raw of items) {
				if (listed.size >= maxGists) break;
				const item = asRecord(raw);
				const id = asString(item?.id);
				const updatedAt = asString(item?.updated_at);
				if (id === undefined || updatedAt === undefined) continue;
				listed.set(id, { updatedAt });
			}
			if (items.length < 100) break;
		}

		// 2. Diff against the persisted cursor.
		const previous = this.loadState();
		const changedIds: string[] = [];
		for (const [id, entry] of listed) {
			if (previous.gists[id] !== entry.updatedAt) changedIds.push(id);
		}
		const deletedDocIds = Object.keys(previous.gists).filter((id) => !listed.has(id));

		// 3. Fetch full content only for new/changed gists.
		const documents: ConnectorDocument[] = [];
		const warnings: string[] = [];
		for (const id of changedIds) {
			const result = await httpJson(`${baseUrl}/gists/${encodeURIComponent(id)}`, request);
			if (!result.ok) {
				warnings.push(`gist ${id} fetch failed: ${result.message}`);
				continue;
			}
			const document = toDocument(asRecord(result.json));
			if (document !== undefined) documents.push(document);
		}

		// 4. Persist the new cursor only when the list completed.
		this.saveState({
			version: STATE_VERSION,
			gists: Object.fromEntries([...listed].map(([id, e]) => [id, e.updatedAt])),
		});

		const changed = changedIds.length > 0 || deletedDocIds.length > 0;
		return {
			ok: true,
			documents,
			changed,
			deletedDocIds,
			...(warnings.length > 0 ? { warnings } : {}),
		};
	}

	private async resolveAuth(): Promise<string | undefined> {
		const direct = resolveToken(this.options.token, this.options.tokenEnv ?? DEFAULT_TOKEN_ENV);
		if (direct !== undefined) return direct;
		if (this.options.ghCliFallback === false) return undefined;
		const runner = this.options.ghTokenRunner ?? defaultGhTokenRunner;
		try {
			return await runner();
		} catch {
			return undefined;
		}
	}

	private loadState(): GistSyncState {
		const statePath = this.options.statePath;
		if (statePath === undefined || !existsSync(statePath)) return { version: STATE_VERSION, gists: {} };
		try {
			const parsed = JSON.parse(readFileSync(statePath, "utf8")) as GistSyncState;
			if (parsed.version !== STATE_VERSION || typeof parsed.gists !== "object" || parsed.gists === null) {
				return { version: STATE_VERSION, gists: {} };
			}
			return parsed;
		} catch {
			return { version: STATE_VERSION, gists: {} };
		}
	}

	private saveState(state: GistSyncState): void {
		const statePath = this.options.statePath;
		if (statePath === undefined) return;
		try {
			mkdirSync(dirname(statePath), { recursive: true });
			writeFileSync(statePath, `${JSON.stringify(state)}\n`, "utf8");
		} catch {
			// Cursor persistence is best-effort; a lost cursor degrades to a full re-sync.
		}
	}
}

function toDocument(gist: Record<string, unknown> | undefined): ConnectorDocument | undefined {
	if (gist === undefined) return undefined;
	const id = asString(gist.id);
	if (id === undefined) return undefined;
	const description = asString(gist.description) ?? "";
	const filesRecord = asRecord(gist.files) ?? {};
	const fileNames = Object.keys(filesRecord);
	const sections: string[] = [];
	for (const name of fileNames) {
		const file = asRecord(filesRecord[name]);
		const filename = asString(file?.filename) ?? name;
		const content = asString(file?.content) ?? "";
		sections.push(`## ${filename}\n${content}`);
	}
	const title = description.length > 0 ? description : (fileNames[0] ?? id);
	const content = [description, ...sections].filter((part) => part.length > 0).join("\n\n");
	return {
		docId: id,
		hierarchy: ["gists"],
		title,
		content: content.length > 0 ? content : title,
		publishedAt: parseEpochMs(gist.updated_at),
		metadata: {
			gistId: id,
			public: gist.public === true,
			files: fileNames,
			...(asString(gist.html_url) !== undefined ? { htmlUrl: asString(gist.html_url) } : {}),
		},
	};
}
