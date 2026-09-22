import { CrawlerCliClient } from "../../crawler-client.ts";
import type {
	CrawlerCliOptions,
	CrawlerFailure,
	CrawlerHit,
	CrawlerProfile,
	CrawlerSearchResult,
} from "../../crawler-types.ts";

const MESSAGE_PAGE_MAX = 50;
const DOC_PAGE_MAX = 20;
const RATE_LIMIT_CODE = "99991400";
const MAX_ATTEMPTS = 7;

export interface LarkClientOptions {
	readonly binaryPath?: string;
	readonly env?: Readonly<Record<string, string | undefined>>;
	readonly timeoutMs?: number;
	readonly chatIds?: readonly string[];
	readonly sleep?: (ms: number) => Promise<void>;
}

export class LarkClient {
	private readonly version: CrawlerCliClient;
	private readonly auth: CrawlerCliClient;
	private readonly messages: CrawlerCliClient;
	private readonly docs: CrawlerCliClient;
	private readonly sleep: (ms: number) => Promise<void>;

	constructor(options: LarkClientOptions = {}) {
		const cli: CrawlerCliOptions = {
			binaryPath: options.binaryPath ?? "lark-cli",
			...(options.env !== undefined ? { env: options.env } : {}),
			...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
		};
		const chatIds = (options.chatIds ?? []).filter(isSafeId);
		this.version = new CrawlerCliClient(VERSION_PROFILE, cli);
		this.auth = new CrawlerCliClient(AUTH_PROFILE, cli);
		this.messages = new CrawlerCliClient(messageProfile(chatIds), cli);
		this.docs = new CrawlerCliClient(DOC_PROFILE, cli);
		this.sleep = options.sleep ?? delay;
	}

	probeVersion(): Promise<CrawlerSearchResult | Awaited<ReturnType<CrawlerCliClient["sync"]>>> {
		return this.version.sync();
	}

	probeAuth(): Promise<Awaited<ReturnType<CrawlerCliClient["sync"]>>> {
		return this.auth.sync();
	}

	searchMessages(query: string, topK: number | undefined): Promise<CrawlerSearchResult> {
		return this.retry(() => this.messages.search(query, topK === undefined ? {} : { topK }));
	}

	searchDocs(query: string, topK: number | undefined): Promise<CrawlerSearchResult> {
		return this.retry(() => this.docs.search(query, topK === undefined ? {} : { topK }));
	}

	private async retry(run: () => Promise<CrawlerSearchResult>): Promise<CrawlerSearchResult> {
		let failure: CrawlerFailure | undefined;
		for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
			const result = await run();
			if (result.ok) return result;
			failure = result;
			if (!isRateLimited(result) || attempt === MAX_ATTEMPTS - 1) return result;
			await this.sleep(200 * 2 ** attempt);
		}
		return failure ?? { ok: false, reason: "spawn-error", stdout: "", stderr: "", code: null };
	}
}

const VERSION_PROFILE: CrawlerProfile = {
	binaryName: "lark-cli",
	allowedEnvPrefixes: [],
	syncArgs: () => ["--version"],
	searchArgs: () => [],
	parseSyncCount: () => 0,
	parseHits: () => [],
};

const AUTH_PROFILE: CrawlerProfile = {
	binaryName: "lark-cli",
	allowedEnvPrefixes: [],
	syncArgs: () => ["auth", "status", "--json"],
	searchArgs: () => [],
	parseSyncCount: () => 0,
	parseHits: () => [],
};

const DOC_PROFILE: CrawlerProfile = {
	binaryName: "lark-cli",
	allowedEnvPrefixes: [],
	syncArgs: () => [],
	searchArgs: (_options, query, topK) => [
		"drive",
		"+search",
		"--query",
		query,
		"--format",
		"json",
		"--page-size",
		String(clampPageSize(topK, DOC_PAGE_MAX)),
	],
	parseSyncCount: () => 0,
	parseHits: parseDocs,
};

function messageProfile(chatIds: readonly string[]): CrawlerProfile {
	return {
		binaryName: "lark-cli",
		allowedEnvPrefixes: [],
		syncArgs: () => [],
		searchArgs: (_options, query, topK) => {
			const args = [
				"im",
				"+messages-search",
				"--query",
				query,
				"--format",
				"json",
				"--page-size",
				String(clampPageSize(topK, MESSAGE_PAGE_MAX)),
			];
			if (chatIds.length > 0) args.push("--chat-id", chatIds.join(","));
			return args;
		},
		parseSyncCount: () => 0,
		parseHits: parseMessages,
	};
}

function parseMessages(stdout: string): readonly CrawlerHit[] | undefined {
	const data = envelopeData(stdout);
	if (data === undefined) return undefined;
	const rows = data.messages;
	if (!Array.isArray(rows)) return undefined;
	const hits: CrawlerHit[] = [];
	for (const row of rows) {
		if (!isRecord(row)) return undefined;
		const id = scalarString(row.message_id);
		const content = scalarString(row.content);
		if (id === undefined || content === undefined || !isSafeId(id)) continue;
		const chatId = scalarString(row.chat_id);
		const observedAt = scalarString(row.create_time);
		hits.push({
			id,
			content,
			score: 1 / (hits.length + 1),
			...(scalarString(row.chat_name) !== undefined ? { title: scalarString(row.chat_name) } : {}),
			metadata: {
				...(chatId !== undefined ? { chatId } : {}),
				...(scalarString(row.chat_type) !== undefined ? { chatType: scalarString(row.chat_type) } : {}),
				...(scalarString(row.chat_name) !== undefined ? { chatName: scalarString(row.chat_name) } : {}),
				...(observedAt !== undefined ? { observedAt } : {}),
				coverage: "server-side-unverified",
			},
		});
	}
	return hits;
}

function parseDocs(stdout: string): readonly CrawlerHit[] | undefined {
	const data = envelopeData(stdout);
	if (data === undefined) return undefined;
	const rows = data.results;
	if (!Array.isArray(rows)) return undefined;
	const hits: CrawlerHit[] = [];
	for (const row of rows) {
		if (!isRecord(row)) return undefined;
		const meta = isRecord(row.result_meta) ? row.result_meta : {};
		const token =
			firstString(meta, ["token", "docs_token"]) ??
			firstString(row, ["token", "docs_token"]) ??
			tokenFromUrl(firstString(meta, ["url"]));
		const content = stripHighlight(
			firstString(row, ["summary_highlighted", "title_highlighted", "summary", "title"]) ?? "",
		);
		if (token === undefined || !isSafeId(token) || content.length === 0) continue;
		const permalink = firstString(meta, ["url"]);
		const observedAt = firstString(meta, ["update_time_iso", "update_time"]) ?? scalarString(meta.update_time);
		hits.push({
			id: token,
			content,
			score: 1 / (hits.length + 1),
			...(firstString(row, ["title", "title_highlighted"]) !== undefined
				? { title: stripHighlight(firstString(row, ["title", "title_highlighted"]) ?? "") }
				: {}),
			metadata: {
				...(permalink !== undefined ? { permalink } : {}),
				...(observedAt !== undefined ? { observedAt } : {}),
				coverage: "server-side-unverified",
			},
		});
	}
	return hits;
}

function envelopeData(stdout: string): Record<string, unknown> | undefined {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) return undefined;
	let parsed: unknown;
	try {
		parsed = JSON.parse(trimmed);
	} catch {
		return undefined;
	}
	if (!isRecord(parsed) || parsed.ok !== true || !isRecord(parsed.data)) return undefined;
	return parsed.data;
}

function firstString(record: Record<string, unknown>, keys: readonly string[]): string | undefined {
	for (const key of keys) {
		const value = scalarString(record[key]);
		if (value !== undefined) return value;
	}
	return undefined;
}

function scalarString(value: unknown): string | undefined {
	if (typeof value === "string" && value.length > 0) return value;
	if (typeof value === "number" && Number.isFinite(value)) return String(value);
	return undefined;
}

function tokenFromUrl(url: string | undefined): string | undefined {
	if (url === undefined) return undefined;
	const path = url.split("?")[0]?.split("#")[0] ?? "";
	const parts = path.split("/").filter((part) => part.length > 0);
	const segment = parts[parts.length - 1];
	return segment !== undefined && isSafeId(segment) ? segment : undefined;
}

function stripHighlight(value: string): string {
	return value.replaceAll("<h>", "").replaceAll("</h>", "").replaceAll("<hb>", "").replaceAll("</hb>", "").trim();
}

export function isSafeId(value: string): boolean {
	return value.length > 0 && value !== "." && value !== ".." && !value.includes("#") && !value.includes("/");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isRateLimited(failure: CrawlerFailure): boolean {
	return failure.stderr.includes(RATE_LIMIT_CODE) || failure.stdout.includes(RATE_LIMIT_CODE);
}

function clampPageSize(topK: number, max: number): number {
	if (!Number.isFinite(topK)) return Math.min(20, max);
	const truncated = Math.trunc(topK);
	if (truncated < 1) return 1;
	if (truncated > max) return max;
	return truncated;
}

function delay(ms: number): Promise<void> {
	return new Promise((resolve) => {
		setTimeout(resolve, ms);
	});
}
