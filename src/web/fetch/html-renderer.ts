/**
 * HTML rendering backend chain — ported from oh-my-pi's
 * `packages/coding-agent/src/tools/fetch.ts` (MIT licensed).
 *
 * FetchProvider order: native → lynx → firecrawl → jina.
 * trafilatura is dropped (requires oh-my-pi's pip ensureTool).
 * parallel is dropped (requires oh-my-pi's keyed client port).
 * lynx is used only when found on PATH via a `which` check.
 * firecrawl via FIRECRAWL_API_KEY env.
 * jina via keyless https://r.jina.ai/ with optional JINA_API_KEY.
 */

import { spawnSync } from "node:child_process";
import { gfm, TurndownService } from "./turndown/index.ts";

export type FetchProvider = "native" | "lynx" | "firecrawl" | "jina";

const FETCH_PROVIDER_ORDER: readonly FetchProvider[] = ["native", "lynx", "firecrawl", "jina"];

/** Per-remote-backend budget so a stalled endpoint cannot starve local renderers. */
const REMOTE_READER_MAX_MS = 10_000;

const JINA_MARKDOWN_MARKER = "Markdown Content:";
const JINA_READER_MAX_BYTES = 2 * 1024 * 1024;

export interface RenderHtmlToTextOptions {
	/** Overall render budget in seconds (default 30). */
	timeoutSeconds?: number;
	signal?: AbortSignal;
	fetch?: typeof fetch;
	firecrawlApiKey?: string;
	jinaApiKey?: string;
	/**
	 * Inject custom backend runners for testing. Keys must cover every
	 * FetchProvider in FETCH_PROVIDER_ORDER. A runner returns the rendered
	 * markdown or null to signal "not available / low quality / error."
	 */
	runners?: Record<FetchProvider, () => Promise<string | null>>;
}

/**
 * Parse the `Markdown Content:` marker from a Jina Reader response body.
 */
export function parseJinaReaderContent(responseBody: string): string | null {
	const markerStart = responseBody.indexOf(JINA_MARKDOWN_MARKER);
	if (markerStart < 0) return null;

	const content = responseBody.slice(markerStart + JINA_MARKDOWN_MARKER.length).trim();
	if (content.length < 100 || content.startsWith("Loading...") || content.startsWith("Please enable JavaScript")) {
		return null;
	}
	return content;
}

/**
 * Check if rendered output looks JS-gated or mostly navigation.
 */
export function isLowQualityOutput(content: string): boolean {
	const lower = content.toLowerCase();

	const jsGated = [
		"enable javascript",
		"javascript required",
		"turn on javascript",
		"please enable javascript",
		"browser not supported",
	];
	if (content.length < 1024 && jsGated.some((t) => lower.includes(t))) {
		return true;
	}

	const lines = content.split("\n").filter((l) => l.trim());
	const shortLines = lines.filter((l) => l.trim().length < 40);
	if (lines.length > 10 && shortLines.length / lines.length > 0.7) {
		return true;
	}

	return false;
}

function combineSignals(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
	if (!signal) return AbortSignal.timeout(timeoutMs);
	return AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
}

function hasCommand(cmd: string): boolean {
	try {
		return spawnSync(cmd, ["--version"], { stdio: "ignore" }).status === 0;
	} catch {
		return false;
	}
}

/**
 * Convert HTML to markdown using the vendored Turndown with GFM support.
 * Strips script/style tags before conversion.
 */
export function htmlToBasicMarkdown(html: string): string {
	const cleaned = html.replace(/<script[\s\S]*?<\/script>/gi, "").replace(/<style[\s\S]*?<\/style>/gi, "");
	const turndown = new TurndownService({
		headingStyle: "atx",
		codeBlockStyle: "fenced",
		bulletListMarker: "-",
	});
	turndown.use(gfm);
	turndown.addRule("strikethrough", {
		filter: ["del", "s", "strike"],
		replacement(content) {
			return `~~${content}~~`;
		},
	});
	return turndown.turndown(cleaned).trim();
}

/**
 * Render HTML to markdown by trying reader backends in priority order:
 * native (in-process), lynx, firecrawl, then jina.
 *
 * Every backend's output must clear the quality gate (>100 non-whitespace
 * chars and not isLowQualityOutput) before it is accepted; otherwise the
 * next backend is tried.
 */
export async function renderHtmlToText(
	url: string,
	html: string,
	options: RenderHtmlToTextOptions,
): Promise<{ content: string; ok: boolean; method: FetchProvider | "none" }> {
	const { signal, fetch: fetchImpl = fetch, firecrawlApiKey, jinaApiKey, runners } = options;
	const timeoutSeconds = options.timeoutSeconds ?? 30;
	const fetchFn = fetchImpl;

	const remoteBudgetMs = Math.min(timeoutSeconds * 1000, REMOTE_READER_MAX_MS);

	const defaultRunners: Record<FetchProvider, () => Promise<string | null>> = {
		native: async () => htmlToBasicMarkdown(html),
		lynx: async () => {
			if (!hasCommand("lynx")) return null;
			try {
				const result = spawnSync("lynx", ["-dump", "-nolist", "-width", "250", url], {
					signal: combineSignals(signal, remoteBudgetMs),
					encoding: "utf-8",
					maxBuffer: 10 * 1024 * 1024,
				});
				return result.status === 0 ? (result.stdout ?? null) : null;
			} catch {
				return null;
			}
		},
		firecrawl: async () => {
			const apiKey = firecrawlApiKey ?? process.env.FIRECRAWL_API_KEY;
			if (!apiKey) return null;
			try {
				const response = await fetchFn("https://api.firecrawl.dev/v2/scrape", {
					method: "POST",
					headers: {
						Accept: "application/json",
						"Content-Type": "application/json",
						Authorization: `Bearer ${apiKey}`,
					},
					body: JSON.stringify({ url, formats: ["markdown"] }),
					signal: combineSignals(signal, remoteBudgetMs),
				});
				if (!response.ok) return null;
				const payload = (await response.json()) as { success?: boolean; data?: { markdown?: string | null } };
				if (payload.success === false) return null;
				return payload.data?.markdown ?? null;
			} catch {
				return null;
			}
		},
		jina: async () => {
			const apiKey = jinaApiKey ?? process.env.JINA_API_KEY;
			const headers: Record<string, string> = {
				Accept: "text/markdown",
				"X-No-Cache": "true",
			};
			if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
			try {
				const response = await fetchFn(`https://r.jina.ai/${url}`, {
					headers,
					signal: combineSignals(signal, remoteBudgetMs),
				});
				if (!response.ok) return null;
				const contentLength = Number(response.headers.get("content-length"));
				if (Number.isFinite(contentLength) && contentLength > JINA_READER_MAX_BYTES) return null;
				return parseJinaReaderContent(await response.text());
			} catch {
				return null;
			}
		},
	};

	const activeRunners = runners ?? defaultRunners;

	let lowQuality: { content: string; method: FetchProvider } | null = null;

	for (const method of FETCH_PROVIDER_ORDER) {
		signal?.throwIfAborted();
		try {
			const content = await activeRunners[method]();
			if (!content || content.replace(/\s/g, "").length <= 100) continue;
			if (!isLowQualityOutput(content)) {
				return { content, ok: true, method };
			}
			lowQuality ??= { content, method };
		} catch {
			signal?.throwIfAborted();
		}
	}

	if (lowQuality) {
		return { content: lowQuality.content, ok: true, method: lowQuality.method };
	}
	return { content: "", ok: false, method: "none" };
}
