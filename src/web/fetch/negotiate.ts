/**
 * Content negotiation and LLM endpoint discovery — ported from oh-my-pi's
 * `packages/coding-agent/src/tools/fetch.ts` (MIT licensed).
 */

import { type LoadPageResult, loadPage, looksLikeHtml } from "./page-loader.ts";
import { normalizeMime } from "./url-target.ts";

/**
 * Build llms.txt / llms.md candidates scoped to the requested URL.
 */
export function buildLlmEndpointCandidates(url: string): string[] {
	try {
		const parsed = new URL(url);
		if (parsed.pathname === "/") {
			return [`${parsed.origin}/.well-known/llms.txt`, `${parsed.origin}/llms.txt`, `${parsed.origin}/llms.md`];
		}

		const trimmedPath = parsed.pathname.replace(/\/+$/, "");
		const segments = trimmedPath.split("/").filter(Boolean);
		const scopeDepth = parsed.pathname.endsWith("/") ? segments.length : Math.max(segments.length - 1, 1);
		const endpoints: string[] = [];

		for (let depth = scopeDepth; depth >= 1; depth--) {
			const scope = `/${segments.slice(0, depth).join("/")}/`;
			endpoints.push(`${parsed.origin}${scope}llms.txt`, `${parsed.origin}${scope}llms.md`);
		}

		return endpoints;
	} catch {
		return [];
	}
}

/**
 * Try fetching URL with .md appended (llms.txt convention).
 */
export async function tryMdSuffix(url: string, timeout: number, signal?: AbortSignal): Promise<string | null> {
	const candidates: string[] = [];

	try {
		const parsed = new URL(url);
		const pathname = parsed.pathname;

		if (pathname.endsWith("/")) {
			candidates.push(`${parsed.origin}${pathname}index.html.md`);
		} else if (pathname.includes(".")) {
			candidates.push(`${parsed.origin}${pathname}.md`);
		} else {
			candidates.push(`${parsed.origin}${pathname}.md`);
		}
	} catch {
		return null;
	}

	if (signal?.aborted) {
		return null;
	}

	for (const candidate of candidates) {
		if (signal?.aborted) {
			return null;
		}
		const result = await loadPage(candidate, { timeout, signal });
		if (result.ok && result.content.trim().length > 100 && !looksLikeHtml(result.content)) {
			return result.content;
		}
	}

	return null;
}

/**
 * Try to fetch LLM-friendly endpoints (llms.txt, llms.md).
 */
export async function tryLlmEndpoints(
	url: string,
	timeout: number,
	signal?: AbortSignal,
): Promise<{ content: string; endpoint: string } | null> {
	const endpoints = buildLlmEndpointCandidates(url);

	if (signal?.aborted || endpoints.length === 0) {
		return null;
	}

	for (const endpoint of endpoints) {
		if (signal?.aborted) {
			return null;
		}
		const result = await loadPage(endpoint, { timeout: Math.min(timeout, 5), signal });
		if (result.ok && result.content.trim().length > 100 && !looksLikeHtml(result.content)) {
			return { content: result.content, endpoint };
		}
	}
	return null;
}

/**
 * Try content negotiation for markdown/plain.
 */
export async function tryContentNegotiation(
	url: string,
	timeout: number,
	signal?: AbortSignal,
): Promise<{ content: string; type: string } | null> {
	if (signal?.aborted) {
		return null;
	}

	let result: LoadPageResult;
	try {
		result = await loadPage(url, {
			timeout,
			headers: { Accept: "text/markdown, text/plain;q=0.9, text/html;q=0.8" },
			signal,
		});
	} catch {
		return null;
	}

	if (!result.ok) return null;

	const mime = normalizeMime(result.contentType);
	if ((mime.includes("markdown") || mime === "text/plain") && !looksLikeHtml(result.content)) {
		return { content: result.content, type: result.contentType };
	}

	return null;
}
