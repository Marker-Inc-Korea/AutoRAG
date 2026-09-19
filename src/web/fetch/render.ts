/**
 * renderUrl — the full URL-fetch/render pipeline — ported from oh-my-pi's
 * `packages/coding-agent/src/tools/fetch.ts` (MIT licensed).
 *
 * Special site handlers (step 1), images, sqlite, archives, and notebooks
 * are NOT ported. Binary payloads (zip, pdf, etc.) return a textual binary
 * notice. The markit conversion step uses AutoRAG's parser registry for
 * PDF/DOCX/PPTX/XLSX; EPUB and other unsupported types fall through to the
 * binary-notice fallback.
 */

import { convertBinaryPayload, fetchBinary } from "./convert.ts";
import { extractDocumentLinks, formatJson, parseAlternateLinks, parseFeedToMarkdown } from "./feeds.ts";
import { isLowQualityOutput, renderHtmlToText } from "./html-renderer.ts";
import { tryContentNegotiation, tryLlmEndpoints, tryMdSuffix } from "./negotiate.ts";
import { finalizeOutput, loadPage, looksLikeHtml, MAX_BYTES, type RenderResult } from "./page-loader.ts";
import {
	buildBinaryNotice,
	getExtensionHint,
	isConvertible,
	normalizeMime,
	normalizeUrl,
	sampleLooksBinary,
} from "./url-target.ts";

export interface RenderUrlOptions {
	timeoutSeconds?: number;
	raw?: boolean;
	signal?: AbortSignal;
	fetch?: typeof fetch;
	firecrawlApiKey?: string;
	jinaApiKey?: string;
}

/** MIME types that are always binary (never text-renderable). */
const BINARY_MIMES = new Set([
	"application/zip",
	"application/x-zip-compressed",
	"application/x-tar",
	"application/tar",
	"application/gzip",
	"application/x-gzip",
	"application/x-7z-compressed",
	"application/x-rar-compressed",
	"application/x-bzip2",
	"application/x-xz",
	"application/octet-stream",
	"application/pdf",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	"application/vnd.openxmlformats-officedocument.presentationml.presentation",
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	"application/vnd.ms-excel",
	"application/msword",
	"application/vnd.ms-powerpoint",
	"application/epub+zip",
	"application/x-sqlite3",
	"application/sqlite3",
	"application/sqlite",
	"application/x-ipynb+json",
]);

function formatBytes(bytes: number): string {
	if (bytes < 1024) return `${bytes} B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
	return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

/**
 * Fetch and render a URL to markdown or text.
 */
export async function renderUrl(url: string, options: RenderUrlOptions = {}): Promise<RenderResult> {
	const { timeoutSeconds = 30, raw = false, signal, fetch: fetchImpl, firecrawlApiKey, jinaApiKey } = options;

	const notes: string[] = [];
	const fetchedAt = new Date().toISOString();
	const timeout = timeoutSeconds;

	if (signal?.aborted) {
		throw new Error("aborted");
	}

	// Step 0: Normalize URL
	url = normalizeUrl(url);

	// Step 2: Fetch page
	const load = (target: string, extra: { headers?: Record<string, string> } = {}) =>
		loadPage(target, { timeout, signal, ...(fetchImpl ? { fetch: fetchImpl } : {}), ...extra });

	const response = await load(url).catch((err: unknown) => {
		if (signal?.aborted) throw err;
		return {
			content: "",
			contentType: "",
			finalUrl: url,
			ok: false as const,
			error: err instanceof Error ? err.message : String(err),
		};
	});

	if (signal?.aborted) {
		throw new Error("aborted");
	}

	if (!response.ok) {
		const status = "status" in response ? response.status : undefined;
		return {
			url,
			finalUrl: response.finalUrl || url,
			contentType: response.contentType || "unknown",
			method: "failed",
			content: "",
			fetchedAt,
			truncated: false,
			notes: [
				status ? `Failed to fetch URL (HTTP ${status})` : "Failed to fetch URL",
				...(response.error ? [`Cause: ${response.error}`] : []),
			],
		};
	}

	const { finalUrl, content: rawContent } = response;
	if (response.truncated) {
		notes.push(`Response body exceeded ${formatBytes(MAX_BYTES)} and was cut mid-stream; content is incomplete`);
	}
	const mime = normalizeMime(response.contentType);
	const extHint = getExtensionHint(finalUrl);

	// Step 3: Handle convertible binary files (PDF, DOCX, etc.)
	if (!raw && isConvertible(mime, extHint)) {
		const binary = await fetchBinary(finalUrl, timeout, signal, fetchImpl);
		if (binary.ok) {
			const ext = getExtensionHint(finalUrl, binary.contentDisposition) || extHint;
			const converted = await convertBinaryPayload(binary.buffer, ext, finalUrl, signal);
			if (converted.ok && converted.content.trim().length > 50) {
				notes.push("Converted with parser registry");
				const output = finalizeOutput(converted.content);
				return {
					url,
					finalUrl,
					contentType: mime,
					method: "markit",
					content: output.content,
					fetchedAt,
					truncated: output.truncated,
					notes,
				};
			}
			if (!converted.ok && converted.error) {
				notes.push(`Conversion failed: ${converted.error}`);
			} else if (converted.ok) {
				notes.push("Conversion produced no usable output");
			}
		} else if (binary.error) {
			notes.push(`Binary fetch failed: ${binary.error}`);
		} else {
			notes.push("Binary fetch failed");
		}
	}

	// Step 3b: Binary notice for binary content (non-convertible or conversion-failed)
	const looksBinary = sampleLooksBinary(rawContent) || BINARY_MIMES.has(mime);
	if (looksBinary) {
		const output = finalizeOutput(buildBinaryNotice(finalUrl, mime));
		return {
			url,
			finalUrl,
			contentType: mime || "application/octet-stream",
			method: "binary",
			content: output.content,
			fetchedAt,
			truncated: output.truncated,
			notes,
		};
	}

	// Step 4: Handle non-HTML text content
	const isHtml = mime.includes("html") || mime.includes("xhtml");
	const isJson = mime.includes("json");
	const isXml = mime.includes("xml") && !isHtml;
	const isText = mime.includes("text/plain") || mime.includes("text/markdown");
	const isFeed = mime.includes("rss") || mime.includes("atom") || mime.includes("feed");

	// Raw mode: return the response body verbatim
	if (raw) {
		const output = finalizeOutput(rawContent);
		return {
			url,
			finalUrl,
			contentType: mime,
			method: "raw",
			content: output.content,
			fetchedAt,
			truncated: output.truncated,
			notes,
		};
	}

	if (isJson) {
		const output = finalizeOutput(formatJson(rawContent));
		return {
			url,
			finalUrl,
			contentType: mime,
			method: "json",
			content: output.content,
			fetchedAt,
			truncated: output.truncated,
			notes,
		};
	}

	if (isFeed || (isXml && (rawContent.includes("<rss") || rawContent.includes("<feed")))) {
		const parsed = parseFeedToMarkdown(rawContent);
		const output = finalizeOutput(parsed);
		return {
			url,
			finalUrl,
			contentType: mime,
			method: "feed",
			content: output.content,
			fetchedAt,
			truncated: output.truncated,
			notes,
		};
	}

	if (isText && !looksLikeHtml(rawContent)) {
		const output = finalizeOutput(rawContent);
		return {
			url,
			finalUrl,
			contentType: mime,
			method: "text",
			content: output.content,
			fetchedAt,
			truncated: output.truncated,
			notes,
		};
	}

	// Step 5: For HTML, try digestible formats first
	if (isHtml && !raw) {
		// 5A: Check for page-specific markdown alternate
		const alternates = parseAlternateLinks(rawContent, finalUrl);
		const markdownAlt = alternates.find((alt) => alt.endsWith(".md") || alt.includes("markdown"));
		if (markdownAlt) {
			const resolved = markdownAlt.startsWith("http") ? markdownAlt : new URL(markdownAlt, finalUrl).href;
			const altResult = await load(resolved);
			if (altResult.ok && altResult.content.trim().length > 100 && !looksLikeHtml(altResult.content)) {
				notes.push(`Used markdown alternate: ${resolved}`);
				const output = finalizeOutput(altResult.content);
				return {
					url,
					finalUrl,
					contentType: "text/markdown",
					method: "alternate-markdown",
					content: output.content,
					fetchedAt,
					truncated: output.truncated,
					notes,
				};
			}
		}

		// 5B: Try URL.md suffix
		const mdSuffix = await tryMdSuffix(finalUrl, timeout, signal, fetchImpl);
		if (mdSuffix) {
			notes.push("Found .md suffix version");
			const output = finalizeOutput(mdSuffix);
			return {
				url,
				finalUrl,
				contentType: "text/markdown",
				method: "md-suffix",
				content: output.content,
				fetchedAt,
				truncated: output.truncated,
				notes,
			};
		}

		// 5C: Content negotiation
		const negotiated = await tryContentNegotiation(url, timeout, signal, fetchImpl);
		if (negotiated) {
			notes.push(`Content negotiation returned ${negotiated.type}`);
			const output = finalizeOutput(negotiated.content);
			return {
				url,
				finalUrl,
				contentType: normalizeMime(negotiated.type),
				method: "content-negotiation",
				content: output.content,
				fetchedAt,
				truncated: output.truncated,
				notes,
			};
		}

		// 5D: Check for feed alternates
		const feedAlternates = alternates.filter((alt) => !alt.endsWith(".md") && !alt.includes("markdown"));
		for (const altUrl of feedAlternates.slice(0, 2)) {
			const resolved = altUrl.startsWith("http") ? altUrl : new URL(altUrl, finalUrl).href;
			const altResult = await load(resolved);
			if (altResult.ok && altResult.content.trim().length > 200) {
				notes.push(`Used feed alternate: ${resolved}`);
				const parsed = parseFeedToMarkdown(altResult.content);
				const output = finalizeOutput(parsed);
				return {
					url,
					finalUrl,
					contentType: "application/feed",
					method: "alternate-feed",
					content: output.content,
					fetchedAt,
					truncated: output.truncated,
					notes,
				};
			}
		}

		if (signal?.aborted) {
			throw new Error("aborted");
		}

		// 5E: Render HTML via the reader-backend chain
		const htmlResult = await renderHtmlToText(finalUrl, rawContent, {
			timeoutSeconds: timeout,
			signal,
			fetch: fetchImpl,
			firecrawlApiKey,
			jinaApiKey,
		});
		if (!htmlResult.ok) {
			notes.push("html rendering failed (no reader backend produced usable output)");

			const llmResult = await tryLlmEndpoints(finalUrl, timeout, signal, fetchImpl);
			if (llmResult) {
				notes.push(`Used llms.txt fallback: ${llmResult.endpoint}`);
				const output = finalizeOutput(llmResult.content);
				return {
					url,
					finalUrl,
					contentType: "text/plain",
					method: "llms.txt",
					content: output.content,
					fetchedAt,
					truncated: output.truncated,
					notes,
				};
			}

			const output = finalizeOutput(rawContent);
			return {
				url,
				finalUrl,
				contentType: mime,
				method: "raw-html",
				content: output.content,
				fetchedAt,
				truncated: output.truncated,
				notes,
			};
		}

		// Step 6: If rendered output is low quality, try more targeted fallbacks
		if (isLowQualityOutput(htmlResult.content)) {
			const docLinks = extractDocumentLinks(rawContent, finalUrl);
			if (docLinks.length > 0) {
				const docUrl = docLinks[0];
				const binary = await fetchBinary(docUrl, timeout, signal, fetchImpl);
				if (binary.ok) {
					const ext = getExtensionHint(docUrl, binary.contentDisposition);
					if (isConvertible("", ext)) {
						const converted = await convertBinaryPayload(binary.buffer, ext, docUrl, signal);
						if (converted.ok && converted.content.trim().length > htmlResult.content.length) {
							notes.push(`Extracted and converted document: ${docUrl}`);
							const output = finalizeOutput(converted.content);
							return {
								url,
								finalUrl,
								contentType: "application/document",
								method: "extracted-document",
								content: output.content,
								fetchedAt,
								truncated: output.truncated,
								notes,
							};
						}
						if (!converted.ok && converted.error) {
							notes.push(`Conversion failed: ${converted.error}`);
						}
					}
				} else if (binary.error) {
					notes.push(`Binary fetch failed: ${binary.error}`);
				}
			}

			const llmResult = await tryLlmEndpoints(finalUrl, timeout, signal, fetchImpl);
			if (llmResult) {
				notes.push(`Used llms.txt fallback: ${llmResult.endpoint}`);
				const output = finalizeOutput(llmResult.content);
				return {
					url,
					finalUrl,
					contentType: "text/plain",
					method: "llms.txt",
					content: output.content,
					fetchedAt,
					truncated: output.truncated,
					notes,
				};
			}

			notes.push("Page appears to require JavaScript or is mostly navigation");
		}

		const output = finalizeOutput(htmlResult.content);
		return {
			url,
			finalUrl,
			contentType: mime,
			method: htmlResult.method,
			content: output.content,
			fetchedAt,
			truncated: output.truncated,
			notes,
		};
	}

	// Fallback: return raw content
	const output = finalizeOutput(rawContent);
	return {
		url,
		finalUrl,
		contentType: mime,
		method: "raw",
		content: output.content,
		fetchedAt,
		truncated: output.truncated,
		notes,
	};
}
