/**
 * `web_fetch` agent tool — read a public web page as text/markdown.
 *
 * LLM-facing wrapper around the oh-my-pi-style render pipeline in
 * `src/web/fetch/`: content-negotiation (markdown alternates, `.md` suffix,
 * llms.txt), feed/JSON/text handling, and HTML rendered to markdown through
 * the reader-backend chain. Output is head-truncated like oh-my-pi's
 * read-url integration so one page cannot flood the context.
 */
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { renderUrl } from "../web/fetch/render.ts";

export const WEB_FETCH_TOOL_NAME = "web_fetch";

/** Mirror of oh-my-pi's read-url output bounds. */
const FETCH_DEFAULT_MAX_LINES = 300;
const FETCH_MAX_OUTPUT_BYTES = 50 * 1024;

const webFetchSchema = Type.Object({
	url: Type.String({ description: "The http(s) URL to fetch and read." }),
	raw: Type.Optional(
		Type.Boolean({ description: "Return the raw response body without markdown rendering or formatting." }),
	),
});

export interface WebFetchToolOptions {
	/** Total fetch/render timeout in seconds (default 30). */
	readonly timeoutSeconds?: number;
}

export interface WebFetchToolDetails {
	readonly method: "web_fetch";
	readonly url: string;
	readonly finalUrl: string;
	readonly contentType: string;
	readonly fetchMethod: string;
	readonly truncated: boolean;
	readonly available: boolean;
	readonly error?: string;
}

interface HeadTruncation {
	content: string;
	truncated: boolean;
}

/** Head-truncate tool output by lines and bytes, keeping the leading context. */
function truncateHead(output: string, maxBytes: number, maxLines: number): HeadTruncation {
	const lines = output.split("\n");
	let content = lines.length > maxLines ? `${lines.slice(0, maxLines).join("\n")}\n` : output;
	let truncated = lines.length > maxLines;
	if (Buffer.byteLength(content, "utf-8") > maxBytes) {
		const buf = Buffer.from(content, "utf-8");
		content = buf.subarray(0, maxBytes).toString("utf-8");
		truncated = true;
	}
	if (truncated) {
		content += `\n[... truncated: showing the first ${maxLines} lines / ${Math.floor(maxBytes / 1024)}KB. Fetch a more specific URL or use raw mode for the full body ...]`;
	}
	return { content, truncated };
}

export function createWebFetchTool(
	options: WebFetchToolOptions = {},
): AgentTool<typeof webFetchSchema, WebFetchToolDetails> {
	return {
		name: WEB_FETCH_TOOL_NAME,
		label: "Web Fetch",
		description:
			"Fetch and read a public web page as markdown/text. Use it to read specific URLs — pages found via web_search, official docs, papers, GitHub files. Handles HTML→markdown rendering, feeds, JSON, and content negotiation. Only http(s) URLs: never local file paths or datasource virtual ids.",
		parameters: webFetchSchema,
		async execute(_toolCallId, params, signal): Promise<AgentToolResult<WebFetchToolDetails>> {
			const url = params.url.trim();
			if (!/^https?:\/\//i.test(url)) {
				return {
					content: [
						{
							type: "text",
							text: `web_fetch only reads http(s) URLs; received "${url}". Local files belong to bash and datasource virtual ids to their dedicated datasource search tools.`,
						},
					],
					details: {
						method: WEB_FETCH_TOOL_NAME,
						url,
						finalUrl: url,
						contentType: "unknown",
						fetchMethod: "rejected",
						truncated: false,
						available: false,
						error: "non-http URL",
					},
				};
			}
			try {
				const rendered = await renderUrl(url, {
					timeoutSeconds: options.timeoutSeconds,
					raw: params.raw,
					signal,
				});
				const header =
					`URL: ${rendered.finalUrl}\n` +
					`Content-Type: ${rendered.contentType}\n` +
					`Method: ${rendered.method}\n` +
					(rendered.notes.length > 0 ? `Notes: ${rendered.notes.join("; ")}\n` : "") +
					`\n---\n\n`;
				const truncation = truncateHead(header + rendered.content, FETCH_MAX_OUTPUT_BYTES, FETCH_DEFAULT_MAX_LINES);
				return {
					content: [{ type: "text", text: truncation.content }],
					details: {
						method: WEB_FETCH_TOOL_NAME,
						url,
						finalUrl: rendered.finalUrl,
						contentType: rendered.contentType,
						fetchMethod: rendered.method,
						truncated: rendered.truncated || truncation.truncated,
						available: rendered.method !== "failed",
					},
				};
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				return {
					content: [{ type: "text", text: `web_fetch failed for ${url}: ${message}` }],
					details: {
						method: WEB_FETCH_TOOL_NAME,
						url,
						finalUrl: url,
						contentType: "unknown",
						fetchMethod: "failed",
						truncated: false,
						available: false,
						error: message,
					},
				};
			}
		},
	};
}
