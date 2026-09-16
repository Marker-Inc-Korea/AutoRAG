/**
 * URL fetch/render pipeline — self-contained module backing the future
 * `web_fetch` agent tool. Ported from oh-my-pi (MIT licensed).
 */

export { type BinaryFetchResult, type ConvertResult, convertBinaryPayload, fetchBinary } from "./convert.ts";
export {
	cleanFeedText,
	extractDocumentLinks,
	extractHeadHtml,
	formatJson,
	getHtmlAttribute,
	parseAlternateLinks,
	parseFeedToMarkdown,
} from "./feeds.ts";
export {
	type FetchProvider,
	htmlToBasicMarkdown,
	isLowQualityOutput,
	parseJinaReaderContent,
	type RenderHtmlToTextOptions,
	renderHtmlToText,
} from "./html-renderer.ts";
export { buildLlmEndpointCandidates, tryContentNegotiation, tryLlmEndpoints, tryMdSuffix } from "./negotiate.ts";
export type { LoadPageOptions, LoadPageResult, RenderResult } from "./page-loader.ts";
export {
	decodeHtmlEntities,
	finalizeOutput,
	loadPage,
	looksLikeHtml,
	MAX_BYTES,
	MAX_OUTPUT_CHARS,
} from "./page-loader.ts";
export { type RenderUrlOptions, renderUrl } from "./render.ts";
export {
	buildBinaryNotice,
	getExtensionHint,
	getFilenameExtensionHint,
	isConvertible,
	normalizeMime,
	normalizeUrl,
	sampleLooksBinary,
} from "./url-target.ts";
