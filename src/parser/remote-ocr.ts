import { createHash } from "node:crypto";
import { normalizeMarkdown } from "./text.ts";
import type { VisualExtractInput, VisualExtractResult, VisualPageExtractor } from "./visual-extractor.ts";

/**
 * Official dedicated remote OCR / document-parsing providers supported natively via their provider APIs.
 *
 * Unlike general chat LLMs, these are purpose-built OCR / document parsing APIs:
 * - `mistral`: Mistral OCR (`POST /v1/ocr`) — `mistral-ocr-latest`, `mistral-ocr-2512`
 * - `nvidia`: NVIDIA NIM Nemotron Parse (`POST /v1/chat/completions` with `markdown_no_bbox` tool) — `nvidia/nemotron-parse`
 * - `upstage`: Upstage Document Parse (`POST /v1/document-digitization` multipart) — `document-parse`, `ocr`
 * - `zai`: Z.AI GLM-OCR Layout Parsing (`POST /api/paas/v4/layout_parsing`) — `glm-ocr`
 */
export type RemoteOcrProviderId = "mistral" | "nvidia" | "upstage" | "zai";

export interface OfficialRemoteOcrModelSpec {
	readonly provider: RemoteOcrProviderId;
	readonly id: string;
	readonly label: string;
	readonly defaultBaseUrl: string;
	readonly defaultApiKeyEnv: string;
}

export const OFFICIAL_REMOTE_OCR_MODELS: Readonly<Record<RemoteOcrProviderId, readonly OfficialRemoteOcrModelSpec[]>> =
	{
		mistral: [
			{
				provider: "mistral",
				id: "mistral-ocr-latest",
				label: "Mistral OCR (latest)",
				defaultBaseUrl: "https://api.mistral.ai",
				defaultApiKeyEnv: "MISTRAL_API_KEY",
			},
			{
				provider: "mistral",
				id: "mistral-ocr-2512",
				label: "Mistral OCR 3 (2512)",
				defaultBaseUrl: "https://api.mistral.ai",
				defaultApiKeyEnv: "MISTRAL_API_KEY",
			},
		],
		nvidia: [
			{
				provider: "nvidia",
				id: "nvidia/nemotron-parse",
				label: "NVIDIA Nemotron Parse",
				defaultBaseUrl: "https://integrate.api.nvidia.com/v1",
				defaultApiKeyEnv: "NVIDIA_API_KEY",
			},
		],
		upstage: [
			{
				provider: "upstage",
				id: "document-parse",
				label: "Upstage Document Parse",
				defaultBaseUrl: "https://api.upstage.ai",
				defaultApiKeyEnv: "UPSTAGE_API_KEY",
			},
			{
				provider: "upstage",
				id: "ocr",
				label: "Upstage Document OCR",
				defaultBaseUrl: "https://api.upstage.ai",
				defaultApiKeyEnv: "UPSTAGE_API_KEY",
			},
		],
		zai: [
			{
				provider: "zai",
				id: "glm-ocr",
				label: "GLM-OCR (Z.AI)",
				defaultBaseUrl: "https://api.z.ai",
				defaultApiKeyEnv: "ZAI_API_KEY",
			},
		],
	};

export function isOfficialRemoteOcrProvider(provider: string): provider is RemoteOcrProviderId {
	return Object.hasOwn(OFFICIAL_REMOTE_OCR_MODELS, provider);
}

export function officialRemoteOcrModelsFor(provider: string): readonly OfficialRemoteOcrModelSpec[] {
	return isOfficialRemoteOcrProvider(provider) ? OFFICIAL_REMOTE_OCR_MODELS[provider] : [];
}

export function findOfficialRemoteOcrModel(provider: string, modelId: string): OfficialRemoteOcrModelSpec | undefined {
	return officialRemoteOcrModelsFor(provider).find((entry) => entry.id === modelId);
}

export interface RemoteApiOcrExtractorOptions {
	readonly provider: RemoteOcrProviderId;
	readonly model: string;
	readonly apiKey: string;
	readonly baseUrl?: string;
	readonly timeoutMs?: number;
	readonly maxRetries?: number;
	readonly retryBaseMs?: number;
	readonly fetch?: typeof globalThis.fetch;
	readonly sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

const REMOTE_API_OCR_VERSION = "v1-api";
const DEFAULT_REMOTE_OCR_TIMEOUT_MS = 60_000;
const DEFAULT_REMOTE_OCR_MAX_RETRIES = 2;
const DEFAULT_REMOTE_OCR_RETRY_BASE_MS = 750;

function isRetryableStatus(status: number): boolean {
	return status === 429 || status === 500 || status === 502 || status === 503 || status === 504 || status === 529;
}

function isRetryableError(error: unknown): boolean {
	const message = error instanceof Error ? error.message : String(error);
	return /\b(429|500|502|503|504|529)\b|rate[\s-]?limit|too many requests|overloaded|temporar|econnreset|etimedout|fetch failed/iu.test(
		message,
	);
}

function defaultRetrySleep(ms: number, signal?: AbortSignal): Promise<void> {
	if (ms <= 0) return Promise.resolve();
	return new Promise<void>((resolve, reject) => {
		if (signal?.aborted) {
			reject(signal.reason ?? new Error("Aborted"));
			return;
		}
		const onAbort = () => {
			clearTimeout(timer);
			reject(signal?.reason ?? new Error("Aborted"));
		};
		const timer = setTimeout(() => {
			signal?.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

function cleanMarkdown(raw: string): string {
	return normalizeMarkdown(
		raw
			.replace(/\r\n?/gu, "\n")
			.split("\n")
			.map((line) => line.trimEnd())
			.join("\n")
			.trim(),
	);
}

function trimTrailingSlash(url: string): string {
	let end = url.length;
	while (end > 0 && url.charCodeAt(end - 1) === 47) end -= 1;
	return end === url.length ? url : url.slice(0, end);
}

function resolveMistralEndpoint(baseUrl?: string): string {
	const root = trimTrailingSlash(baseUrl ?? "https://api.mistral.ai");
	if (root.endsWith("/v1/ocr")) return root;
	if (root.endsWith("/v1")) return `${root}/ocr`;
	return `${root}/v1/ocr`;
}

function resolveNvidiaEndpoint(baseUrl?: string): string {
	const root = trimTrailingSlash(baseUrl ?? "https://integrate.api.nvidia.com/v1");
	if (root.endsWith("/chat/completions")) return root;
	if (root.endsWith("/v1")) return `${root}/chat/completions`;
	return `${root}/v1/chat/completions`;
}

function resolveUpstageEndpoint(baseUrl?: string): string {
	const root = trimTrailingSlash(baseUrl ?? "https://api.upstage.ai");
	if (root.endsWith("/v1/document-digitization")) return root;
	if (root.endsWith("/v1")) return `${root}/document-digitization`;
	return `${root}/v1/document-digitization`;
}

function resolveZaiEndpoint(baseUrl?: string): string {
	const root = trimTrailingSlash(baseUrl ?? "https://api.z.ai");
	if (root.endsWith("/layout_parsing")) return root;
	// pi-ai's zai provider baseUrl is https://api.z.ai/api/coding/paas/v4 — layout_parsing lives on /api/paas/v4/layout_parsing.
	const origin = new URL(root).origin;
	return `${origin}/api/paas/v4/layout_parsing`;
}

function extensionForMime(mimeType: VisualExtractInput["image"]["mimeType"]): string {
	if (mimeType === "image/jpeg") return "jpg";
	if (mimeType === "image/webp") return "webp";
	return "png";
}

function extractMistralMarkdown(payload: unknown): string {
	if (typeof payload !== "object" || payload === null) return "";
	const pages = (payload as { pages?: unknown }).pages;
	if (!Array.isArray(pages)) return "";
	return pages
		.map((page) =>
			typeof page === "object" && page !== null && typeof (page as { markdown?: unknown }).markdown === "string"
				? (page as { markdown: string }).markdown
				: "",
		)
		.filter((text) => text.trim().length > 0)
		.join("\n\n");
}

function extractNemotronMarkdown(payload: unknown): string {
	if (typeof payload !== "object" || payload === null) return "";
	const choices = (payload as { choices?: unknown }).choices;
	if (!Array.isArray(choices) || choices.length === 0) return "";
	const message = (choices[0] as { message?: unknown })?.message;
	if (typeof message !== "object" || message === null) return "";
	const msg = message as { content?: unknown; tool_calls?: unknown };
	const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
	if (toolCalls.length > 0) {
		const parts: string[] = [];
		for (const call of toolCalls) {
			const argsRaw = (call as { function?: { arguments?: unknown } })?.function?.arguments;
			if (typeof argsRaw !== "string") continue;
			try {
				const parsed: unknown = JSON.parse(argsRaw);
				if (Array.isArray(parsed)) {
					for (const item of parsed) {
						if (
							typeof item === "object" &&
							item !== null &&
							typeof (item as { text?: unknown }).text === "string"
						) {
							const text = (item as { text: string }).text.trim();
							if (text.length > 0) parts.push(text);
						} else if (typeof item === "string" && item.trim().length > 0) {
							parts.push(item.trim());
						}
					}
				} else if (typeof parsed === "object" && parsed !== null) {
					const text =
						(parsed as { text?: unknown; markdown?: unknown }).text ??
						(parsed as { markdown?: unknown }).markdown;
					if (typeof text === "string" && text.trim().length > 0) parts.push(text.trim());
				} else if (typeof parsed === "string" && parsed.trim().length > 0) {
					parts.push(parsed.trim());
				}
			} catch {
				if (argsRaw.trim().length > 0) parts.push(argsRaw.trim());
			}
		}
		if (parts.length > 0) return parts.join("\n\n");
	}
	return typeof msg.content === "string" ? msg.content : "";
}

function extractUpstageMarkdown(payload: unknown): string {
	if (typeof payload !== "object" || payload === null) return "";
	const record = payload as {
		content?: { markdown?: unknown; text?: unknown; html?: unknown };
		text?: unknown;
		elements?: unknown;
	};
	if (typeof record.content?.markdown === "string" && record.content.markdown.trim().length > 0) {
		return record.content.markdown;
	}
	if (typeof record.content?.text === "string" && record.content.text.trim().length > 0) {
		return record.content.text;
	}
	if (typeof record.text === "string" && record.text.trim().length > 0) {
		return record.text;
	}
	if (Array.isArray(record.elements)) {
		const parts = record.elements
			.map((el) => {
				if (typeof el !== "object" || el === null) return "";
				const content = (el as { content?: { markdown?: unknown; text?: unknown }; text?: unknown }).content;
				if (typeof content?.markdown === "string") return content.markdown;
				if (typeof content?.text === "string") return content.text;
				if (typeof (el as { text?: unknown }).text === "string") return (el as { text: string }).text;
				return "";
			})
			.filter((text) => text.trim().length > 0);
		if (parts.length > 0) return parts.join("\n\n");
	}
	return typeof record.content?.html === "string" ? record.content.html : "";
}

function extractZaiMarkdown(payload: unknown): string {
	if (typeof payload !== "object" || payload === null) return "";
	const record = payload as { md_results?: unknown; layout_details?: unknown };
	if (typeof record.md_results === "string" && record.md_results.trim().length > 0) {
		return record.md_results;
	}
	if (Array.isArray(record.layout_details)) {
		const parts: string[] = [];
		for (const page of record.layout_details) {
			if (!Array.isArray(page)) continue;
			for (const block of page) {
				if (
					typeof block === "object" &&
					block !== null &&
					typeof (block as { content?: unknown }).content === "string"
				) {
					const text = (block as { content: string }).content.trim();
					if (text.length > 0) parts.push(text);
				}
			}
		}
		return parts.join("\n\n");
	}
	return "";
}

async function performRemoteOcrRequest(
	options: RemoteApiOcrExtractorOptions,
	input: VisualExtractInput,
	fetchImpl: typeof globalThis.fetch,
	signal: AbortSignal,
): Promise<string> {
	const base64 = Buffer.from(
		input.image.data.buffer,
		input.image.data.byteOffset,
		input.image.data.byteLength,
	).toString("base64");
	const dataUrl = `data:${input.image.mimeType};base64,${base64}`;

	let url: string;
	let init: RequestInit;

	if (options.provider === "mistral") {
		url = resolveMistralEndpoint(options.baseUrl);
		init = {
			method: "POST",
			headers: {
				authorization: `Bearer ${options.apiKey}`,
				"content-type": "application/json",
				accept: "application/json",
			},
			body: JSON.stringify({
				model: options.model,
				document: {
					type: "image_url",
					image_url: dataUrl,
				},
			}),
			signal,
		};
	} else if (options.provider === "nvidia") {
		url = resolveNvidiaEndpoint(options.baseUrl);
		init = {
			method: "POST",
			headers: {
				authorization: `Bearer ${options.apiKey}`,
				"content-type": "application/json",
				accept: "application/json",
			},
			body: JSON.stringify({
				model: options.model,
				messages: [
					{
						role: "user",
						content: [
							{
								type: "image_url",
								image_url: { url: dataUrl },
							},
						],
					},
				],
				tools: [{ type: "function", function: { name: "markdown_no_bbox" } }],
				tool_choice: { type: "function", function: { name: "markdown_no_bbox" } },
				temperature: 0,
				max_tokens: 4096,
			}),
			signal,
		};
	} else if (options.provider === "upstage") {
		url = resolveUpstageEndpoint(options.baseUrl);
		const form = new FormData();
		const ext = extensionForMime(input.image.mimeType);
		const bytes = new Uint8Array(input.image.data.byteLength);
		bytes.set(input.image.data);
		form.append("document", new Blob([bytes], { type: input.image.mimeType }), `page-${input.page}.${ext}`);
		form.append("model", options.model);
		if (options.model !== "ocr") {
			form.append("ocr", "force");
			form.append("output_formats", '["markdown"]');
		}
		init = {
			method: "POST",
			headers: {
				authorization: `Bearer ${options.apiKey}`,
				accept: "application/json",
			},
			body: form,
			signal,
		};
	} else {
		url = resolveZaiEndpoint(options.baseUrl);
		init = {
			method: "POST",
			headers: {
				authorization: `Bearer ${options.apiKey}`,
				"content-type": "application/json",
				accept: "application/json",
			},
			body: JSON.stringify({
				model: options.model,
				file: dataUrl,
			}),
			signal,
		};
	}

	const response = await fetchImpl(url, init);
	if (!response.ok) {
		const bodyText = await response.text().catch(() => "");
		const detail = bodyText.trim().slice(0, 300);
		const err = new Error(
			`Remote OCR (${options.provider}/${options.model}) HTTP ${response.status}${detail.length > 0 ? `: ${detail}` : ""}`,
		);
		(err as { status?: number }).status = response.status;
		throw err;
	}
	const payload: unknown = await response.json();
	if (options.provider === "mistral") return extractMistralMarkdown(payload);
	if (options.provider === "nvidia") return extractNemotronMarkdown(payload);
	if (options.provider === "upstage") return extractUpstageMarkdown(payload);
	return extractZaiMarkdown(payload);
}

/**
 * Create a dedicated remote OCR API extractor for official provider OCR / document-parsing endpoints
 * (Mistral OCR, NVIDIA Nemotron Parse, Upstage Document Parse, Z.AI GLM-OCR).
 */
export function createRemoteApiOcrExtractor(options: RemoteApiOcrExtractorOptions): VisualPageExtractor {
	const modelId = `${options.provider}/${options.model}`;
	const timeoutMs = options.timeoutMs ?? DEFAULT_REMOTE_OCR_TIMEOUT_MS;
	const maxRetries = Math.max(0, Math.floor(options.maxRetries ?? DEFAULT_REMOTE_OCR_MAX_RETRIES));
	const retryBaseMs = Math.max(0, options.retryBaseMs ?? DEFAULT_REMOTE_OCR_RETRY_BASE_MS);
	const fetchImpl = options.fetch ?? globalThis.fetch;
	const sleepFn = options.sleep ?? defaultRetrySleep;
	const baseSuffix = options.baseUrl
		? `:${createHash("sha256").update(options.baseUrl).digest("hex").slice(0, 8)}`
		: "";

	return {
		kind: "ocr",
		modelId,
		remote: true,
		fingerprint: `ocr:${REMOTE_API_OCR_VERSION}:${modelId}${baseSuffix}`,
		async extract(input: VisualExtractInput): Promise<VisualExtractResult> {
			for (let attempt = 0; ; attempt += 1) {
				if (input.signal?.aborted) throw input.signal.reason ?? new Error("Aborted");
				const controller = new AbortController();
				const onAbort = () => controller.abort(input.signal?.reason);
				input.signal?.addEventListener("abort", onAbort, { once: true });
				const timer = setTimeout(
					() => controller.abort(new Error(`Remote OCR (${modelId}) timed out after ${timeoutMs}ms`)),
					timeoutMs,
				);
				try {
					const raw = await performRemoteOcrRequest(options, input, fetchImpl, controller.signal);
					return {
						markdown: cleanMarkdown(raw),
						extractorKind: "ocr",
						extractorModel: modelId,
					};
				} catch (error) {
					const status = (error as { status?: number })?.status;
					const retryable = status !== undefined ? isRetryableStatus(status) : isRetryableError(error);
					if (input.signal?.aborted || attempt >= maxRetries || !retryable) {
						throw error;
					}
					await sleepFn(retryBaseMs * 2 ** attempt, input.signal);
				} finally {
					clearTimeout(timer);
					input.signal?.removeEventListener("abort", onAbort);
				}
			}
		},
	};
}
