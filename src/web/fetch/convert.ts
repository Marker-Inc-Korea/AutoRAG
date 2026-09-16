/**
 * Binary download and conversion — adapted from oh-my-pi's
 * `packages/coding-agent/src/web/scrapers/utils.ts` (MIT licensed).
 *
 * `convertBinaryPayload` uses AutoRAG's parser registry for PDF/DOCX/PPTX/XLSX.
 * EPUB and other unsupported extensions fall through to the binary-notice
 * fallback, since AutoRAG's default registry has no EPUB parser. This is a
 * deliberate deviation from oh-my-pi (which uses markit for all convertible
 * types including EPUB) — see the report for rationale.
 */

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDefaultParserRegistry, type ParseInput, type ParseOutput, type Parser } from "../../parser/index.ts";
import { MAX_BYTES } from "./page-loader.ts";
import { buildBinaryNotice } from "./url-target.ts";

export interface BinaryFetchSuccess {
	ok: true;
	buffer: Uint8Array;
	contentDisposition?: string;
}

export type BinaryFetchResult = BinaryFetchSuccess | { ok: false; error?: string };

function combineSignals(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
	if (!signal) return AbortSignal.timeout(timeoutMs);
	return AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
}

async function readResponseWithLimit(response: Response, maxBytes: number, signal?: AbortSignal): Promise<Uint8Array> {
	const reader = response.body?.getReader();
	if (!reader) return new Uint8Array(0);

	const chunks: Buffer[] = [];
	let totalBytes = 0;

	try {
		while (true) {
			if (signal?.aborted) {
				await reader.cancel();
				throw new Error("aborted");
			}
			const { done, value } = await reader.read();
			if (done) break;
			if (!value || value.byteLength === 0) continue;

			totalBytes += value.byteLength;
			if (totalBytes > maxBytes) {
				await reader.cancel();
				throw new Error(`response exceeds ${maxBytes} bytes`);
			}

			chunks.push(Buffer.from(value));
		}
	} finally {
		reader.releaseLock();
	}

	return new Uint8Array(Buffer.concat(chunks, totalBytes));
}

/**
 * Fetch binary content from a URL with a size limit.
 */
export async function fetchBinary(
	url: string,
	timeout: number = 20,
	signal?: AbortSignal,
	fetchImpl?: typeof fetch,
): Promise<BinaryFetchResult> {
	const fetchFn = fetchImpl ?? fetch;
	const requestSignal = combineSignals(signal, timeout * 1000);
	try {
		const response = await fetchFn(url, {
			signal: requestSignal,
			headers: { "User-Agent": "Mozilla/5.0 (compatible; TextBot/1.0)" },
			redirect: "follow",
		});

		if (!response.ok) {
			return { ok: false, error: `HTTP ${response.status}` };
		}

		const contentDisposition = response.headers.get("content-disposition") ?? undefined;
		const contentLength = response.headers.get("content-length");
		if (contentLength) {
			const size = Number.parseInt(contentLength, 10);
			if (Number.isFinite(size) && size > MAX_BYTES) {
				return { ok: false, error: `content-length ${size} exceeds ${MAX_BYTES}` };
			}
		}
		const buffer = await readResponseWithLimit(response, MAX_BYTES, requestSignal);
		return { ok: true, buffer, contentDisposition };
	} catch (err) {
		if (signal?.aborted) throw new Error("aborted");
		if (requestSignal?.aborted) return { ok: false, error: "aborted" };
		return { ok: false, error: err instanceof Error ? err.message : "Failed to fetch binary" };
	}
}

let parserRegistryPromise: Promise<Parser[]> | undefined;

async function getConverters(): Promise<Parser[]> {
	parserRegistryPromise ??= createDefaultParserRegistry().list();
	return parserRegistryPromise;
}

export interface ConvertResult {
	content: string;
	ok: boolean;
	error?: string;
}

/**
 * Convert a binary payload to markdown using AutoRAG's parser registry.
 * Falls back to the binary-notice string when no parser handles the extension.
 */
export async function convertBinaryPayload(
	buffer: Uint8Array,
	extension: string,
	_finalUrl: string,
	signal?: AbortSignal,
): Promise<ConvertResult> {
	const ext = extension.toLowerCase().startsWith(".") ? extension.toLowerCase() : `.${extension.toLowerCase()}`;

	const parsers = await getConverters();
	const parser = parsers.find((p) => p.extensions.includes(ext));
	if (!parser) {
		return { content: "", ok: false, error: `No parser registered for ${ext}` };
	}

	const tempDir = await mkdtemp(join(tmpdir(), "autorag-fetch-"));
	const tempPath = join(tempDir, `payload${ext}`);
	try {
		await writeFile(tempPath, buffer);
		const input: ParseInput = {
			virtualPath: `payload${ext}`,
			sourcePath: tempPath,
			bytes: buffer,
		};
		signal?.throwIfAborted();
		const output: ParseOutput = await parser.parse(input);
		return { content: output.markdown, ok: true };
	} catch (err) {
		signal?.throwIfAborted();
		return {
			content: "",
			ok: false,
			error: err instanceof Error ? err.message : "Conversion failed",
		};
	} finally {
		await rm(tempDir, { recursive: true, force: true });
	}
}

export { buildBinaryNotice };
